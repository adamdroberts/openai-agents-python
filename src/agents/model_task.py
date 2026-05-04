from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Generic

from openai.types.responses import ResponseOutputMessage

from .agent import Agent
from .agent_output import AgentOutputSchemaBase
from .items import ItemHelpers, ModelResponse, TResponseInputItem
from .lifecycle import RunHooks
from .model_settings import ModelSettings
from .run_config import RunConfig
from .run_context import RunContextWrapper, TContext
from .run_internal.agent_runner_helpers import ensure_context_wrapper
from .run_internal.items import deduplicate_input_items_preferring_latest
from .run_internal.model_retry import get_response_with_retry
from .run_internal.turn_preparation import (
    get_model,
    get_output_schema,
    maybe_filter_model_input,
    validate_run_hooks,
)
from .tool import Tool
from .tracing.model_tracing import get_model_tracing_impl
from .usage import Usage


@dataclass
class ModelTaskResult(Generic[TContext]):
    """Result from a direct model task.

    A model task is a single LLM call made without exposing tools, MCP tools, or handoffs. The
    caller keeps control of what to do with the output.
    """

    input: str | list[TResponseInputItem]
    """The caller-provided input."""

    model_input: list[TResponseInputItem]
    """The final input items sent to the model after any input filter runs."""

    instructions: str | None
    """The instructions sent to the model."""

    raw_response: ModelResponse
    """The raw model response."""

    output_text: str
    """Text extracted from the last assistant message, if present."""

    final_output: Any
    """Parsed structured output when `agent.output_type` is set, otherwise `output_text`."""

    refusal: str | None
    """Refusal text extracted from the last assistant message, if present."""

    context_wrapper: RunContextWrapper[TContext]
    """The context wrapper used for the call, including accumulated usage."""

    @property
    def usage(self) -> Usage:
        """Usage reported by the model response."""
        return self.raw_response.usage

    @property
    def last_response_id(self) -> str | None:
        """The response ID from the model response, if available."""
        return self.raw_response.response_id


def _last_message(response: ModelResponse) -> ResponseOutputMessage | None:
    for item in reversed(response.output):
        if isinstance(item, ResponseOutputMessage):
            return item
    return None


def _extract_output_text(response: ModelResponse) -> str:
    message = _last_message(response)
    if message is None:
        return ""
    return ItemHelpers.extract_text(message) or ""


def _extract_refusal(response: ModelResponse) -> str | None:
    message = _last_message(response)
    if message is None:
        return None
    return ItemHelpers.extract_refusal(message)


def _resolve_final_output(
    *,
    output_schema: AgentOutputSchemaBase | None,
    output_text: str,
) -> Any:
    if output_schema is not None and not output_schema.is_plain_text():
        if not output_text:
            return None
        return output_schema.validate_json(output_text)
    return output_text


async def ask_model(
    agent: Agent[TContext],
    input: str | list[TResponseInputItem],
    *,
    context: TContext | None = None,
    hooks: RunHooks[TContext] | None = None,
    run_config: RunConfig | None = None,
    previous_response_id: str | None = None,
    conversation_id: str | None = None,
) -> ModelTaskResult[TContext]:
    """Ask an agent's model to perform one scoped task without running the agent loop."""

    if run_config is None:
        run_config = RunConfig()
    validated_hooks = validate_run_hooks(hooks)
    context_wrapper = ensure_context_wrapper(context)
    model_input = ItemHelpers.input_to_new_input_list(input)
    context_wrapper.turn_input = list(model_input)

    system_prompt, prompt_config = await asyncio.gather(
        agent.get_system_prompt(context_wrapper),
        agent.get_prompt(context_wrapper),
    )
    output_schema = get_output_schema(agent)

    filtered = await maybe_filter_model_input(
        agent=agent,
        run_config=run_config,
        context_wrapper=context_wrapper,
        input_items=model_input,
        system_instructions=system_prompt,
    )
    if isinstance(filtered.input, list):
        filtered.input = deduplicate_input_items_preferring_latest(filtered.input)

    model = get_model(agent, run_config)
    model_settings: ModelSettings = agent.model_settings.resolve(run_config.model_settings)
    empty_tools: list[Tool] = []

    await asyncio.gather(
        validated_hooks.on_llm_start(context_wrapper, agent, filtered.instructions, filtered.input),
        (
            agent.hooks.on_llm_start(
                context_wrapper,
                agent,
                filtered.instructions,
                filtered.input,
            )
            if agent.hooks
            else asyncio.sleep(0)
        ),
    )

    raw_response = await get_response_with_retry(
        get_response=lambda: model.get_response(
            system_instructions=filtered.instructions,
            input=filtered.input,
            model_settings=model_settings,
            tools=empty_tools,
            output_schema=output_schema,
            handoffs=[],
            tracing=get_model_tracing_impl(
                run_config.tracing_disabled,
                run_config.trace_include_sensitive_data,
            ),
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt_config,
        ),
        rewind=lambda: asyncio.sleep(0),
        retry_settings=model_settings.retry,
        get_retry_advice=model.get_retry_advice,
        previous_response_id=previous_response_id,
        conversation_id=conversation_id,
    )
    context_wrapper.usage.add(raw_response.usage)

    await asyncio.gather(
        (
            agent.hooks.on_llm_end(context_wrapper, agent, raw_response)
            if agent.hooks
            else asyncio.sleep(0)
        ),
        validated_hooks.on_llm_end(context_wrapper, agent, raw_response),
    )

    output_text = _extract_output_text(raw_response)
    final_output = _resolve_final_output(
        output_schema=output_schema,
        output_text=output_text,
    )
    return ModelTaskResult(
        input=input,
        model_input=filtered.input,
        instructions=filtered.instructions,
        raw_response=raw_response,
        output_text=output_text,
        final_output=final_output,
        refusal=_extract_refusal(raw_response),
        context_wrapper=context_wrapper,
    )


__all__ = ["ModelTaskResult", "ask_model"]
