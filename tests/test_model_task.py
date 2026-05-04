from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from agents import Agent, RunConfig, Runner, function_tool
from agents.model_task import ask_model
from agents.run_config import ModelInputData
from tests.fake_model import FakeModel
from tests.test_responses import get_text_message


class Decision(BaseModel):
    answer: str
    rationale: str


@function_tool
def unsafe_tool() -> str:
    """A tool that should not be visible to direct model tasks."""
    return "tool result"


@pytest.mark.asyncio
async def test_ask_model_makes_single_model_call_without_tools_or_handoffs() -> None:
    model = FakeModel()
    model.set_next_output([get_text_message("approved")])
    delegate = Agent(name="delegate", model=FakeModel())
    agent = Agent(
        name="reviewer",
        instructions="Review the request.",
        model=model,
        tools=[unsafe_tool],
        handoffs=[delegate],
    )

    result = await Runner.ask_model(agent, "Review this request.")

    assert result.output_text == "approved"
    assert result.final_output == "approved"
    assert model.get_next_output() == []
    assert model.last_turn_args["tools"] == []
    assert model.last_turn_args["handoffs"] == []
    assert model.last_turn_args["system_instructions"] == "Review the request."
    assert model.last_turn_args["input"] == [{"content": "Review this request.", "role": "user"}]


@pytest.mark.asyncio
async def test_ask_model_applies_call_model_input_filter() -> None:
    model = FakeModel()
    model.set_next_output([get_text_message("filtered")])
    agent = Agent(name="filter-test", model=model)

    def add_context(data: Any) -> ModelInputData:
        return ModelInputData(
            input=[*data.model_data.input, {"role": "user", "content": "extra context"}],
            instructions="filtered instructions",
        )

    result = await ask_model(
        agent,
        "original",
        run_config=RunConfig(call_model_input_filter=add_context),
    )

    assert result.instructions == "filtered instructions"
    assert result.model_input[-1] == {"role": "user", "content": "extra context"}
    assert model.last_turn_args["system_instructions"] == "filtered instructions"
    assert model.last_turn_args["input"][-1] == {"role": "user", "content": "extra context"}


@pytest.mark.asyncio
async def test_ask_model_parses_structured_output() -> None:
    model = FakeModel()
    model.set_next_output([get_text_message('{"answer": "approve", "rationale": "valid"}')])
    agent = Agent(name="structured", model=model, output_type=Decision)

    result = await Runner.ask_model(agent, "Decide.")

    assert isinstance(result.final_output, Decision)
    assert result.final_output.answer == "approve"
    assert result.final_output.rationale == "valid"


def test_ask_model_sync() -> None:
    model = FakeModel()
    model.set_next_output([get_text_message("sync answer")])
    agent = Agent(name="sync", model=model)

    result = Runner.ask_model_sync(agent, "one call")

    assert result.final_output == "sync answer"
    assert model.last_turn_args["input"] == [{"content": "one call", "role": "user"}]
