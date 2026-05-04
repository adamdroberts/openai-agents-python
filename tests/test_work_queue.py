from __future__ import annotations

import asyncio
import time

import pytest

from agents import InMemoryWorkQueue, QueuedWorkItem, RedisWorkQueue, run_work_queue_batch


def test_in_memory_work_queue_orders_by_score_and_finalizes() -> None:
    queue = InMemoryWorkQueue()
    queue.enqueue("later", {"value": 2}, score=20)
    queue.enqueue("sooner", {"value": 1}, score=10)

    assert queue.peek(count=2) == [{"value": 1}, {"value": 2}]
    item = queue.reserve(timeout_seconds=0)

    assert item is not None
    assert item.item_id == "sooner"
    assert item.payload == {"value": 1}
    assert queue.depths().ready == 1
    assert queue.depths().processing == 1
    assert queue.finalize(item) == "acked"
    assert queue.depths().total == 1


def test_in_memory_work_queue_requeues_newer_payload_on_finalize() -> None:
    queue = InMemoryWorkQueue()
    queue.enqueue("item", {"version": 1}, score=1)
    item = queue.reserve(timeout_seconds=0)
    assert item is not None

    result = queue.enqueue("item", {"version": 2}, score=0)

    assert result["state"] == "deferred"
    assert queue.finalize(item) == "requeued"
    next_item = queue.reserve(timeout_seconds=0)
    assert next_item is not None
    assert next_item.payload == {"version": 2}
    assert queue.finalize(next_item) == "acked"


def test_in_memory_work_queue_requeues_expired_leases() -> None:
    queue = InMemoryWorkQueue(lease_seconds=1)
    queue.enqueue("item", {"value": 1}, score=1)
    item = queue.reserve(timeout_seconds=0)
    assert item is not None

    time.sleep(1.01)

    assert queue.requeue_expired() == 1
    assert queue.depths().ready == 1
    assert queue.depths().processing == 0
    next_item = queue.reserve(timeout_seconds=0)
    assert next_item is not None
    assert next_item.item_id == "item"


def test_in_memory_work_queue_discard_ready_item() -> None:
    queue = InMemoryWorkQueue()
    queue.enqueue("item", {"value": 1}, score=1)

    assert queue.discard("item") == "dropped"
    assert queue.reserve(timeout_seconds=0) is None


def test_in_memory_work_queue_discard_processing_item_reports_processing() -> None:
    queue = InMemoryWorkQueue()
    queue.enqueue("item", {"value": 1}, score=1)
    item = queue.reserve(timeout_seconds=0)
    assert item is not None

    assert queue.discard("item") == "processing"
    assert queue.finalize(item) == "acked"


@pytest.mark.asyncio
async def test_run_work_queue_batch_processes_items_concurrently() -> None:
    queue = InMemoryWorkQueue()
    for index in range(4):
        queue.enqueue(f"item-{index}", {"value": index}, score=index)

    active = 0
    max_active = 0
    processed: list[str] = []
    lock = asyncio.Lock()

    async def handle(item: QueuedWorkItem) -> None:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        processed.append(item.item_id)
        async with lock:
            active -= 1

    result = await run_work_queue_batch(
        queue,
        handle,
        concurrency=2,
        reserve_timeout_seconds=0,
    )

    assert result.reserved == 4
    assert result.succeeded == 4
    assert result.failed == 0
    assert result.finalize_statuses == {"acked": 4}
    assert max_active == 2
    assert sorted(processed) == ["item-0", "item-1", "item-2", "item-3"]
    assert queue.depths().total == 0


@pytest.mark.asyncio
async def test_run_work_queue_batch_respects_max_items() -> None:
    queue = InMemoryWorkQueue()
    for index in range(5):
        queue.enqueue(f"item-{index}", {"value": index}, score=index)

    processed: list[str] = []

    async def handle(item: QueuedWorkItem) -> None:
        processed.append(item.item_id)

    result = await run_work_queue_batch(
        queue,
        handle,
        concurrency=3,
        max_items=2,
        reserve_timeout_seconds=0,
    )

    assert result.reserved == 2
    assert result.succeeded == 2
    assert result.finalize_statuses == {"acked": 2}
    assert len(processed) == 2
    assert queue.depths().total == 3


@pytest.mark.asyncio
async def test_run_work_queue_batch_leaves_failed_items_unfinalized() -> None:
    queue = InMemoryWorkQueue()
    queue.enqueue("item", {"value": 1}, score=1)

    async def handle(_item: QueuedWorkItem) -> None:
        raise RuntimeError("handler failed")

    result = await run_work_queue_batch(
        queue,
        handle,
        concurrency=2,
        reserve_timeout_seconds=0,
    )

    assert result.reserved == 1
    assert result.succeeded == 0
    assert result.failed == 1
    assert result.errors[0].phase == "handler"
    assert isinstance(result.errors[0].error, RuntimeError)
    assert queue.depths().ready == 0
    assert queue.depths().processing == 1


def test_redis_work_queue_with_fakeredis() -> None:
    fakeredis = pytest.importorskip("fakeredis")

    client = fakeredis.FakeRedis()
    queue = RedisWorkQueue(client, queue_name="test:work", lease_seconds=1)
    try:
        queue.enqueue("later", {"value": 2}, score=20)
    except Exception as exc:
        if "evalsha" in str(exc).lower():
            pytest.skip("fakeredis backend does not support Redis scripts in this environment.")
        raise
    queue.enqueue("sooner", {"value": 1}, score=10)

    assert queue.peek(count=2) == [{"value": 1}, {"value": 2}]
    item = queue.reserve(timeout_seconds=0)
    assert item is not None
    assert item.item_id == "sooner"
    assert item.payload == {"value": 1}
    assert queue.finalize(item) == "acked"
