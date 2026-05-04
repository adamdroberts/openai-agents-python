from __future__ import annotations

import time

import pytest

from agents import InMemoryWorkQueue, RedisWorkQueue


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
