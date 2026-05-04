from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

_MAX_QUEUE_SCORE = 32_503_680_000_000.0


@dataclass(frozen=True)
class QueuedWorkItem:
    """A reserved work item from a work queue."""

    item_id: str
    payload: dict[str, Any]
    version: int
    score: float


@dataclass(frozen=True)
class WorkQueueDepths:
    """Queue depth counts."""

    ready: int
    processing: int

    @property
    def total(self) -> int:
        """The combined ready and processing count."""
        return self.ready + self.processing


def _normalize_score(score: float) -> float:
    if isinstance(score, bool):
        return float(int(score))
    try:
        normalized = float(score)
    except (TypeError, ValueError):
        return _MAX_QUEUE_SCORE
    if math.isnan(normalized) or math.isinf(normalized):
        return _MAX_QUEUE_SCORE
    return normalized


class InMemoryWorkQueue:
    """Thread-safe lease-based work queue for local workers and tests."""

    def __init__(self, *, lease_seconds: int = 60) -> None:
        self._lease_seconds = max(int(lease_seconds), 1)
        self._condition = threading.Condition(threading.RLock())
        self._ready: set[str] = set()
        self._processing: dict[str, float] = {}
        self._versions: dict[str, int] = {}
        self._dirty: dict[str, int] = {}
        self._payloads: dict[str, dict[str, Any]] = {}

    def enqueue(self, item_id: str, payload: dict[str, Any], *, score: float) -> dict[str, Any]:
        """Enqueue or update an item."""
        if not item_id:
            raise ValueError("item_id is required")
        normalized_score = _normalize_score(score)
        with self._condition:
            version = self._versions.get(item_id, 0) + 1
            self._versions[item_id] = version
            self._payloads[item_id] = {
                "item_id": item_id,
                "payload": dict(payload),
                "version": version,
                "score": normalized_score,
                "updated_at_ms": int(time.time() * 1000),
            }
            if item_id in self._processing:
                self._dirty[item_id] = version
                state = "deferred"
            else:
                self._ready.add(item_id)
                self._dirty.pop(item_id, None)
                state = "ready"
            self._condition.notify()
            return {"version": version, "state": state, "score": normalized_score}

    def reserve(self, *, timeout_seconds: float = 1.0) -> QueuedWorkItem | None:
        """Reserve the next ready item until `finalize` or lease expiry."""
        deadline = time.monotonic() + max(timeout_seconds, 0.0)
        with self._condition:
            while True:
                self.requeue_expired()
                item_id = self._next_ready_item_id()
                if item_id is not None:
                    self._ready.remove(item_id)
                    self._processing[item_id] = time.time() + self._lease_seconds
                    wrapper = self._payloads.get(item_id)
                    if not isinstance(wrapper, dict):
                        self._drop_processing(item_id)
                        continue
                    payload = wrapper.get("payload")
                    if not isinstance(payload, dict):
                        self._drop_processing(item_id)
                        continue
                    return QueuedWorkItem(
                        item_id=item_id,
                        payload=dict(payload),
                        version=int(wrapper.get("version") or 0),
                        score=_normalize_score(wrapper.get("score", _MAX_QUEUE_SCORE)),
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(0.25, remaining))

    def finalize(self, item: QueuedWorkItem) -> str:
        """Acknowledge a reserved item or requeue it if a newer version arrived."""
        with self._condition:
            self._processing.pop(item.item_id, None)
            wrapper = self._payloads.get(item.item_id)
            current_version = int(wrapper.get("version") or 0) if wrapper else 0
            dirty_version = int(self._dirty.get(item.item_id) or 0)
            if current_version > item.version or dirty_version > item.version:
                self._ready.add(item.item_id)
                self._dirty.pop(item.item_id, None)
                self._condition.notify()
                return "requeued"
            self._versions.pop(item.item_id, None)
            self._dirty.pop(item.item_id, None)
            self._payloads.pop(item.item_id, None)
            return "acked"

    def requeue_expired(self) -> int:
        """Move expired processing leases back to the ready queue."""
        with self._condition:
            now = time.time()
            moved = 0
            expired = [item_id for item_id, deadline in self._processing.items() if deadline <= now]
            for item_id in expired:
                self._processing.pop(item_id, None)
                if item_id in self._payloads:
                    self._ready.add(item_id)
                    moved += 1
                else:
                    self._versions.pop(item_id, None)
                    self._dirty.pop(item_id, None)
            if moved:
                self._condition.notify_all()
            return moved

    def discard(self, item_id: str) -> str:
        """Discard a ready item, or report that a reserved item is still processing."""
        with self._condition:
            self._ready.discard(item_id)
            if item_id in self._processing:
                return "processing"
            self._versions.pop(item_id, None)
            self._dirty.pop(item_id, None)
            self._payloads.pop(item_id, None)
            return "dropped"

    def depths(self) -> WorkQueueDepths:
        """Return ready and processing depths."""
        with self._condition:
            return WorkQueueDepths(ready=len(self._ready), processing=len(self._processing))

    def peek(self, count: int = 4) -> list[dict[str, Any]]:
        """Return payloads for upcoming ready items without reserving them."""
        with self._condition:
            item_ids = self._sorted_ready_item_ids()[: max(int(count), 0)]
            payloads: list[dict[str, Any]] = []
            for item_id in item_ids:
                wrapper = self._payloads.get(item_id)
                payload = wrapper.get("payload") if isinstance(wrapper, dict) else None
                if isinstance(payload, dict):
                    payloads.append(dict(payload))
            return payloads

    def _next_ready_item_id(self) -> str | None:
        item_ids = self._sorted_ready_item_ids()
        return item_ids[0] if item_ids else None

    def _sorted_ready_item_ids(self) -> list[str]:
        return sorted(
            self._ready,
            key=lambda item_id: (
                _normalize_score(self._payloads.get(item_id, {}).get("score", _MAX_QUEUE_SCORE)),
                item_id,
            ),
        )

    def _drop_processing(self, item_id: str) -> None:
        self._processing.pop(item_id, None)
        self._versions.pop(item_id, None)
        self._dirty.pop(item_id, None)
        self._payloads.pop(item_id, None)


_ENQUEUE_SCRIPT = """
local version = redis.call('HINCRBY', KEYS[3], ARGV[1], 1)
local wrapper = cjson.decode(ARGV[2])
wrapper["version"] = version
wrapper["score"] = tonumber(ARGV[3])
wrapper["updated_at_ms"] = tonumber(ARGV[4])
redis.call('SET', KEYS[5], cjson.encode(wrapper))
if redis.call('ZSCORE', KEYS[2], ARGV[1]) then
  redis.call('HSET', KEYS[4], ARGV[1], version)
  return {tostring(version), "deferred"}
end
redis.call('ZADD', KEYS[1], tonumber(ARGV[3]), ARGV[1])
redis.call('HDEL', KEYS[4], ARGV[1])
return {tostring(version), "ready"}
"""


_RESERVE_SCRIPT = """
local items = redis.call('ZRANGE', KEYS[1], 0, 0)
if not items[1] then
  return nil
end
local item = items[1]
redis.call('ZREM', KEYS[1], item)
redis.call('ZADD', KEYS[2], tonumber(ARGV[1]), item)
return item
"""


_FINALIZE_SCRIPT = """
redis.call('ZREM', KEYS[2], ARGV[1])
local payload_json = redis.call('GET', KEYS[5])
local current_version = 0
local current_score = tonumber(ARGV[3])
if payload_json then
  local wrapper = cjson.decode(payload_json)
  current_version = tonumber(wrapper["version"] or 0)
  current_score = tonumber(wrapper["score"] or current_score)
end
local dirty_version = tonumber(redis.call('HGET', KEYS[4], ARGV[1]) or "0")
if current_version > tonumber(ARGV[2]) or dirty_version > tonumber(ARGV[2]) then
  redis.call('ZADD', KEYS[1], current_score, ARGV[1])
  redis.call('HDEL', KEYS[4], ARGV[1])
  return "requeued"
end
redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('HDEL', KEYS[4], ARGV[1])
redis.call('DEL', KEYS[5])
return "acked"
"""


_REQUEUE_EXPIRED_SCRIPT = """
local expired = redis.call(
  'ZRANGEBYSCORE',
  KEYS[2],
  '-inf',
  tonumber(ARGV[1]),
  'LIMIT',
  0,
  tonumber(ARGV[2])
)
local moved = 0
for _, item in ipairs(expired) do
  redis.call('ZREM', KEYS[2], item)
  local payload_json = redis.call('GET', ARGV[3] .. item)
  if payload_json then
    local wrapper = cjson.decode(payload_json)
    local score = tonumber(wrapper["score"] or 0)
    redis.call('ZADD', KEYS[1], score, item)
    moved = moved + 1
  else
    redis.call('HDEL', KEYS[3], item)
    redis.call('HDEL', KEYS[4], item)
  end
end
return moved
"""


_DISCARD_SCRIPT = """
redis.call('ZREM', KEYS[1], ARGV[1])
if redis.call('ZSCORE', KEYS[2], ARGV[1]) then
  return "processing"
end
redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('HDEL', KEYS[4], ARGV[1])
redis.call('DEL', KEYS[5])
return "dropped"
"""


_DROP_PROCESSING_SCRIPT = """
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('HDEL', KEYS[2], ARGV[1])
redis.call('HDEL', KEYS[3], ARGV[1])
redis.call('DEL', KEYS[4])
return "dropped"
"""


class RedisWorkQueue:
    """Redis-backed lease-based work queue."""

    def __init__(
        self,
        redis_client: Any,
        *,
        queue_name: str,
        lease_seconds: int = 60,
        reclaim_batch_size: int = 50,
    ) -> None:
        self._redis = redis_client
        self._queue_name = queue_name
        self._lease_seconds = max(int(lease_seconds), 1)
        self._reclaim_batch_size = max(int(reclaim_batch_size), 1)
        self._ready_key = f"{queue_name}:ready"
        self._processing_key = f"{queue_name}:processing"
        self._versions_key = f"{queue_name}:versions"
        self._dirty_key = f"{queue_name}:dirty"
        self._payload_prefix = f"{queue_name}:payload:"
        self._enqueue = self._redis.register_script(_ENQUEUE_SCRIPT)
        self._reserve = self._redis.register_script(_RESERVE_SCRIPT)
        self._finalize = self._redis.register_script(_FINALIZE_SCRIPT)
        self._requeue_expired = self._redis.register_script(_REQUEUE_EXPIRED_SCRIPT)
        self._discard = self._redis.register_script(_DISCARD_SCRIPT)
        self._drop_processing = self._redis.register_script(_DROP_PROCESSING_SCRIPT)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        queue_name: str,
        lease_seconds: int = 60,
        reclaim_batch_size: int = 50,
        redis_kwargs: dict[str, Any] | None = None,
    ) -> RedisWorkQueue:
        """Create a Redis work queue from a Redis URL."""
        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisWorkQueue requires the 'redis' package. Install it with: pip install redis"
            ) from exc
        client = redis.Redis.from_url(url, **(redis_kwargs or {}))
        return cls(
            client,
            queue_name=queue_name,
            lease_seconds=lease_seconds,
            reclaim_batch_size=reclaim_batch_size,
        )

    def enqueue(self, item_id: str, payload: dict[str, Any], *, score: float) -> dict[str, Any]:
        """Enqueue or update an item."""
        if not item_id:
            raise ValueError("item_id is required")
        normalized_score = _normalize_score(score)
        payload_wrapper = {
            "item_id": item_id,
            "payload": payload,
        }
        result = self._enqueue(
            keys=[
                self._ready_key,
                self._processing_key,
                self._versions_key,
                self._dirty_key,
                self._payload_key(item_id),
            ],
            args=[
                item_id,
                json.dumps(payload_wrapper, ensure_ascii=False, separators=(",", ":")),
                normalized_score,
                int(time.time() * 1000),
            ],
        )
        version = int(result[0]) if isinstance(result, list | tuple) and len(result) >= 1 else 0
        state = (
            self._decode_redis_value(result[1])
            if isinstance(result, list | tuple) and len(result) >= 2
            else None
        ) or "ready"
        return {"version": version, "state": state, "score": normalized_score}

    def reserve(self, *, timeout_seconds: float = 1.0) -> QueuedWorkItem | None:
        """Reserve the next ready item until `finalize` or lease expiry."""
        deadline = time.monotonic() + max(timeout_seconds, 0.0)
        while True:
            self.requeue_expired()
            lease_deadline_ms = int((time.time() + self._lease_seconds) * 1000)
            raw_item_id = self._reserve(
                keys=[self._ready_key, self._processing_key],
                args=[lease_deadline_ms],
            )
            item_id = self._decode_redis_value(raw_item_id)
            if item_id:
                payload_json = self._redis.get(self._payload_key(item_id))
                payload_text = self._decode_redis_value(payload_json)
                if not payload_text:
                    self._drop_reserved(item_id)
                    continue
                try:
                    decoded = json.loads(payload_text)
                except json.JSONDecodeError:
                    self._drop_reserved(item_id)
                    continue
                payload = decoded.get("payload")
                if not isinstance(payload, dict):
                    self._drop_reserved(item_id)
                    continue
                return QueuedWorkItem(
                    item_id=item_id,
                    payload=payload,
                    version=int(decoded.get("version") or 0),
                    score=_normalize_score(decoded.get("score", _MAX_QUEUE_SCORE)),
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.25, remaining))

    def finalize(self, item: QueuedWorkItem) -> str:
        """Acknowledge a reserved item or requeue it if a newer version arrived."""
        result = self._finalize(
            keys=[
                self._ready_key,
                self._processing_key,
                self._versions_key,
                self._dirty_key,
                self._payload_key(item.item_id),
            ],
            args=[item.item_id, item.version, item.score],
        )
        return self._decode_redis_value(result) or "acked"

    def requeue_expired(self) -> int:
        """Move expired processing leases back to the ready queue."""
        result = self._requeue_expired(
            keys=[self._ready_key, self._processing_key, self._versions_key, self._dirty_key],
            args=[int(time.time() * 1000), self._reclaim_batch_size, self._payload_prefix],
        )
        return int(result or 0)

    def discard(self, item_id: str) -> str:
        """Discard a ready item, or report that a reserved item is still processing."""
        result = self._discard(
            keys=[
                self._ready_key,
                self._processing_key,
                self._versions_key,
                self._dirty_key,
                self._payload_key(item_id),
            ],
            args=[item_id],
        )
        return self._decode_redis_value(result) or "dropped"

    def depths(self) -> WorkQueueDepths:
        """Return ready and processing depths."""
        ready = int(self._redis.zcard(self._ready_key))
        processing = int(self._redis.zcard(self._processing_key))
        return WorkQueueDepths(ready=ready, processing=processing)

    def peek(self, count: int = 4) -> list[dict[str, Any]]:
        """Return payloads for upcoming ready items without reserving them."""
        try:
            item_ids = self._redis.zrange(self._ready_key, 0, count - 1)
        except Exception:
            return []
        results: list[dict[str, Any]] = []
        for raw_id in item_ids or ():
            item_id = self._decode_redis_value(raw_id)
            if not item_id:
                continue
            try:
                payload_json = self._redis.get(self._payload_key(item_id))
                payload_text = self._decode_redis_value(payload_json)
                if not payload_text:
                    continue
                decoded = json.loads(payload_text)
                payload = decoded.get("payload")
                if isinstance(payload, dict):
                    results.append(payload)
            except Exception:
                continue
        return results

    def _payload_key(self, item_id: str) -> str:
        return f"{self._payload_prefix}{item_id}"

    def _drop_reserved(self, item_id: str) -> str:
        result = self._drop_processing(
            keys=[
                self._processing_key,
                self._versions_key,
                self._dirty_key,
                self._payload_key(item_id),
            ],
            args=[item_id],
        )
        return self._decode_redis_value(result) or "dropped"

    @staticmethod
    def _decode_redis_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)


__all__ = [
    "InMemoryWorkQueue",
    "QueuedWorkItem",
    "RedisWorkQueue",
    "WorkQueueDepths",
]
