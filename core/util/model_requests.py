"""Per-provider request pacing and priority concurrency gate."""

import asyncio
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from email.utils import parsedate_to_datetime

_REQUEST_CONCURRENCY: ContextVar[int] = ContextVar("stealer_model_concurrency", default=1)
_REQUEST_PRIORITY: ContextVar[int] = ContextVar("stealer_model_priority", default=0)


@contextmanager
def model_request_context(*, concurrency: int = 1, priority: int = 0):
    """Set temporary concurrency and priority for nested model_request_gate calls."""
    tok_c = _REQUEST_CONCURRENCY.set(max(1, int(concurrency or 1)))
    tok_p = _REQUEST_PRIORITY.set(int(priority or 0))
    try:
        yield
    finally:
        _REQUEST_CONCURRENCY.reset(tok_c)
        _REQUEST_PRIORITY.reset(tok_p)


def is_rate_limited(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    status = status or getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or any(word in str(exc).lower() for word in
                                ("429", "ratelimit", "rate limit", "request limit", "too many requests"))


def retry_after(exc: Exception, default: float = 30) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None) or getattr(exc, "headers", None) or {}
    value = headers.get("Retry-After", headers.get("retry-after", ""))
    try:
        return max(0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return default


@dataclass
class _Waiter:
    future: asyncio.Future
    limit: int


@dataclass
class _ProviderState:
    in_flight: int = 0
    active_limit: int = 1
    next_time: float = 0.0
    pace_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    urgent_waiters: deque[_Waiter] = field(default_factory=deque)
    bg_waiters: deque[_Waiter] = field(default_factory=deque)


class ModelRequestGate:
    """Paces start times per provider while bounding in-flight concurrency and prioritizing realtime work."""

    def __init__(self, plugin):
        self.plugin = plugin
        self._states: dict[str, _ProviderState] = {}

    def _state(self, provider_id: str) -> _ProviderState:
        state = self._states.get(provider_id)
        if state is None:
            state = _ProviderState()
            self._states[provider_id] = state
        return state

    @staticmethod
    def _has_live_waiters(queue: deque[_Waiter]) -> bool:
        while queue and queue[0].future.done():
            queue.popleft()
        return bool(queue)

    def _drain_waiters(self, state: _ProviderState) -> None:
        while self._has_live_waiters(state.urgent_waiters):
            waiter = state.urgent_waiters[0]
            eff_limit = max(waiter.limit, state.active_limit)
            if state.in_flight >= eff_limit:
                return
            state.urgent_waiters.popleft()
            if not waiter.future.done():
                state.in_flight += 1
                waiter.future.set_result(True)

        if self._has_live_waiters(state.urgent_waiters):
            return

        while self._has_live_waiters(state.bg_waiters):
            waiter = state.bg_waiters[0]
            if state.in_flight >= waiter.limit:
                return
            state.bg_waiters.popleft()
            if not waiter.future.done():
                state.in_flight += 1
                state.active_limit = max(state.active_limit, waiter.limit)
                waiter.future.set_result(True)

    async def _acquire_slot(self, state: _ProviderState, limit: int, priority: int) -> None:
        is_urgent = priority <= 0
        if is_urgent:
            eff_limit = max(limit, state.active_limit)
            if not self._has_live_waiters(state.urgent_waiters) and state.in_flight < eff_limit:
                state.in_flight += 1
                return
            fut = asyncio.get_running_loop().create_future()
            state.urgent_waiters.append(_Waiter(fut, limit))
        else:
            if (
                not self._has_live_waiters(state.urgent_waiters)
                and not self._has_live_waiters(state.bg_waiters)
                and state.in_flight < limit
            ):
                state.in_flight += 1
                state.active_limit = max(state.active_limit, limit)
                return
            fut = asyncio.get_running_loop().create_future()
            state.bg_waiters.append(_Waiter(fut, limit))

        try:
            await fut
        except BaseException:
            if fut.done() and not fut.cancelled():
                self._release_slot(state)
            raise

    def _release_slot(self, state: _ProviderState) -> None:
        state.in_flight = max(0, state.in_flight - 1)
        if state.in_flight == 0 and not self._has_live_waiters(state.bg_waiters):
            state.active_limit = 1
        self._drain_waiters(state)

    async def call(
        self,
        provider_id: str,
        operation,
        *,
        concurrency: int | None = None,
        priority: int | None = None,
    ):
        eff_concurrency = max(1, int(concurrency if concurrency is not None else _REQUEST_CONCURRENCY.get()))
        eff_priority = int(priority if priority is not None else _REQUEST_PRIORITY.get())
        state = self._state(provider_id)

        await self._acquire_slot(state, eff_concurrency, eff_priority)
        try:
            async with state.pace_lock:
                delay = state.next_time - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                cfg = getattr(self.plugin, "plugin_config", None)
                interval = max(0.0, float(getattr(cfg, "model_request_interval", 5.0)))
                state.next_time = time.monotonic() + interval
            try:
                return await operation()
            except Exception as exc:
                if is_rate_limited(exc):
                    state.next_time = max(state.next_time, time.monotonic() + retry_after(exc))
                raise
        finally:
            self._release_slot(state)


def model_request_gate(plugin) -> ModelRequestGate:
    gate = getattr(plugin, "_model_request_gate", None)
    if gate is None:
        gate = ModelRequestGate(plugin)
        plugin._model_request_gate = gate
    return gate
