"""自动配图被事件误判为新消息的回归测试。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.test_auto_emoji_flow import DummyEvent, _build_main


def event_for(message_id, *, post_type="message", sender="user", self_id="bot"):
    event = DummyEvent("这是一条完整回复")
    event.message_obj = SimpleNamespace(
        message_id=message_id, raw_message={"post_type": post_type},
        sender=SimpleNamespace(user_id=sender), self_id=self_id,
    )
    return event


@pytest.mark.asyncio
async def test_notice_self_and_same_message_do_not_cancel_pending_task():
    main = _build_main(1)
    main._get_event_handler = lambda **kwargs: SimpleNamespace(on_message=AsyncMock())
    original = event_for("42")
    state = main._emoji_sender_engine.emoji_turn_state(original)
    state.set_llm_reply("这是一条完整回复")
    task = asyncio.create_task(asyncio.Event().wait())
    main._emoji_sender_engine.schedule_auto_emoji_task(original, task)
    for event in (original, event_for("42"), event_for("notice", post_type="notice"),
                  event_for("request", post_type="request"), event_for("own", sender="bot")):
        await main.on_message(event)
        assert not task.cancelled() and not task.done()
        assert state.get_llm_reply() == "这是一条完整回复"
    await main.on_message(event_for("43"))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not main._emoji_sender_engine._pending_auto_emoji_sources


@pytest.mark.asyncio
async def test_replacing_task_does_not_let_old_callback_forget_new_source():
    main = _build_main(1)
    original, next_event = event_for("one"), event_for("two")
    engine = main._emoji_sender_engine
    first = asyncio.create_task(asyncio.Event().wait())
    second = asyncio.create_task(asyncio.Event().wait())
    engine.schedule_auto_emoji_task(original, first)
    engine.schedule_auto_emoji_task(next_event, second)
    await asyncio.sleep(0)
    assert engine._pending_auto_emoji_sources["session-1"][1] is next_event
    assert not engine.is_new_message_turn(next_event)
    second.cancel()
    await asyncio.gather(first, second, return_exceptions=True)
    await asyncio.sleep(0)
    assert not engine._pending_auto_emoji_sources


@pytest.mark.asyncio
async def test_cancellation_log_identifies_events_without_exposing_message_ids(monkeypatch):
    import importlib

    main = _build_main(1)
    engine = main._emoji_sender_engine
    module = importlib.import_module(type(engine).__module__)
    captured = []
    monkeypatch.setattr(module.logger, "debug", lambda message: captured.append(message))
    first = event_for("sensitive-original")
    next_event = event_for("sensitive-next")
    task = asyncio.create_task(asyncio.Event().wait())
    engine.schedule_auto_emoji_task(first, task)
    assert engine.cancel_pending_auto_emoji(next_event)
    await asyncio.gather(task, return_exceptions=True)
    assert len(captured) == 1
    assert "event_type=message" in captured[0] and "source_id=" in captured[0] and "event_id=" in captured[0]
    assert "sensitive-original" not in captured[0] and "sensitive-next" not in captured[0]
