"""LLM 回复来源及异步期间被命令输出替换的回归。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.events.meme_sender_engine import MemeSenderEngine
from core.processing.natural_emotion_analyzer import EmotionQuery
from core.search.meme_selector import MemeSelector
from tests.test_auto_emoji_flow import DummyEvent, DummyLLMResponse, DummyResult, _build_main


def general_result(text):
    result = DummyResult(text)
    result.is_llm_result = lambda: False
    result.result_content_type = SimpleNamespace(name="GENERAL_RESULT")
    return result


def sender(text="这是一条完整回复"):
    analyzer = SimpleNamespace(analyze_for_reply=AsyncMock(
        return_value=EmotionQuery(True, "检索词", ["happy"]),
    ))
    plugin = SimpleNamespace(
        plugin_config=SimpleNamespace(enable_jev=False, enable_natural_emotion_analysis=True),
        enable_natural_emotion_analysis=True, emotion_analyzer=analyzer,
    )
    engine = MemeSenderEngine(plugin)
    engine.try_send_emoji = AsyncMock(return_value=False)
    engine.get_meme_send_delay = lambda *args: 0
    event = DummyEvent(text)
    engine.emoji_turn_state(event).set_llm_reply(text)
    return engine, event, analyzer


@pytest.mark.asyncio
async def test_command_event_skips_both_hooks_even_when_result_is_marked_llm():
    main = _build_main(1)
    main._safe_create_task = AsyncMock()
    event = DummyEvent("操作完成，今天好开心！")
    # 参数为空的命令也会在框架映射中占一项。
    event.set_extra("handlers_parsed_params", {"other_plugin_status": {}})
    assert not await main._prepare_emoji_after_llm(event, DummyLLMResponse(event.get_result().get_plain_text()))
    assert not await main._prepare_emoji_response(event)
    main._safe_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_ordinary_plugin_output_skips_analysis():
    engine, event, analyzer = sender()
    event.set_result(general_result("这是一条完整回复"))
    await engine.async_analyze_and_send_emoji(event, "这是一条完整回复", [])
    analyzer.analyze_for_reply.assert_not_awaited()
    engine.try_send_emoji.assert_not_awaited()


@pytest.mark.asyncio
async def test_general_result_does_not_schedule_through_decorating_hook():
    main = _build_main(1)
    main._safe_create_task = AsyncMock()
    event = DummyEvent()
    event.set_result(general_result("插件操作成功！"))
    assert not await main._prepare_emoji_response(event)
    main._safe_create_task.assert_not_called()


@pytest.mark.asyncio
async def test_source_replaced_during_analysis_skips_send():
    engine, event, analyzer = sender()

    async def analyze(*args, **kwargs):
        event.set_result(general_result("另一个插件操作成功！"))
        return EmotionQuery(True, "检索词", ["happy"])

    analyzer.analyze_for_reply.side_effect = analyze
    await engine.async_analyze_and_send_emoji(event, "这是一条完整回复", [])
    analyzer.analyze_for_reply.assert_awaited_once()
    engine.try_send_emoji.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_replaced_during_candidate_selection_skips_platform_send():
    main = _build_main(1)
    selector = MemeSelector(main)
    event = DummyEvent("这是一条完整回复")
    state = main._emoji_sender_engine.emoji_turn_state(event)
    state.set_llm_reply("这是一条完整回复")
    state.set_auto_decision(True)
    state.claim_auto_send_meme()

    async def select(*args, **kwargs):
        event.set_result(general_result("另一个插件操作成功！"))
        return "candidate.png"

    selector.select_emoji = AsyncMock(side_effect=select)
    selector.send_emoji_message = AsyncMock(return_value="file_image")
    assert not await selector.try_send_emoji(event, ["happy"], "检索词")
    selector.select_emoji.assert_awaited_once()
    selector.send_emoji_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_replaced_during_delay_skips_send(monkeypatch):
    engine, event, _ = sender()
    entered, release = asyncio.Event(), asyncio.Event()
    engine.get_meme_send_delay = lambda *args: 1

    async def delay(_seconds):
        entered.set()
        await release.wait()

    monkeypatch.setattr(asyncio, "sleep", delay)
    task = asyncio.create_task(engine.async_analyze_and_send_emoji(event, "这是一条完整回复", []))
    await asyncio.wait_for(entered.wait(), timeout=1)
    event.set_result(general_result("另一个插件操作成功！"))
    release.set()
    await task
    engine.try_send_emoji.assert_not_awaited()


@pytest.mark.asyncio
async def test_decorating_command_output_cancels_pending_task_and_keeps_source_invalid():
    main = _build_main(1)
    main._safe_create_task = lambda coro, name: asyncio.create_task(coro)
    event = DummyEvent("这是一条完整回复")
    assert await main._prepare_emoji_after_llm(event, DummyLLMResponse("这是一条完整回复"))
    engine = main._emoji_sender_engine
    task = next(iter(engine._pending_auto_emoji_tasks.values()))
    event.set_result(general_result("操作成功！"))
    assert not await main._prepare_emoji_response(event)
    with pytest.raises(asyncio.CancelledError):
        await task
    event.set_result(None)
    assert not engine.is_auto_emoji_source_current(event, "这是一条完整回复")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["LLM_RESULT", "STREAMING_RESULT", "STREAMING_FINISH", "sent"])
async def test_confirmed_normal_and_streaming_replies_survive_pipeline_cleanup(kind):
    engine, event, analyzer = sender()
    if kind == "sent":
        event.set_result(None)
    elif kind != "LLM_RESULT":
        result = general_result("" if kind == "STREAMING_RESULT" else "这是一条完整回复")
        result.result_content_type.name = kind
        event.set_result(result)
    await engine.async_analyze_and_send_emoji(event, "这是一条完整回复", [])
    analyzer.analyze_for_reply.assert_awaited_once()
    engine.try_send_emoji.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_turn_discards_captured_llm_reply():
    engine, event, analyzer = sender()
    engine.reset_turn_state(event)
    await engine.async_analyze_and_send_emoji(event, "这是一条完整回复", [])
    analyzer.analyze_for_reply.assert_not_awaited()
    engine.try_send_emoji.assert_not_awaited()


def test_repeated_questions_reach_matching_and_technical_errors_keep_gate():
    engine = MemeSenderEngine(SimpleNamespace(plugin_config=SimpleNamespace(auto_meme_intent_gate=True)))
    assert not engine.should_skip_auto_emoji_by_gate("你在说什么？？")
    assert engine.should_skip_auto_emoji_by_gate("连接失败？？")


@pytest.mark.asyncio
async def test_tts_record_replaced_llm_reply_keeps_source_current():
    engine, event, analyzer = sender()
    result = DummyResult("")
    result.chain = [SimpleNamespace(type=SimpleNamespace(name="Record"))]
    result.result_content_type = SimpleNamespace(name="LLM_RESULT")
    event.set_result(result)
    await engine.async_analyze_and_send_emoji(event, "这是一条完整回复", [])
    analyzer.analyze_for_reply.assert_awaited_once()
    engine.try_send_emoji.assert_awaited_once()
