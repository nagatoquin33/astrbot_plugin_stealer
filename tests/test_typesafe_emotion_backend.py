"""JEV Top 10 决策、历史边界、排序及两条发送链路回归。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.config.config import PluginConfig
from core.events.meme_sender_engine import MemeSenderEngine
from core.processing.natural_emotion_analyzer import EmotionQuery
from core.search.jev_selector import JevSelector, JevSelectionError
from core.search.meme_selector import MemeSelector
from core.search.meme_smart_select_service import MemeSmartSelectService


def config(**kwargs):
    return PluginConfig(
        {"enable_jev": True, "typesafe_api_key": "test-key", **kwargs}
    )


def install_http(monkeypatch, payload=None, status=200, error=None):
    captured = {}

    class Response:
        async def __aenter__(self):
            if error:
                raise error
            self.status = status
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return payload

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            captured.update(url=url, **kwargs)
            return Response()

    monkeypatch.setattr("core.search.jev_selector.aiohttp.ClientSession", Session)
    return captured


def answer(choice):
    return {"answers": {"meme": {"type": "choice", "choice": choice}}}


@pytest.mark.parametrize(
    "overrides,enabled",
    [
        ({}, True),
        ({"enable_jev": False}, False),
        ({"typesafe_api_key": " "}, False),
        ({"enable_natural_emotion_analysis": False}, False),
    ],
)
def test_backend_gate(overrides, enabled):
    assert JevSelector.enabled(config(**overrides)) is enabled


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("", "https://api.typesafe.ai/v1/systemone"),
        ("https://proxy.example", "https://proxy.example/v1/systemone"),
        ("https://proxy.example/", "https://proxy.example/v1/systemone"),
        ("https://proxy.example/v1", "https://proxy.example/v1/systemone"),
        (
            "https://proxy.example/v1/systemone/",
            "https://proxy.example/v1/systemone",
        ),
    ],
)
def test_api_url_normalization(base_url, expected):
    assert JevSelector.api_url(config(typesafe_api_base_url=base_url)) == expected


@pytest.mark.parametrize("old_backend,explicit,expected", [
    ("typesafe", None, True), ("llm", None, False),
    ("typesafe", False, False), ("llm", True, True),
])
def test_legacy_backend_migration(old_backend, explicit, expected):
    values = {"emotion_backend": old_backend}
    if explicit is not None:
        values["enable_jev"] = explicit
    cfg = PluginConfig(values)
    assert cfg.enable_jev is expected
    cfg.update_config({"enable_jev": False})
    assert not cfg.enable_jev


@pytest.mark.asyncio
@pytest.mark.parametrize("master,jev,route", [
    (False, False, "raw"), (False, True, "raw"),
    (True, False, "llm"), (True, True, "jev"),
])
async def test_checkbox_routing_matrix(master, jev, route):
    analyzer = SimpleNamespace(analyze_for_reply=AsyncMock(
        return_value=EmotionQuery(True, "检索词", ["happy"]),
    ))
    selector = SimpleNamespace(
        select_emoji_with_jev=AsyncMock(return_value="a.png"),
        send_jev_selection=AsyncMock(return_value=True),
    )
    plugin = SimpleNamespace(
        plugin_config=config(enable_natural_emotion_analysis=master, enable_jev=jev),
        enable_natural_emotion_analysis=master,
        emotion_analyzer=analyzer, meme_selector=selector,
    )
    engine = MemeSenderEngine(plugin)
    engine.get_meme_send_delay = lambda *args: 0
    engine.try_send_emoji = AsyncMock(return_value=True)
    engine.mark_auto_emoji_sent = AsyncMock()
    engine.emoji_turn_state = lambda event: SimpleNamespace(mark_active_sent=Mock())
    event = SimpleNamespace(get_result=lambda: None)
    await engine.async_analyze_and_send_emoji(event, "回复原文", [], user_message="用户")
    assert analyzer.analyze_for_reply.await_count == (route == "llm")
    assert selector.select_emoji_with_jev.await_count == (route == "jev")
    assert selector.send_jev_selection.await_count == (route == "jev")
    if route == "raw":
        engine.try_send_emoji.assert_awaited_once_with(event, [], "回复原文")
    elif route == "llm":
        engine.try_send_emoji.assert_awaited_once_with(event, ["happy"], "检索词")
    else:
        engine.try_send_emoji.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_contains_ten_metadata_options_and_context(monkeypatch):
    captured = install_http(monkeypatch, answer("m10"))
    svc = JevSelector(SimpleNamespace(plugin_config=config()))
    candidates = [
        (
            f"private/path/{i}.png",
            {
                "overlay_text": "下班了",
                "desc": "猫咪欢呼",
                "scenes": ["下班"],
                "origin_url": "private",
            },
        )
        for i in range(12)
    ]
    selected = await svc.select(
        candidates,
        history=[{"role": "user", "content": "加班结束了"}],
        user_message="走吧",
        reply="终于可以回家啦",
    )
    assert selected == candidates[9][0]
    body = captured["json"]
    assert captured["url"] == "https://api.typesafe.ai/v1/systemone"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"].total == 10
    assert body["model"] == "jev-latest"
    assert body["state"]["recent_conversation"] == [
        {"role": "user", "content": "加班结束了"}
    ]
    assert body["state"]["current_assistant_reply"] == "终于可以回家啦"
    options = body["questions"]["meme"]["criteria"]
    assert len(options) == 11 and "none" in options and "m11" not in options
    assert options["m01"]["overlay_text"] == "下班了"
    assert "private" not in json.dumps(body)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,status,error,expected",
    [
        (answer("none"), 200, None, None),
        (answer("m01"), 200, None, "a.png"),
        (answer("m99"), 200, None, "error"),
        (answer([]), 200, None, "error"),
        ({"answers": []}, 200, None, "error"),
        ({}, 200, None, "error"),
        (None, 429, None, "error"),
        (None, 401, None, "error"),
        (None, 200, asyncio.TimeoutError(), "error"),
    ],
)
async def test_response_and_failure_contract(
    monkeypatch, payload, status, error, expected
):
    install_http(monkeypatch, payload, status, error)
    svc = JevSelector(SimpleNamespace(plugin_config=config()))
    call = svc.select([("a.png", {})], history=[], user_message="用户", reply="回复")
    if expected == "error":
        with pytest.raises(JevSelectionError):
            await call
    else:
        assert await call == expected


@pytest.mark.asyncio
async def test_empty_candidates_and_cancellation(monkeypatch):
    captured = install_http(monkeypatch, error=asyncio.CancelledError())
    svc = JevSelector(SimpleNamespace(plugin_config=config()))
    assert await svc.select([], history=[], user_message="", reply="") is None
    assert not captured
    with pytest.raises(asyncio.CancelledError):
        await svc.select([("a", {})], history=[], user_message="", reply="")


def test_history_filters_roles_media_and_limits():
    raw = [{"role": "user", "content": str(i) * 600} for i in range(10)]
    raw += [
        {"role": "system", "content": "secret"},
        {"role": "tool", "content": "secret"},
        {
            "role": "assistant",
            "content": [
                {"type": "image_url", "image_url": "secret"},
                {"type": "text", "text": "回复"},
            ],
        },
    ]
    recent = JevSelector.normalize_history(json.dumps(raw))
    assert len(recent) == 6
    assert all(len(item["content"]) <= 400 for item in recent)
    assert recent[-1] == {"role": "assistant", "content": "回复"}
    assert "secret" not in str(recent)
    assert JevSelector.normalize_history("invalid") == []


@pytest.mark.asyncio
async def test_history_uses_current_session():
    manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="cid"),
        get_conversation=AsyncMock(
            return_value=SimpleNamespace(history='[{"role":"user","content":"历史"}]')
        ),
    )
    svc = JevSelector(
        SimpleNamespace(context=SimpleNamespace(conversation_manager=manager))
    )
    assert await svc.get_history(SimpleNamespace(unified_msg_origin="session-a")) == [
        {"role": "user", "content": "历史"}
    ]
    manager.get_conversation.assert_awaited_once_with("session-a", "cid")
    manager.get_conversation.side_effect = RuntimeError()
    assert await svc.get_history(SimpleNamespace(unified_msg_origin="session-a")) == []


def build_ranking(tmp_path):
    idx = {}
    for i in range(14):
        path = tmp_path / f"{i}.png"
        if i != 13:
            path.touch()
        idx[str(path)] = {
            "category": "happy",
            "overlay_text": "下班",
            "desc": "下班",
            "scenes": ["下班"],
            "blocked": i == 12,
        }
    search = SimpleNamespace(
        get_index=lambda: idx,
        _bm25_dirty=False,
        _bm25_index=SimpleNamespace(get_top_k=lambda *a, **k: []),
    )
    strategy = SimpleNamespace(
        _calculate_recent_penalty=lambda *a: 0, _update_recent_usage=Mock()
    )
    scope = SimpleNamespace(
        _is_entry_allowed_for_event=lambda data, event: not data.get("blocked")
    )
    svc = MemeSmartSelectService(None, search, strategy, scope)
    return svc, idx, strategy, scope


@pytest.mark.asyncio
async def test_real_ranking_no_shadowing_randomness_or_early_stop(tmp_path):
    svc, idx, strategy, _ = build_ranking(tmp_path)
    with patch(
        "core.search.meme_smart_select_service.random.uniform",
        side_effect=AssertionError("random"),
    ):
        ranked = await svc._rank_emoji_candidates("", "下班", deterministic=True)
    assert len(ranked) == 12
    assert all(not idx[item[0]]["blocked"] for item in ranked)
    strategy._update_recent_usage.assert_not_called()
    # LLM 链路仍可选择并记录最近使用；过去在 entry_category(data) 处直接失败。
    assert await svc._select_emoji_smart_impl("happy", "下班") in idx
    strategy._update_recent_usage.assert_called_once()


@pytest.mark.asyncio
async def test_top_ten_facade_and_send_revalidation(tmp_path):
    svc, idx, strategy, scope = build_ranking(tmp_path)
    selector = MemeSelector.__new__(MemeSelector)
    selector.plugin = SimpleNamespace(is_send_enabled_for_event=lambda event: True)
    selector._selection_lock = asyncio.Lock()
    selector._smart_select_service = svc
    selector._selection_strategy = strategy
    selector._scope_service = scope
    selector._get_index = lambda: idx
    path = next(iter(idx))
    selector._jev_selector = SimpleNamespace(
        get_history=AsyncMock(return_value=[]), select=AsyncMock(return_value=path)
    )
    assert (
        await selector.select_emoji_with_jev(object(), "下班了", user_message="下班")
        == path
    )
    args = selector._jev_selector.select.await_args
    assert len(args.args[0]) == 10
    assert len({p for p, _ in args.args[0]}) == 10
    selector.send_emoji_with_text = AsyncMock(return_value=True)
    idx[path]["blocked"] = True
    assert not await selector.send_jev_selection(object(), path, "下班")
    selector.send_emoji_with_text.assert_not_awaited()
    idx[path]["blocked"] = False
    assert await selector.send_jev_selection(object(), path, "下班")
    strategy._update_recent_usage.assert_called_once()
    idx.pop(path)
    assert not await selector.send_jev_selection(object(), path, "下班")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,result,fail,expected",
    [
        ("typesafe", "a.png", False, "jev"),
        ("typesafe", None, False, "none"),
        ("typesafe", None, True, "llm"),
        ("llm", "a.png", False, "llm"),
    ],
)
async def test_sender_routes_preserve_llm_and_abstention(mode, result, fail, expected):
    cfg = config(enable_jev=mode == "typesafe")
    analyzer = SimpleNamespace(
        analyze_for_reply=AsyncMock(
            return_value=EmotionQuery(True, "检索词", ["happy"])
        )
    )
    selector = SimpleNamespace(
        select_emoji_with_jev=AsyncMock(return_value=result),
        send_jev_selection=AsyncMock(return_value=True),
    )
    if fail:
        selector.select_emoji_with_jev.side_effect = JevSelectionError("timeout")
    plugin = SimpleNamespace(
        plugin_config=cfg,
        enable_natural_emotion_analysis=True,
        emotion_analyzer=analyzer,
        meme_selector=selector,
    )
    engine = MemeSenderEngine(plugin)
    engine.get_meme_send_delay = lambda *a: 0
    engine.try_send_emoji = AsyncMock(return_value=True)
    engine.mark_auto_emoji_sent = AsyncMock()
    state = SimpleNamespace(mark_active_sent=Mock())
    engine.emoji_turn_state = lambda event: state
    event = SimpleNamespace(get_result=lambda: None)
    await engine.async_analyze_and_send_emoji(
        event, "这是一条完整回复", [], user_message="用户问题"
    )
    if expected == "llm":
        analyzer.analyze_for_reply.assert_awaited_once()
        engine.try_send_emoji.assert_awaited_once_with(event, ["happy"], "检索词")
        selector.send_jev_selection.assert_not_awaited()
    else:
        analyzer.analyze_for_reply.assert_not_awaited()
        engine.try_send_emoji.assert_not_awaited()
        if expected == "jev":
            selector.send_jev_selection.assert_awaited_once_with(
                event, "a.png", "这是一条完整回复"
            )
        else:
            selector.send_jev_selection.assert_not_awaited()
    assert engine.mark_auto_emoji_sent.await_count == (expected != "none")


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["m02", "none"])
async def test_rank_http_decision_to_sender(tmp_path, monkeypatch, choice):
    captured = install_http(monkeypatch, answer(choice))
    svc, idx, strategy, scope = build_ranking(tmp_path)
    plugin = SimpleNamespace(
        plugin_config=config(), enable_natural_emotion_analysis=True,
        is_send_enabled_for_event=lambda event: True,
        emotion_analyzer=SimpleNamespace(analyze_for_reply=AsyncMock()),
    )
    selector = MemeSelector.__new__(MemeSelector)
    selector.plugin = plugin
    selector._selection_lock = asyncio.Lock()
    selector._smart_select_service = svc
    selector._selection_strategy = strategy
    selector._scope_service = scope
    selector._get_index = lambda: idx
    selector._jev_selector = JevSelector(plugin)
    selector._jev_selector.get_history = AsyncMock(return_value=[
        {"role": "user", "content": "今天又加班了"},
        {"role": "assistant", "content": "辛苦啦"},
    ])
    selector.send_emoji_with_text = AsyncMock(return_value=True)
    plugin.meme_selector = selector
    engine = MemeSenderEngine(plugin)
    engine.get_meme_send_delay = lambda *args: 0
    engine.mark_auto_emoji_sent = AsyncMock()
    state = SimpleNamespace(mark_active_sent=Mock())
    engine.emoji_turn_state = lambda event: state
    event = SimpleNamespace(get_result=lambda: None)
    await engine.async_analyze_and_send_emoji(event, "终于下班了", [], user_message="下班啦")
    assert len(captured["json"]["questions"]["meme"]["criteria"]) == 11
    plugin.emotion_analyzer.analyze_for_reply.assert_not_awaited()
    if choice == "m02":
        selector.send_emoji_with_text.assert_awaited_once_with(event, list(idx)[1], "终于下班了")
        engine.mark_auto_emoji_sent.assert_awaited_once()
    else:
        selector.send_emoji_with_text.assert_not_awaited()
        engine.mark_auto_emoji_sent.assert_not_awaited()


def test_backend_config_defaults_match_schema():
    from pathlib import Path

    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))
    for key in ("enable_jev", "typesafe_api_key", "typesafe_api_base_url"):
        assert schema[key]["default"] == PluginConfig.model_fields[key].default
    assert schema["enable_jev"]["default"] is False
    assert schema["enable_jev"]["type"] == "bool"
    assert "emotion_backend" not in schema
