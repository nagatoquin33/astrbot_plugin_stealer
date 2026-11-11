"""Offline behavioral checks for relevance, tool contracts and library search."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.db.database_service import DatabaseService
from core.processing.natural_emotion_analyzer import (
    NaturalEmotionAnalyzer,
    _EMOTION_ANALYSIS_DEFAULT_TEMPLATE,
)
from core.search.meme_scope_service import MemeScopeService
from core.search.meme_selector import MemeSelector
from tests.test_auto_emoji_flow import DummyEvent, _build_main


def selector_for(index, tmp_path):
    config = SimpleNamespace(
        categories=["happy", "tired", "thank", "sigh"], category_info={},
        character_info={"cat": {"name": "猫猫"}, "dog": {"name": "狗狗"}},
        get_keyword_map=lambda: {"开心": "happy", "感谢": "thank"},
        enable_embedding_search=False,
    )
    plugin = SimpleNamespace(plugin_config=config, cache_dir=tmp_path)
    selector = MemeSelector(plugin)
    selector._search_engine.get_index = lambda: index
    return selector


@pytest.mark.parametrize("query,expected", [
    ("安慰刚加班的朋友 辛苦了早点休息", "comfort"),
    ("谢谢你帮忙 感谢", "thanks"),
    ("我今天不开心", "unhappy"),
    ("猫猫 谢谢你", "cat"),
])
def test_intent_and_overlay_rank_above_distractors(query, expected, tmp_path):
    index = {
        "comfort": {"category": "tired", "overlay_text": "辛苦了早点休息",
                    "desc": "温柔关心", "scenes": ["朋友刚加班"]},
        "taunt": {"category": "happy", "overlay_text": "哈哈活该", "desc": "嘲笑加班",
                  "is_favorite": True, "use_count": 1000},
        "thanks": {"category": "thank", "overlay_text": "谢谢你帮忙", "desc": "感谢帮助"},
        "happy": {"category": "happy", "overlay_text": "开心", "desc": "开心大笑"},
        "unhappy": {"category": "sigh", "overlay_text": "不开心", "desc": "低落"},
        "cat": {"category": "thank", "character": "cat", "overlay_text": "谢谢你"},
        "dog": {"category": "thank", "character": "dog", "overlay_text": "谢谢你"},
    }
    results = asyncio.run(selector_for(index, tmp_path).smart_search(query, idx=index))
    assert results and results[0][0] == expected
    if "不开心" in query:
        assert "happy" not in [row[0] for row in results]


def test_unrelated_query_does_not_fall_back_to_emotion_bucket(tmp_path):
    index = {"happy": {"category": "happy", "desc": "开心大笑"}}
    results = asyncio.run(selector_for(index, tmp_path).smart_search("量子纠缠公式", idx=index))
    assert results == []


def test_semantic_embedding_recall_passes_without_character_or_literal_overlap(tmp_path):
    index = {
        "sleepy": {
            "category": "tired",
            "desc": "猫耳少女抱着枕头打瞌睡",
            "tags": ["睡觉"],
            "scenes": ["深夜困了"],
        }
    }
    selector = selector_for(index, tmp_path)
    service = selector._smart_select_service
    service._embedding_service = SimpleNamespace(
        is_available=lambda: True,
        search=AsyncMock(return_value=[("sleepy", 0.40)]),
    )
    ranked = asyncio.run(
        service._rank_emoji_candidates("tired", "熬夜折腾完代码赶紧去休息", idx=index)
    )
    assert [item[0] for item in ranked] == ["sleepy"]


def test_automatic_selection_only_randomizes_close_scores(tmp_path):
    selector = selector_for({}, tmp_path)
    service = selector._smart_select_service
    service._rank_emoji_candidates = AsyncMock(return_value=[
        ("best", .8, .8, 0, 0, "happy"), ("weak", .4, .4, 0, 0, "happy"),
    ])
    paths = [asyncio.run(service._select_emoji_smart_impl("happy", "hello")) for _ in range(20)]
    assert set(paths) == {"best"}


def tool_main(index, tmp_path):
    main = _build_main(0)
    main.db_service = SimpleNamespace(count_total=lambda: len(index), get_index_cache_readonly=lambda: index)
    main.meme_selector = selector_for(index, tmp_path)
    main.plugin_config = main.meme_selector.plugin.plugin_config
    main.plugin_config.get_categories = lambda: main.plugin_config.categories
    return main


async def output(generator):
    return "\n".join([chunk async for chunk in generator])


@pytest.mark.asyncio
async def test_tool_skips_missing_files_without_breaking_ids_and_expires_after_send(tmp_path):
    image = tmp_path / "ok.png"
    image.write_bytes(b"test")
    path = str(image)
    main = tool_main({path: {"scenes": ["朋友加班"]}}, tmp_path)
    main._search_meme_candidates = AsyncMock(return_value=[
        (str(tmp_path / "missing.png"), "missing", "tired", ""),
        (path, "安慰", "tired", ""),
    ])
    event = DummyEvent()
    text = await output(main.search_meme(event, "安慰朋友"))
    assert "[1]" in text and "[2]" not in text
    assert "适合回应：朋友加班" in text
    main.meme_selector.is_path_allowed_for_event = lambda *args: True
    main.meme_selector.send_emoji_message = AsyncMock(return_value="image")
    main.meme_selector.record_emoji_usage = AsyncMock()
    assert "发送成功" in await output(main.send_meme(event, 1))
    assert "candidate_expired" in await output(main.send_meme(event, 1))
    assert main.meme_selector.send_emoji_message.await_count == 1


@pytest.mark.asyncio
async def test_empty_search_clears_previous_candidates(tmp_path):
    main = tool_main({}, tmp_path)
    event = DummyEvent()
    state = main._emoji_sender_engine.emoji_turn_state(event)
    state.set_candidates([{"path": __file__}])
    assert "缺少 query" in await output(main.search_meme(event, ""))
    assert state.get_candidates() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", [True, 1.9, "1.9"])
async def test_tool_rejects_non_integer_ids(bad_id, tmp_path):
    main = tool_main({}, tmp_path)
    assert "invalid_id" in await output(main.send_meme(DummyEvent(), bad_id))


@pytest.mark.asyncio
async def test_optional_reply_and_character_filter_use_existing_metadata(tmp_path):
    index = {
        "cat": {"character": "cat", "overlay_text": "辛苦了"},
        "dog": {"character": "dog", "overlay_text": "辛苦了"},
    }
    main = tool_main(index, tmp_path)
    results = await main._search_meme_candidates(DummyEvent(), "安慰", reply="辛苦了", character="猫猫")
    assert [row[0] for row in results] == ["cat"]


@pytest.mark.asyncio
async def test_llm_can_react_with_meme_without_user_request_or_planned_text(tmp_path):
    image = tmp_path / "reaction.png"
    image.write_bytes(b"test")
    path = str(image)
    index = {path: {"category": "happy", "overlay_text": "笑死", "scenes": ["这事太离谱了"]}}
    main = tool_main(index, tmp_path)
    event = DummyEvent()
    event.get_message_str = lambda: "今天上班忘了带电脑，坐了一天"
    main.meme_selector.is_path_allowed_for_event = lambda *args: True
    main.meme_selector.send_emoji_message = AsyncMock(return_value="image")
    main.meme_selector.record_emoji_usage = AsyncMock()
    assert "[1]" in await output(main.search_meme(event, "笑死"))
    assert "发送成功" in await output(main.send_meme(event, 1))
    assert main._emoji_sender_engine.emoji_turn_state(event).is_active_sent()
    assert main.meme_selector.send_emoji_message.await_count == 1


def test_local_scope_requires_known_origin_and_current_session():
    scope = MemeScopeService(SimpleNamespace(plugin_config=SimpleNamespace(
        get_event_target=lambda event: ("", ""))))
    assert not scope._is_entry_allowed_for_event({"scope_mode": "local", "origin_target": "group:1"}, DummyEvent())
    assert not scope._is_entry_allowed_for_event({"scope_mode": "local"}, DummyEvent())
    assert scope._is_entry_allowed_for_event({"scope_mode": "public"}, DummyEvent())


def test_library_search_combines_fields_and_treats_wildcards_literally(tmp_path):
    db = DatabaseService(tmp_path / "library.db")
    asyncio.run(db.insert_batch([
        {"path": "a.png", "category": "thank", "overlay_text": "谢谢", "tags": ["猫猫"], "desc": "100%确定_a"},
        {"path": "b.png", "category": "thank", "overlay_text": "谢谢", "tags": ["狗狗"], "desc": "100分确定ba"},
    ]))
    for query in ("猫猫 谢谢", "%", "_a"):
        rows, count, categories = db.get_emojis_paginated(search_query=query)
        assert [row["path"] for row in rows] == ["a.png"]
        assert count == 1 and categories == {"thank": 1}


def test_existing_custom_emotion_template_is_preserved():
    analyzer = NaturalEmotionAnalyzer.__new__(NaturalEmotionAnalyzer)
    analyzer.plugin_config = SimpleNamespace(emotion_analysis_prompt="my custom {llm_reply}")
    assert analyzer._load_emotion_analysis_template() == "my custom {llm_reply}"
    analyzer.plugin_config.emotion_analysis_prompt = ""
    assert analyzer._load_emotion_analysis_template() == _EMOTION_ANALYSIS_DEFAULT_TEMPLATE


def test_basic_settings_hide_advanced_fields_without_removing_config_keys():
    schema = json.loads((Path(__file__).parents[1] / "_conf_schema.json").read_text(encoding="utf-8"))
    defaults = {key: value.get("default") for key, value in schema.items()}
    visible = [key for key, value in schema.items() if not value.get("invisible") and all(
        defaults.get(other) == expected for other, expected in value.get("condition", {}).items())]
    assert "show_advanced_settings" in visible
    assert "emotion_analysis_prompt" not in visible
    assert "vision_provider_id" in visible
    assert len(visible) <= 18
    defaults["show_advanced_settings"] = True
    assert all(defaults.get(key) == value for key, value in schema["emotion_analysis_prompt"]["condition"].items())
