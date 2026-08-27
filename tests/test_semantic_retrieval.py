"""语义检索：VLM 新字段、查询改写、嵌入文档拼接。"""

import types

from core.processing.classification_parser import ClassificationParser
from core.processing.natural_emotion_analyzer import (
    EmotionQuery,
    NaturalEmotionAnalyzer,
)
from core.processing.semantic_schema import build_meme_search_text
from core.search.embedding_service import EmbeddingService
from core.search.meme_smart_select_service import MemeSmartSelectService


def test_parser_reads_overlay_and_emotions():
    parser = ClassificationParser(plugin_instance=None)
    payload = """
    {"category": "sigh", "emotions": ["sigh", "tired"], "tags": ["熊猫头", "躺平"],
     "description": "熊猫头平躺闭眼一脸放弃", "overlay_text": "算了",
     "scenes": ["又被安排加班", "算了不想干了"]}
    """
    category, tags, desc, emotion, scenes, overlay, emotions = (
        parser._parse_classification_response(payload, "x.png")
    )
    assert category == "sigh"
    assert overlay == "算了"
    assert "tired" in emotions
    assert emotions[0] == "sigh"
    assert "算了" in scenes
    assert "熊猫头" in tags
    assert "放弃" in desc


def test_parser_unknown_category_falls_back_to_closest():
    class _Proc:
        def _normalize_category(self, raw, fallback_other=True):
            text = str(raw or "").strip().lower()
            if text in {"happy", "sad"}:
                return text
            return "confused" if fallback_other else ""

    plugin = types.SimpleNamespace(image_processor_service=_Proc())
    parser = ClassificationParser(plugin_instance=plugin)
    category, *_rest, overlay, emotions = parser._parse_classification_response(
        '{"category": "whatever", "description": "文字梗", "overlay_text": "我不听"}',
        "x.png",
    )
    assert category == "confused"
    assert overlay == "我不听"
    assert "confused" in emotions


def test_sanitize_scenes_keeps_overlay_and_short_phrases():
    parser = ClassificationParser(plugin_instance=None)
    long_scene = "这是一句明显超过长度限制的编造对话情境" * 3
    scenes = parser.sanitize_scenes(
        ["算了", long_scene],
        overlay_text="算了",
    )
    assert scenes[0] == "算了"
    assert len(scenes) == 1
    assert all(len(item) <= 40 for item in scenes)


def test_search_text_includes_manual_character():
    text = build_meme_search_text(
        {
            "category": "happy",
            "desc": "棕发小女孩唱歌",
            "character": "neurosama",
            "overlay_text": "heart",
        },
        character_info={"neurosama": {"name": "Neuro-sama"}},
    )
    assert "neurosama" in text
    assert "Neuro-sama" in text
    assert text.index("heart") < text.index("neurosama")


def test_search_text_puts_overlay_first():
    text = build_meme_search_text(
        {
            "category": "sigh",
            "desc": "熊猫头躺平",
            "overlay_text": "算了",
            "tags": ["摆烂"],
            "scenes": ["被安排加班"],
            "emotions": ["sigh", "tired"],
        },
        category_info={"sigh": {"name": "无奈", "desc": "叹气、摆烂"}},
    )
    assert text.startswith("算了")
    assert "被安排加班" in text
    assert "无奈" in text
    assert "tired" in text


def test_build_bm25_search_text_weights_overlay_and_keeps_category_weak():
    text = build_meme_search_text(
        {
            "category": "sigh",
            "desc": "熊猫头躺平",
            "overlay_text": "算了",
            "tags": ["摆烂"],
            "scenes": ["被安排加班"],
            "emotions": ["sigh", "tired"],
        },
        category_info={"sigh": {"name": "无奈", "desc": "叹气、摆烂"}},
        bm25=True,
    )
    assert text.count("算了") == 3
    assert "无奈" not in text
    assert "叹气" not in text
    assert "sigh" in text
    assert text.count("被安排加班") == 2


def test_embedding_service_uses_overlay_in_search_text():
    plugin = types.SimpleNamespace(plugin_config=types.SimpleNamespace(category_info={}))
    svc = EmbeddingService(plugin)
    text = svc._build_search_text(
        {"category": "dumb", "desc": "猫猫瞪眼", "overlay_text": "啊？", "tags": [], "scenes": []}
    )
    assert "啊？" in text
    assert "猫猫瞪眼" in text


def test_overlay_recall_matches_context_substring():
    svc = MemeSmartSelectService(plugin_instance=None)
    svc._is_entry_allowed_for_event = lambda data, event: True  # type: ignore[method-assign]
    idx = {
        "/a.png": {"overlay_text": "算了", "category": "sigh"},
        "/b.png": {"overlay_text": "我超爱", "category": "love"},
    }
    hits = svc._overlay_recall_paths(idx, "又被安排加班，算了不想干了", event=None)
    assert "/a.png" in hits
    assert "/b.png" not in hits


def test_scene_recall_matches_context_substring():
    svc = MemeSmartSelectService(plugin_instance=None)
    svc._is_entry_allowed_for_event = lambda data, event: True  # type: ignore[method-assign]
    idx = {
        "/a.png": {"scenes": ["又被安排加班"], "category": "sigh"},
        "/b.png": {"scenes": ["哈哈哈哈"], "category": "happy"},
    }
    hits = svc._scene_recall_paths(idx, "又被安排加班，不想干了", event=None)
    assert "/a.png" in hits
    assert "/b.png" not in hits


def test_emotion_query_parses_json_and_legacy_word():
    analyzer = NaturalEmotionAnalyzer(
        types.SimpleNamespace(
            plugin_config=types.SimpleNamespace(
                get_categories=lambda: ["happy", "sigh", "tired", "dumb"],
                category_info={},
                emotion_analysis_provider_id="",
                enable_natural_emotion_analysis=True,
            ),
            context=types.SimpleNamespace(),
        )
    )
    analyzer.categories = ["happy", "sigh", "tired", "dumb"]
    parsed = analyzer._parse_emotion_query(
        '{"should_send": true, "query": "被安排加班摆烂", "emotions": ["sigh", "tired"]}',
        fallback_query="原文",
    )
    assert isinstance(parsed, EmotionQuery)
    assert parsed.search_query == "被安排加班摆烂"
    assert parsed.emotion_priors == ["sigh", "tired"]

    skipped = analyzer._parse_emotion_query(
        '{"should_send": false}', fallback_query="原文"
    )
    assert isinstance(skipped, EmotionQuery)
    assert skipped.should_send is True
    assert skipped.search_query == "原文"
    assert skipped.emotion_priors == []

    legacy = analyzer._parse_emotion_query("sigh", fallback_query="原文")
    assert isinstance(legacy, EmotionQuery)
    assert legacy.primary == "sigh"
    assert legacy.search_query == "原文"


def test_bundled_prompts_are_category_neutral_and_tag_light():
    import json
    from pathlib import Path

    data = json.loads(
        (Path(__file__).resolve().parents[1] / "prompts.json").read_text(encoding="utf-8")
    )
    for key in (
        "EMOJI_CLASSIFICATION_PROMPT",
        "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT",
    ):
        text = data[key]
        assert "每个分类机会均等" in text
        assert "不要把某一类当默认" in text
        assert "0~3" in text
        assert "2~6" not in text
        assert "troll 不是" not in text


def test_get_prompts_falls_back_to_bundled_when_custom_blank():
    from core.config.config import PluginConfig

    bundled = {
        "EMOJI_CLASSIFICATION_PROMPT": "PLUGIN A {emotion_list}",
        "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT": "PLUGIN B {emotion_list}",
    }
    fake_cfg = types.SimpleNamespace(
        custom_meme_classification_prompt="",
        custom_meme_classification_with_filter_prompt="",
    )
    result = PluginConfig.get_prompts(fake_cfg, bundled)
    assert result["emoji_classification_prompt"] == "PLUGIN A {emotion_list}"
    assert result["emoji_classification_with_filter_prompt"] == "PLUGIN B {emotion_list}"


def test_get_prompts_prefers_custom_vlm_prompts():
    from core.config.config import PluginConfig

    bundled = {
        "EMOJI_CLASSIFICATION_PROMPT": "PLUGIN A {emotion_list}",
        "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT": "PLUGIN B {emotion_list}",
    }
    fake_cfg = types.SimpleNamespace(
        custom_meme_classification_prompt="CUSTOM A {emotion_list}",
        custom_meme_classification_with_filter_prompt="CUSTOM B {emotion_list}",
    )
    result = PluginConfig.get_prompts(fake_cfg, bundled)
    assert result["emoji_classification_prompt"] == "CUSTOM A {emotion_list}"
    assert result["emoji_classification_with_filter_prompt"] == "CUSTOM B {emotion_list}"
