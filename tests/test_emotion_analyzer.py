"""PR #91: NaturalEmotionAnalyzer / SmartEmotionMatcher unit tests.

Covers:
- emotion analysis template is the bundled default (user config is ignored)
- {emotion_list} / {llm_reply} / {user_message} placeholders
- model "none" abstain behavior (_EMOTION_ABSTAIN / last_analysis_abstained)
- _parse_emotion_result parsing
- SmartEmotionMatcher abstain short-circuit
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# PluginConfig.__init__ 调用 StarTools.get_data_dir；给每个实例一个独立临时目录，
# 避免 categories.json 等状态在不同测试间互相污染。
_star_module = sys.modules.get("astrbot.api.star")
if _star_module is not None:
    def _unique_data_dir(name):
        # 每个 PluginConfig 实例都使用全新的临时目录，避免上次运行残留的
        # categories.json / category_info.json 污染本次测试的初始状态。
        return os.path.join(tempfile.mkdtemp(prefix="astrbot_stealer_cfg_test_"), name)

    _star_module.StarTools = types.SimpleNamespace(get_data_dir=_unique_data_dir)
    _star_module.Context = object
    _star_module.Star = object

from core.config.config import PluginConfig
from core.processing.natural_emotion_analyzer import (
    EmotionQuery,
    NaturalEmotionAnalyzer,
    SmartEmotionMatcher,
    _EMOTION_ABSTAIN,
    _EMOTION_ANALYSIS_DEFAULT_TEMPLATE,
)


def _build_plugin(config=None, llm_result="happy"):
    config = config if config is not None else PluginConfig(None)
    context = types.SimpleNamespace(
        llm_generate=AsyncMock(
            return_value=types.SimpleNamespace(completion_text=llm_result)
        ),
        get_current_chat_provider_id=AsyncMock(return_value="provider-test"),
    )
    return types.SimpleNamespace(plugin_config=config, context=context)


def _build_analyzer(config=None, llm_result="happy"):
    return NaturalEmotionAnalyzer(_build_plugin(config, llm_result))


def _dummy_event():
    return types.SimpleNamespace(unified_msg_origin="test-origin")


class TestTemplateLoading:
    def test_clean_text_does_not_parse_removed_emotion_markers(self):
        analyzer = _build_analyzer()
        assert analyzer._clean_text("  &&happy&&   hello  ") == "&&happy&& hello"

    def test_default_template_used_when_prompt_empty(self):
        analyzer = _build_analyzer(PluginConfig(None))
        assert analyzer._emotion_analysis_template == _EMOTION_ANALYSIS_DEFAULT_TEMPLATE

    def test_default_template_has_required_placeholders(self):
        for key in ("{emotion_list}", "{llm_reply}", "{user_message}"):
            assert key in _EMOTION_ANALYSIS_DEFAULT_TEMPLATE

    def test_default_template_is_category_neutral(self):
        text = _EMOTION_ANALYSIS_DEFAULT_TEMPLATE
        assert "每个分类机会均等" in text
        assert "不要把某一类当默认" in text
        assert "troll 仅" not in text
        assert "无法判断" not in text

    def test_default_template_renders_without_key_error(self):
        analyzer = _build_analyzer()
        prompt = analyzer._render_emotion_analysis_template(
            _EMOTION_ANALYSIS_DEFAULT_TEMPLATE,
            emotion_list="happy, sigh",
            llm_reply="又被安排加班了",
            user_message="今天好累",
        )
        assert '{"query": "摸鱼 下班 辛苦了"' in prompt
        assert "{emotion_list}" not in prompt
        assert "{llm_reply}" not in prompt
        assert "{user_message}" not in prompt

    def test_legacy_unescaped_prompt_renders_without_key_error(self):
        # 旧版本 _conf_schema.json 默认值里的输出示例带有未转义 JSON 花括号，
        # 使用 .format() 会把 {query} 当成占位符导致 KeyError。
        cfg = PluginConfig(
            {"emotion_analysis_prompt": _EMOTION_ANALYSIS_DEFAULT_TEMPLATE}
        )
        analyzer = _build_analyzer(cfg)
        prompt = analyzer._render_emotion_analysis_template(
            analyzer._emotion_analysis_template,
            emotion_list="happy, sigh",
            llm_reply="又被安排加班了",
            user_message="今天好累",
        )
        assert '{"query": "摸鱼 下班 辛苦了"' in prompt
        assert "又被安排加班了" in prompt

    def test_custom_template_is_loaded_and_stripped(self):
        cfg = PluginConfig(
            {"emotion_analysis_prompt": "  {emotion_list} | {llm_reply} | {user_message}  "}
        )
        analyzer = _build_analyzer(cfg)
        assert analyzer._emotion_analysis_template == "{emotion_list} | {llm_reply} | {user_message}"
        assert analyzer._emotion_analysis_template != _EMOTION_ANALYSIS_DEFAULT_TEMPLATE

    def test_whitespace_only_prompt_falls_back_to_default(self):
        analyzer = _build_analyzer(PluginConfig({"emotion_analysis_prompt": "   "}))
        assert analyzer._emotion_analysis_template == _EMOTION_ANALYSIS_DEFAULT_TEMPLATE

    def test_analyzer_tracks_last_analysis_abstained_flag(self):
        analyzer = _build_analyzer()
        assert analyzer.last_analysis_abstained is False


class TestEmotionListText:
    def test_build_emotion_list_uses_default_categories(self):
        analyzer = _build_analyzer()
        text = analyzer._build_emotion_list_text()
        assert "happy(" in text
        assert "angry(" in text
        assert ", " in text

    def test_build_emotion_list_with_custom_category_info(self):
        cfg = PluginConfig(
            {
                "categories": ["custom1", "custom2"],
                "category_info": {
                    "custom1": {"name": "first", "desc": "desc one"},
                    "custom2": {"name": "second", "desc": ""},
                },
            }
        )
        analyzer = _build_analyzer(cfg)
        text = analyzer._build_emotion_list_text()
        assert text == "custom1(desc one), custom2(second)"

    def test_build_emotion_list_falls_back_to_key(self):
        cfg = PluginConfig({"categories": ["alpha", "beta"], "category_info": {}})
        analyzer = _build_analyzer(cfg)
        assert analyzer._build_emotion_list_text() == "alpha(alpha), beta(beta)"


def test_closest_category_stays_within_configured_categories():
    cfg = PluginConfig({"categories": ["alpha", "beta"], "category_info": {}})
    for raw in ("", "happy", "other", "unknown", "beta"):
        assert cfg.closest_category(raw) in {"alpha", "beta"}


def test_update_config_keeps_all_custom_prompts():
    cfg = PluginConfig({})
    ok = cfg.update_config(
        {
            "custom_meme_classification_prompt": "custom vlm a",
            "custom_meme_classification_with_filter_prompt": "custom vlm b",
            "emotion_analysis_prompt": "custom emotion",
            "steal_meme": True,
        }
    )
    assert ok
    assert cfg.steal_meme is True
    assert cfg.custom_meme_classification_prompt == "custom vlm a"
    assert cfg.custom_meme_classification_with_filter_prompt == "custom vlm b"
    assert cfg.emotion_analysis_prompt == "custom emotion"


class TestParseEmotionResult:
    def test_direct_category(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_emotion_result("happy") == "happy"

    def test_category_inside_sentence(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_emotion_result("分类：angry") == "angry"
        assert analyzer._parse_emotion_result("我觉得是 sad 情绪") == "sad"

    def test_whitespace_and_case_insensitive(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_emotion_result("  Happy  ") == "happy"

    def test_none_returns_abstain_sentinel(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_emotion_result("none") is _EMOTION_ABSTAIN

    def test_none_variants_return_abstain(self):
        analyzer = _build_analyzer()
        for variant in ("none.", "none！", "none，", "None", "none "):
            assert analyzer._parse_emotion_result(variant) is _EMOTION_ABSTAIN

    def test_unparseable_returns_none(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_emotion_result("???") is None

    def test_empty_input_returns_none(self):
        analyzer = _build_analyzer()
        assert analyzer._parse_emotion_result("") is None


class TestAnalyzeEmotion(unittest.IsolatedAsyncioTestCase):
    async def test_short_reply_returns_none_without_llm(self):
        analyzer = _build_analyzer()
        with patch.object(
            analyzer, "_analyze_with_llm", new=AsyncMock(return_value="happy")
        ) as mocked:
            result = await analyzer.analyze_emotion(_dummy_event(), "ok")
        self.assertIsNone(result)
        mocked.assert_not_called()

    async def test_empty_reply_returns_none(self):
        analyzer = _build_analyzer()
        result = await analyzer.analyze_emotion(_dummy_event(), "")
        self.assertIsNone(result)

    async def test_normal_result_clears_abstain_flag_and_updates_stats(self):
        analyzer = _build_analyzer()
        with patch.object(
            analyzer, "_analyze_with_llm", new=AsyncMock(return_value="happy")
        ):
            result = await analyzer.analyze_emotion(_dummy_event(), "reply text here")
        self.assertIsInstance(result, EmotionQuery)
        self.assertEqual(result.primary, "happy")
        self.assertFalse(analyzer.last_analysis_abstained)
        self.assertEqual(analyzer.stats["total_analyses"], 1)
        self.assertEqual(analyzer.stats["successful_analyses"], 1)

    async def test_none_result_sets_abstain_flag_and_skips_stats(self):
        analyzer = _build_analyzer()
        with patch.object(
            analyzer, "_analyze_with_llm", new=AsyncMock(return_value=_EMOTION_ABSTAIN)
        ):
            result = await analyzer.analyze_emotion(_dummy_event(), "reply text here")
        self.assertIsNone(result)
        self.assertTrue(analyzer.last_analysis_abstained)
        self.assertEqual(analyzer.stats["total_analyses"], 0)

    async def test_cache_hit_returns_cached_result(self):
        analyzer = _build_analyzer()
        cache_key = analyzer._get_cache_key("msg|||reply text here")
        async with analyzer._cache_lock:
            analyzer._cache_result(cache_key, "sad")
        with patch.object(
            analyzer, "_analyze_with_llm", new=AsyncMock(return_value="happy")
        ) as mocked:
            result = await analyzer.analyze_emotion(
                _dummy_event(), "reply text here", user_message="msg"
            )
        self.assertIsInstance(result, EmotionQuery)
        self.assertEqual(result.primary, "sad")
        mocked.assert_not_called()
        self.assertEqual(analyzer.stats["cache_hits"], 1)

    async def test_fast_path_local_match_skips_llm(self):
        analyzer = _build_analyzer()
        with patch.object(
            analyzer, "_analyze_with_llm", new=AsyncMock(return_value="troll")
        ) as mocked:
            result = await analyzer.analyze_emotion(_dummy_event(), "哈哈笑死我了")
        self.assertIsInstance(result, EmotionQuery)
        self.assertEqual(result.primary, "happy")
        mocked.assert_not_called()

    async def test_llm_failure_falls_through_without_abstain(self):
        analyzer = _build_analyzer()
        with patch.object(analyzer, "_analyze_with_llm", new=AsyncMock(return_value=None)):
            result = await analyzer.analyze_emotion(_dummy_event(), "reply text here")
        self.assertIsNone(result)
        self.assertFalse(analyzer.last_analysis_abstained)
        self.assertEqual(analyzer.stats["total_analyses"], 1)

    async def test_legacy_unescaped_default_prompt_llm_path(self):
        cfg = PluginConfig(
            {"emotion_analysis_prompt": _EMOTION_ANALYSIS_DEFAULT_TEMPLATE}
        )
        analyzer = _build_analyzer(cfg)
        result = await analyzer._analyze_with_llm(
            _dummy_event(), "reply text here", user_message="msg"
        )
        self.assertIsInstance(result, EmotionQuery)
        call = analyzer.plugin.context.llm_generate.await_args
        self.assertIn('{"query": "摸鱼 下班 辛苦了"', call.kwargs["prompt"])


class TestSmartEmotionMatcher(unittest.IsolatedAsyncioTestCase):
    def _build_matcher(self, config=None):
        return SmartEmotionMatcher(_build_plugin(config))

    async def test_short_reply_skips_analysis(self):
        matcher = self._build_matcher()
        result = await matcher.analyze_and_match_emotion(_dummy_event(), "ok")
        self.assertIsNone(result)

    async def test_disabled_natural_analysis_returns_none(self):
        cfg = PluginConfig({"enable_natural_emotion_analysis": False})
        matcher = self._build_matcher(cfg)
        result = await matcher.analyze_and_match_emotion(_dummy_event(), "reply text here")
        self.assertIsNone(result)

    async def test_abstain_propagates_to_none(self):
        matcher = self._build_matcher()
        with patch.object(
            matcher.natural_analyzer,
            "analyze_emotion",
            new=AsyncMock(return_value=None),
        ):
            result = await matcher.analyze_and_match_emotion(_dummy_event(), "reply text here")
        self.assertIsNone(result)

    async def test_emotion_returned_from_analyzer(self):
        matcher = self._build_matcher()
        with patch.object(
            matcher.natural_analyzer,
            "analyze_emotion",
            new=AsyncMock(return_value=EmotionQuery(True, "happy reply", ["happy"])),
        ):
            result = await matcher.analyze_and_match_emotion(_dummy_event(), "reply text here")
        self.assertIsInstance(result, EmotionQuery)
        self.assertEqual(result.primary, "happy")
