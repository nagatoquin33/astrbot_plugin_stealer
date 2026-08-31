"""
meme_selector 模块单元测试
测试表情包选择器的去重机制和评分逻辑
"""

import pytest
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock


def _install_astrbot_stubs():
    # 检查是否已安装兼容的 stubs (由 conftest.py 安装)
    if "astrbot.api.message_components" in sys.modules:
        existing_image = sys.modules["astrbot.api.message_components"].Image
        if hasattr(existing_image, "fromBase64") and hasattr(
            existing_image, "convert_to_file_path"
        ):
            test_result = existing_image.fromBase64("test")
            if test_result == "b64:test":
                return  # stubs 已兼容，跳过安装

    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    astrbot_module = sys.modules.get("astrbot") or types.ModuleType("astrbot")
    api_module = sys.modules.get("astrbot.api") or types.ModuleType("astrbot.api")
    event_module = sys.modules.get("astrbot.api.event") or types.ModuleType("astrbot.api.event")
    star_module = sys.modules.get("astrbot.api.star") or types.ModuleType("astrbot.api.star")
    message_components_module = (
        sys.modules.get("astrbot.api.message_components")
        or types.ModuleType("astrbot.api.message_components")
    )

    api_module.logger = logger
    event_module.AstrMessageEvent = object
    event_module.MessageChain = list
    star_module.Context = object
    star_module.StarTools = object
    class Image:
        @classmethod
        def fromBase64(cls, value):
            return f"b64:{value}"

        async def convert_to_file_path(self):
            return ""

    message_components_module.Image = Image
    message_components_module.Plain = object

    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot.api.event"] = event_module
    sys.modules["astrbot.api.star"] = star_module
    sys.modules["astrbot.api.message_components"] = message_components_module


_install_astrbot_stubs()

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.search.meme_selector import MemeSelector


class MockPluginConfig:
    categories = ["happy", "sad", "angry", "tired", "dumb", "confused"]
    category_info = {}
    keyword_map = {}

    def get_keyword_map(self):
        return self.keyword_map


class MockCacheService:
    def __init__(self):
        self._index = {}
        self._cache = {}

    def get_index_cache_readonly(self):
        return self._index

    def get_cache(self, cache_name):
        return self._cache.get(cache_name)

    async def set_cache(self, cache_name, cache_data, persist=True):
        self._cache[cache_name] = cache_data


class MockPlugin:
    def __init__(self):
        self.categories = ["happy", "sad", "angry", "tired", "dumb", "confused"]
        self.plugin_config = MockPluginConfig()
        self.cache_service = MockCacheService()


class TestRecentUsage:
    """测试最近使用记录功能"""

    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)

    def test_update_recent_usage_adds_new_entry(self):
        self.selector._update_recent_usage("happy", "/path/to/emoji1.png")
        recent = self.selector._get_recent_usage("happy")
        assert "/path/to/emoji1.png" in recent

    def test_update_recent_usage_removes_duplicate(self):
        self.selector._update_recent_usage("happy", "/path/to/emoji1.png")
        self.selector._update_recent_usage("happy", "/path/to/emoji2.png")
        self.selector._update_recent_usage("happy", "/path/to/emoji1.png")
        recent = self.selector._get_recent_usage("happy")
        assert recent.count("/path/to/emoji1.png") == 1
        assert recent[-1] == "/path/to/emoji1.png"

    def test_update_recent_usage_respects_max_limit(self):
        for i in range(15):
            self.selector._update_recent_usage("happy", f"/path/emoji_{i}.png")
        recent = self.selector._get_recent_usage("happy")
        assert len(recent) <= self.selector.MAX_RECENT_USAGE

    def test_different_categories_have_separate_history(self):
        self.selector._update_recent_usage("happy", "/path/to/happy_emoji.png")
        self.selector._update_recent_usage("sad", "/path/to/sad_emoji.png")
        assert len(self.selector._get_recent_usage("happy")) == 1
        assert len(self.selector._get_recent_usage("sad")) == 1
        assert self.selector._get_recent_usage("happy")[0] == "/path/to/happy_emoji.png"
        assert self.selector._get_recent_usage("sad")[0] == "/path/to/sad_emoji.png"


class TestRecentPenalty:
    """测试历史惩罚机制"""

    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)

    def test_no_penalty_for_first_use(self):
        penalty = self.selector._calculate_recent_penalty("happy", "/path/to/emoji.png")
        assert penalty == 0.0

    def test_penalty_for_recently_used(self):
        for i in range(5):
            self.selector._update_recent_usage("happy", f"/path/to/emoji_{i}.png")
        penalty = self.selector._calculate_recent_penalty("happy", "/path/to/emoji_4.png")
        assert penalty > 0

    def test_penalty_decreases_with_recency(self):
        for i in range(4):
            self.selector._update_recent_usage("happy", f"/path/to/emoji_{i}.png")
        penalty_recent = self.selector._calculate_recent_penalty("happy", "/path/to/emoji_3.png")
        penalty_old = self.selector._calculate_recent_penalty("happy", "/path/to/emoji_0.png")
        assert penalty_recent > penalty_old

    def test_no_penalty_for_unused_path(self):
        self.selector._update_recent_usage("happy", "/path/to/emoji1.png")
        penalty = self.selector._calculate_recent_penalty("happy", "/path/to/emoji2.png")
        assert penalty == 0.0

    def test_no_penalty_for_different_category(self):
        self.selector._update_recent_usage("happy", "/path/to/emoji.png")
        penalty = self.selector._calculate_recent_penalty("sad", "/path/to/emoji.png")
        assert penalty == 0.0


class TestCandidateCategories:
    """测试候选分类获取"""

    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)

    def test_exact_match_returns_self(self):
        cats = self.selector._get_candidate_categories("happy")
        assert "happy" in cats

    def test_fuzzy_match_returns_similar(self):
        cats = self.selector._get_candidate_categories("开心")
        assert len(cats) > 0

    def test_limit_respected(self):
        cats = self.selector._get_candidate_categories("a", limit=2)
        assert len(cats) <= 2

    def test_empty_input_returns_empty(self):
        cats = self.selector._get_candidate_categories("")
        assert cats == []


class TestCanonPath:
    """测试路径规范化"""

    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)

    def test_backslash_to_forward_slash(self):
        result = self.selector._canon_path("C:\\path\\to\\emoji.png")
        assert "\\" not in result

    def test_case_insensitive(self):
        result1 = self.selector._canon_path("/Path/To/Emoji.png")
        result2 = self.selector._canon_path("/path/to/emoji.png")
        assert result1 == result2

    def test_slash_normalized(self):
        result = self.selector._canon_path("/path/to\\emoji.png")
        assert "\\" not in result


class TestSearchImagesDeduplication:
    """测试搜索结果去重"""

    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)
        self.plugin.cache_service._index = {
            "/path/happy_1.png": {
                "category": "happy",
                "desc": "开心大笑",
                "tags": ["开心"],
                "scenes": ["高兴"],
            },
            "/path/happy_2.png": {
                "category": "happy",
                "desc": "微笑",
                "tags": ["微笑"],
                "scenes": ["开心"],
            },
            "/path/sad_1.png": {
                "category": "sad",
                "desc": "难过哭泣",
                "tags": ["难过"],
                "scenes": ["伤心"],
            },
        }
        self.selector._bm25_dirty = True

    def test_recently_used_paths_excluded_from_search(self):
        self.selector._update_recent_usage("happy", "/path/happy_1.png")
        self.selector._bm25_dirty = True
        import asyncio
        results = asyncio.run(
            self.selector.search_images("开心", limit=3, idx=self.plugin.cache_service._index)
        )
        result_paths = [r[0] for r in results]
        assert "/path/happy_1.png" not in result_paths

    def test_unused_paths_included_in_search(self):
        self.selector._bm25_dirty = True
        import asyncio
        results = asyncio.run(
            self.selector.search_images("开心", limit=3, idx=self.plugin.cache_service._index)
        )
        assert len(results) > 0


class TestBm25Signature:
    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)
        self.plugin.cache_service._index = {
            "/path/happy_1.png": {
                "category": "happy",
                "desc": "旧描述",
                "tags": ["旧标签1", "旧标签2"],
                "scenes": ["旧场景"],
            }
        }

    def test_signature_changes_when_searchable_content_changes(self):
        original = self.selector._compute_bm25_signature(self.plugin.cache_service._index)
        self.plugin.cache_service._index["/path/happy_1.png"]["desc"] = "新描述"
        self.plugin.cache_service._index["/path/happy_1.png"]["tags"] = ["新标签1", "新标签2"]

        updated = self.selector._compute_bm25_signature(self.plugin.cache_service._index)
        assert original != updated


class TestBm25IndexBuild:
    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)

        class FakeDbService:
            def count_total(self):
                return 1

            def get_corpus_signature(self):
                return "db-signature"

            def get_index_cache_readonly(self):
                return {
                    "/db/path.png": {
                        "category": "sad",
                        "desc": "数据库旧描述",
                        "tags": ["旧库标签"],
                        "scenes": ["旧库场景"],
                    }
                }

        self.plugin.db_service = FakeDbService()
        self.plugin.cache_service._cache["bm25_cache"] = {
            "signature": "db-signature",
            "documents": [["old", "cached", "doc"]],
            "doc_paths": ["/db/path.png"],
        }

    def test_explicit_idx_rebuilds_instead_of_reusing_db_cached_signature(self):
        import asyncio

        explicit_idx = {
            "/fresh/path.png": {
                "category": "happy",
                "desc": "新的显式索引",
                "tags": ["新标签"],
                "scenes": ["新场景"],
            }
        }

        asyncio.run(self.selector._build_bm25_index(explicit_idx))

        assert self.selector._bm25_doc_paths == ["/fresh/path.png"]
        assert self.selector._bm25_signature != "db-signature"


class TestFallbackSearch:
    """测试降级搜索去重"""

    def setup_method(self):
        self.plugin = MockPlugin()
        self.selector = MemeSelector(self.plugin)
        self.plugin.cache_service._index = {
            "/path/happy_1.png": {
                "category": "happy",
                "desc": "开心大笑",
                "tags": ["开心", "大笑"],
                "scenes": ["高兴"],
            },
            "/path/happy_2.png": {
                "category": "happy",
                "desc": "微笑",
                "tags": ["微笑"],
                "scenes": ["开心"],
            },
        }

    def test_recently_used_excluded_in_fallback(self):
        self.selector._update_recent_usage("happy", "/path/happy_1.png")
        import asyncio
        results = asyncio.run(
            self.selector._search_images_fallback(
                "开心", limit=5, idx=self.plugin.cache_service._index
            )
        )
        result_paths = [r[0] for r in results]
        assert "/path/happy_1.png" not in result_paths


class TestSmartSearchQueryExpansion:
    def setup_method(self):
        self.plugin = MockPlugin()
        self.plugin.plugin_config.keyword_map = {"无语": "dumb"}
        self.selector = MemeSelector(self.plugin)

    def test_natural_phrase_expands_to_category_key(self):
        calls = []
        expected = [("/path/dumb.png", "无语", "dumb", "沉默")]

        async def _search_images(query, *, limit, idx, event):
            calls.append(query)
            return expected if query == "来个无语的表情包 dumb" else []

        self.selector._smart_select_service.search_images = _search_images

        import asyncio

        results = asyncio.run(
            self.selector.smart_search("来个无语的表情包", limit=5, idx={})
        )

        assert results == expected
        assert calls == ["来个无语的表情包 dumb"]


class _DummyTurnState:
    def is_active_sent(self):
        return False


class _DummyResult:
    def __init__(self):
        self.chain = []
        self.file_images = []
        self.base64_images = []
        self.messages = []
        self.result_content_type = "mixed"

    def set_result_content_type(self, value):
        self.result_content_type = value
        return self

    def message(self, value):
        self.messages.append(value)
        return self

    def file_image(self, value):
        self.file_images.append(value)
        return self

    def base64_image(self, value):
        self.base64_images.append(value)
        return self

    def stop_event(self):
        return self


class _DummyEvent:
    def __init__(self, platform_name="discord"):
        self.platform_name = platform_name
        self.sent = []
        self._result = _DummyResult()

    async def send(self, payload):
        self.sent.append(payload)

    def make_result(self):
        return _DummyResult()

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def get_platform_name(self):
        return self.platform_name


class TestSendPathOptimization:
    def setup_method(self):
        # 重新安装 stubs 以确保使用正确的版本
        _install_astrbot_stubs()
        self.plugin = MockPlugin()
        self.plugin.send_meme_as_gif = False
        self.plugin.image_processor_service = types.SimpleNamespace(
            _file_to_gif_base64=AsyncMock(return_value="encoded-image")
        )
        self.plugin._emoji_turn_state = lambda event: _DummyTurnState()
        self.selector = MemeSelector(self.plugin)
        self.selector._check_group_allowed = lambda event: True
        self.selector.record_emoji_usage = AsyncMock()
        self.selector._try_send_telegram_sticker = AsyncMock(return_value=False)

    def test_send_emoji_with_text_prefers_file_image_when_supported(self, tmp_path):
        image_path = tmp_path / "emoji.png"
        image_path.write_bytes(b"fake")
        event = _DummyEvent(platform_name="discord")

        import asyncio

        asyncio.run(self.selector.send_emoji_with_text(event, str(image_path), "hello"))

        assert len(event.sent) == 1
        assert event.sent[0].file_images == [str(image_path)]
        self.plugin.image_processor_service._file_to_gif_base64.assert_not_awaited()

    def test_send_emoji_with_text_falls_back_to_base64_for_aiocqhttp(self, tmp_path):
        image_path = tmp_path / "emoji.png"
        image_path.write_bytes(b"fake")
        event = _DummyEvent(platform_name="aiocqhttp")

        import asyncio

        asyncio.run(self.selector.send_emoji_with_text(event, str(image_path), "hello"))

        assert event.sent == [["b64:encoded-image"]]
        self.plugin.image_processor_service._file_to_gif_base64.assert_awaited_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
