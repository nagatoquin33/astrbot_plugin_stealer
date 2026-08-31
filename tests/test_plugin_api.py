"""PR #90: PluginAPI 待审核分类列表构建（_build_categories_list）。"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from astrbot_plugin_stealer.plugin_api import PluginAPI


def _build_api(category_info):
    cfg = types.SimpleNamespace(get_category_info=lambda: category_info)
    plugin = types.SimpleNamespace(plugin_config=cfg)
    return PluginAPI(plugin)


class TestBuildCategoriesList:
    def test_known_categories_with_zero_counts(self):
        api = _build_api(
            [
                {"key": "happy", "name": "开心", "desc": "快乐"},
                {"key": "angry", "name": "生气", "desc": "愤怒"},
            ]
        )
        result = api._build_categories_list({"happy": 3, "angry": 1})
        assert result == [
            {"key": "happy", "name": "开心", "count": 3},
            {"key": "angry", "name": "生气", "count": 1},
        ]

    def test_unknown_category_in_counts_is_appended(self):
        api = _build_api([{"key": "happy", "name": "开心", "desc": "快乐"}])
        result = api._build_categories_list({"happy": 2, "custom_x": 5})
        keys = [item["key"] for item in result]
        assert "custom_x" in keys
        custom = next(item for item in result if item["key"] == "custom_x")
        assert custom == {"key": "custom_x", "name": "custom_x", "count": 5}

    def test_sorted_by_count_desc(self):
        api = _build_api(
            [
                {"key": "happy", "name": "开心", "desc": ""},
                {"key": "sad", "name": "难过", "desc": ""},
                {"key": "angry", "name": "生气", "desc": ""},
            ]
        )
        result = api._build_categories_list({"happy": 1, "sad": 9, "angry": 4})
        assert [item["key"] for item in result] == ["sad", "angry", "happy"]
        assert [item["count"] for item in result] == [9, 4, 1]

    def test_empty_counts_returns_known_categories(self):
        api = _build_api([{"key": "happy", "name": "开心", "desc": ""}])
        result = api._build_categories_list({})
        assert result == [{"key": "happy", "name": "开心", "count": 0}]

    def test_empty_category_info_returns_counts_only(self):
        api = _build_api([])
        result = api._build_categories_list({"a": 2, "b": 1})
        assert result == [
            {"key": "a", "name": "a", "count": 2},
            {"key": "b", "name": "b", "count": 1},
        ]


class TestDashboardPrefs:
    @pytest.mark.asyncio
    async def test_page_override_persists_when_config_theme_changes(self):
        store: dict = {}

        async def get_kv(key, default=None):
            return store.get(key, default)

        async def put_kv(key, value):
            store[key] = value

        config = types.SimpleNamespace(webui_theme="minecraft")
        plugin = types.SimpleNamespace(
            plugin_config=config,
            get_kv_data=get_kv,
            put_kv_data=put_kv,
        )
        api = PluginAPI(plugin)
        loaded = await api._load_dashboard_prefs()
        assert loaded["theme"] == "minecraft"
        assert loaded["view"] == "grid"

        updated = await api._update_dashboard_prefs({"theme": "fallout", "view": "list"})
        assert updated == {"theme": "fallout", "view": "list"}
        loaded = await api._load_dashboard_prefs()
        assert loaded == updated
        assert store[api.DASHBOARD_PREFS_KEY]["theme"] == "fallout"

        config.webui_theme = "dark"
        loaded = await api._load_dashboard_prefs()
        assert loaded == {"theme": "fallout", "view": "list"}
        assert store[api.DASHBOARD_PREFS_KEY] == {
            "theme": "fallout",
            "view": "list",
        }

    @pytest.mark.asyncio
    async def test_saved_theme_does_not_require_config_snapshot(self):
        store = {"dashboard_prefs": {"theme": "auto", "view": "list"}}

        async def get_kv(key, default=None):
            return store.get(key, default)

        async def put_kv(key, value):
            store[key] = value

        plugin = types.SimpleNamespace(
            plugin_config=types.SimpleNamespace(webui_theme="minecraft"),
            get_kv_data=get_kv,
            put_kv_data=put_kv,
        )
        api = PluginAPI(plugin)

        loaded = await api._load_dashboard_prefs()
        assert loaded == {"theme": "auto", "view": "list"}
        assert store[api.DASHBOARD_PREFS_KEY] == {"theme": "auto", "view": "list"}

    @pytest.mark.asyncio
    async def test_empty_kv_read_uses_in_memory_saved_preferences(self):
        async def get_kv(_key, default=None):
            return default

        plugin = types.SimpleNamespace(
            plugin_config=types.SimpleNamespace(webui_theme="light"),
            get_kv_data=get_kv,
            _dashboard_prefs={"theme": "fallout", "view": "list"},
        )
        api = PluginAPI(plugin)

        assert await api._load_dashboard_prefs() == {
            "theme": "fallout",
            "view": "list",
        }

    @pytest.mark.asyncio
    async def test_invalid_theme_update_keeps_valid_override(self):
        store: dict = {}

        async def get_kv(key, default=None):
            return store.get(key, default)

        async def put_kv(key, value):
            store[key] = value

        plugin = types.SimpleNamespace(
            plugin_config=types.SimpleNamespace(webui_theme="light"),
            get_kv_data=get_kv,
            put_kv_data=put_kv,
        )
        api = PluginAPI(plugin)
        await api._update_dashboard_prefs({"theme": "fallout"})

        updated = await api._update_dashboard_prefs({"theme": "not-a-theme"})
        assert updated["theme"] == "fallout"

    def test_normalize_theme_and_unknown(self):
        api = _build_api([])
        assert api._normalize_theme("dark") == "dark"
        assert api._normalize_theme("midnight") == "auto"
        assert api._normalize_theme("nope") == "auto"
        assert api._normalize_view("list") == "list"
        assert api._normalize_view("other") == "grid"
