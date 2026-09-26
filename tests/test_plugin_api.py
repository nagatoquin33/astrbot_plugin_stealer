"""PR #90: PluginAPI 待审核分类列表构建（_build_categories_list）。"""

import io
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import astrbot_plugin_stealer.api.taxonomy as taxonomy_module

from astrbot_plugin_stealer.plugin_api import PluginAPI
from astrbot_plugin_stealer.core.sources.models import ExternalSourceSecurityError


def _build_api(category_info):
    cfg = types.SimpleNamespace(get_category_info=lambda: category_info)
    plugin = types.SimpleNamespace(plugin_config=cfg)
    return PluginAPI(plugin)


def test_split_route_groups_keep_all_registered_endpoints():
    routes = []
    api = PluginAPI(types.SimpleNamespace())
    context = types.SimpleNamespace(
        register_web_api=lambda path, handler, methods, description: routes.append(
            (path, handler, methods, description)
        )
    )

    api.register(context)

    assert len(routes) == 39
    assert len({path for path, *_ in routes}) == len(routes)
    assert all(handler.__self__ is api for _, handler, _, _ in routes)
    assert {path for path, *_ in routes} >= {
        "/astrbot_plugin_stealer/images",
        "/astrbot_plugin_stealer/pending",
        "/astrbot_plugin_stealer/sources",
        "/astrbot_plugin_stealer/categories",
    }


class TestCategoryUpdateSafety:
    @staticmethod
    def _api(monkeypatch, payload):
        class FakeRequest:
            async def get_json(self):
                return payload

        config = types.SimpleNamespace(
            category_info={"happy": {"name": "开心", "desc": ""}},
            ensure_category_dirs=lambda keys: None,
            save_category_info=lambda: None,
        )
        updates = []
        plugin = types.SimpleNamespace(
            plugin_config=config,
            update_config=lambda value: updates.append(value),
        )
        monkeypatch.setattr(taxonomy_module, "request", FakeRequest())
        monkeypatch.setattr(taxonomy_module, "jsonify", lambda value: value)
        return PluginAPI(plugin), updates

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", ["../escape", "a/b", "中文", "2bad", "con", "other", "unknown", "x" * 49])
    async def test_update_rejects_unsafe_category_key(self, monkeypatch, key):
        api, updates = self._api(monkeypatch, {"categories": [{"key": key}]})
        body, status = await api._categories_update()
        assert status == 400
        assert body["success"] is False
        assert updates == []

    @pytest.mark.asyncio
    async def test_update_rejects_case_insensitive_duplicate(self, monkeypatch):
        api, updates = self._api(
            monkeypatch,
            {"categories": [{"key": "Happy"}, {"key": "happy"}]},
        )
        body, status = await api._categories_update()
        assert status == 400
        assert "重复" in body["error"]
        assert updates == []

    @pytest.mark.asyncio
    async def test_update_normalizes_key_and_bounds_display_fields(self, monkeypatch):
        api, updates = self._api(
            monkeypatch,
            {"categories": [{"key": " Custom_Tag ", "name": "名" * 60, "desc": "述" * 260}]},
        )
        body = await api._categories_update()
        assert body["success"] is True
        assert body["categories"] == ["custom_tag"]
        assert updates == [{"categories": ["custom_tag"]}]
        assert len(api._cfg.category_info["custom_tag"]["name"]) == 40
        assert len(api._cfg.category_info["custom_tag"]["desc"]) == 200

    @pytest.mark.asyncio
    async def test_update_rejects_duplicate_display_names(self, monkeypatch):
        api, updates = self._api(
            monkeypatch,
            {
                "categories": [
                    {"key": "cat_one", "name": "开心"},
                    {"key": "cat_two", "name": "开心"},
                ]
            },
        )
        body, status = await api._categories_update()
        assert status == 400
        assert "显示名称重复" in body["error"]
        assert updates == []


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


def test_build_image_item_includes_emotions_for_reanalysis_comparison():
    api = PluginAPI(types.SimpleNamespace())
    item = api._build_image_item(
        "meme.gif",
        {"hash": "h", "emotions": ["happy", "surprised"], "tags": [], "scenes": []},
    )
    assert item["emotions"] == ["happy", "surprised"]


def test_pending_and_library_share_multi_emotion_fields():
    api = _build_api([])
    metadata = {
        "path": "sample.png", "hash": "sample", "category": "dumb",
        "emotions": ["dumb", "sigh", "tired"], "overlay_text": "算了",
        "desc": "三种情绪", "tags": ["熊猫头"], "scenes": ["不想干了"], "id": 7,
    }
    pending = api._build_pending_item(metadata)
    stored = api._build_image_item(metadata["path"], metadata)
    for field in ("category", "emotions", "desc", "tags", "scenes", "overlay_text"):
        assert pending[field] == stored[field] == metadata[field]
    assert pending["id"] == 7


class TestExternalSourceUpload:
    def test_bounded_stream_write(self, tmp_path):
        target = tmp_path / "pack.zip"
        upload = types.SimpleNamespace(stream=io.BytesIO(b"zip bytes"))
        written = PluginAPI._save_source_upload_limited(upload, target, 32)
        assert written == 9
        assert target.read_bytes() == b"zip bytes"

    def test_bounded_stream_rejects_oversized_upload(self, tmp_path):
        target = tmp_path / "pack.zip"
        upload = types.SimpleNamespace(stream=io.BytesIO(b"too large"))
        with pytest.raises(ExternalSourceSecurityError):
            PluginAPI._save_source_upload_limited(upload, target, 4)


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
        assert loaded["sidebar"] == "expanded"

        updated = await api._update_dashboard_prefs(
            {"theme": "fallout", "view": "list", "sidebar": "collapsed"}
        )
        assert updated == {"theme": "fallout", "view": "list", "sidebar": "collapsed"}
        loaded = await api._load_dashboard_prefs()
        assert loaded == updated
        assert store[api.DASHBOARD_PREFS_KEY]["theme"] == "fallout"

        config.webui_theme = "dark"
        loaded = await api._load_dashboard_prefs()
        assert loaded == {"theme": "fallout", "view": "list", "sidebar": "collapsed"}
        assert store[api.DASHBOARD_PREFS_KEY] == {
            "theme": "fallout",
            "view": "list",
            "sidebar": "collapsed",
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
        assert loaded == {"theme": "auto", "view": "list", "sidebar": "expanded"}
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
            "sidebar": "expanded",
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
        assert api._normalize_sidebar("collapsed") == "collapsed"
        assert api._normalize_sidebar("other") == "expanded"
