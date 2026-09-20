import types
from types import SimpleNamespace

import pytest

from core.events.event_context import (
    get_event_platform_name,
    get_event_session_key,
    normalize_event_value,
)
from core.util.blacklist import add_blacklist_hash
from core.util.normalization import (
    normalize_category_key,
    canonicalize_path,
    normalize_character_key,
    normalize_label_list,
    normalize_scope_mode,
)
from core.config.config import PluginConfig


@pytest.mark.parametrize("raw, expected", [(" Happy ", "happy"), ("meme_2", "meme_2"), ("foo-bar", "foo-bar")])
def test_normalize_category_key_accepts_portable_slugs(raw, expected):
    assert normalize_category_key(raw) == expected


@pytest.mark.parametrize("raw", ["", "../outside", "a/b", "中文", "2happy", "a.b", "con", "COM1", "other", "unknown", "a" * 49])
def test_normalize_category_key_rejects_unsafe_names(raw):
    with pytest.raises(ValueError):
        normalize_category_key(raw)


def test_ensure_category_dir_stays_under_storage_root(tmp_path):
    holder = SimpleNamespace(categories_dir=tmp_path / "categories")
    holder.categories_dir.mkdir()
    created = PluginConfig.ensure_category_dir(holder, "custom_tag")
    assert created == holder.categories_dir / "custom_tag"
    assert created.is_dir()
    with pytest.raises(ValueError):
        PluginConfig.ensure_category_dir(holder, "../outside")
    assert not (tmp_path / "outside").exists()


def test_metadata_normalizers_share_alias_and_list_rules():
    assert normalize_scope_mode("global") == "public"
    assert normalize_scope_mode("private") == "local"
    assert normalize_scope_mode("invalid", default=None) is None
    assert normalize_character_key("  Hatsune_Miku ") == "hatsune_miku"
    assert normalize_label_list("开心，猫、猫；大笑") == ["开心", "猫", "大笑"]
    assert normalize_label_list("a,b,a", allow_duplicates=True, csv_only=True) == [
        "a",
        "b",
        "a",
    ]
    assert normalize_label_list(["a", "b", "a"], max_count=2) == ["a", "b"]


def test_canonicalize_path_unifies_separators_and_segments():
    assert canonicalize_path(r"C:\memes\happy\..\cat.gif") == canonicalize_path(
        "C:/memes/cat.gif"
    )
    assert "\\" not in canonicalize_path(r"C:\memes\cat.gif")


def test_event_context_uses_consistent_platform_and_session_fallbacks():
    event = types.SimpleNamespace(
        get_platform_name=lambda: "",
        get_platform_id=lambda: "`Telegram`",
        get_session_id=lambda: "",
        unified_msg_origin="telegram:group:42",
    )

    assert normalize_event_value(" `value` ") == "value"
    assert get_event_platform_name(event) == "telegram"
    assert get_event_session_key(event) == "telegram:group:42"
    assert get_event_session_key(None) == "global"


def test_event_context_prefers_unified_origin_over_local_session_id():
    event = types.SimpleNamespace(
        get_session_id=lambda: "42",
        unified_msg_origin="telegram:group:42",
    )

    assert get_event_session_key(event) == "telegram:group:42"


@pytest.mark.asyncio
async def test_blacklist_writer_prefers_database():
    calls = []

    class Database:
        async def add_blacklist(self, image_hash, timestamp):
            calls.append((image_hash, timestamp))

    class Cache:
        async def set(self, *_args, **_kwargs):
            raise AssertionError("database path must not write legacy cache")

    plugin = types.SimpleNamespace(db_service=Database(), cache_service=Cache())
    assert await add_blacklist_hash(plugin, " abc ", timestamp=123)
    assert calls == [("abc", 123)]


@pytest.mark.asyncio
async def test_blacklist_writer_reports_missing_database():
    plugin = types.SimpleNamespace(db_service=None)
    assert not await add_blacklist_hash(plugin, "abc", timestamp=456)
    assert not await add_blacklist_hash(plugin, "")
