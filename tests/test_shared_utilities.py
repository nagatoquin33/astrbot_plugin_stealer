import types

import pytest

from core.events.event_context import (
    get_event_platform_name,
    get_event_session_key,
    normalize_event_value,
)
from core.util.blacklist import add_blacklist_hash
from core.util.normalization import (
    canonicalize_path,
    normalize_character_key,
    normalize_label_list,
    normalize_scope_mode,
)


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
async def test_blacklist_writer_falls_back_to_legacy_cache():
    calls = []

    class Cache:
        async def set(self, *args, **kwargs):
            calls.append((args, kwargs))

    plugin = types.SimpleNamespace(db_service=None, cache_service=Cache())
    assert await add_blacklist_hash(plugin, "abc", timestamp=456)
    assert calls == [
        (("blacklist_cache", "abc", 456), {"persist": True}),
    ]
    assert not await add_blacklist_hash(plugin, "")
