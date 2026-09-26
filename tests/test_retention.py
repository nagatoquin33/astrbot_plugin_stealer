"""Capacity ranking, library isolation, and real-file deletion regressions."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from astrbot_plugin_stealer.core.maintenance.retention import (
    eviction_candidates, library_counts, library_group,
)
from astrbot_plugin_stealer.core.events.event_handler import EventHandler
from astrbot_plugin_stealer.core.db.database_service import DatabaseService


def test_protected_libraries_never_consume_quota_or_change_ranking():
    index = {
        "old": {"created_at": 1},
        "new": {"created_at": 2},
        "favorite": {"is_favorite": True},
        "character": {"character": "alice"},
        "both": {"is_favorite": True, "character": "alice"},
        "external": {"retention_class": "external"},
        "pinned": {"retention_class": "pinned"},
    }
    assert eviction_candidates(index, 2) == []
    assert eviction_candidates(index, 1) == [("old", 1)]
    assert library_counts(index) == dict(general=4, favorites=2, characters=1, automatic=2)
    assert index["both"]["character"] == "alice"


@pytest.mark.parametrize("weight,expected", [(0.7, "new_unused"), (0.3, "old_used"), (0, "old_used"), (1, "new_unused")])
def test_weight_changes_tradeoff(weight, expected):
    index = {
        "old_used": {"use_count": 100, "created_at": 1},
        "new_unused": {"use_count": 0, "created_at": 100},
    }
    assert eviction_candidates(index, 1, weight)[0][0] == expected


def test_combined_score_and_exact_overflow():
    index = {
        "old_used": {"use_count": 10, "created_at": 10},
        "new_unused": {"use_count": 0, "created_at": 110},
        "middle": {"use_count": 1, "created_at": 20},
    }
    # Middle scores .90, new_unused .70, old_used .30.
    assert eviction_candidates(index, 2) == [("middle", 20)]
    assert eviction_candidates(index, 1) == [("middle", 20), ("new_unused", 110)]


def test_equal_missing_and_invalid_values_are_deterministic():
    index = {"z": {}, "a": {"use_count": "bad", "created_at": None}, "b": {"use_count": -1, "created_at": float("nan")}}
    assert eviction_candidates(index, 1) == [("a", 0), ("b", 0)]
    assert eviction_candidates(index, 0) == []
    assert eviction_candidates(index, -1) == []
    assert eviction_candidates({"bad": None}, 1) == []


def test_removing_protection_returns_entry_to_quota_with_history():
    meta = {"is_favorite": True, "character": "alice", "use_count": 4, "created_at": 12}
    assert library_group(meta) == "favorites"
    meta["is_favorite"] = False
    assert library_group(meta) == "characters"
    meta["character"] = ""
    assert library_counts({"item": meta})["automatic"] == 1
    assert meta["use_count"] == 4 and meta["created_at"] == 12


@pytest.mark.asyncio
async def test_enforcement_removes_only_ranked_general_files(tmp_path):
    index = {}
    for name, meta in {
        "old_popular": {"use_count": 100, "created_at": 1},
        "new_unused": {"use_count": 0, "created_at": 100},
        "favorite": {"is_favorite": True},
        "character": {"character": "alice"},
        "external": {"retention_class": "external"},
    }.items():
        path = tmp_path / name
        path.write_bytes(b"image")
        index[str(path)] = meta
    handler = EventHandler(SimpleNamespace(plugin_config=SimpleNamespace(max_reg_num=1, eviction_usage_weight=0.7)))
    removed = await handler._enforce_capacity(index)
    assert removed == [str(tmp_path / "new_unused")]
    assert not (tmp_path / "new_unused").exists()
    assert len(index) == 4
    assert all(Path(path).exists() for path in index)


@pytest.mark.asyncio
async def test_failed_file_delete_keeps_index(tmp_path, monkeypatch):
    from astrbot_plugin_stealer.core.events import event_handler
    async def fail(_):
        return False
    monkeypatch.setattr(event_handler, "safe_remove_file", fail)
    index = {}
    for i in range(2):
        path = tmp_path / str(i)
        path.write_bytes(b"image")
        index[str(path)] = {"created_at": i}
    handler = EventHandler(SimpleNamespace(plugin_config=SimpleNamespace(max_reg_num=1)))
    assert await handler._enforce_capacity(index) == []
    assert len(index) == 2


@pytest.mark.asyncio
async def test_database_library_filters_counts_and_least_used(tmp_path):
    db = DatabaseService(tmp_path / "library.db")
    entries = [
        dict(path="a", hash="a", category="happy", use_count=5, created_at=1),
        dict(path="b", hash="b", category="happy", use_count=0, created_at=2),
        dict(path="c", hash="c", category="happy", character="alice"),
        dict(path="d", hash="d", category="happy", character="alice", is_favorite=True),
        dict(path="e", hash="e", category="happy", retention_class="external"),
    ]
    await db.insert_batch(entries)
    assert db.get_library_counts() == dict(general=3, favorites=1, characters=1, automatic=2)
    for library, expected in [("general", {"a", "b", "e"}), ("favorites", {"d"}), ("characters", {"c"})]:
        rows, count, categories = db.get_emojis_paginated(library=library, sort_order="least_used")
        assert {row["path"] for row in rows} == expected
        assert count == len(expected) == categories["happy"]
        if library == "general":
            assert rows[-1]["path"] == "a"
    assert db.get_character_counts(exclude_favorites=True)["alice"] == 1
    assert db.get_character_counts()["alice"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_db", [False, True])
@pytest.mark.parametrize("library,expected", [("general", {"a", "b"}), ("favorites", {"d"}), ("characters", {"c"})])
async def test_list_api_library_contract(tmp_path, monkeypatch, use_db, library, expected):
    from astrbot_plugin_stealer import plugin_api
    from astrbot_plugin_stealer.api import library as library_routes
    from werkzeug.datastructures import MultiDict
    db = DatabaseService(tmp_path / "api.db")
    index = {}
    for key, metadata in {
        "a": {"use_count": 5}, "b": {"use_count": 0},
        "c": {"character": "alice"},
        "d": {"character": "alice", "is_favorite": True},
    }.items():
        path = tmp_path / (key + ".png")
        path.write_bytes(b"test")
        index[str(path)] = dict(path=str(path), hash=key, category="happy", **metadata)
    await db.insert_batch(list(index.values()))
    cfg = SimpleNamespace(max_reg_num=100, get_category_info=lambda: [], get_character_info_list=lambda: [])
    api = plugin_api.PluginAPI(SimpleNamespace(db_service=db if use_db else None, plugin_config=cfg))
    monkeypatch.setattr(api, "_get_index", lambda: index)
    monkeypatch.setattr(library_routes, "request", SimpleNamespace(args=MultiDict({"library": library, "sort": "least_used"})))
    monkeypatch.setattr(library_routes, "jsonify", lambda data: data)
    result = await api.handle_list_images()
    assert result["success"], result
    assert {item["hash"] for item in result["images"]} == expected
    assert result["total"] == len(expected)
    assert result["libraries"] == dict(general=2, favorites=1, characters=1, automatic=2)
    assert result["automatic_limit"] == 100
    if library == "general":
        assert [item["hash"] for item in result["images"]] == ["b", "a"]


@pytest.mark.asyncio
async def test_rebuild_preserves_protection_and_usage_before_capacity(tmp_path):
    from unittest.mock import AsyncMock
    from astrbot_plugin_stealer.core.commands.index_rebuild_command import IndexRebuildCommand
    old = {
        "fav.png": {"hash": "f", "is_favorite": True, "use_count": 9, "created_at": 10},
        "char.png": {"hash": "c", "character": "alice", "last_used_at": 20},
        "external.png": {"hash": "e", "retention_class": "external"},
        "general.png": {"hash": "g", "use_count": 7, "created_at": 30},
    }
    rebuilt = {path: {"hash": meta["hash"]} for path, meta in old.items()}
    manager = SimpleNamespace(rebuild_index_from_files=AsyncMock(return_value=rebuilt), save_index=AsyncMock())
    handler = SimpleNamespace(_enforce_capacity=AsyncMock())
    plugin = SimpleNamespace(
        index_manager=manager, event_handler=handler, cache_dir=None, base_dir=None,
        plugin_config=SimpleNamespace(max_reg_num=1),
        db_service=SimpleNamespace(count_total=lambda: len(old), get_index_cache_readonly=lambda: old),
    )
    messages = [message async for message in IndexRebuildCommand(plugin).rebuild_index(SimpleNamespace(plain_result=lambda text: text))]
    assert "索引重建完成" in messages[-1]
    handler._enforce_capacity.assert_not_awaited()
    manager.save_index.assert_awaited_once()
    for path, meta in old.items():
        for key, value in meta.items():
            assert rebuilt[path][key] == value
