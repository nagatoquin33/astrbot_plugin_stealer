import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.db.index_manager import IndexManager
from core.db.database_service import DatabaseService
from core.db.index_manager import refresh_search_entry, delete_index_paths


def build_manager(tmp_path):
    return IndexManager(SimpleNamespace(
        base_dir=tmp_path, cache_dir=tmp_path / "cache", categories_dir=tmp_path / "categories",
    ))


def test_migrate_legacy_data_returns_loaded_records(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    backup_path = cache_dir / "index_cache.json.backup"
    legacy_index = {
        "/legacy/a.gif": {
            "hash": "h1",
            "category": "happy",
            "desc": "legacy-desc",
            "tags": ["legacy-tag"],
            "scenes": ["legacy-scene"],
        }
    }
    backup_path.write_text(json.dumps(legacy_index, ensure_ascii=False), encoding="utf-8")

    service = build_manager(tmp_path)
    migrated = asyncio.run(service.migrate_legacy_data(tmp_path))

    assert migrated == legacy_index


def test_load_legacy_index_data_prefers_richer_metadata(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    primary_path = cache_dir / "index_cache.json"
    backup_path = cache_dir / "index_cache.json.backup"
    primary_path.write_text(
        json.dumps(
            {
                "/legacy/a.gif": {
                    "hash": "h1",
                    "category": "happy",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    backup_path.write_text(
        json.dumps(
            {
                "/legacy/a.gif": {
                    "hash": "h1",
                    "category": "happy",
                    "desc": "rich-desc",
                    "tags": ["rich-tag"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    service = build_manager(tmp_path)
    merged, loaded_paths = asyncio.run(service.load_legacy_index_data(tmp_path))

    assert primary_path in loaded_paths
    assert backup_path in loaded_paths
    assert merged["/legacy/a.gif"]["desc"] == "rich-desc"
    assert merged["/legacy/a.gif"]["tags"] == ["rich-tag"]


def test_legacy_blacklist_is_imported_idempotently_and_kept_as_backup(tmp_path):
    manager = build_manager(tmp_path)
    manager.db_service = DatabaseService(tmp_path / "emoji.db")
    manager.cache_dir.mkdir()
    legacy = manager.cache_dir / "blacklist_cache.json"
    legacy.write_text('{"blocked-image": 123}', encoding="utf-8")
    asyncio.run(manager.migrate_blacklist())
    asyncio.run(manager.migrate_blacklist())
    assert manager.db_service.blacklisted_hashes() == {"blocked-image"}
    assert legacy.read_text(encoding="utf-8") == '{"blocked-image": 123}'


@pytest.mark.asyncio
async def test_index_commit_refreshes_changed_metadata_and_removes_deleted_vectors(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([
        {"path": "a.png", "hash": "a", "category": "dumb", "desc": "旧描述"},
        {"path": "b.png", "hash": "b", "category": "happy", "desc": "不变"},
    ])

    async def insert_after_commit(path, entry):
        assert db.get_emoji(path)["desc"] == entry["desc"]

    embedding = SimpleNamespace(delete_by_path=AsyncMock(), insert_emoji=AsyncMock(side_effect=insert_after_commit))
    plugin = SimpleNamespace(
        db_service=db,
        meme_selector=SimpleNamespace(
            _invalidate_bm25_index=Mock(),
            _smart_select_service=SimpleNamespace(_embedding_service=embedding, _invalidate_embedding_index=Mock()),
        ),
    )
    manager = IndexManager(plugin)
    index = db.get_index_cache_readonly()
    index["a.png"]["desc"] = "新描述"
    index["a.png"]["emotions"] = ["dumb", "sigh"]
    index["b.png"]["use_count"] = 5
    await manager.save_index(index)
    embedding.insert_emoji.assert_awaited_once()
    assert embedding.insert_emoji.await_args.args[1]["emotions"] == ["dumb", "sigh"]
    embedding.delete_by_path.assert_awaited_once_with("a.png")

    embedding.insert_emoji.reset_mock()
    embedding.delete_by_path.reset_mock()
    del index["a.png"]
    await manager.save_index(index)
    assert db.get_emoji("a.png") is not None  # 缺失条目不能当作删除，保护并发提交。
    embedding.delete_by_path.assert_not_awaited()
    await delete_index_paths(plugin, ["a.png"])
    assert db.get_emoji("a.png") is None
    embedding.delete_by_path.assert_awaited_once_with("a.png")
    embedding.insert_emoji.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_move_removes_old_path_and_failure_does_not_raise():
    embedding = SimpleNamespace(delete_by_path=AsyncMock(), insert_emoji=AsyncMock())
    plugin = SimpleNamespace(meme_selector=SimpleNamespace(
        _invalidate_bm25_index=Mock(),
        _smart_select_service=SimpleNamespace(_embedding_service=embedding, _invalidate_embedding_index=Mock()),
    ))
    entry = {"desc": "新路径"}
    await refresh_search_entry(plugin, "new.png", entry, previous_path="old.png")
    assert [call.args[0] for call in embedding.delete_by_path.await_args_list] == ["old.png", "new.png"]
    embedding.insert_emoji.assert_awaited_once_with("new.png", entry)
    embedding.insert_emoji.side_effect = RuntimeError("provider unavailable")
    await refresh_search_entry(plugin, "new.png", entry)
