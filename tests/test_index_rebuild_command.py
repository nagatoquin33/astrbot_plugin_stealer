"""Rebuild removes missing records without dropping files outside the scan root."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.commands.index_rebuild_command import IndexRebuildCommand
from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager


@pytest.mark.asyncio
async def test_rebuild_prunes_missing_index_and_preserves_existing_external_file(
    tmp_path: Path,
):
    categories = tmp_path / "categories"
    valid = categories / "happy" / "valid.png"
    missing = categories / "happy" / "missing.png"
    external = tmp_path / "imported" / "outside.png"
    valid.parent.mkdir(parents=True)
    external.parent.mkdir(parents=True)
    valid.write_bytes(b"valid")
    external.write_bytes(b"external")

    db = DatabaseService(tmp_path / "emojis.db")
    await db.insert_batch([
        {"path": str(valid), "hash": "valid", "category": "happy", "created_at": 1},
        {"path": str(missing), "hash": "missing", "category": "happy", "created_at": 2},
        {"path": str(external), "hash": "external", "category": "happy", "created_at": 3},
    ])
    plugin = SimpleNamespace(
        db_service=db,
        base_dir=tmp_path,
        cache_dir=None,
        categories_dir=categories,
        plugin_config=SimpleNamespace(categories_dir=categories, max_reg_num=100),
    )
    plugin.index_manager = IndexManager(plugin)
    event = SimpleNamespace(plain_result=lambda text: text)

    messages = [
        message async for message in IndexRebuildCommand(plugin).rebuild_index(event)
    ]
    assert messages[-1].startswith("✅")
    assert "重建后索引/文件: 2 个" in messages[-1]
    assert db.get_emoji(str(valid)) is not None
    assert db.get_emoji(str(missing)) is None
    assert db.get_emoji(str(external)) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_configured", [False, True])
async def test_rebuild_keeps_rich_legacy_metadata_and_counts_unique_records(tmp_path, cache_configured):
    categories = tmp_path / "categories"
    image = categories / "happy" / "legacy.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    cache = tmp_path / "cache"
    cache.mkdir()
    rich = {
        str(image): {
            "desc": "完整描述", "tags": ["标签"], "scenes": ["聊天"],
            "scope_mode": "local", "origin_target": "group:100",
            "is_favorite": True, "use_count": 7,
        },
    }
    primary = cache / "index_cache.json"
    backup = cache / "index_cache.json.backup"
    primary.write_text(json.dumps(rich, ensure_ascii=False), encoding="utf-8")
    backup_text = json.dumps({str(image): {"desc": "简略描述"}})
    backup.write_text(backup_text, encoding="utf-8")
    db = DatabaseService(tmp_path / "emoji.db")
    plugin = SimpleNamespace(
        db_service=db, base_dir=tmp_path, cache_dir=cache if cache_configured else None,
        categories_dir=categories,
        plugin_config=SimpleNamespace(categories_dir=categories, max_reg_num=100),
    )
    plugin.index_manager = IndexManager(plugin)

    messages = [message async for message in IndexRebuildCommand(plugin).rebuild_index(
        SimpleNamespace(plain_result=lambda text: text)
    )]

    assert messages[-1].startswith("✅"), messages
    assert "旧版备份数据: 1 条" in messages[-1]
    stored = db.get_emoji(str(image))
    for key, value in rich[str(image)].items():
        assert stored[key] == value
    assert primary.is_file()
    assert backup.read_text(encoding="utf-8") == backup_text
