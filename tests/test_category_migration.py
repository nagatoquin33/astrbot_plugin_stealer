"""旧版中文分类 key 的启动兼容迁移。"""

import hashlib
import json
from types import SimpleNamespace

import pytest

import core.config.config as config_module
from core.config.config import PluginConfig
from core.db.database_service import DatabaseService
from astrbot_plugin_stealer.main import Main


def _write_legacy_state(tmp_path):
    (tmp_path / "categories.json").write_text(
        json.dumps(["开心", "我的分类", "happy"], ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "category_info.json").write_text(
        json.dumps(
            {
                "开心": {"name": "开心", "desc": "旧版中文分类"},
                "我的分类": {"name": "我的中文分类", "desc": "用户自定义"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_legacy_chinese_keys_are_mapped_to_safe_keys_and_display_names_survive(
    tmp_path, monkeypatch
):
    _write_legacy_state(tmp_path)
    monkeypatch.setattr(config_module.StarTools, "get_data_dir", lambda _: tmp_path)

    config = PluginConfig({})

    assert "开心" not in config.categories
    assert "我的分类" not in config.categories
    assert "happy" in config.categories
    custom_key = next(key for key in config.categories if key.startswith("legacy_"))
    assert config.category_info["happy"]["name"] == "开心"
    assert config.category_info[custom_key]["name"] == "我的中文分类"
    assert config.normalize_category_strict("开心") == "happy"
    assert config.normalize_category_strict("我的分类") == custom_key
    assert config.get_legacy_category_key_map() == {
        "开心": "happy",
        "我的分类": custom_key,
    }

    persisted_categories = json.loads((tmp_path / "categories.json").read_text(encoding="utf-8"))
    assert all("开心" != key and "我的分类" != key for key in persisted_categories)
    assert hashlib.sha256("我的分类".encode("utf-8")).hexdigest()[:12] in custom_key


@pytest.mark.asyncio
async def test_startup_storage_migration_moves_files_db_rows_and_pending_rows(tmp_path, monkeypatch):
    _write_legacy_state(tmp_path)
    monkeypatch.setattr(config_module.StarTools, "get_data_dir", lambda _: tmp_path)
    config = PluginConfig({})
    mapping = config.get_legacy_category_key_map()
    custom_key = mapping["我的分类"]

    old_dir = config.categories_dir / "我的分类"
    old_dir.mkdir(parents=True)
    old_path = old_dir / "old.png"
    old_path.write_bytes(b"old-image")
    db = DatabaseService(config.cache_dir / "emoji.db")
    await db.insert_batch(
        [
            {
                "path": str(old_path),
                "hash": "old-hash",
                "category": "我的分类",
                "desc": "旧图",
            }
        ]
    )
    pending_id = await db.insert_pending(
        {"path": str(config.pending_dir / "pending.png"), "hash": "pending", "category": "我的分类"}
    )
    assert pending_id

    plugin = SimpleNamespace(
        plugin_config=config,
        categories_dir=config.categories_dir,
        db_service=db,
    )
    await Main._migrate_legacy_category_storage(plugin)

    new_path = config.categories_dir / custom_key / "old.png"
    assert new_path.is_file()
    assert not old_path.exists()
    assert db.get_emoji(str(new_path))["category"] == custom_key
    assert db.get_emoji(str(old_path)) is None
    assert db.get_pending(pending_id)["category"] == custom_key
