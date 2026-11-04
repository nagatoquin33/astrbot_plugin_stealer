"""主副情绪移动与分类删除的真实 SQLite / HTTP 回归。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from quart import Quart

import astrbot_plugin_stealer.core.config.config as config_module
from astrbot_plugin_stealer.core.config.config import PluginConfig
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.plugin_api import PluginAPI


@pytest.fixture
def setup_library(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "StarTools", SimpleNamespace(get_data_dir=lambda _: tmp_path))
    cfg = PluginConfig({})
    db = DatabaseService(tmp_path / "test.db")
    plugin = SimpleNamespace(plugin_config=cfg, db_service=db, base_dir=tmp_path, update_config=cfg.update_config)
    app = Quart(__name__)
    PluginAPI(plugin).register(SimpleNamespace(register_web_api=lambda path, handler, methods, desc:
        app.add_url_rule(path, endpoint=handler.__name__, view_func=handler, methods=methods)))
    return SimpleNamespace(cfg=cfg, db=db, plugin=plugin, client=app.test_client(), path=tmp_path)


@pytest.mark.asyncio
async def test_delete_category_clears_secondary_labels_and_search_jobs(setup_library):
    env = setup_library
    await env.db.insert_batch([{"path": str(env.path / "a.png"), "hash": "a", "category": "happy",
                               "emotions": ["happy", "sad", "tired"], "tags": ["猫"]}])
    pending_id = await env.db.insert_pending({"path": str(env.path / "pending.png"), "hash": "p",
                                              "category": "happy", "emotions": ["happy", "sad"]})
    pending_primary = await env.db.insert_pending({"path": str(env.path / "pending-primary.png"), "hash": "p2",
                                                   "category": "happy", "emotions": ["sad", "tired"]})
    await env.db.update_pending(pending_primary, {"category": "sad"})
    for job in env.db.get_search_refresh_jobs():
        env.db.finish_search_refresh(job["path"], job["generation"])
    resp = await env.client.post('/astrbot_plugin_stealer/categories/delete', json={"key": "sad"})
    assert (await resp.get_json())["success"] is True
    row = env.db.get_emoji(str(env.path / "a.png"))
    assert row["category"] == "happy" and row["emotions"] == ["happy", "tired"] and row["tags"] == ["猫"]
    assert env.db.get_pending(pending_id)["emotions"] == ["happy"]
    pending_row = env.db.get_pending(pending_primary)
    assert pending_row["category"] == "" and pending_row["emotions"] == ["tired"]
    assert "category" not in pending_row["manual_fields"]
    assert env.db.get_emojis_paginated(category="sad", include_secondary=True)[1] == 0
    stale = await env.client.post('/astrbot_plugin_stealer/images/update', json={"hash": "a", "emotions": ["happy", "sad"]})
    assert (await stale.get_json())["success"] is False
    assert env.db.get_emoji(str(env.path / "a.png"))["emotions"] == ["happy", "tired"]
    assert {job["path"] for job in env.db.get_search_refresh_jobs()} == {str(env.path / "a.png")}


@pytest.mark.asyncio
async def test_delete_category_restores_config_if_label_cleanup_fails(setup_library, monkeypatch):
    env = setup_library
    monkeypatch.setattr(env.db, "remove_emotion_label", AsyncMock(side_effect=OSError("failed")))
    resp = await env.client.post('/astrbot_plugin_stealer/categories/delete', json={"key": "sad"})
    assert (await resp.get_json())["success"] is False
    assert "sad" in env.cfg.categories


@pytest.mark.asyncio
async def test_batch_move_replaces_old_primary_and_preserves_secondary(setup_library):
    env = setup_library
    path = env.cfg.ensure_category_dir("happy") / "a.png"
    Image.new("RGB", (8, 8), "red").save(path)
    await env.db.insert_batch([{"path": str(path), "hash": "a", "category": "happy",
                                "emotions": ["happy", "sad", "angry"]}])
    resp = await env.client.post('/astrbot_plugin_stealer/images/batch-move', json={"hashes": ["a"], "category": "angry"})
    assert (await resp.get_json())["count"] == 1
    _, row = env.db.get_emoji_by_hash("a")
    assert row["category"] == "angry" and row["emotions"] == ["angry", "sad"]
    assert env.db.get_emojis_paginated(category="happy", include_secondary=True)[1] == 0
