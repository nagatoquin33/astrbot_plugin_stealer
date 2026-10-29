"""配置失败、旧快照及入库清理并发的实际回归。"""

import asyncio
import base64
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from astrbot_plugin_stealer.core.config.config import PluginConfig
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.core.db.index_manager import IndexManager
from astrbot_plugin_stealer.core.util.safe_io import protect_file
from astrbot_plugin_stealer.plugin_api import PluginAPI
from astrbot_plugin_stealer.task_scheduler import TaskScheduler
from astrbot_plugin_stealer.api import maintenance, taxonomy, image_mutations


@pytest.fixture
def flow(tmp_path):
    cfg = PluginConfig({})
    db = DatabaseService(tmp_path / "audit.db")
    plugin = SimpleNamespace(base_dir=cfg.data_dir, plugin_config=cfg, db_service=db,
                             task_scheduler=TaskScheduler(), update_config=cfg.update_config)
    api = PluginAPI(plugin)
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(stream, "PNG")
    return SimpleNamespace(cfg=cfg, db=db, plugin=plugin, api=api, content=stream.getvalue())


def payload(monkeypatch, module, data):
    monkeypatch.setattr(module, "request", SimpleNamespace(get_json=AsyncMock(return_value=data)))
    monkeypatch.setattr(module, "jsonify", lambda result: result)


def test_failed_backing_write_restores_runtime_cache_and_backing(tmp_path):
    class Backing(dict):
        def save_config(self, updates):
            self.update(updates)
            raise PermissionError("write denied")

    backing = Backing(steal_meme=False, send_target_whitelist=[])
    cfg = PluginConfig(backing)
    assert not cfg.update_config({"steal_meme": True, "send_target_whitelist": ["group:1"]})
    assert cfg.steal_meme is False and cfg.send_target_whitelist == []
    assert cfg._target_policy_cache["send"]["whitelist"] == frozenset()
    assert backing == {"steal_meme": False, "send_target_whitelist": []}


def test_backing_write_that_truncates_file_is_restored(tmp_path):
    class Backing(dict):
        def save_config(self, updates):
            self.update(updates)
            Path(self.config_path).write_bytes(b"partial JSON")
            raise OSError("disk write failed")

    path = tmp_path / "framework-config.json"
    original = b'{"steal_meme":false}'
    path.write_bytes(original)
    backing = Backing(steal_meme=False)
    backing.config_path = str(path)
    cfg = PluginConfig(backing)
    assert not cfg.update_config({"steal_meme": True})
    assert path.read_bytes() == original and not cfg.steal_meme
    assert backing == {"steal_meme": False}


def test_invalid_runtime_policy_update_does_not_leave_partial_state(flow):
    assert not flow.cfg.update_config({"steal_meme": True, "send_target_whitelist": True})
    assert not flow.cfg.steal_meme and flow.cfg.send_target_whitelist == []


def test_multi_file_configuration_failure_restores_every_file(flow, monkeypatch):
    cfg = flow.cfg
    before = cfg.categories_path.read_bytes()
    info_before = cfg.category_info_path.read_bytes()
    categories = list(cfg.categories)
    original = cfg._write_json_file

    def fail_info(path, value):
        return False if path == cfg.category_info_path else original(path, value)

    monkeypatch.setattr(cfg, "_write_json_file", fail_info)
    assert not cfg.update_config({"categories": ["happy"], "category_info": {"happy": {"name": "新"}}})
    assert cfg.categories == categories
    assert cfg.categories_path.read_bytes() == before
    assert cfg.category_info_path.read_bytes() == info_before


@pytest.mark.asyncio
async def test_taxonomy_does_not_report_success_when_persistence_fails(flow, monkeypatch):
    payload(monkeypatch, taxonomy, {"categories": [{"key": "happy", "name": "新名称"}]})
    monkeypatch.setattr(flow.plugin, "update_config", lambda updates: False)
    before = dict(flow.cfg.category_info)
    result = await flow.api._categories_update()
    assert result["success"] is False and flow.cfg.category_info == before


@pytest.mark.asyncio
async def test_stale_snapshot_only_applies_its_own_changes(flow):
    db = flow.db
    await db.insert_batch([{"path": "a", "hash": "a", "desc": "old"}, {"path": "b", "hash": "b"}])
    snapshot = db.get_index_cache_readonly()
    snapshot["a"]["scope_mode"] = "local"
    await db.update_path("a", {"desc": "new edit", "is_favorite": True})
    await db.delete_paths(["b"])
    await IndexManager(flow.plugin).save_index(snapshot)
    assert db.get_emoji("b") is None
    stored = db.get_emoji("a")
    assert stored["desc"] == "new edit" and stored["is_favorite"]
    assert stored["scope_mode"] == "local"


@pytest.mark.asyncio
async def test_unchanged_snapshot_is_noop_and_missing_new_file_is_not_reinserted(flow):
    await flow.db.insert_batch([{"path": "a", "hash": "a"}])
    snapshot = flow.db.get_index_cache_readonly()
    revision = flow.db.get_corpus_signature()
    await IndexManager(flow.plugin).save_index(snapshot)
    assert flow.db.get_corpus_signature() == revision
    missing = str(flow.cfg.categories_dir / "removed.png")
    snapshot[missing] = {"hash": "removed", "category": "happy"}
    await IndexManager(flow.plugin).save_index(snapshot)
    assert flow.db.get_emoji(missing) is None


@pytest.mark.asyncio
async def test_cleanup_waits_for_upload_commit_and_preserves_new_file(flow, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    original = flow.db.insert_batch

    async def delayed(rows):
        started.set()
        await release.wait()
        return await original(rows)

    monkeypatch.setattr(flow.db, "insert_batch", delayed)
    payload(monkeypatch, maintenance, {"sections": ["orphan_files"]})
    upload = asyncio.create_task(flow.api._persist_image(
        file_content=flow.content, file_ext=".png", category="happy",
    ))
    await started.wait()
    cleanup = asyncio.create_task(flow.api.handle_storage_cleanup())
    await asyncio.sleep(0)
    assert not cleanup.done()
    release.set()
    image = await upload
    result = await cleanup
    assert result["removed"]["orphan_files"] == 0
    path, _ = flow.db.get_emoji_by_hash(image["hash"])
    assert Path(path).read_bytes() == flow.content


@pytest.mark.asyncio
async def test_cleanup_keeps_active_model_input(flow, monkeypatch):
    path = flow.cfg.ensure_raw_dir() / "active.png"
    path.write_bytes(flow.content)
    payload(monkeypatch, maintenance, {"sections": ["raw_files"]})
    with protect_file(path):
        result = await flow.api.handle_storage_cleanup()
    assert result["removed"]["raw_files"] == 0 and path.exists()


@pytest.mark.asyncio
async def test_delete_category_waits_for_upload_and_removes_both_record_and_file(flow, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    original = flow.db.insert_batch

    async def delayed(rows):
        started.set()
        await release.wait()
        return await original(rows)

    monkeypatch.setattr(flow.db, "insert_batch", delayed)
    payload(monkeypatch, taxonomy, {"key": "happy"})
    upload = asyncio.create_task(flow.api._persist_image(
        file_content=flow.content, file_ext=".png", category="happy",
    ))
    await started.wait()
    deletion = asyncio.create_task(flow.api.handle_delete_category())
    await asyncio.sleep(0)
    assert not deletion.done()
    release.set()
    image = await upload
    assert (await deletion)["success"]
    assert flow.db.get_emoji_by_hash(image["hash"]) is None
    assert "happy" not in flow.cfg.categories


@pytest.mark.asyncio
async def test_unload_cancels_registered_batch_upload_without_writing(flow, monkeypatch):
    started = asyncio.Event()

    async def classify(**kwargs):
        started.set()
        await asyncio.Event().wait()

    flow.plugin.image_processor_service = SimpleNamespace(classify_image=classify)
    payload(monkeypatch, image_mutations, {
        "_files": [{"name": "input.png", "base64": base64.b64encode(flow.content).decode()}],
        "auto_analyze": True,
    })
    result = await flow.api.handle_batch_upload()
    assert result["success"]
    await started.wait()
    assert "batch_upload_" + result["task_id"] in flow.plugin.task_scheduler._tasks
    await flow.plugin.task_scheduler.cleanup()
    assert flow.db.count_total() == 0
    assert flow.api.batch_upload_tasks[result["task_id"]]["status"] == "failed"
