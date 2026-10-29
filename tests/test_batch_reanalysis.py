"""真实 SQLite 上验证后台重分析、并发编辑和停止边界。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_stealer.api import reanalysis as routes
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.core.processing.library_reanalysis import BatchReanalysisService
from astrbot_plugin_stealer.plugin_api import PluginAPI
from astrbot_plugin_stealer.task_scheduler import TaskScheduler


@pytest.fixture
def batch(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    processor = SimpleNamespace(
        CATEGORY_FILTERED="filtered", CATEGORY_NOT_EMOJI="not-emoji",
        classify_image=AsyncMock(return_value=("happy", ["new"], "new description", "happy", ["chat"], "text", ["happy"])),
    )
    config = SimpleNamespace(categories=["happy", "sad"])
    scheduler = TaskScheduler()
    updated = AsyncMock()
    service = BatchReanalysisService(db, processor, config, scheduler, updated)
    return SimpleNamespace(db=db, processor=processor, service=service, scheduler=scheduler,
                           updated=updated, root=tmp_path)


async def seed(batch, key="a"):
    path = batch.root / f"{key}.png"
    path.write_bytes(b"original image")
    await batch.db.insert_batch([{
        "path": str(path), "hash": key, "category": "sad", "desc": "old", "tags": ["old"],
        "character": "alice", "is_favorite": True, "scope_mode": "local",
        "origin_target": "group:1", "source": "external:test", "use_count": 7,
    }])
    return path


async def finish(batch):
    await batch.scheduler._tasks[batch.service.TASK_NAME]
    return batch.service.snapshot()


@pytest.mark.asyncio
async def test_success_preserves_manual_fields_and_never_scans_library(batch, monkeypatch):
    path = await seed(batch)
    before = batch.db.get_emoji(str(path))
    monkeypatch.setattr(batch.db, "get_index_cache_readonly", lambda: pytest.fail("不应扫描全库"))
    job = batch.service.start(["a", "a"])
    assert job["total"] == 1 and job["processed"] == 0
    result = await finish(batch)
    assert result["status"] == "completed" and result["succeeded"] == 1
    assert batch.db.get_emoji(str(path)) == before
    batch.updated.assert_not_awaited()
    assert result["pending"] == 1 and result["saved"] == 0
    assert "before" not in result["results"][0]
    detail = batch.service.result_detail(job["id"], "a")
    expected = {
        "desc": "new description", "tags": ["new"], "scenes": ["chat"],
        "overlay_text": "text", "emotions": ["happy"],
    }
    assert detail["before"] == {key: before[key] for key in expected}
    assert detail["after"] == expected
    saved = await batch.service.review(job["id"], "a", "save")
    assert saved["saved"] == 1 and saved["pending"] == 0
    after = batch.db.get_emoji(str(path))
    assert {key: after[key] for key in expected} == expected
    for field in ("category", "character", "is_favorite", "scope_mode", "origin_target", "source", "use_count"):
        assert after[field] == before[field]
    assert path.read_bytes() == b"original image"
    batch.updated.assert_awaited_once_with(str(path))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["during_analysis", "before_confirmation"])
@pytest.mark.parametrize("change", ["edit", "delete", "usage"])
async def test_changes_are_checked_only_when_user_confirms(batch, change, phase):
    path = await seed(batch)
    started, release = asyncio.Event(), asyncio.Event()

    async def classify(**kwargs):
        started.set()
        await release.wait()
        return "happy", ["new"], "new description", "happy", [], "", []

    batch.processor.classify_image.side_effect = classify
    batch.service.start(["a"])
    await started.wait()
    if phase == "before_confirmation":
        release.set()
        await finish(batch)
    if change == "delete":
        await batch.db.delete_paths([str(path)])
    else:
        await batch.db.update_path(str(path), {"desc": "manual"} if change == "edit" else {"use_count": 99})
    release.set()
    result = await finish(batch)
    assert result["pending"] == 1 and result["saved"] == 0
    if change == "usage":
        saved = await batch.service.review(result["id"], "a", "save")
        assert saved["saved"] == 1
        assert batch.db.get_emoji(str(path))["use_count"] == 99
    else:
        with pytest.raises(RuntimeError, match="修改或删除"):
            await batch.service.review(result["id"], "a", "save")
        assert batch.service.snapshot()["pending"] == 0
        batch.updated.assert_not_awaited()
        after = batch.db.get_emoji(str(path))
        assert after is None if change == "delete" else after["desc"] == "manual"


@pytest.mark.asyncio
async def test_stop_finishes_current_image_and_prevents_next_call(batch):
    await seed(batch, "a")
    await seed(batch, "b")
    started, release = asyncio.Event(), asyncio.Event()

    async def classify(**kwargs):
        started.set()
        await release.wait()
        return "happy", [], "new", "happy", [], "", []

    batch.processor.classify_image.side_effect = classify
    job = batch.service.start(["a", "b"])
    await started.wait()
    with pytest.raises(RuntimeError):
        batch.service.start(["b"])
    with pytest.raises(ValueError):
        batch.service.cancel("wrong-id")
    assert batch.service.cancel(job["id"])["status"] == "stopping"
    release.set()
    result = await finish(batch)
    assert result["status"] == "cancelled" and result["processed"] == 1
    assert batch.processor.classify_image.await_count == 1
    assert batch.db.get_emoji_by_hash("b")[1]["desc"] == "old"


@pytest.mark.asyncio
async def test_failed_and_missing_items_do_not_abort_batch_and_can_retry(batch):
    first = await seed(batch, "a")
    await seed(batch, "b")
    batch.processor.classify_image.side_effect = [
        ("", [], "", "", [], "", []),
        ("happy", [], "success", "happy", [], "", []),
    ]
    batch.service.start(["a", "missing", "b"])
    result = await finish(batch)
    assert (result["failed"], result["skipped"], result["succeeded"]) == (1, 1, 1)
    assert batch.db.get_emoji(str(first))["desc"] == "old"
    batch.processor.classify_image.side_effect = None
    batch.service.retry_failed(result["id"])
    retried = await finish(batch)
    assert retried["id"] == result["id"] and retried["pending"] == 2
    assert retried["succeeded"] == 2 and retried["failed"] == 0
    assert batch.service.result_detail(result["id"], "b")["after"]["desc"] == "success"


@pytest.mark.asyncio
async def test_shutdown_cancels_model_request_and_rejects_new_work(batch):
    path = await seed(batch)
    started = asyncio.Event()

    async def classify(**kwargs):
        started.set()
        await asyncio.Event().wait()

    batch.processor.classify_image.side_effect = classify
    batch.service.start(["a"])
    await started.wait()
    await batch.service.close()
    assert batch.service.snapshot()["status"] == "cancelled"
    assert batch.db.get_emoji(str(path))["desc"] == "old"
    assert batch.service.TASK_NAME not in batch.scheduler._tasks
    with pytest.raises(RuntimeError):
        batch.service.start(["a"])


@pytest.mark.asyncio
async def test_shutdown_waits_for_an_already_started_commit(batch, monkeypatch):
    path = await seed(batch)
    started, release = asyncio.Event(), asyncio.Event()
    original_update = batch.db.update_path

    async def update(*args, **kwargs):
        started.set()
        await release.wait()
        return await original_update(*args, **kwargs)

    monkeypatch.setattr(batch.db, "update_path", update)
    job = batch.service.start(["a"])
    await finish(batch)
    saving = asyncio.create_task(batch.service.review(job["id"], "a", "save"))
    await started.wait()
    closing = asyncio.create_task(batch.service.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await saving
    await closing
    assert batch.db.get_emoji(str(path))["desc"] == "new description"
    batch.updated.assert_awaited_once_with(str(path))


@pytest.mark.asyncio
async def test_http_submission_returns_before_model_and_restores_same_task(batch, monkeypatch):
    await seed(batch)
    api = PluginAPI(SimpleNamespace(batch_reanalysis=batch.service))
    monkeypatch.setattr(routes, "jsonify", lambda data: data)
    monkeypatch.setattr(routes, "request", SimpleNamespace(get_json=AsyncMock(return_value={"hashes": ["a"]})))
    result = await api.handle_batch_reanalyze()
    assert result["success"] and result["task"]["processed"] == 0
    restored = await api.handle_batch_reanalyze_status()
    assert restored["task"]["id"] == result["task"]["id"]
    _, status = await api.handle_batch_reanalyze()
    assert status == 409
    await finish(batch)


@pytest.mark.asyncio
async def test_discard_does_not_save_and_new_batch_cannot_replace_pending_results(batch):
    path = await seed(batch)
    job = batch.service.start(["a"])
    await finish(batch)
    with pytest.raises(RuntimeError, match="待确认"):
        batch.service.start(["a"])
    with pytest.raises(ValueError):
        await batch.service.review("old-job", "a", "save")
    discarded = await batch.service.review(job["id"], "a", "discard")
    assert discarded["discarded"] == 1 and discarded["pending"] == 0
    assert batch.db.get_emoji(str(path))["desc"] == "old"
    batch.updated.assert_not_awaited()
    batch.service.start(["a"])
    await finish(batch)


@pytest.mark.asyncio
async def test_repeated_confirmation_writes_only_once(batch):
    await seed(batch)
    job = batch.service.start(["a"])
    await finish(batch)
    responses = await asyncio.gather(*[
        batch.service.review(job["id"], "a", "save") for _ in range(2)
    ])
    assert all(result["saved"] == 1 for result in responses)
    batch.updated.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_endpoint_saves_only_the_explicitly_selected_image(batch, monkeypatch):
    await seed(batch, "a")
    await seed(batch, "b")
    job = batch.service.start(["a", "b"])
    await finish(batch)
    api = PluginAPI(SimpleNamespace(batch_reanalysis=batch.service))
    monkeypatch.setattr(routes, "jsonify", lambda data: data)
    monkeypatch.setattr(routes, "request", SimpleNamespace(get_json=AsyncMock(return_value={
        "task_id": job["id"], "hash": "b", "decision": "save",
    })))
    response = await api.handle_batch_reanalyze_review()
    assert response["success"] and response["task"]["pending"] == 1
    assert batch.db.get_emoji_by_hash("a")[1]["desc"] == "old"
    assert batch.db.get_emoji_by_hash("b")[1]["desc"] == "new description"


@pytest.mark.asyncio
async def test_failed_save_keeps_result_ready_for_explicit_retry(batch, monkeypatch):
    path = await seed(batch)
    job = batch.service.start(["a"])
    await finish(batch)
    api = PluginAPI(SimpleNamespace(batch_reanalysis=batch.service))
    monkeypatch.setattr(routes, "jsonify", lambda data: data)
    monkeypatch.setattr(routes, "request", SimpleNamespace(get_json=AsyncMock(return_value={
        "task_id": job["id"], "hash": "a", "decision": "save",
    })))
    original_update = batch.db.update_path
    monkeypatch.setattr(batch.db, "update_path", AsyncMock(side_effect=OSError("disk full")))
    response, status = await api.handle_batch_reanalyze_review()
    assert status == 500 and response["error"] == "disk full"
    assert response["task"]["pending"] == 1 and response["task"]["saved"] == 0
    assert batch.db.get_emoji(str(path))["desc"] == "old"
    monkeypatch.setattr(batch.db, "update_path", original_update)
    assert (await api.handle_batch_reanalyze_review())["task"]["saved"] == 1
