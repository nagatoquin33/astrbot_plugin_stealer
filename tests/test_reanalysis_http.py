"""真实注册路由、Quart、分析器和 SQLite；只替换外部模型响应。"""

import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from PIL import Image
from quart import Quart

from astrbot_plugin_stealer.core.config.config import PluginConfig
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.core.db.index_manager import refresh_search_entry
from astrbot_plugin_stealer.core.processing.image_processor_service import ImageProcessorService
from astrbot_plugin_stealer.core.processing.library_reanalysis import BatchReanalysisService
from astrbot_plugin_stealer.plugin_api import PluginAPI
from astrbot_plugin_stealer.task_scheduler import TaskScheduler


BASE = "/astrbot_plugin_stealer/images/reanalyze"
EXPECTED = {
    "desc": "新的描述", "tags": ["新标签"], "scenes": ["打招呼"],
    "overlay_text": "你好", "emotions": ["happy"],
}


@pytest_asyncio.fixture
async def http_flow(tmp_path):
    cfg = PluginConfig({"vision_provider_id": "vision", "model_request_interval": 0})
    db = DatabaseService(tmp_path / "http.db")
    response = dict(category="happy", description=EXPECTED["desc"], **{
        key: value for key, value in EXPECTED.items() if key != "desc"
    })
    llm = AsyncMock(return_value=SimpleNamespace(completion_text=json.dumps(response)))
    plugin = SimpleNamespace(
        db_service=db, plugin_config=cfg, task_scheduler=TaskScheduler(),
        vision_max_retries=1, vision_retry_delay=0,
        context=SimpleNamespace(llm_generate=llm),
    )
    processor = ImageProcessorService(plugin)
    service = BatchReanalysisService(db, processor, cfg, plugin.task_scheduler,
                                     lambda path: refresh_search_entry(plugin, path))
    plugin.batch_reanalysis = service
    app = Quart(__name__)
    app.config["TESTING"] = True

    def register(path, handler, methods, _description):
        app.add_url_rule(path, endpoint=handler.__name__, view_func=handler, methods=methods)

    PluginAPI(plugin).register(SimpleNamespace(register_web_api=register))
    paths = []
    hashes = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32, 32), color).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        await db.insert_batch([{
            "path": str(path), "hash": digest, "category": "sad", "desc": "原描述",
            "tags": ["旧标签"], "scenes": ["原场景"], "overlay_text": "原文字",
            "emotions": ["sad"], "is_favorite": True, "character": "alice",
            "scope_mode": "local", "origin_target": "group:1", "use_count": 9,
        }])
        paths.append(str(path))
        hashes.append(digest)
    for job in db.get_search_refresh_jobs():
        db.finish_search_refresh(job["path"], job["generation"])
    try:
        yield SimpleNamespace(client=app.test_client(), db=db, service=service,
                              scheduler=plugin.task_scheduler, llm=llm, paths=paths, hashes=hashes)
    finally:
        await service.close()


async def finish(flow):
    await flow.scheduler._tasks[flow.service.TASK_NAME]


@pytest.mark.asyncio
async def test_registered_http_analysis_review_and_discard(http_flow):
    flow = http_flow
    before = [flow.db.get_emoji(path) for path in flow.paths]
    started = await flow.client.post(BASE, json={"hashes": flow.hashes})
    assert started.status_code == 200
    task_id = (await started.get_json())["task"]["id"]
    await finish(flow)
    assert [flow.db.get_emoji(path) for path in flow.paths] == before
    assert flow.db.get_search_refresh_jobs() == []
    for digest, original in zip(flow.hashes, before):
        detail = await flow.client.get(BASE + "/result", query_string={"task_id": task_id, "hash": digest})
        assert detail.status_code == 200
        result = (await detail.get_json())["result"]
        assert result["before"] == {key: original[key] for key in EXPECTED}
        assert result["after"] == EXPECTED
    saved = await flow.client.post(BASE + "/review", json={
        "task_id": task_id, "hash": flow.hashes[0], "decision": "save",
    })
    assert saved.status_code == 200
    assert (await saved.get_json())["task"]["saved"] == 1
    after = flow.db.get_emoji(flow.paths[0])
    assert {key: after[key] for key in EXPECTED} == EXPECTED
    for key in ("category", "character", "is_favorite", "scope_mode", "origin_target", "use_count"):
        assert after[key] == before[0][key]
    discarded = await flow.client.post(BASE + "/review", json={
        "task_id": task_id, "hash": flow.hashes[1], "decision": "discard",
    })
    assert discarded.status_code == 200
    assert (await discarded.get_json())["task"]["pending"] == 0
    assert flow.db.get_emoji(flow.paths[1]) == before[1]
    assert {job["path"] for job in flow.db.get_search_refresh_jobs()} == {flow.paths[0]}


@pytest.mark.asyncio
async def test_http_retry_preserves_ready_results(http_flow):
    flow = http_flow
    valid = flow.llm.return_value
    flow.llm.side_effect = [RuntimeError("temporary failure"), valid, valid]
    task_id = (await (await flow.client.post(BASE, json={"hashes": flow.hashes})).get_json())["task"]["id"]
    await finish(flow)
    status = (await (await flow.client.get(BASE + "/status")).get_json())["task"]
    assert status["failed"] == 1 and status["pending"] == 1
    retried = await flow.client.post(BASE + "/retry", json={"task_id": task_id})
    assert retried.status_code == 200
    await finish(flow)
    status = (await (await flow.client.get(BASE + "/status")).get_json())["task"]
    assert status["id"] == task_id and status["pending"] == 2 and status["failed"] == 0
    assert all(flow.db.get_emoji(path)["desc"] == "原描述" for path in flow.paths)


@pytest.mark.asyncio
async def test_http_stop_and_conflicting_confirmation(http_flow):
    flow = http_flow
    started, release = asyncio.Event(), asyncio.Event()
    valid = flow.llm.return_value

    async def model(**kwargs):
        started.set()
        await release.wait()
        return valid

    flow.llm.side_effect = model
    task_id = (await (await flow.client.post(BASE, json={"hashes": flow.hashes})).get_json())["task"]["id"]
    await started.wait()
    stopped = await flow.client.post(BASE + "/cancel", json={"task_id": task_id})
    assert stopped.status_code == 200
    release.set()
    await finish(flow)
    status = (await (await flow.client.get(BASE + "/status")).get_json())["task"]
    assert status["status"] == "cancelled" and status["processed"] == 1
    await flow.db.update_path(flow.paths[0], {"desc": "手工修改"})
    conflict = await flow.client.post(BASE + "/review", json={
        "task_id": task_id, "hash": flow.hashes[0], "decision": "save",
    })
    assert conflict.status_code == 409
    assert flow.db.get_emoji(flow.paths[0])["desc"] == "手工修改"


@pytest.mark.asyncio
async def test_http_errors_keep_transport_contract(http_flow, monkeypatch):
    flow = http_flow
    assert (await flow.client.get(BASE)).status_code == 405
    assert (await flow.client.post(BASE, json=[])).status_code == 400
    missing = await flow.client.get(BASE + "/result", query_string={"task_id": "missing", "hash": "a"})
    assert missing.status_code == 404

    def broken_start(hashes):
        raise RuntimeError("internal failure")

    monkeypatch.setattr(flow.service, "start", broken_start)
    failure = await flow.client.post(BASE, json={"hashes": flow.hashes})
    assert failure.status_code == 500
    assert (await failure.get_json())["error"] == "internal failure"
