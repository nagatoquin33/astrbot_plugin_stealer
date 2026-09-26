"""生产分析与写回链路：真实配置/SQLite，替换外部模型响应。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

import astrbot_plugin_stealer.core.config.config as config_module
import astrbot_plugin_stealer.api.maintenance as maintenance_module
import astrbot_plugin_stealer.api.image_mutations as image_mutations_module
import astrbot_plugin_stealer.plugin_api as plugin_api_module
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.core.processing.image_processor_service import (
    ImageProcessorService,
)
from astrbot_plugin_stealer.plugin_api import PluginAPI
from astrbot_plugin_stealer.core.db.index_manager import IndexManager
from astrbot_plugin_stealer.core.commands.image_mgmt_command import ImageManagementCommand


RESULT = {
    "category": "dumb",
    "tags": ["熊猫头"],
    "description": "熊猫头闭嘴不语",
    "overlay_text": "龙哑人",
    "scenes": ["说不出话"],
    "emotions": ["dumb"],
}


@pytest.fixture
def flow(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config_module, "StarTools", SimpleNamespace(get_data_dir=lambda _: tmp_path)
    )
    cfg = config_module.PluginConfig({"vision_provider_id": "vision-a"})
    path = cfg.ensure_category_dir("dumb") / "probe.png"
    Image.new("RGB", (40, 40), "red").save(path)
    image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    db = DatabaseService(tmp_path / "audit.db")
    llm = AsyncMock(return_value=SimpleNamespace(completion_text=json.dumps(RESULT)))
    framework = {
        "provider_settings": {"default_image_caption_provider_id": "framework-a"}
    }
    plugin = SimpleNamespace(
        plugin_config=cfg,
        db_service=db,
        vision_max_retries=3,
        vision_retry_delay=0,
        context=SimpleNamespace(llm_generate=llm, get_config=lambda: framework),
    )
    processor = ImageProcessorService(plugin)
    plugin.image_processor_service = processor
    api = PluginAPI(plugin)
    refresh = AsyncMock()
    monkeypatch.setattr(plugin_api_module, "refresh_search_entry", refresh)
    payload = {"hash": image_hash}
    fake_request = SimpleNamespace(get_json=AsyncMock(side_effect=lambda: payload))
    for module in (maintenance_module, image_mutations_module):
        monkeypatch.setattr(module, "request", fake_request)
        monkeypatch.setattr(module, "jsonify", lambda value: value)
    return SimpleNamespace(
        api=api, refresh=refresh,
        plugin=plugin,
        processor=processor,
        db=db,
        path=str(path),
        image_hash=image_hash,
        payload=payload,
        llm=llm,
        framework=framework,
    )


async def seed(flow):
    await flow.db.insert_batch(
        [
            {
                "path": flow.path,
                "hash": flow.image_hash,
                "category": "dumb",
                "scope_mode": "local",
                "origin_target": "group:123",
                "character": "panda",
                "is_favorite": 1,
                "use_count": 8,
                "source": "audit",
            }
        ]
    )


@pytest.mark.asyncio
async def test_preview_and_apply_preserve_unrelated_metadata(flow, monkeypatch):
    await seed(flow)
    before = flow.db.get_emoji(flow.path)

    def no_full_scan():
        raise AssertionError("单图分析不应读取完整图库")

    monkeypatch.setattr(flow.db, "get_index_cache_readonly", no_full_scan)
    result = await flow.api.handle_analyze_image()
    assert result["success"] is True
    assert result["overlay_text"] == RESULT["overlay_text"]
    assert flow.llm.await_args.kwargs["image_urls"] == [flow.path]
    assert flow.db.get_emoji(flow.path) == before

    flow.payload.update(
        {
            key: result[key]
            for key in ("category", "tags", "scenes", "overlay_text", "emotions")
        }
    )
    flow.payload["desc"] = result["description"]
    assert await flow.api.handle_update_image() == {"success": True}
    after = flow.db.get_emoji(flow.path)
    for field in (
        "scope_mode",
        "origin_target",
        "character",
        "is_favorite",
        "use_count",
        "source",
    ):
        assert after[field] == before[field]
    assert after["overlay_text"] == RESULT["overlay_text"]
    flow.refresh.assert_awaited_once_with(flow.plugin, flow.path)


@pytest.mark.asyncio
async def test_provider_updates_reach_the_actual_model_call(flow):
    await seed(flow)
    await flow.api.handle_analyze_image()
    assert flow.llm.await_args.kwargs["chat_provider_id"] == "vision-a"
    flow.processor.update_config(vision_provider_id="vision-b")
    await flow.api.handle_analyze_image()
    assert flow.llm.await_args.kwargs["chat_provider_id"] == "vision-b"
    flow.processor.update_config(vision_provider_id="")
    await flow.api.handle_analyze_image()
    assert flow.llm.await_args.kwargs["chat_provider_id"] == "framework-a"
    flow.framework["provider_settings"]["default_image_caption_provider_id"] = (
        "framework-b"
    )
    await flow.api.handle_analyze_image()
    assert flow.llm.await_args.kwargs["chat_provider_id"] == "framework-b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", ["{}", "图片完全空白，无法识别", '{"category": "dumb"}']
)
async def test_invalid_model_responses_exhaust_retries_without_writing(flow, response):
    await seed(flow)
    before = flow.db.get_emoji(flow.path)
    flow.llm.return_value = SimpleNamespace(completion_text=response)
    result = await flow.api.handle_analyze_image()
    assert result["success"] is False
    assert flow.llm.await_count == 3
    assert flow.db.get_emoji(flow.path) == before


@pytest.mark.asyncio
async def test_retry_accepts_a_later_valid_response(flow):
    await seed(flow)
    flow.llm.side_effect = [
        SimpleNamespace(completion_text="{}"),
        SimpleNamespace(completion_text=json.dumps(RESULT)),
    ]
    result = await flow.api.handle_analyze_image()
    assert result["success"] is True
    assert result["overlay_text"] == RESULT["overlay_text"]
    assert flow.llm.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("desc", "新描述"),
        ("tags", ["新标签"]),
        ("scenes", ["新场景"]),
        ("emotions", ["happy"]),
    ],
)
async def test_semantic_edits_refresh_embedding_without_changing_scope(
    flow, field, value
):
    await seed(flow)
    flow.payload[field] = value
    assert await flow.api.handle_update_image() == {"success": True}
    flow.refresh.assert_awaited_once_with(flow.plugin, flow.path)
    assert flow.db.get_emoji(flow.path)["scope_mode"] == "local"


@pytest.mark.asyncio
async def test_missing_processor_returns_the_intended_error(flow):
    flow.plugin.image_processor_service = None
    result = await flow.api.handle_analyze_image()
    assert result == {"success": False, "error": "图片处理服务不可用"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["duplicate", "blacklisted"])
async def test_database_dedup_works_without_legacy_cache_service(flow, reason):
    if reason == "duplicate":
        await seed(flow)
    else:
        await flow.db.add_blacklist(flow.image_hash)
    assert await flow.processor._is_duplicate_or_blacklisted(
        flow.image_hash, {}, flow.path, False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["delete_image", "blacklist_image"])
async def test_command_deletion_removes_file_and_database_record(flow, action):
    await seed(flow)
    flow.plugin.index_manager = IndexManager(flow.plugin)
    command = getattr(ImageManagementCommand(flow.plugin), action)
    replies = [text async for text in command(SimpleNamespace(plain_result=lambda text: text), "1")]
    assert replies and "✅" in replies[0]
    assert not Path(flow.path).exists()
    assert flow.db.get_emoji(flow.path) is None
    if action == "blacklist_image":
        assert flow.image_hash in flow.db.blacklisted_hashes()


@pytest.mark.asyncio
async def test_provider_errors_do_not_retry_with_a_string_image_list(flow):
    await seed(flow)
    flow.llm.side_effect = RuntimeError("image_urls rejected by provider")
    assert (await flow.api.handle_analyze_image())["success"] is False
    assert flow.llm.await_count == 3
    assert all(
        call.kwargs["image_urls"] == [flow.path] for call in flow.llm.await_args_list
    )
