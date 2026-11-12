"""外部导入与 WebUI 上传的感知哈希去重回归。

复现路径：外部导入的表情包被 bot 发出后，平台重新编码，图片按视觉相同但
SHA256 不同被偷回，正式库出现两张一样的图。导入/上传必须在入库前写 phash
并做视觉比对，历史记录需要补齐 phash 才能被偷图去重命中。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from astrbot_plugin_stealer.api import image_mutations
from astrbot_plugin_stealer.core.config.config import PluginConfig
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.core.processing.image_processor_service import (
    ImageProcessorService,
)
from astrbot_plugin_stealer.core.processing.phash_dedup_service import PHashDedupService
from astrbot_plugin_stealer.core.sources.source_service import SourceService
from astrbot_plugin_stealer.plugin_api import PluginAPI
from tests.test_external_sources import _Config


def _meme_bytes(fmt: str = "PNG", **save_kwargs) -> bytes:
    image = Image.new("RGB", (96, 96), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 60, 60), fill="black")
    draw.ellipse((50, 40, 88, 80), fill=(200, 40, 40))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def _pack(root, relative_path: str, raw: bytes):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "id": root.name,
                "name": root.name,
                "version": "1.0.0",
                "categories": ["happy"],
                "license": "CC0-1.0",
            }
        ),
        encoding="utf-8",
    )
    (root / "memes_data.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": root.name,
                        "relative_path": relative_path,
                        "category": "happy",
                        "caption": "测试表情",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return root


def _source_service(tmp_path) -> tuple[SourceService, DatabaseService]:
    data_dir = tmp_path / "stealer-data"
    data_dir.mkdir()
    db = DatabaseService(data_dir / "emoji.db")
    plugin = SimpleNamespace(
        base_dir=data_dir,
        db_service=db,
        plugin_config=_Config(data_dir),
        meme_selector=None,
    )
    return SourceService(plugin), db


@pytest.mark.asyncio
async def test_import_stores_phash_and_rejects_visual_duplicate(tmp_path):
    service, db = _source_service(tmp_path)
    await service.initialize()

    original = _meme_bytes("PNG")
    first = _pack(tmp_path / "pack-a", "memes/happy/smile.png", original)
    result = await service.import_now(
        {"source_type": "meme_pack", "path": str(first), "review": False}
    )
    assert result["imported"] == 1

    stored_path, meta = db.get_emoji_by_hash(hashlib.sha256(original).hexdigest())
    assert meta.get("phash"), "导入入库必须写入感知哈希"

    reencoded = _meme_bytes("JPEG", quality=92)
    assert reencoded != original
    second = _pack(tmp_path / "pack-b", "memes/happy/smile.jpg", reencoded)
    result = await service.import_now(
        {"source_type": "meme_pack", "path": str(second), "review": False}
    )
    assert result["duplicates"] == 1
    assert result["imported"] == 0
    assert db.count_total() == 1
    await service.close()


@pytest.mark.asyncio
async def test_review_import_stores_phash_and_rejects_pending_visual_duplicate(tmp_path):
    service, db = _source_service(tmp_path)
    await service.initialize()

    original = _meme_bytes("PNG")
    first = _pack(tmp_path / "pack-review", "memes/happy/smile.png", original)
    result = await service.import_now(
        {"source_type": "meme_pack", "path": str(first), "review": True}
    )
    assert result["pending"] == 1
    row = db.get_pending_by_hash(hashlib.sha256(original).hexdigest())
    assert row and row["phash"], "待审核导入也要在审核入库前写入感知哈希"

    reencoded = _meme_bytes("JPEG", quality=92)
    second = _pack(tmp_path / "pack-review-2", "memes/happy/smile.jpg", reencoded)
    result = await service.import_now(
        {"source_type": "meme_pack", "path": str(second), "review": True}
    )
    assert result["duplicates"] == 1
    assert db.count_pending() == 1
    await service.close()


@pytest.mark.asyncio
async def test_upload_stores_phash_skips_exact_and_allows_similar(tmp_path):
    cfg = PluginConfig({})
    db = DatabaseService(tmp_path / "upload.db")
    plugin = SimpleNamespace(base_dir=cfg.data_dir, plugin_config=cfg, db_service=db)
    api = PluginAPI(plugin)

    original = _meme_bytes("PNG")
    first = await api._persist_image(
        file_content=original, file_ext=".png", category="happy"
    )
    assert not first.get("duplicate")
    _, meta = db.get_emoji_by_hash(first["hash"])
    assert meta.get("phash"), "上传入库必须写入感知哈希"

    similar = await api._persist_image(
        file_content=_meme_bytes("JPEG", quality=92),
        file_ext=".jpg",
        category="sad",
    )
    assert similar.get("similar") is True
    assert not similar.get("duplicate")
    assert db.count_total() == 2, "视觉相似只提示，仍按新图入库"

    exact = await api._persist_image(
        file_content=original, file_ext=".png", category="happy"
    )
    assert exact.get("duplicate") is True
    assert db.count_total() == 2


@pytest.mark.asyncio
async def test_upload_route_reports_duplicate_and_similar(tmp_path, monkeypatch):
    cfg = PluginConfig({})
    db = DatabaseService(tmp_path / "route.db")
    plugin = SimpleNamespace(base_dir=cfg.data_dir, plugin_config=cfg, db_service=db)
    api = PluginAPI(plugin)

    class _UploadRequest:
        def __init__(self, payload):
            self._payload = payload

        @property
        def files(self):
            async def _empty():
                return {}

            return _empty()

        @property
        def form(self):
            async def _empty():
                return {}

            return _empty()

        async def get_json(self):
            return self._payload

    def _install(payload):
        monkeypatch.setattr(
            image_mutations, "request", _UploadRequest(payload)
        )
        monkeypatch.setattr(image_mutations, "jsonify", lambda result: result)

    exact = _meme_bytes("PNG")
    _install(
        {"base64": base64.b64encode(exact).decode(), "filename": "a.png", "emotion": "happy"}
    )
    first = await api.handle_upload_image()
    assert first["success"] is True and not first.get("duplicate")

    _install(
        {
            "base64": base64.b64encode(_meme_bytes("JPEG", quality=92)).decode(),
            "filename": "b.jpg",
            "emotion": "sad",
        }
    )
    similar = await api.handle_upload_image()
    assert similar.get("similar") is True and not similar.get("duplicate")

    _install(
        {"base64": base64.b64encode(exact).decode(), "filename": "a.png", "emotion": "happy"}
    )
    duplicate = await api.handle_upload_image()
    assert duplicate.get("duplicate") is True
    assert db.count_total() == 2


@pytest.mark.asyncio
async def test_flat_color_stickers_are_not_false_visual_duplicates(tmp_path):
    cfg = PluginConfig({})
    db = DatabaseService(tmp_path / "flat.db")
    plugin = SimpleNamespace(base_dir=cfg.data_dir, plugin_config=cfg, db_service=db)
    api = PluginAPI(plugin)

    def flat(color: str) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), color).save(buffer, "PNG")
        return buffer.getvalue()

    await api._persist_image(
        file_content=flat("red"), file_ext=".png", category="happy"
    )
    second = await api._persist_image(
        file_content=flat("blue"), file_ext=".png", category="sad"
    )
    assert not second.get("duplicate"), "纯色图片缺乏结构信息，不应判成视觉重复"
    assert db.count_total() == 2


@pytest.mark.asyncio
async def test_backfill_missing_phash_covers_import_rows_and_pending_pool(tmp_path):
    db = DatabaseService(tmp_path / "backfill.db")
    original = _meme_bytes("PNG")
    original_hash = hashlib.sha256(original).hexdigest()

    library_path = tmp_path / "imported.png"
    library_path.write_bytes(original)
    await db.insert_batch(
        [{"path": str(library_path), "hash": original_hash, "category": "happy"}]
    )

    pending_path = tmp_path / "pending.png"
    pending_path.write_bytes(original)
    pending_id = await db.insert_pending(
        {
            "path": str(pending_path),
            "hash": "legacy-pending-hash",
            "analysis_state": "unanalysed",
        }
    )

    processor = ImageProcessorService.__new__(ImageProcessorService)
    processor.plugin = SimpleNamespace(db_service=db)
    processor._phash_service = PHashDedupService()

    assert await processor.backfill_missing_phash() == 2
    _, meta = db.get_emoji_by_hash(original_hash)
    assert meta.get("phash")
    pending = db.get_pending(pending_id)
    assert pending.get("phash")
    assert pending.get("analysis_state") == "unanalysed", "回填不应改动审核状态"

    stolen = _meme_bytes("JPEG", quality=92)
    stolen_path = tmp_path / "stolen.jpg"
    stolen_path.write_bytes(stolen)
    assert await processor._is_duplicate_or_blacklisted(
        hashlib.sha256(stolen).hexdigest(), {}, str(stolen_path), False
    )
