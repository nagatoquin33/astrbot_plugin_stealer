"""用真实数据库验证图库分页和两种图片预览的共享查询。"""

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from werkzeug.datastructures import MultiDict

from astrbot_plugin_stealer.api import library as library_routes
from astrbot_plugin_stealer.api import image_mutations
from unittest.mock import AsyncMock
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.plugin_api import PluginAPI


@pytest.fixture
def api(tmp_path, monkeypatch):
    db = DatabaseService(tmp_path / "emoji.db")
    config = SimpleNamespace(
        max_reg_num=100,
        get_category_info=lambda: [{"key": "happy", "name": "开心"}],
        get_character_info_list=lambda: [{"key": "alice", "name": "Alice"}],
    )

    def reject_snapshot():
        pytest.fail("分页和图片预览不应读取全库快照")

    monkeypatch.setattr(db, "get_index_cache_readonly", reject_snapshot)
    monkeypatch.setattr(library_routes, "jsonify", lambda data: data)
    return PluginAPI(SimpleNamespace(db_service=db, plugin_config=config, base_dir=tmp_path))


def set_query(monkeypatch, **values):
    monkeypatch.setattr(library_routes, "request", SimpleNamespace(args=MultiDict(values)))


@pytest.mark.asyncio
@pytest.mark.parametrize("after_delete", [False, True])
async def test_empty_library_keeps_configured_categories_and_statistics(api, monkeypatch, after_delete):
    if after_delete:
        path = str(api._data_dir / "removed.png")
        await api._db.insert_batch([
            {"path": path, "hash": "removed", "category": "happy", "character": "alice", "is_favorite": True},
        ])
        await api._db.delete_paths([path])
    set_query(monkeypatch)

    result = await api.handle_list_images()

    assert result == {
        "success": True,
        "total": 0,
        "page": 1,
        "size": 50,
        "images": [],
        "categories": [{"key": "happy", "name": "开心", "count": 0}],
        "characters": [{"key": "alice", "name": "Alice", "desc": "", "count": 0}],
        "unassigned_character_count": 0,
        "favorite_count": 0,
        "libraries": {"general": 0, "favorites": 0, "characters": 0, "automatic": 0},
        "automatic_limit": 100,
    }


@pytest.mark.asyncio
async def test_library_filters_and_page_counts_use_database(api, monkeypatch):
    await api._db.insert_batch([
        {"path": str(api._data_dir / f"{key}.png"), "hash": key, "category": category,
         "character": character, "is_favorite": favorite, "overlay_text": "测试关键词", "created_at": created}
        for key, category, character, favorite, created in (
            ("first", "happy", "alice", True, 10),
            ("second", "happy", "alice", True, 20),
            ("other-category", "sad", "alice", True, 30),
            ("other-character", "happy", "bob", True, 40),
            ("not-favorite", "happy", "alice", False, 50),
        )
    ])
    set_query(monkeypatch, category="happy", character="alice", favorite_only="true",
              q="关键词", sort="oldest", size="1", page="2")

    result = await api.handle_list_images()

    assert result["success"], result
    assert result["total"] == 2
    assert result["page"] == 2 and result["size"] == 1
    assert [item["hash"] for item in result["images"]] == ["second"]
    assert result["favorite_count"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,extra,field,value", [
    ("favorite", {"favorite": True}, "is_favorite", 1),
    ("character", {"character": "alice"}, "character", "alice"),
    ("scope", {"scope_mode": "local"}, "scope_mode", "local"),
    ("delete", {}, None, None),
])
async def test_batch_mutation_reads_only_selected_hashes(api, monkeypatch, operation, extra, field, value):
    paths = [api._data_dir / f"batch-{i}.png" for i in range(3)]
    for path in paths:
        path.write_bytes(b"image")
    await api._db.insert_batch([
        {"path": str(path), "hash": "selected" if i < 2 else "other",
         "category": "happy", "origin_target": "group:1"}
        for i, path in enumerate(paths)
    ])
    before = api._db.get_emoji(str(paths[2]))
    api._cfg.get_characters = lambda: ["alice"]
    monkeypatch.setattr(image_mutations, "request", SimpleNamespace(
        get_json=AsyncMock(return_value={"hashes": ["selected"], **extra}),
    ))
    monkeypatch.setattr(image_mutations, "jsonify", lambda data: data)
    response = await getattr(api, "handle_batch_" + operation)()
    assert response["success"] and response["count"] == 2
    assert api._db.get_emoji(str(paths[2])) == before
    for path in paths[:2]:
        if operation == "delete":
            assert api._db.get_emoji(str(path)) is None and not path.exists()
        else:
            assert api._db.get_emoji(str(path))[field] == value


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["handle_image_data", "handle_thumbnail"])
@pytest.mark.parametrize("source", ["library", "pending", "missing"])
async def test_preview_uses_same_hash_lookup_without_snapshot(api, monkeypatch, route, source):
    image_hash = "shared-hash"
    expected_path = api._data_dir / "expected.png"
    Image.new("RGB", (48, 48), (240, 20, 20)).save(expected_path)

    if source == "library":
        other_path = api._data_dir / "other.png"
        Image.new("RGB", (48, 48), (20, 20, 240)).save(other_path)
        # 插入顺序与时间顺序相反，验证两种预览都优先选最早的正式记录。
        await api._db.insert_batch([
            {"path": str(other_path), "hash": image_hash, "created_at": 20},
            {"path": str(expected_path), "hash": image_hash, "created_at": 10},
        ])
        await api._db.insert_pending({"path": str(other_path), "hash": image_hash})
    elif source == "pending":
        await api._db.insert_pending({"path": str(expected_path), "hash": image_hash})
    set_query(monkeypatch, hash=image_hash, size="32")

    result = await getattr(api, route)()

    if source == "missing":
        if route == "handle_thumbnail":
            result, status = result
            assert status == 404
        assert result == {"success": False, "error": "图片未找到"}
        return

    assert result["success"], result
    raw = base64.b64decode(result["url"].split(",", 1)[1])
    if route == "handle_image_data":
        assert raw == expected_path.read_bytes()
    else:
        with Image.open(io.BytesIO(raw)) as thumb:
            assert thumb.size == (32, 32)
            red, _, blue = thumb.getpixel((16, 16))
            assert red > 200 and blue < 50
        assert (await api.handle_thumbnail())["url"] == result["url"]
