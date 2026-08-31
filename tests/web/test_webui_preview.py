"""WebUI 预览服务器（tests/web/test_server.py）的冒烟测试。

验证 mock 桥接注入、静态资源、API 形状与关键交互流（审核/上传/删除），
确保预览服务器与 pages/dashboard 前端、plugin_api.py 的响应结构保持一致。
"""

import base64
import io

import pytest

from tests.web.test_server import DASHBOARD_DIR, PLUGIN_BASE, PreviewState, create_app

try:
    from aiohttp.test_utils import TestClient, TestServer
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"aiohttp 不可用，跳过 WebUI 预览测试: {exc}", allow_module_level=True)


def _png_base64(color=(200, 60, 60)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _client(state: PreviewState | None = None) -> TestClient:
    return TestClient(TestServer(create_app(state)))


@pytest.mark.asyncio
async def test_index_injects_bridge_and_i18n():
    async with _client() as client:
        resp = await client.get("/")
        assert resp.status == 200
        html = await resp.text()
        assert "window.AstrBotPluginPage" not in html  # 桥接走外链脚本而非内联
        assert '<script src="/mock/bridge.js"></script>' in html
        assert "__STEALER_I18N__" in html
        assert '"zh-CN"' in html and '"en-US"' in html
        assert "./app.js" in html
        # 快速项：prod 构建 + favicon
        assert "vue.global.prod.min.js" in html
        assert 'rel="icon"' in html


@pytest.mark.asyncio
async def test_bridge_js_and_static_assets_served():
    async with _client() as client:
        bridge = await client.get("/mock/bridge.js")
        assert bridge.status == 200
        bridge_text = await bridge.text()
        for fn in ("apiGet", "apiPost", "upload", "getLocale", "getI18n", "_files"):
            assert fn in bridge_text

        for path, keyword in (
            ("/app.js", "createApp"),
            ("/template.js", "TEMPLATE"),
            ("/app.css", "{"),
        ):
            resp = await client.get(path)
            assert resp.status == 200, path
            assert await resp.text()

        logo = await client.get("/logo.png")
        assert logo.status == 200
        assert logo.content_type == "image/png"
        assert await logo.read()

        vaultboy = await client.get("/vaultboy.png")
        assert vaultboy.status == 200
        assert vaultboy.content_type == "image/png"
        assert await vaultboy.read()


def test_grid_does_not_prefetch_originals_on_hover():
    """issue #101：列表只渲染缩略图，hover 不得预取原图（大 GIF 会卡死）。"""
    template = (DASHBOARD_DIR / "template.js").read_text(encoding="utf-8")
    app_js = (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")
    assert '@mouseenter="loadOriginalImage' not in template
    assert "originalDataUrls[img.hash] || imageDataUrls[img.hash]" not in template
    assert "originalDataUrls[item.hash] || imageDataUrls[item.hash]" not in template
    assert "originalDataUrls[previewItem?.hash]" in template
    assert "if (img?.hash) loadOriginalImage(img.hash)" not in app_js
    assert "requestOriginalForPreview" in app_js


def test_fallout_theme_avoids_fullpage_compositing():
    """辐射 4 主题不得用全屏 mix-blend / 图片 filter，避免详情弹窗合成卡死。"""
    css = (DASHBOARD_DIR / "app.css").read_text(encoding="utf-8")
    template = (DASHBOARD_DIR / "template.js").read_text(encoding="utf-8")
    start = css.find('[data-theme="fallout"]')
    assert start != -1
    fallout = css[start:]
    assert "mix-blend-mode" not in fallout
    assert "filter: contrast" not in fallout
    assert "html[data-theme=\"fallout\"]::after" not in css
    assert ".fo-chassis" in css
    assert ".fo-pip-tabs" in css
    assert ".fo-hud" in css
    assert "vaultboy.png" in css
    assert "fo-vaultboy" in template
    assert "PIP-BOY 3000 MK IV" in template
    assert "ROBCO INDUSTRIES" in template
    assert "loadDashboardPrefs" in (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")


def test_server_theme_overrides_host_query_after_prefs_load():
    """AstrBot 必带的 ?theme=dark/light 只代表宿主状态，不能覆盖页面偏好。"""
    app_js = (DASHBOARD_DIR / "app.js").read_text(encoding="utf-8")
    prefs_block = app_js[app_js.index("const loadDashboardPrefs"):]
    assert "!readStored(THEME_STORAGE_KEY)" not in prefs_block
    assert "hasQueryTheme" not in prefs_block
    assert "requestRevision === themePreferenceRevision" in prefs_block
    assert "setThemeMode(resolveThemeValue(data.theme), false)" in prefs_block
    assert "hostThemeFromQuery" in app_js
    assert "themeMode = ref(resolveThemeValue(readStored" in app_js


@pytest.mark.asyncio
async def test_prefs_persist_theme_and_view():
    state = PreviewState(seed=False)
    async with _client(state) as client:
        before = await (await client.get(f"{PLUGIN_BASE}/prefs")).json()
        assert before["success"] is True
        assert before["theme"] == "auto"

        saved = await (await client.post(
            f"{PLUGIN_BASE}/prefs", json={"theme": "fallout", "view": "list"}
        )).json()
        assert saved == {"success": True, "theme": "fallout", "view": "list"}

        again = await (await client.get(f"{PLUGIN_BASE}/prefs")).json()
        assert again["theme"] == "fallout"
        assert again["view"] == "list"

        ignored = await (await client.post(
            f"{PLUGIN_BASE}/prefs", json={"theme": "not-a-theme"}
        )).json()
        assert ignored["theme"] == "fallout"


@pytest.mark.asyncio
async def test_health_and_stats_shapes():
    async with _client() as client:
        health = await client.get(f"{PLUGIN_BASE}/health")
        assert (await health.json())["status"] == "ok"

        stats = (await (await client.get(f"{PLUGIN_BASE}/stats")).json())["stats"]
        seeded = PreviewState(seed=True)
        assert stats == {
            "total": len(seeded.library),
            "categories": len({m["category"] for m in seeded.library}),
            "today": stats["today"],
        }
        assert isinstance(stats["today"], int)


@pytest.mark.asyncio
async def test_images_list_pagination_and_categories():
    async with _client() as client:
        data = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=5")).json())
        assert data["success"] is True
        assert len(data["images"]) <= 5
        assert data["total"] > 0
        assert {"key", "name", "count"} <= set(data["categories"][0])
        counts_sum = sum(c["count"] for c in data["categories"])
        assert counts_sum == data["total"]

        # 第二页不与第一页重叠（种子数据足够多）
        page2 = (await (await client.get(f"{PLUGIN_BASE}/images?page=2&size=5")).json())
        hashes1 = {i["hash"] for i in data["images"]}
        hashes2 = {i["hash"] for i in page2["images"]}
        assert hashes1.isdisjoint(hashes2)


@pytest.mark.asyncio
async def test_images_search_filter_by_query():
    async with _client() as client:
        all_items = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=100")).json())
        target = next(m for m in all_items["images"] if m["tags"])
        keyword = target["tags"][0]

        found = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=100&q={keyword}")).json())
        assert found["total"] >= 1
        assert any(m["hash"] == target["hash"] for m in found["images"])

        by_cat = (await (await client.get(f"{PLUGIN_BASE}/images?category={target['category']}")).json())
        assert all(m["category"] == target["category"] for m in by_cat["images"])


@pytest.mark.asyncio
async def test_thumbnail_and_image_data_return_data_urls():
    async with _client() as client:
        listing = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=1")).json())
        img_hash = listing["images"][0]["hash"]

        thumb = (await (await client.get(f"{PLUGIN_BASE}/thumbnail?hash={img_hash}&size=120")).json())
        assert thumb["success"] is True
        assert thumb["url"].startswith("data:image/png;base64,")

        original = (await (await client.get(f"{PLUGIN_BASE}/image-data?hash={img_hash}")).json())
        assert original["success"] is True
        assert original["url"].startswith("data:image/png;base64,")

        missing = (await (await client.get(f"{PLUGIN_BASE}/thumbnail?hash=nope")).json())
        assert missing["success"] is False


@pytest.mark.asyncio
async def test_pending_list_stats_and_approve_flow():
    state = PreviewState(seed=True)
    async with _client(state) as client:
        before_total = (await (await client.get(f"{PLUGIN_BASE}/stats")).json())["stats"]["total"]

        pending = (await (await client.get(f"{PLUGIN_BASE}/pending?page=1&size=10")).json())
        assert pending["success"] is True
        first_id = pending["images"][0]["id"]
        pending_before = len(state.pending)

        pstats = (await (await client.get(f"{PLUGIN_BASE}/pending/stats")).json())["stats"]
        assert pstats == {"pending": pending_before, "capacity": 200, "paused": False}

        approved = (await (await client.post(
            f"{PLUGIN_BASE}/pending/approve", json={"id": first_id}
        )).json())
        assert approved["success"] is True and approved["approved"] == 1

        after_total = (await (await client.get(f"{PLUGIN_BASE}/stats")).json())["stats"]["total"]
        assert after_total == before_total + 1
        assert len(state.pending) == pending_before - 1

        missing = (await (await client.post(
            f"{PLUGIN_BASE}/pending/approve", json={"id": 999999}
        )).json())
        assert missing["success"] is False


@pytest.mark.asyncio
async def test_pending_update_then_reject_with_blacklist():
    state = PreviewState(seed=True)
    async with _client(state) as client:
        pending = (await (await client.get(f"{PLUGIN_BASE}/pending?page=1&size=1")).json())
        row = pending["images"][0]
        pid = row["id"]

        updated = (await (await client.post(
            f"{PLUGIN_BASE}/pending/update",
            json={"id": pid, "desc": "改过的描述", "tags": ["新标签"], "scope_mode": "local"},
        )).json())
        assert updated["success"] is True
        assert updated["item"]["desc"] == "改过的描述"
        assert updated["item"]["tags"] == ["新标签"]

        bad_cat = (await (await client.post(
            f"{PLUGIN_BASE}/pending/update", json={"id": pid, "category": "nope"}
        )).json())
        assert bad_cat["success"] is False

        rejected = (await (await client.post(
            f"{PLUGIN_BASE}/pending/reject", json={"id": pid, "blacklist": True}
        )).json())
        assert rejected == {"success": True, "deleted": 1, "blacklisted": 1}
        assert row["hash"] in state.blacklist


@pytest.mark.asyncio
async def test_library_update_scope_validation_delete():
    async with _client() as client:
        listing = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=50")).json())
        item = listing["images"][0]

        ok = (await (await client.post(
            f"{PLUGIN_BASE}/images/update",
            json={"hash": item["hash"], "desc": "更新后的描述", "is_favorite": True},
        )).json())
        assert ok["success"] is True

        refreshed = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=50&q=更新后的描述")).json())
        hit = next(m for m in refreshed["images"] if m["hash"] == item["hash"])
        assert hit["is_favorite"] is True

        # 无 origin_target 时设为 local 应报错（与 plugin_api 行为一致）
        local_fail = (await (await client.post(
            f"{PLUGIN_BASE}/images/update", json={"hash": item["hash"], "scope_mode": "local"}
        )).json())
        assert local_fail == {"success": False, "error": "Origin target missing"}

        deleted = (await (await client.post(
            f"{PLUGIN_BASE}/images/delete", json={"hash": item["hash"]}
        )).json())
        assert deleted["success"] is True and deleted["count"] == 1

        gone = (await (await client.get(f"{PLUGIN_BASE}/images?q={item['hash']}&page=1&size=50")).json())
        assert gone["total"] == 0


@pytest.mark.asyncio
async def test_batch_operations_move_scope_favorite_delete():
    async with _client() as client:
        listing = (await (await client.get(f"{PLUGIN_BASE}/images?page=1&size=3")).json())
        hashes = [m["hash"] for m in listing["images"]][:2]

        moved = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-move", json={"hashes": hashes, "category": "troll"}
        )).json())
        assert moved["success"] is True and moved["count"] == 2

        scoped = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-scope", json={"hashes": hashes, "scope_mode": "public"}
        )).json())
        assert scoped["success"] is True and scoped["skipped"] == 0

        fav = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-favorite", json={"hashes": hashes, "favorite": True}
        )).json())
        assert fav["success"] is True and fav["count"] == 2

        deleted = (await (await client.post(
            f"{PLUGIN_BASE}/images/batch-delete", json={"hashes": hashes}
        )).json())
        assert deleted["success"] is True and deleted["count"] == 2


@pytest.mark.asyncio
async def test_upload_single_and_duplicate_detection():
    async with _client() as client:
        b64 = _png_base64()
        first = (await (await client.post(
            f"{PLUGIN_BASE}/images/upload",
            json={"base64": b64, "filename": "hello.png", "emotion": "shy", "tags": "测试,上传"},
        )).json())
        assert first["success"] is True
        assert first["image"]["category"] == "shy"

        duplicate = (await (await client.post(
            f"{PLUGIN_BASE}/images/upload",
            json={"base64": b64, "filename": "hello.png", "emotion": "shy"},
        )).json())
        assert duplicate["hash"] == first["hash"]

        empty = (await (await client.post(f"{PLUGIN_BASE}/images/upload", json={})).json())
        assert empty["success"] is False


@pytest.mark.asyncio
async def test_batch_upload_task_lifecycle():
    async with _client() as client:
        payload = {
            "_files": [
                {"key": "file", "name": "a.png", "base64": _png_base64((10, 180, 90))},
                {"key": "file", "name": "b.png", "base64": _png_base64((40, 80, 220))},
            ],
            "emotion": "love",
        }
        created = (await (await client.post(f"{PLUGIN_BASE}/images/batch-upload", json=payload)).json())
        assert created["success"] is True
        assert created["task_id"]

        status = (await (await client.get(
            f"{PLUGIN_BASE}/images/batch-upload-status?task_id={created['task_id']}"
        )).json())
        assert status["success"] is True
        assert status["status"] == "completed"
        assert status["processed"] == 2
        assert status["success_count"] == 2
        assert status["failed_count"] == 0

        unknown = (await (await client.get(
            f"{PLUGIN_BASE}/images/batch-upload-status?task_id=nope"
        )).json())
        assert unknown["success"] is False


@pytest.mark.asyncio
async def test_analyze_emotions_storage_scan_cleanup():
    async with _client() as client:
        analyzed = (await (await client.post(
            f"{PLUGIN_BASE}/analyze", json={"base64": _png_base64()}
        )).json())
        assert analyzed["success"] is True
        assert {"category", "tags", "desc"} <= set(analyzed)

        emotions = (await (await client.get(f"{PLUGIN_BASE}/emotions")).json())
        assert emotions["success"] is True and len(emotions["emotions"]) == 17

        scan = (await (await client.get(f"{PLUGIN_BASE}/storage/scan")).json())
        assert scan["success"] is True
        for section in ("stale_index", "orphan_files", "thumb_cache", "temp_files"):
            assert {"count", "bytes", "samples"} <= set(scan[section])

        cleanup = (await (await client.post(
            f"{PLUGIN_BASE}/storage/cleanup", json={"strategy": "balanced"}
        )).json())
        assert cleanup["success"] is True
        assert sum(int(v) for v in cleanup["removed"].values()) >= 0


@pytest.mark.asyncio
async def test_unknown_endpoint_returns_404_envelope():
    async with _client() as client:
        resp = await client.get(f"{PLUGIN_BASE}/definitely-not-a-route")
        assert resp.status == 404
        body = await resp.json()
        assert body["success"] is False
