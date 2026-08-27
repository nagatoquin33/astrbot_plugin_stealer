#!/usr/bin/env python3
"""WebUI 预览测试服务器 - 在无 AstrBot 环境下预览 pages/dashboard。

用法:
    python tests/web/test_server.py [--port 8091]

- 静态托管 pages/dashboard/{index.html,app.js,app.css,template.js}
- 注入 mock 版 window.AstrBotPluginPage 桥接（apiGet/apiPost/upload/getLocale/getI18n）
- 按 plugin_api.py 的路由在 /astrbot_plugin_stealer/* 提供 mock API
- 内存数据：库内表情 + 待审核池，支持增删改/审核/批量操作的真实交互预览
"""

import argparse
import asyncio
import base64
import colorsys
import io
import json
import random
import time
import uuid
from pathlib import Path

from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = REPO_ROOT / "pages" / "dashboard"
I18N_DIR = REPO_ROOT / ".astrbot-plugin" / "i18n"

PLUGIN_BASE = "/astrbot_plugin_stealer"
DEFAULT_PORT = 8091
PENDING_CAPACITY = 200

EMOTIONS = [
    {"key": "happy", "name": "开心", "desc": "快乐、愉悦、满足、好心情"},
    {"key": "sad", "name": "难过", "desc": "悲伤、沮丧、失落、emo"},
    {"key": "angry", "name": "生气", "desc": "愤怒、恼火、不满、暴躁"},
    {"key": "shy", "name": "害羞", "desc": "羞涩、不好意思、腼腆"},
    {"key": "surprised", "name": "惊讶", "desc": "意外、震惊、惊奇、啊？"},
    {"key": "troll", "name": "整活", "desc": "调皮、搞怪、发癫、抽象"},
    {"key": "cry", "name": "哭哭", "desc": "哭泣、流泪、委屈、破防"},
    {"key": "confused", "name": "困惑", "desc": "迷茫、不解、疑惑、问号脸"},
    {"key": "embarrassed", "name": "尴尬", "desc": "社死、窘迫、为难、脚趾抠地"},
    {"key": "love", "name": "喜欢", "desc": "喜爱、爱慕、宠溺、心动"},
    {"key": "disgust", "name": "嫌弃", "desc": "厌恶、反感、讨厌、yue"},
    {"key": "fear", "name": "害怕", "desc": "恐惧、担心、紧张、怂"},
    {"key": "excitement", "name": "兴奋", "desc": "激动、亢奋、嗨、上头"},
    {"key": "tired", "name": "困倦", "desc": "疲惫、困、无力、想躺"},
    {"key": "sigh", "name": "无奈", "desc": "叹气、摆烂、算了、心累"},
    {"key": "thank", "name": "感谢", "desc": "道谢、感恩、收到、爱了"},
    {"key": "dumb", "name": "无语", "desc": "呆住、傻眼、离谱、沉默"},
]

BRIDGE_JS = """(function () {
'use strict';
var BASE = '%BASE%';

function buildQuery(params) {
    if (!params) return '';
    var sp = new URLSearchParams();
    Object.keys(params).forEach(function (k) {
        if (params[k] !== undefined && params[k] !== null) sp.append(k, params[k]);
    });
    var s = sp.toString();
    return s ? '?' + s : '';
}

async function apiGet(endpoint, params) {
    var res = await fetch(BASE + '/' + endpoint + buildQuery(params), { credentials: 'same-origin' });
    return res.json();
}

function base64ToFile(b64, name) {
    var parts = String(b64).split(',');
    var meta = parts[0].match(/:(.*?);/);
    var mime = meta ? meta[1] : 'application/octet-stream';
    var bin = atob(parts[parts.length - 1]);
    var u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    try { return new File([u8], name, { type: mime }); } catch (e) {
        return new Blob([u8], { type: mime });
    }
}

async function postMultipart(endpoint, fields, files) {
    var fd = new FormData();
    Object.keys(fields || {}).forEach(function (k) {
        if (k === '_files') return;
        var v = fields[k];
        fd.append(k, (v !== null && typeof v === 'object') ? JSON.stringify(v) : v);
    });
    (files || []).forEach(function (f) {
        fd.append(f.key || 'file', base64ToFile(f.base64, f.name || 'file.png'), f.name || 'file.png');
    });
    var res = await fetch(BASE + '/' + endpoint, { method: 'POST', credentials: 'same-origin', body: fd });
    return res.json();
}

async function apiPost(endpoint, payload) {
    if (payload && Array.isArray(payload._files)) {
        return postMultipart(endpoint, payload, payload._files);
    }
    var res = await fetch(BASE + '/' + endpoint, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
    });
    return res.json();
}

async function upload(endpoint, file) {
    var fd = new FormData();
    fd.append('file', file, file.name);
    var res = await fetch(BASE + '/' + endpoint, { method: 'POST', credentials: 'same-origin', body: fd });
    return res.json();
}

window.AstrBotPluginPage = {
    apiGet: apiGet,
    apiPost: apiPost,
    upload: upload,
    getLocale: function () {
        try { return localStorage.getItem('mock_locale') || navigator.language || 'zh-CN'; }
        catch (e) { return 'zh-CN'; }
    },
    setLocale: function (locale) {
        try { localStorage.setItem('mock_locale', locale); } catch (e) {}
        location.reload();
    },
    getI18n: function () { return window.__STEALER_I18N__ || {}; },
    getContext: function () {
        return { locale: this.getLocale(), i18n: this.getI18n() };
    }
};
})();
""".replace("%BASE%", PLUGIN_BASE)


class PreviewState:
    """内存数据源，模拟 db_service + 文件系统。"""

    def __init__(self, seed: bool = True) -> None:
        self.library: list[dict] = []
        self.pending: list[dict] = []
        self.blacklist: set[str] = set()
        self.batch_tasks: dict[str, dict] = {}
        self.prefs: dict[str, str] = {"theme": "auto", "view": "grid"}
        self._next_pending_id = 1
        self._now = int(time.time())
        if seed:
            self._seed()

    # ── 种子数据 ──────────────────────────────────────────────

    def _seed(self) -> None:
        rng = random.Random(42)
        now = self._now
        for idx, emotion in enumerate(EMOTIONS):
            for j in range(rng.randint(1, 3)):
                created = now - (idx * 7 + j * 13) * 3600
                uses = rng.randint(0, 30)
                item = self._new_library_item(
                    category=emotion["key"],
                    desc=f"{emotion['name']}示例表情包 #{idx}{j}",
                    tags=[emotion["name"], "示例"],
                    scenes=rng.choice([["聊天"], ["群聊", "斗图"], []]),
                    created_at=created,
                    use_count=uses,
                    last_used_at=created + uses * 600,
                    is_favorite=1 if rng.random() < 0.15 else 0,
                )
                self.library.append(item)

        for k in range(6):
            emotion = EMOTIONS[k % len(EMOTIONS)]
            self.pending.append(
                {
                    "id": self._next_pending_id,
                    "hash": f"pend{self._next_pending_id:04d}" + uuid.uuid4().hex[:8],
                    "category": emotion["key"],
                    "tags": [emotion["name"], "待审核"],
                    "desc": f"待审核示例 #{k}",
                    "scenes": ["聊天"],
                    "scope_mode": "public",
                    "origin_target": f"group:1000{k}",
                    "source": "steal",
                    "review_status": "pending",
                    "created_at": now - (k + 1) * 1800,
                    "width": 240,
                    "height": 240,
                    "format": "PNG",
                    "bytes": 4096 + k * 128,
                    "add_method": "auto_steal",
                    "source_url": "",
                    "original_name": f"pending_{k}.png",
                }
            )
            self._next_pending_id += 1

    def _new_library_item(self, **kwargs) -> dict:
        ts = kwargs.get("created_at") or self._now
        data = {
            "hash": kwargs.get("hash") or uuid.uuid4().hex[:12],
            "category": kwargs.get("category") or "happy",
            "tags": list(kwargs.get("tags") or []),
            "desc": str(kwargs.get("desc") or ""),
            "scenes": list(kwargs.get("scenes") or []),
            "scope_mode": str(kwargs.get("scope_mode") or "public"),
            "origin_target": str(kwargs.get("origin_target") or ""),
            "created_at": ts,
            "is_favorite": int(kwargs.get("is_favorite") or 0),
            "use_count": int(kwargs.get("use_count") or 0),
            "last_used_at": int(kwargs.get("last_used_at") or 0),
            "width": 240,
            "height": 240,
            "format": "PNG",
            "bytes": 4096,
            "add_method": str(kwargs.get("add_method") or "upload"),
            "reviewed_at": ts,
            "source_url": "",
            "original_name": "",
        }
        return data

    # ── 查询辅助 ──────────────────────────────────────────────

    @staticmethod
    def _match(item: dict, *, q: str, category: str, favorite_only: bool) -> bool:
        if category and item.get("category") != category:
            return False
        if favorite_only and not item.get("is_favorite"):
            return False
        if q:
            haystack = " ".join(
                [
                    " ".join(str(t) for t in item.get("tags") or []),
                    str(item.get("desc") or ""),
                    " ".join(str(s) for s in item.get("scenes") or []),
                ]
            ).lower()
            if q not in haystack:
                return False
        return True

    def _category_counts(self, items: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            cat = str(item.get("category") or "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _categories_list(self, items: list[dict]) -> list[dict]:
        name_map = {e["key"]: e["name"] for e in EMOTIONS}
        result = [
            {"key": key, "name": name_map.get(key, key), "count": count}
            for key, count in self._category_counts(items).items()
        ]
        result.sort(key=lambda c: c["count"], reverse=True)
        return result


def generate_placeholder_png(seed_text: str, label: str, size: int = 300) -> bytes:
    """生成确定性彩色占位图。"""
    from PIL import Image, ImageDraw, ImageFont

    rng = random.Random(seed_text)
    h1 = rng.random()
    top = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(h1, 0.45, 0.92))
    bottom = tuple(int(c * 255) for c in colorsys.hsv_to_rgb((h1 + 0.13) % 1.0, 0.55, 0.65))

    img = Image.new("RGB", (size, size), top)
    draw = ImageDraw.Draw(img)
    for y in range(size):
        ratio = y / max(1, size - 1)
        row = tuple(int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3))
        draw.line([(0, y), (size, y)], fill=row)

    accent_h = (h1 + 0.5) % 1.0
    accent = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(accent_h, 0.7, 0.95))
    cx, cy = size // 2, int(size * 0.42)
    r = int(size * 0.26)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=accent, outline=(255, 255, 255), width=max(2, size // 60))
    eye_r = max(2, size // 40)
    off = r // 3
    for ex in (cx - off, cx + off):
        draw.ellipse([ex - eye_r, cy - eye_r, ex + eye_r, cy + eye_r], fill=(30, 32, 44))
    draw.arc([cx - r // 2, cy, cx + r // 2, cy + r], start=20, end=160, fill=(30, 32, 44), width=max(2, size // 50))

    font = None
    for candidate in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, size=max(12, size // 7))
            break
        except OSError:
            continue
    text = label[:12]
    if font is not None:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((size - tw) / 2 - bbox[0], size * 0.78 - th / 2 - bbox[1]), text, font=font, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def to_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


class PreviewServer:
    """aiohttp 应用工厂 + mock API 实现。"""

    def __init__(self, state: PreviewState | None = None, host: str = "127.0.0.1", port: int = DEFAULT_PORT):
        self.state = state or PreviewState()
        self.host = host
        self.port = port
        self._png_cache: dict[tuple[str, int], bytes] = {}
        self.app = web.Application()
        self._setup_routes()

    # ── 路由 ──────────────────────────────────────────────────

    def _setup_routes(self) -> None:
        r = self.app.router
        r.add_get("/", self.handle_index)
        r.add_get("/index.html", self.handle_index)
        r.add_get("/mock/bridge.js", self.handle_bridge_js)
        r.add_get("/logo.png", self.handle_logo)
        r.add_get("/vaultboy.png", self.handle_vaultboy)
        for filename in ("app.js", "app.css", "template.js"):
            r.add_get("/" + filename, self._make_static_handler(filename))
        r.add_route("*", PLUGIN_BASE + "/{endpoint:.+}", self.handle_api)
        r.add_get("/{tail:.+}", self.handle_fallback_404)

    async def handle_logo(self, request: web.Request) -> web.Response:
        path = DASHBOARD_DIR / "logo.png"
        if not path.is_file():
            return web.Response(status=404, text="missing logo.png")
        return web.Response(body=path.read_bytes(), content_type="image/png")

    async def handle_vaultboy(self, request: web.Request) -> web.Response:
        path = DASHBOARD_DIR / "vaultboy.png"
        if not path.is_file():
            return web.Response(status=404, text="missing vaultboy.png")
        return web.Response(body=path.read_bytes(), content_type="image/png")

    async def handle_index(self, request: web.Request) -> web.Response:
        html = (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
        i18n: dict[str, dict] = {}
        for locale_file in I18N_DIR.glob("*.json"):
            try:
                i18n[locale_file.stem] = json.loads(locale_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        inject = (
            "<script>window.__STEALER_I18N__ = "
            + json.dumps(i18n, ensure_ascii=False)
            + ";</script>\n"
            + '<script src="/mock/bridge.js"></script>\n'
        )
        html = html.replace("</head>", inject + "</head>", 1)
        return web.Response(text=html, content_type="text/html", charset="utf-8")

    async def handle_bridge_js(self, request: web.Request) -> web.Response:
        return web.Response(text=BRIDGE_JS, content_type="application/javascript", charset="utf-8")

    def _make_static_handler(self, filename: str):
        async def handler(request: web.Request) -> web.Response:
            path = DASHBOARD_DIR / filename
            if not path.is_file():
                return web.Response(status=404, text=f"missing {filename}")
            ctype = "text/css" if filename.endswith(".css") else "application/javascript"
            return web.Response(text=path.read_text(encoding="utf-8"), content_type=ctype, charset="utf-8")

        return handler

    async def handle_fallback_404(self, request: web.Request) -> web.Response:
        return web.json_response({"success": False, "error": f"not found: {request.match_info['tail']}"}, status=404)

    # ── API 分发 ──────────────────────────────────────────────

    async def handle_api(self, request: web.Request) -> web.Response:
        endpoint = request.match_info["endpoint"].rstrip("/")
        payload: dict = {}
        if request.method == "POST":
            if request.content_type.startswith("multipart/"):
                form = await request.post()
                files = [(k, v) for k, v in form.items() if hasattr(v, "file")]
                payload = {k: v for k, v in form.items() if not hasattr(v, "file")}
                payload["_multipart_files"] = files
            else:
                try:
                    payload = await request.json()
                except (ValueError, TypeError):
                    payload = {}
        handler = getattr(self, "api_" + endpoint.replace("-", "_").replace("/", "_"), None)
        if handler is None:
            return web.json_response({"success": False, "error": f"no mock for {endpoint}"}, status=404)
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(request, payload)
            else:
                result = handler(request, payload)
        except Exception as exc:  # noqa: BLE001 - mock 层统一兜底
            return web.json_response({"success": False, "error": str(exc)}, status=500)
        return web.json_response(result)

    # ── 图片渲染 ──────────────────────────────────────────────

    def _label_for(self, item: dict) -> str:
        name_map = {e["key"]: e["name"] for e in EMOTIONS}
        return name_map.get(item.get("category"), str(item.get("category", "?")))

    def _png_for(self, item: dict, size: int) -> bytes:
        key = (str(item["hash"]), int(size))
        cached = self._png_cache.get(key)
        if cached is None:
            cached = generate_placeholder_png(str(item["hash"]), self._label_for(item), size)
            self._png_cache[key] = cached
        return cached

    # ── GET handlers ──────────────────────────────────────────

    def api_health(self, request, payload):
        return {"success": True, "status": "ok", "service": "emoji-manager-webui-mock"}

    def api_prefs(self, request, payload):
        valid_themes = {"auto", "dark", "light", "minecraft", "fallout"}
        aliases = {"midnight": "dark", "sakura": "light"}
        if request.method == "POST":
            theme = aliases.get(str(payload.get("theme") or ""), str(payload.get("theme") or "").strip())
            if theme in valid_themes:
                self.state.prefs["theme"] = theme
            if "view" in payload:
                self.state.prefs["view"] = "list" if payload.get("view") == "list" else "grid"
        return {"success": True, **self.state.prefs}

    def api_stats(self, request, payload):
        st = self.state
        today_start = int(time.time()) - int(time.time()) % 86400
        return {
            "success": True,
            "stats": {
                "total": len(st.library),
                "categories": len(st._category_counts(st.library)),
                "today": sum(1 for m in st.library if m.get("created_at", 0) >= today_start),
            },
        }

    def api_images(self, request, payload):
        st = self.state
        page = max(1, int(request.query.get("page", 1)))
        page_size = max(1, min(int(request.query.get("size", 24)), 200))
        q = str(request.query.get("q", "")).lower().strip()
        category = str(request.query.get("category", "") or "")
        sort_order = str(request.query.get("sort", "newest"))
        favorite_only = str(request.query.get("favorite_only", "false")).lower() == "true"
        if category == "__favorite__":
            category = ""
            favorite_only = True

        items = [m for m in st.library if st._match(m, q=q, category=category, favorite_only=favorite_only)]
        sort_keys = {
            "oldest": lambda m: (m.get("created_at", 0), m["hash"]),
            "most_used": lambda m: (m.get("use_count", 0), m.get("last_used_at", 0)),
            "recent_used": lambda m: (m.get("last_used_at", 0), m.get("use_count", 0)),
        }
        if sort_order in sort_keys:
            items.sort(key=sort_keys[sort_order], reverse=(sort_order != "oldest"))
        else:
            items.sort(key=lambda m: (m.get("created_at", 0), m["hash"]), reverse=True)

        total = len(items)
        paged = items[(page - 1) * page_size : (page - 1) * page_size + page_size]
        return {
            "success": True,
            "total": total,
            "page": page,
            "size": page_size,
            "images": [self._library_view(m) for m in paged],
            "categories": st._categories_list(st.library),
            "favorite_count": sum(1 for m in st.library if m.get("is_favorite")),
        }

    @staticmethod
    def _library_view(item: dict) -> dict:
        """与 plugin_api._build_image_item 对齐：is_favorite 输出 bool。"""
        view = dict(item)
        view["is_favorite"] = bool(view.get("is_favorite"))
        return view

    def api_thumbnail(self, request, payload):
        img_hash = request.query.get("hash", "").strip()
        size = int(request.query.get("size", 300))
        item = next((m for m in self.state.library if m["hash"] == img_hash), None)
        if item is None:
            item = next((m for m in self.state.pending if m["hash"] == img_hash), None)
        if item is None:
            return {"success": False, "error": "图片未找到"}
        return {"success": True, "hash": img_hash, "url": to_data_url(self._png_for(item, min(max(size, 32), 600)))}

    def api_image_data(self, request, payload):
        img_hash = request.query.get("hash", "").strip()
        item = next((m for m in self.state.library + self.state.pending if m["hash"] == img_hash), None)
        if item is None:
            return {"success": False, "error": "图片未找到"}
        return {"success": True, "hash": img_hash, "url": to_data_url(self._png_for(item, 480))}

    def api_emotions(self, request, payload):
        return {"success": True, "emotions": [dict(e) for e in EMOTIONS]}

    def api_categories(self, request, payload):
        return {"success": True, "categories": self.state._categories_list(self.state.library)}

    def api_categories_delete(self, request, payload):
        key = str(payload.get("key") or payload.get("category") or "").strip()
        if not key:
            return {"success": False, "error": "缺少分类 key"}
        if any(m["category"] == key for m in self.state.library):
            return {"success": False, "error": "分类下仍有表情包，无法删除"}
        return {"success": True}

    def api_pending(self, request, payload):
        st = self.state
        page = max(1, int(request.query.get("page", 1)))
        page_size = max(1, min(int(request.query.get("size", 24)), 200))
        q = str(request.query.get("q", "")).lower().strip()
        category = str(request.query.get("category", "") or "")
        items = [m for m in st.pending if st._match(m, q=q, category=category, favorite_only=False)]
        items.sort(key=lambda m: m.get("created_at", 0), reverse=True)
        total = len(items)
        paged = items[(page - 1) * page_size : (page - 1) * page_size + page_size]
        return {
            "success": True,
            "images": [dict(m) for m in paged],
            "total": total,
            "category_total": len(st.pending),
            "categories": st._categories_list(st.pending),
        }

    def api_pending_stats(self, request, payload):
        pending = len(self.state.pending)
        return {
            "success": True,
            "stats": {"pending": pending, "capacity": PENDING_CAPACITY, "paused": pending >= PENDING_CAPACITY},
        }

    def api_storage_scan(self, request, payload):
        orphan = max(0, (len(self.state.library) // 7) % 5)
        return {
            "success": True,
            "stale_index": {"count": 0, "bytes": 0, "samples": []},
            "orphan_files": {"count": orphan, "bytes": orphan * 4096, "samples": []},
            "thumb_cache": {"count": len(self._png_cache), "bytes": sum(len(v) for v in self._png_cache.values()), "samples": []},
            "temp_files": {"count": 1, "bytes": 1024, "samples": []},
            "raw_files": {"count": 0, "bytes": 0, "samples": []},
        }

    def api_images_batch_upload_status(self, request, payload):
        task_id = request.query.get("task_id", "")
        task = self.state.batch_tasks.get(task_id)
        if not task:
            return {"success": False, "error": "任务不存在"}
        return {"success": True, **task}

    # ── POST handlers ─────────────────────────────────────────

    def _resolve_ids(self, payload: dict) -> list[int]:
        raw = payload.get("ids")
        if raw is None:
            single = payload.get("id")
            raw = [single] if single is not None else []
        if not isinstance(raw, list):
            raw = []
        return sorted({int(i) for i in raw if i is not None})

    def _approve_one(self, pending_id: int) -> tuple[bool, str]:
        st = self.state
        idx = next((i for i, m in enumerate(st.pending) if m["id"] == pending_id), -1)
        if idx < 0:
            return False, "pending not found"
        row = st.pending.pop(idx)
        if not any(e["key"] == row["category"] for e in EMOTIONS):
            st.pending.insert(idx, row)
            return False, f"invalid category: {row['category']!r}"
        item = st._new_library_item(**{k: row[k] for k in (
            "hash", "category", "tags", "desc", "scenes", "scope_mode", "origin_target", "add_method"
        )})
        item["reviewed_at"] = int(time.time())
        st.library.append(item)
        return True, ""

    def api_pending_approve(self, request, payload):
        ids = self._resolve_ids(payload)
        if not ids:
            return {"success": False, "error": "缺少 id/ids"}
        approved, errors = 0, []
        for pid in ids:
            ok, msg = self._approve_one(pid)
            if ok:
                approved += 1
            elif msg != "pending not found":
                errors.append(f"id={pid}: {msg}")
        return {"success": approved > 0, "approved": approved, "errors": errors}

    def api_pending_reject(self, request, payload):
        ids = self._resolve_ids(payload)
        if not ids:
            return {"success": False, "error": "缺少 id/ids"}
        blacklist = bool(payload.get("blacklist"))
        removed = [m for m in self.state.pending if m["id"] in set(ids)]
        self.state.pending = [m for m in self.state.pending if m["id"] not in set(ids)]
        blacklisted = 0
        if blacklist:
            for m in removed:
                self.state.blacklist.add(m["hash"])
                blacklisted += 1
        return {"success": True, "deleted": len(removed), "blacklisted": blacklisted}

    def api_pending_update(self, request, payload):
        try:
            pid = int(payload.get("id") or 0)
        except (TypeError, ValueError):
            pid = 0
        item = next((m for m in self.state.pending if m["id"] == pid), None)
        if item is None:
            return {"success": False, "error": "pending 不存在"}
        fields = {}
        valid_cats = {e["key"] for e in EMOTIONS}
        if "category" in payload:
            cat = str(payload.get("category") or "").strip()
            if cat and cat not in valid_cats:
                return {"success": False, "error": f"分类无效: {cat!r}"}
            if cat:
                fields["category"] = cat
        if "desc" in payload:
            fields["desc"] = str(payload.get("desc") or "").strip()
        if "scope_mode" in payload:
            fields["scope_mode"] = str(payload.get("scope_mode") or "public").strip()
        for field in ("tags", "scenes"):
            if field in payload:
                raw = payload.get(field)
                if isinstance(raw, list):
                    fields[field] = [str(t).strip() for t in raw if str(t or "").strip()]
                else:
                    fields[field] = [t.strip() for t in str(raw or "").split(",") if t.strip()]
        if not fields:
            return {"success": False, "error": "没有可更新字段"}
        item.update(fields)
        return {"success": True, "item": dict(item)}

    def _find_library_item(self, img_hash: str) -> dict | None:
        return next((m for m in self.state.library if m["hash"] == img_hash), None)

    def api_images_update(self, request, payload):
        img_hash = str(payload.get("hash") or "").strip()
        if not img_hash:
            return {"success": False, "error": "缺少 hash"}
        item = self._find_library_item(img_hash)
        if item is None:
            return {"success": False, "error": "Image not found"}
        if payload.get("category") is not None and payload["category"] != item["category"]:
            new_cat = str(payload["category"]).strip()
            if new_cat:
                item["category"] = new_cat
        if payload.get("tags") is not None:
            tags = payload["tags"]
            item["tags"] = (
                [t.strip() for t in tags.split(",") if t.strip()]
                if isinstance(tags, str)
                else [str(t).strip() for t in tags if str(t).strip()]
            )
        if payload.get("desc") is not None:
            item["desc"] = str(payload["desc"])
        if payload.get("scenes") is not None:
            raw = payload["scenes"]
            parts = raw if isinstance(raw, list) else str(raw).replace("，", ",").replace("、", ",").split(",")
            item["scenes"] = [p.strip() for p in parts if str(p).strip()]
        scope = str(payload.get("scope_mode") or "").strip().lower()
        if scope:
            if scope == "local" and not str(item.get("origin_target", "")).strip():
                return {"success": False, "error": "Origin target missing"}
            item["scope_mode"] = "local" if scope == "local" else "public"
        if payload.get("is_favorite") is not None:
            item["is_favorite"] = 1 if payload["is_favorite"] else 0
        return {"success": True}

    def api_images_delete(self, request, payload):
        img_hash = str(payload.get("hash") or "").strip()
        if not img_hash:
            return {"success": False, "error": "缺少 hash"}
        before = len(self.state.library)
        self.state.library = [m for m in self.state.library if m["hash"] != img_hash]
        count = before - len(self.state.library)
        if not count:
            return {"success": False, "error": "图片未找到"}
        if payload.get("blacklist"):
            self.state.blacklist.add(img_hash)
        return {"success": True, "count": count}

    def api_images_batch_delete(self, request, payload):
        hashes = set(payload.get("hashes") or [])
        before = len(self.state.library)
        self.state.library = [m for m in self.state.library if m["hash"] not in hashes]
        return {"success": True, "count": before - len(self.state.library)}

    def api_images_batch_move(self, request, payload):
        hashes = set(payload.get("hashes") or [])
        category = str(payload.get("category") or "").strip()
        if not category:
            return {"success": False, "error": "缺少目标分类"}
        moved = 0
        for m in self.state.library:
            if m["hash"] in hashes:
                m["category"] = category
                moved += 1
        return {"success": True, "count": moved}

    def api_images_batch_scope(self, request, payload):
        hashes = set(payload.get("hashes") or [])
        scope = "local" if str(payload.get("scope_mode", "")).lower() == "local" else "public"
        count, skipped = 0, 0
        for m in self.state.library:
            if m["hash"] in hashes:
                if scope == "local" and not str(m.get("origin_target", "")).strip():
                    skipped += 1
                    continue
                m["scope_mode"] = scope
                count += 1
        return {"success": True, "count": count, "skipped": skipped}

    def api_images_batch_favorite(self, request, payload):
        hashes = set(payload.get("hashes") or [])
        fav = 1 if payload.get("favorite") else 0
        count = 0
        for m in self.state.library:
            if m["hash"] in hashes:
                m["is_favorite"] = fav
                count += 1
        return {"success": True, "count": count}

    def api_images_scope_repair(self, request, payload):
        hashes = set(payload.get("hashes") or [])
        origin_target = str(payload.get("origin_target") or "").strip()
        scope_mode = "local" if str(payload.get("scope_mode", "")).lower() == "local" else "public"
        if not origin_target:
            return {"success": False, "error": "缺少 origin_target"}
        count = 0
        for m in self.state.library:
            if m["hash"] in hashes and (not payload.get("only_missing") or not m.get("origin_target")):
                m["origin_target"] = origin_target
                m["scope_mode"] = scope_mode
                count += 1
        return {"success": True, "count": count}

    def _parse_upload_meta(self, source: dict) -> dict:
        category = str(source.get("category", source.get("emotion", "")) or "").strip()
        tags_raw = source.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        else:
            tags = []
        scenes_raw = source.get("scenes", source.get("scene"))
        if isinstance(scenes_raw, list):
            scenes = [str(s).strip() for s in scenes_raw if str(s).strip()]
        else:
            scenes = [s.strip() for s in str(scenes_raw or "").replace("，", ",").replace("、", ",").split(",") if s.strip()]
        return {"category": category or "unknown", "tags": tags, "desc": str(source.get("desc", "") or ""), "scenes": scenes}

    def _store_upload(self, content: bytes, filename: str, meta: dict) -> dict:
        import hashlib

        img_hash = hashlib.md5(content).hexdigest()[:16]
        existing = self._find_library_item(img_hash)
        if existing:
            return {"hash": img_hash, "category": existing["category"], "duplicate": True}
        item = self.state._new_library_item(hash=img_hash, category=meta["category"], tags=meta["tags"], desc=meta["desc"], scenes=meta["scenes"], add_method="webui_upload")
        item["bytes"] = len(content)
        self.state.library.append(item)
        return {"hash": img_hash, "category": meta["category"]}

    async def api_images_upload(self, request, payload):
        files = payload.pop("_multipart_files", None)
        if files:
            _, file_field = files[0]
            content = file_field.file.read()
            filename = file_field.filename or "upload.png"
        else:
            b64 = str(payload.get("base64", ""))
            if not b64:
                return {"success": False, "error": "没有上传文件"}
            raw = b64.split(",", 1)[1] if "," in b64 else b64
            try:
                content = base64.b64decode(raw)
            except Exception:
                return {"success": False, "error": "图片数据无效"}
            filename = str(payload.get("filename", "upload.png"))
        if not content:
            return {"success": False, "error": "文件内容为空"}
        image = self._store_upload(content, filename, self._parse_upload_meta(payload))
        return {"success": True, "image": image, "hash": image["hash"]}

    async def api_images_batch_upload(self, request, payload):
        entries: list[tuple[bytes, str]] = []
        files = payload.pop("_multipart_files", None)
        if files:
            for _, field in files:
                entries.append((field.file.read(), field.filename or "batch.png"))
        else:
            for entry in payload.get("_files") or []:
                raw = str(entry.get("base64", "")).split(",", 1)[-1]
                try:
                    entries.append((base64.b64decode(raw), str(entry.get("name", "batch.png"))))
                except Exception:
                    continue
        task_id = uuid.uuid4().hex[:12]
        success_count = failed_count = 0
        for content, filename in entries:
            try:
                self._store_upload(content, filename, self._parse_upload_meta(payload))
                success_count += 1
            except Exception:
                failed_count += 1
        total = len(entries)
        self.state.batch_tasks[task_id] = {
            "status": "completed" if failed_count == 0 else ("failed" if success_count == 0 else "completed"),
            "processed": total,
            "total": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "updated_at": int(time.time()),
        }
        return {"success": True, "task_id": task_id}

    def api_analyze(self, request, payload):
        emotion = EMOTIONS[int(time.time()) % len(EMOTIONS)]
        return {
            "success": True,
            "category": emotion["key"],
            "tags": [emotion["name"], "AI分析"],
            "desc": f"[mock VLM 分析] 看起来很「{emotion['name']}」",
            "scenes": ["聊天"],
        }

    def api_storage_cleanup(self, request, payload):
        removed_thumbs = len(self._png_cache)
        self._png_cache.clear()
        return {
            "success": True,
            "removed": {"stale_index": 0, "orphan_files": 0, "thumb_cache": removed_thumbs, "temp_files": 1},
        }

    # ── 启动 ──────────────────────────────────────────────────

    def run(self) -> None:
        print("\n" + "=" * 56)
        print("  表情包小偷 WebUI 预览服务器 (pages/dashboard)")
        print(f"  预览地址: http://{self.host}:{self.port}/")
        print(f"  库内表情: {len(self.state.library)}  待审核: {len(self.state.pending)}")
        print("=" * 56 + "\n")
        web.run_app(self.app, host=self.host, port=self.port, print=None)


def create_app(state: PreviewState | None = None) -> web.Application:
    """供 pytest / aiohttp test_utils 使用。"""
    return PreviewServer(state=state).app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="表情包小偷 WebUI 预览测试服务器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-seed", action="store_true", help="不生成种子数据")
    args = parser.parse_args()
    server = PreviewServer(state=PreviewState(seed=not args.no_seed), host=args.host, port=args.port)
    server.run()
