import asyncio
import hashlib
import hmac
import logging
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

logger = logging.getLogger("astrbot")


class WebServer:
    """Web UI 服务器类，提供表情包管理界面。"""

    # 常量定义
    CLIENT_MAX_SIZE = 50 * 1024 * 1024  # 50MB 最大请求大小
    SESSION_CLEANUP_INTERVAL = 300  # Session 清理间隔（秒）
    SESSION_MAX_COUNT = 1000  # 最大 Session 数量

    def __init__(self, plugin: Any, host: str = "0.0.0.0", port: int = 8899):
        self.plugin = plugin
        self.host: str = host
        self.port: int = port
        self.app: web.Application = web.Application(
            client_max_size=self.CLIENT_MAX_SIZE,
            middlewares=[self._error_middleware, self._auth_middleware],
        )
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self._started: bool = False

        # Web UI 静态文件目录 (插件目录下的 web 文件夹)
        self.static_dir: Path = Path(__file__).resolve().parent / "web"

        # 图片数据目录 (数据目录下的 plugin_data/...)
        self.data_dir: Path = self.plugin.base_dir

        self._cookie_name: str = "stealer_webui_session"
        self._sessions: dict[str, float] = {}
        self._last_session_cleanup: float = 0.0  # 上次 session 清理时间
        self._session_cleanup_interval: int = self.SESSION_CLEANUP_INTERVAL

        self._setup_routes()

    # ── 响应快捷方法 ──────────────────────────────────────────

    @staticmethod
    def _ok(data: dict | None = None, **kwargs) -> web.Response:
        """返回成功 JSON 响应。

        用法:
            self._ok()                          → {"success": True}
            self._ok({"count": 3})              → {"success": True, "count": 3}
            self._ok(count=3)                   → {"success": True, "count": 3}
        """
        body: dict = {"success": True}
        if data:
            body.update(data)
        if kwargs:
            body.update(kwargs)
        return web.json_response(body)

    @staticmethod
    def _err(msg: str, status: int = 500) -> web.Response:
        """返回失败 JSON 响应。"""
        return web.json_response({"success": False, "error": msg}, status=status)

    # ── 中间件 ────────────────────────────────────────────────

    async def _error_middleware(self, app: web.Application, handler):
        async def middleware_handler(request: web.Request):
            try:
                return await handler(request)
            except web.HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unhandled WebUI error: {e}", exc_info=True)
                if (request.path or "").startswith("/api/"):
                    return WebServer._err("Internal Server Error")
                return web.Response(text="500 Internal Server Error", status=500)

        return middleware_handler

    def _is_auth_enabled(self) -> bool:
        try:
            auth_enabled = getattr(self.plugin, "webui_auth_enabled", True)
            return bool(auth_enabled)
        except Exception:
            return True

    def _get_expected_secret(self) -> str:
        if not self._is_auth_enabled():
            return ""

        password = ""
        # 1. 尝试从 self.plugin.webui_password 获取 (如果 Main 类里有这个属性)
        try:
            password = str(getattr(self.plugin, "webui_password", "") or "").strip()
        except Exception:
            password = ""
        if password:
            return password

        # 2. 尝试从 self.plugin.plugin_config.webui.password 获取
        try:
            plugin_config = getattr(self.plugin, "plugin_config", None)
            if plugin_config and hasattr(plugin_config, "webui"):
                password = str(plugin_config.webui.password or "").strip()
        except Exception:
            password = ""

        if password:
            return password
        return ""

    def _get_session_timeout(self) -> int:
        timeout = 3600
        # 1. 尝试从 self.plugin.webui_session_timeout 获取
        try:
            timeout = int(getattr(self.plugin, "webui_session_timeout", 3600) or 3600)
        except Exception:
            timeout = 3600

        # 如果是默认值，尝试从配置获取
        if timeout == 3600:
            try:
                plugin_config = getattr(self.plugin, "plugin_config", None)
                if plugin_config and hasattr(plugin_config, "webui"):
                    timeout = int(plugin_config.webui.session_timeout or 3600)
            except Exception:
                timeout = 3600

        if timeout <= 0:
            timeout = 3600
        return timeout

    async def _auth_middleware(self, app: web.Application, handler):
        async def middleware_handler(request: web.Request):
            if request.method == "OPTIONS":
                return await handler(request)

            expected = self._get_expected_secret()
            if not expected:
                return await handler(request)

            path = request.path or "/"
            if (
                path in ("/", "/index.html")
                or path.startswith("/web")
                or path in ("/auth/info", "/auth/login", "/auth/logout")
            ):
                return await handler(request)

            sid = str(request.cookies.get(self._cookie_name, "") or "").strip()
            now = time.time()

            # 定期清理所有过期 session，防止内存泄漏
            if now - self._last_session_cleanup > self._session_cleanup_interval:
                expired = [k for k, v in self._sessions.items() if v < now]
                for k in expired:
                    self._sessions.pop(k, None)
                self._last_session_cleanup = now

                # 额外检查：如果 session 数量超过上限，清理最旧的一半
                if len(self._sessions) > self.SESSION_MAX_COUNT:
                    sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1])
                    to_remove = len(self._sessions) - self.SESSION_MAX_COUNT // 2
                    for k, _ in sorted_sessions[:to_remove]:
                        self._sessions.pop(k, None)
                    logger.warning(
                        f"Session 数量超过上限 {self.SESSION_MAX_COUNT}，已清理 {to_remove} 个最旧的 session"
                    )

            exp = self._sessions.get(sid)
            if not exp or exp < now:
                if sid:
                    self._sessions.pop(sid, None)
                if path.startswith("/api/"):
                    return WebServer._err("Unauthorized", 401)
                raise web.HTTPUnauthorized(text="Unauthorized")

            return await handler(request)

        return middleware_handler

    def _setup_routes(self):
        self.app.router.add_get("/api/images", self.handle_list_images)
        self.app.router.add_delete("/api/images/{hash}", self.handle_delete_image)
        self.app.router.add_put("/api/images/{hash}", self.handle_update_image)
        self.app.router.add_post("/api/images/batch/delete", self.handle_batch_delete)
        self.app.router.add_post("/api/images/batch/move", self.handle_batch_move)
        self.app.router.add_post("/api/images/upload", self.handle_upload_image)
        self.app.router.add_get("/api/stats", self.handle_get_stats)
        self.app.router.add_get("/api/config", self.handle_get_config)
        self.app.router.add_get("/api/categories", self.handle_get_categories)
        self.app.router.add_post("/api/categories", self.handle_update_categories)
        self.app.router.add_delete("/api/categories/{key}", self.handle_delete_category)
        self.app.router.add_get("/api/emotions", self.handle_get_emotions)
        self.app.router.add_get("/api/health", self.handle_health_check)
        self.app.router.add_get("/auth/info", self.handle_auth_info)
        self.app.router.add_post("/auth/login", self.handle_auth_login)
        self.app.router.add_post("/auth/logout", self.handle_auth_logout)

        # 静态文件路由
        # 1. 前端页面 - 首页
        self.app.router.add_get("/", self.handle_index)
        # 某些客户端/代理在遇到 FileResponse 异常时会报 "HTTP/0.9"，提供显式入口便于排障
        self.app.router.add_get("/index.html", self.handle_index)

        # 2. 静态资源
        # 插件 web/index.html 如果引用了本地资源（js/css/img），这里提供静态托管。
        # 兼容直接打包在 web/ 目录下的资源结构。
        self.app.router.add_get("/web/{path:.*}", self.handle_web_static)

        self.app.router.add_get("/images/{path:.*}", self.handle_images_static)

    def _resolve_safe_path(
        self, raw: str, base_dir: Path
    ) -> tuple[Path | None, str | None]:
        """安全解析请求路径，防止路径遍历攻击。

        Args:
            raw: 原始请求路径
            base_dir: 基础目录

        Returns:
            tuple: (解析后的安全路径, 错误类型) 或 (None, 错误类型)
        """
        raw = str(raw or "").lstrip("/")
        if not raw:
            return None, "not_found"

        # 安全检查：禁止路径遍历和绝对路径
        if (
            ".." in raw
            or raw.startswith(("/", "\\"))
            or ":" in raw  # Windows 驱动器字母
            or "\x00" in raw  # 空字节注入
        ):
            logger.warning(f"可疑路径请求被拒绝: {raw!r}")
            return None, "bad_request"

        # Windows 特殊设备名检查
        win_reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }
        name_part = raw.split("/")[0].split("\\")[0].upper()
        if name_part in win_reserved:
            logger.warning(f"Windows 保留设备名请求被拒绝: {raw!r}")
            return None, "bad_request"

        base_dir = base_dir.resolve()
        try:
            # 标准化路径并验证是否在基础目录内
            abs_path = (base_dir / raw).resolve()

            # 双重检查：确保解析后的路径确实在基础目录内
            # 使用 os.path.commonpath 更可靠
            try:
                abs_path.relative_to(base_dir)
            except ValueError:
                logger.warning(f"路径遍历尝试被阻止: {raw!r} -> {abs_path}")
                return None, "not_found"

        except Exception as e:
            logger.debug(f"路径解析失败: {raw!r}, 错误: {e}")
            return None, "not_found"

        return abs_path, None

    async def handle_web_static(self, request: web.Request) -> web.StreamResponse:
        abs_path, error = self._resolve_safe_path(
            request.match_info.get("path", ""), self.static_dir
        )
        if error == "bad_request":
            raise web.HTTPBadRequest(text="invalid path")
        if not abs_path:
            raise web.HTTPNotFound()

        if not abs_path.exists() or not abs_path.is_file():
            raise web.HTTPNotFound()

        # 改用手动读取并构造 Response，避免 Windows 下 FileResponse 可能的协议问题 (ERR_INVALID_HTTP_RESPONSE)
        try:
            import mimetypes

            content_type, _ = mimetypes.guess_type(abs_path)
            if not content_type:
                content_type = "application/octet-stream"

            # 读取文件内容
            # 注意：对于大文件这可能会占用内存，但 web 目录下的静态资源通常较小
            if content_type.startswith("text/") or content_type in (
                "application/javascript",
                "application/json",
                "application/xml",
            ):
                try:
                    content = await asyncio.to_thread(
                        abs_path.read_text, encoding="utf-8"
                    )
                    return web.Response(text=content, content_type=content_type)
                except UnicodeDecodeError:
                    # 如果不是 UTF-8，尝试二进制
                    pass

            content = await asyncio.to_thread(abs_path.read_bytes)
            return web.Response(body=content, content_type=content_type)

        except Exception as e:
            logger.error(f"Failed to serve static file {abs_path}: {e}")
            raise web.HTTPNotFound()

    async def handle_images_static(self, request: web.Request) -> web.StreamResponse:
        abs_path, error = self._resolve_safe_path(
            request.match_info.get("path", ""), self.data_dir
        )
        if error == "bad_request":
            raise web.HTTPBadRequest(text="invalid path")
        if not abs_path:
            raise web.HTTPNotFound()

        if not abs_path.exists() or not abs_path.is_file():
            raise web.HTTPNotFound()

        try:
            return web.FileResponse(path=abs_path)
        except Exception as e:
            logger.warning(f"Failed to serve image {abs_path}: {e}")
            raise web.HTTPNotFound()

    async def start(self) -> bool:
        """启动 Web 服务器

        Returns:
            bool: 是否启动成功
        """
        try:
            # 检查静态文件目录
            if not self.static_dir.exists():
                logger.warning(f"WebUI static directory not found: {self.static_dir}")

            # 创建并启动服务器
            # access_log=None 防止日志系统冲突
            self.runner = web.AppRunner(self.app, access_log=None)
            await self.runner.setup()

            # 标准绑定
            self.site = web.TCPSite(self.runner, str(self.host), int(self.port))
            await self.site.start()

            self._started = True

            # 显示实际监听地址
            protocol = "http"
            if self.host == "0.0.0.0":
                logger.info(
                    f"Emoji Manager WebUI started - listening on all interfaces (0.0.0.0:{self.port})"
                )
                logger.info(f"  → Local access: {protocol}://127.0.0.1:{self.port}")
            else:
                logger.info(
                    f"Emoji Manager WebUI started at {protocol}://{self.host}:{self.port}"
                )

            return True

        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 98 or e.errno == 10048:
                logger.error(
                    f"WebUI 端口 {self.port} 已被占用，请更换端口或关闭占用该端口的程序"
                )
            else:
                logger.error(f"Failed to start WebUI (OS error): {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to start WebUI: {e}", exc_info=True)
            return False

    async def stop(self):
        """停止 Web 服务器"""
        if not self._started:
            return
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        self._started = False
        logger.info("Emoji Manager WebUI stopped")

    async def handle_index(self, request):
        """返回首页"""
        try:
            index_file = self.static_dir / "index.html"
            if not index_file.exists():
                return web.Response(
                    text="<h1>Emoji Manager WebUI</h1><p>index.html not found</p>",
                    content_type="text/html",
                    status=404,
                )
            # 这里不直接使用 FileResponse：
            # - 在部分环境/代理下，如果传输过程中异常中断，curl 可能会报 Received HTTP/0.9
            # - 显式构造 Response 能确保状态行和头部稳定输出
            try:
                content = await asyncio.to_thread(
                    index_file.read_text, encoding="utf-8"
                )
            except UnicodeDecodeError:
                # 兼容被意外写入非 UTF-8 的情况（尽量仍返回合法 HTTP 响应）
                content = await asyncio.to_thread(
                    index_file.read_text, encoding="utf-8", errors="replace"
                )
                logger.warning(
                    "WebUI index.html is not valid UTF-8, returned with replacement characters.",
                )
            return web.Response(text=content, content_type="text/html", status=200)
        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return web.Response(text=f"Error: {e}", status=500)

    async def handle_list_images(self, request):
        """获取图片列表"""
        try:
            page = int(request.query.get("page", 1))
            page_size = int(request.query.get("size", 50))
            category_filter = request.query.get("category", None)
            search_query = request.query.get("q", "").lower()
            sort_order = request.query.get("sort", "newest")
            include_meta = request.query.get("meta", "0") in ("1", "true", "yes")

            # 获取所有图片数据
            # 优先从 ImageProcessorService 的缓存或索引中获取
            # index 结构: path -> {hash, category, tags, desc, ...}
            # 但我们需要列表形式

            # 从 cache_service 获取持久化索引
            index = self.plugin.cache_service.get_index_cache()

            images = []
            category_counts = {}

            for path_str, meta in index.items():
                # 转换路径为 web 可访问的 URL
                # 假设 path_str 是绝对路径，我们需要将其转换为相对于 data_dir 的路径
                try:
                    abs_path = Path(path_str)
                    rel_path = abs_path.relative_to(self.data_dir)
                    # /images 路由会负责将相对路径映射到插件数据目录
                    # 使用正斜杠作为路径分隔符（Web 标准）
                    url = f"/images/{rel_path.as_posix()}"

                    # 检查文件是否存在
                    if not abs_path.exists():
                        continue

                    item = {
                        "hash": meta.get("hash", ""),
                        "url": url,
                        "category": meta.get("category", "unknown"),
                        "tags": meta.get("tags", []),
                        "desc": meta.get("desc", ""),
                        "created_at": meta.get("created_at", 0),
                    }

                    # 预览增强：按需返回一些元信息（默认关闭，避免无谓 I/O）
                    if include_meta:
                        try:
                            stat = await asyncio.to_thread(abs_path.stat)
                            item.update(
                                {
                                    "rel_path": rel_path.as_posix(),
                                    "filename": abs_path.name,
                                    "size_bytes": stat.st_size,
                                    # 若索引里没写 created_at，则用 mtime 兜底
                                    "mtime": int(stat.st_mtime),
                                }
                            )
                            if not item.get("created_at"):
                                item["created_at"] = int(stat.st_mtime)
                        except Exception:
                            # 忽略单个文件的 stat 失败
                            pass

                    # 搜索过滤
                    if search_query:
                        # 简单的搜索匹配
                        in_tags = any(search_query in t.lower() for t in item["tags"])
                        in_desc = search_query in item["desc"].lower()
                        if not (in_tags or in_desc):
                            continue

                    # 统计符合搜索条件的分类数量（不受当前选中的分类影响）
                    cat = item["category"]
                    category_counts[cat] = category_counts.get(cat, 0) + 1

                    # 分类过滤
                    if category_filter and item["category"] != category_filter:
                        continue

                    images.append(item)
                except ValueError:
                    # 路径不在 data_dir 下，可能是旧数据或异常
                    continue

            # 排序
            if sort_order == "oldest":
                images.sort(key=lambda x: x["created_at"], reverse=False)
            else:  # newest (default)
                images.sort(key=lambda x: x["created_at"], reverse=True)

            # 分页
            total = len(images)
            start = (page - 1) * page_size
            end = start + page_size
            paged_images = images[start:end]

            # 获取分类详细信息并附带数量
            categories_list = []
            if hasattr(self.plugin, "plugin_config"):
                # 获取配置中的分类列表
                base_categories = self.plugin.plugin_config.get_category_info()
                for cat_info in base_categories:
                    key = cat_info["key"]
                    count = category_counts.get(key, 0)
                    categories_list.append(
                        {"key": key, "name": cat_info["name"], "count": count}
                    )

                # 处理未在配置中但存在的分类（如 unknown 或旧数据）
                known_keys = {c["key"] for c in categories_list}
                for cat_key, count in category_counts.items():
                    if cat_key not in known_keys:
                        categories_list.append(
                            {
                                "key": cat_key,
                                "name": cat_key,  # Fallback name
                                "count": count,
                            }
                        )

            # 按数量倒序排序分类列表
            categories_list.sort(key=lambda x: x["count"], reverse=True)

            return self._ok(
                {
                    "total": total,
                    "page": page,
                    "size": page_size,
                    "images": paged_images,
                    "categories": categories_list,
                }
            )
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return self._err(str(e))

    async def handle_delete_image(self, request):
        """删除图片"""
        image_hash = request.match_info["hash"]
        add_to_blacklist = request.query.get("blacklist", "false").lower() == "true"

        removed_paths: list[str] = []
        await self.plugin.cache_service.update_index(
            lambda current: self._remove_by_hashes(current, {image_hash}, removed_paths)
        )
        target_path = removed_paths[0] if removed_paths else None

        if target_path:
            try:
                # 1. 删除文件（通过统一入口）
                await self.plugin._safe_remove_file(target_path)

                # 2. 添加到黑名单（如果请求）
                if add_to_blacklist:
                    await self.plugin.cache_service.set(
                        "blacklist_cache", image_hash, int(time.time()), persist=True
                    )
                    logger.info(f"Added image {image_hash} to blacklist")

                # 失效缓存
                if hasattr(self.plugin, "image_processor_service"):
                    self.plugin.image_processor_service.invalidate_cache(image_hash)

                return self._ok()
            except Exception as e:
                logger.error(f"Failed to delete image {target_path}: {e}")
                return self._err(str(e))

        return self._err("Image not found", 404)

    def _remove_by_hashes(
        self, current: dict, hashes: set[str], removed_paths: list[str]
    ):
        for path_str, meta in list(current.items()):
            if isinstance(meta, dict) and meta.get("hash") in hashes:
                removed_paths.append(path_str)
                del current[path_str]

    def _collect_items_by_hashes(
        self, index_map: dict, hashes: set[str]
    ) -> list[tuple[str, dict]]:
        items: list[tuple[str, dict]] = []
        for path_str, meta in index_map.items():
            if isinstance(meta, dict) and meta.get("hash") in hashes:
                items.append((path_str, meta))
        return items

    async def _move_items_to_category(
        self, index_map: dict, items: list[tuple[str, dict]], target_category: str
    ) -> tuple[int, list[str]]:
        moved_count = 0
        moved_hashes: list[str] = []
        target_dir = self.plugin.plugin_config.ensure_category_dir(target_category)

        for old_path_str, meta in items:
            try:
                old_path = Path(old_path_str)
                if not old_path.exists():
                    continue

                new_path = target_dir / old_path.name
                await asyncio.to_thread(shutil.move, str(old_path), str(new_path))

                if old_path_str in index_map:
                    del index_map[old_path_str]
                meta["path"] = str(new_path)
                meta["category"] = target_category
                index_map[str(new_path)] = meta
                moved_count += 1
                if meta.get("hash"):
                    moved_hashes.append(meta["hash"])
            except Exception as e:
                logger.warning(f"Failed to move {old_path_str}: {e}")

        return moved_count, moved_hashes

    async def handle_update_image(self, request):
        """更新图片信息 (Category, Tags, Desc)"""
        try:
            image_hash = request.match_info["hash"]
            data = await request.json()

            new_category = data.get("category")
            new_tags = data.get("tags")  # list or comma separated string
            new_desc = data.get("desc")

            updated = {"ok": False, "error": ""}

            async def updater(current: dict):
                target_path = None
                meta = None
                for path_str, m in current.items():
                    if isinstance(m, dict) and m.get("hash") == image_hash:
                        target_path = path_str
                        meta = m
                        break

                if not target_path or not meta:
                    updated["error"] = "Image not found"
                    return

                if new_tags is not None:
                    if isinstance(new_tags, str):
                        meta["tags"] = [
                            t.strip() for t in new_tags.split(",") if t.strip()
                        ]
                    else:
                        meta["tags"] = new_tags

                if new_desc is not None:
                    meta["desc"] = new_desc

                if new_category and new_category != meta.get("category"):
                    old_path = Path(target_path)
                    if not old_path.exists():
                        updated["error"] = "Source file not found"
                        return
                    moved, _ = await self._move_items_to_category(
                        current, [(target_path, meta)], new_category
                    )
                    if moved <= 0:
                        updated["error"] = "Move failed"
                        return
                else:
                    current[target_path] = meta

                updated["ok"] = True

            await self.plugin.cache_service.update_index(updater)
            if not updated["ok"]:
                return self._err(updated["error"] or "Update failed", 404)
            return self._ok()

        except Exception as e:
            logger.error(f"Failed to update image: {e}")
            return self._err(str(e))

    async def handle_batch_delete(self, request):
        """批量删除"""
        try:
            data = await request.json()
            hashes = data.get("hashes", [])
            if not hashes:
                return self._ok(count=0)

            removed_paths: list[str] = []
            hash_set = set(hashes)
            await self.plugin.cache_service.update_index(
                lambda current: self._remove_by_hashes(current, hash_set, removed_paths)
            )

            deleted_count = 0
            for path_str in removed_paths:
                try:
                    await self.plugin._safe_remove_file(path_str)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {path_str}: {e}")

            return self._ok(count=deleted_count)
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            return self._err(str(e))

    async def handle_batch_move(self, request):
        """批量移动"""
        try:
            data = await request.json()
            hashes = data.get("hashes", [])
            target_category = data.get("category")

            if not hashes or not target_category:
                return self._err("Missing hashes or category", 400)

            moved_hashes: list[str] = []
            moved_count = 0
            hash_set = set(hashes)

            async def updater(current: dict):
                nonlocal moved_count, moved_hashes
                items = self._collect_items_by_hashes(current, hash_set)
                items = [
                    (path_str, meta)
                    for path_str, meta in items
                    if meta.get("category") != target_category
                ]
                moved_count, moved_hashes = await self._move_items_to_category(
                    current, items, target_category
                )

            await self.plugin.cache_service.update_index(updater)

            if moved_count > 0 and hasattr(self.plugin, "image_processor_service"):
                for h in moved_hashes:
                    self.plugin.image_processor_service.invalidate_cache(h)

            return self._ok(count=moved_count)
        except Exception as e:
            logger.error(f"Batch move failed: {e}")
            return self._err(str(e))

    async def handle_get_stats(self, request):
        try:
            index = self.plugin.cache_service.get_index_cache()

            # 计算今日新增
            today_start = (
                datetime.now()
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .timestamp()
            )
            today_count = 0
            for meta in index.values():
                if meta.get("created_at", 0) >= today_start:
                    today_count += 1

            # 获取分类数量（配置的分类总数）
            categories_count = 0
            if hasattr(self.plugin, "plugin_config"):
                categories_count = len(self.plugin.plugin_config.categories)

            # 前端期望的数据结构是 { stats: { total, categories, today } }
            return self._ok(
                {
                    "stats": {
                        "total": len(index),
                        "categories": categories_count,
                        "today": today_count,
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return self._err(str(e))

    async def handle_get_config(self, request):
        return self._ok({"version": "1.0.0", "plugin_version": "0.1.0"})

    async def handle_health_check(self, request):
        return self._ok({"status": "ok", "service": "emoji-manager-webui"})

    async def handle_auth_info(self, request):
        expected = self._get_expected_secret()
        return self._ok(
            {
                "requires_auth": bool(expected),
                "session_timeout": self._get_session_timeout(),
            }
        )

    async def handle_auth_login(self, request):
        expected = self._get_expected_secret()
        if not expected:
            return self._ok(requires_auth=False)

        try:
            payload = await request.json()
        except Exception:
            payload = {}

        provided = str((payload or {}).get("password", "") or "").strip()
        if not provided or not hmac.compare_digest(provided, expected):
            return self._err("Unauthorized", 401)

        timeout = self._get_session_timeout()
        sid = uuid.uuid4().hex
        exp = time.time() + float(timeout)
        self._sessions[sid] = exp

        resp = self._ok(expires_at=int(exp))
        resp.set_cookie(
            self._cookie_name,
            sid,
            max_age=timeout,
            httponly=True,
            samesite="Lax",
            path="/",
        )
        return resp

    async def handle_auth_logout(self, request):
        sid = str(request.cookies.get(self._cookie_name, "") or "").strip()
        if sid:
            self._sessions.pop(sid, None)
        resp = self._ok()
        resp.del_cookie(self._cookie_name, path="/")
        return resp

    async def handle_upload_image(self, request):
        try:
            data = await request.post()

            if "file" not in data:
                return self._err("没有上传文件", 400)

            uploaded_file = data["file"]
            if not uploaded_file.filename:
                return self._err("文件名无效", 400)

            category = data.get("emotion", "").strip()
            if not category:
                return self._err("请选择情绪分类", 400)
            tags_raw = data.get("tags", "").strip()
            tags = (
                [t.strip() for t in tags_raw.split(",") if t.strip()]
                if tags_raw
                else []
            )
            desc = data.get("desc", "").strip()

            file_ext = Path(uploaded_file.filename).suffix.lower()
            allowed_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
            if file_ext not in allowed_exts:
                return self._err(
                    f"不支持的文件类型，允许: {', '.join(allowed_exts)}", 400
                )

            file_content = await asyncio.to_thread(uploaded_file.file.read)
            file_hash = hashlib.sha256(file_content).hexdigest()

            timestamp = int(datetime.now().timestamp())
            unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"

            category_dir = self.plugin.plugin_config.ensure_category_dir(category)
            file_path = category_dir / unique_filename

            def _write_file():
                with open(file_path, "wb") as f:
                    f.write(file_content)

            await asyncio.to_thread(_write_file)

            await self.plugin.cache_service.update_index(
                lambda current: current.__setitem__(
                    str(file_path),
                    {
                        "hash": file_hash,
                        "path": str(file_path),
                        "category": category,
                        "tags": tags,
                        "desc": desc,
                        "created_at": timestamp,
                    },
                )
            )

            rel_path = file_path.relative_to(self.data_dir)
            url = f"/images/{rel_path.as_posix()}"

            return self._ok(
                {
                    "image": {
                        "hash": file_hash,
                        "url": url,
                        "category": category,
                        "tags": tags,
                        "desc": desc,
                        "created_at": timestamp,
                    },
                }
            )

        except Exception as e:
            logger.error(f"上传图片失败: {e}", exc_info=True)
            return self._err(str(e))

    async def handle_get_categories(self, request):
        try:
            index = self.plugin.cache_service.get_index_cache()
            categories = {}
            for meta in index.values():
                if isinstance(meta, dict):
                    cat = meta.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
            return self._ok({"categories": categories})
        except Exception as e:
            logger.error(f"获取分类列表失败: {e}")
            return self._err(str(e))

    async def handle_update_categories(self, request):
        try:
            data = await request.json()
            new_categories_data = data.get("categories", [])

            if not isinstance(new_categories_data, list) or not new_categories_data:
                return self._err("分类列表无效", 400)

            # 拆分数据：提取 key 列表和 category_info
            category_keys = []
            category_info = {}

            for item in new_categories_data:
                if isinstance(item, dict) and item.get("key"):
                    key = item["key"]
                    category_keys.append(key)
                    # 只有当有 name 或 desc 时才保存
                    if item.get("name") or item.get("desc"):
                        category_info[key] = {
                            "name": item.get("name", ""),
                            "desc": item.get("desc", ""),
                        }
                elif isinstance(item, str):
                    # 兼容旧格式（直接发送字符串）
                    category_keys.append(item)

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": category_keys})
            else:
                self.plugin.plugin_config.categories = category_keys
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = category_keys

            self.plugin.plugin_config.category_info.update(category_info)
            self.plugin.plugin_config.save_category_info()

            return self._ok(categories=category_keys)
        except Exception as e:
            logger.error(f"更新分类列表失败: {e}", exc_info=True)
            return self._err(str(e))

    async def handle_delete_category(self, request):
        try:
            category_key = (request.match_info.get("key") or "").strip()
            if not category_key:
                return self._err("分类Key无效", 400)

            if (
                not hasattr(self.plugin, "plugin_config")
                or not self.plugin.plugin_config
            ):
                return self._err("配置服务不可用", 500)

            current_categories = list(self.plugin.plugin_config.categories or [])
            if category_key not in current_categories:
                return self._err("分类不存在", 404)

            if len(current_categories) <= 1:
                return self._err("至少需要保留1个分类", 400)

            updated_categories = [c for c in current_categories if c != category_key]

            index_copy = self.plugin.cache_service.get_index_cache()

            deleted_missing_count = 0
            deleted_file_count = 0

            for path_str, meta in list(index_copy.items()):
                if not isinstance(meta, dict):
                    continue
                if meta.get("category") != category_key:
                    continue

                old_path = Path(path_str)
                if not old_path.exists():
                    del index_copy[path_str]
                    deleted_missing_count += 1
                    continue

                try:
                    await self.plugin._safe_remove_file(str(old_path))
                    deleted_file_count += 1

                    if "hash" in meta and hasattr(
                        self.plugin, "image_processor_service"
                    ):
                        try:
                            self.plugin.image_processor_service.invalidate_cache(
                                meta["hash"]
                            )
                        except Exception:
                            pass

                    del index_copy[path_str]
                except Exception as e:
                    logger.warning(f"删除分类文件失败: {old_path}, 错误: {e}")

            # 在 updater 内原子删除属于该分类的条目，避免竞态
            deleted_in_updater: list[str] = []

            def delete_category_updater(current: dict):
                for path_str in list(current.keys()):
                    meta = current[path_str]
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("category") != category_key:
                        continue
                    deleted_in_updater.append(path_str)
                    del current[path_str]

            await self.plugin.cache_service.update_index(delete_category_updater)

            if self.data_dir:
                category_dir = Path(self.data_dir) / "categories" / category_key
                try:
                    base_categories_dir = (Path(self.data_dir) / "categories").resolve()
                    category_dir = (base_categories_dir / category_key).resolve()

                    if (
                        category_dir.parent == base_categories_dir
                        and category_dir.exists()
                        and category_dir.is_dir()
                    ):
                        try:
                            orphan_files = [
                                p for p in category_dir.rglob("*") if p.is_file()
                            ]
                            deleted_file_count += len(orphan_files)
                        except Exception:
                            pass

                        await asyncio.to_thread(shutil.rmtree, category_dir, True)
                except Exception as e:
                    logger.warning(f"删除空分类目录失败: {category_dir}, 错误: {e}")

            if category_key in getattr(self.plugin.plugin_config, "category_info", {}):
                del self.plugin.plugin_config.category_info[category_key]
                self.plugin.plugin_config.save_category_info()

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": updated_categories})
            else:
                self.plugin.plugin_config.categories = updated_categories
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = updated_categories

            return self._ok(
                {
                    "deleted": category_key,
                    "categories": updated_categories,
                    "deleted_files": deleted_file_count,
                    "pruned_missing": deleted_missing_count,
                }
            )
        except Exception as e:
            logger.error(f"删除分类失败: {e}", exc_info=True)
            return self._err(str(e))

    async def handle_get_emotions(self, request):
        try:
            category_info = self.plugin.plugin_config.get_category_info()
            return self._ok({"emotions": category_info})
        except Exception as e:
            logger.error(f"获取情绪分类失败: {e}")
            return self._err(str(e))
