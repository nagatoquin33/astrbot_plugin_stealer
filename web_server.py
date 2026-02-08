import hashlib
import hmac
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

from aiohttp import web

logger = logging.getLogger("astrbot")


class WebServer:
    def __init__(self, plugin, host: str = "0.0.0.0", port: int = 8899):
        self.plugin = plugin
        self.host = host
        self.port = port
        self.app = web.Application(
            client_max_size=50 * 1024 * 1024,
            middlewares=[self._error_middleware, self._auth_middleware],
        )
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self._started = False

        # Web UI 静态文件目录 (插件目录下的 web 文件夹)
        self.static_dir = Path(__file__).parent / "web"

        # 图片数据目录 (数据目录下的 plugin_data/...)
        self.data_dir = self.plugin.base_dir

        self._cookie_name = "stealer_webui_session"
        self._sessions: dict[str, float] = {}

        self._setup_routes()

    async def _error_middleware(self, app: web.Application, handler):
        async def middleware_handler(request: web.Request):
            try:
                return await handler(request)
            except web.HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unhandled WebUI error: {e}", exc_info=True)
                if (request.path or "").startswith("/api/"):
                    return web.json_response(
                        {"success": False, "error": "Internal Server Error"},
                        status=500,
                    )
                return web.Response(text="500 Internal Server Error", status=500)

        return middleware_handler

    def _is_auth_enabled(self) -> bool:
        try:
            enabled = getattr(self.plugin, "webui_auth_enabled", None)
            if enabled is not None:
                return bool(enabled)
        except Exception:
            pass
        try:
            cfg = getattr(self.plugin, "config_service", None)
            enabled = getattr(cfg, "webui_auth_enabled", None)
            if enabled is not None:
                return bool(enabled)
        except Exception:
            pass
        return True

    def _get_expected_secret(self) -> str:
        if not self._is_auth_enabled():
            return ""

        password = ""
        try:
            password = str(getattr(self.plugin, "webui_password", "") or "").strip()
        except Exception:
            password = ""
        if password:
            return password
        try:
            password = str(
                getattr(getattr(self.plugin, "config_service", None), "webui_password", "")
                or ""
            ).strip()
        except Exception:
            password = ""
        if password:
            return password
        return ""

    def _get_session_timeout(self) -> int:
        timeout = 3600
        try:
            timeout = int(getattr(self.plugin, "webui_session_timeout", 3600) or 3600)
        except Exception:
            timeout = 3600
        if timeout == 3600:
            try:
                timeout = int(
                    getattr(getattr(self.plugin, "config_service", None), "webui_session_timeout", 3600)
                    or 3600
                )
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
            exp = self._sessions.get(sid)
            if not exp or exp < now:
                if sid:
                    self._sessions.pop(sid, None)
                if path.startswith("/api/"):
                    return web.json_response(
                        {"success": False, "error": "Unauthorized"},
                        status=401,
                    )
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

    async def handle_web_static(self, request: web.Request) -> web.StreamResponse:
        raw = str(request.match_info.get("path", "") or "")
        raw = raw.lstrip("/")
        if not raw:
            raise web.HTTPNotFound()
        if ".." in raw or raw.startswith(("/", "\\")) or ":" in raw:
            raise web.HTTPBadRequest(text="invalid path")

        base_dir = Path(self.static_dir).resolve()
        try:
            abs_path = (base_dir / Path(raw)).resolve()
            abs_path.relative_to(base_dir)
        except Exception:
            raise web.HTTPNotFound()

        if not abs_path.exists() or not abs_path.is_file():
            raise web.HTTPNotFound()

        return web.FileResponse(path=abs_path)

    async def handle_images_static(self, request: web.Request) -> web.StreamResponse:
        raw = str(request.match_info.get("path", "") or "")
        raw = raw.lstrip("/")
        if not raw:
            raise web.HTTPNotFound()
        if ".." in raw or raw.startswith(("/", "\\")) or ":" in raw:
            raise web.HTTPBadRequest(text="invalid path")

        base_dir = Path(self.data_dir).resolve()
        try:
            abs_path = (base_dir / Path(raw)).resolve()
            abs_path.relative_to(base_dir)
        except Exception:
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
                content = index_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # 兼容被意外写入非 UTF-8 的情况（尽量仍返回合法 HTTP 响应）
                content = index_file.read_text(encoding="utf-8", errors="replace")
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
            index = self.plugin.cache_service.get_cache("index_cache") or {}

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
                            stat = abs_path.stat()
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
            if hasattr(self.plugin, "config_service"):
                # 获取配置中的分类列表
                base_categories = self.plugin.config_service.get_category_info()
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

            return web.json_response(
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
            return web.json_response({"error": str(e)}, status=500)

    async def handle_delete_image(self, request):
        """删除图片"""
        image_hash = request.match_info["hash"]
        add_to_blacklist = request.query.get("blacklist", "false").lower() == "true"

        target = {"path": None}
        await self.plugin.cache_service.update_index(
            lambda current: self._remove_by_hash(current, image_hash, target)
        )
        target_path = target["path"]

        if target_path:
            try:
                # 1. 删除文件
                if os.path.exists(target_path):
                    os.remove(target_path)

                # 2. 添加到黑名单（如果请求）
                if add_to_blacklist:
                    # 使用当前时间戳作为值
                    import time

                    self.plugin.cache_service.set(
                        "blacklist_cache", image_hash, int(time.time()), persist=True
                    )
                    logger.info(f"Added image {image_hash} to blacklist")

                # 失效缓存
                if hasattr(self.plugin, "image_processor_service"):
                    self.plugin.image_processor_service.invalidate_cache(image_hash)

                return web.json_response({"success": True})
            except Exception as e:
                logger.error(f"Failed to delete image {target_path}: {e}")
                return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"error": "Image not found"}, status=404)

    def _remove_by_hash(self, current: dict, image_hash: str, target: dict):
        for path_str, meta in list(current.items()):
            if isinstance(meta, dict) and meta.get("hash") == image_hash:
                target["path"] = path_str
                del current[path_str]
                return

    async def handle_update_image(self, request):
        """更新图片信息 (Category, Tags, Desc)"""
        try:
            image_hash = request.match_info["hash"]
            data = await request.json()

            new_category = data.get("category")
            new_tags = data.get("tags")  # list or comma separated string
            new_desc = data.get("desc")

            updated = {"ok": False, "error": ""}

            def updater(current: dict):
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
                        meta["tags"] = [t.strip() for t in new_tags.split(",") if t.strip()]
                    else:
                        meta["tags"] = new_tags

                if new_desc is not None:
                    meta["desc"] = new_desc

                if new_category and new_category != meta.get("category"):
                    old_path = Path(target_path)
                    if not old_path.exists():
                        updated["error"] = "Source file not found"
                        return

                    new_cat_dir = self.data_dir / "categories" / new_category
                    new_cat_dir.mkdir(parents=True, exist_ok=True)
                    new_path = new_cat_dir / old_path.name
                    shutil.move(str(old_path), str(new_path))

                    del current[target_path]
                    meta["path"] = str(new_path)
                    meta["category"] = new_category
                    current[str(new_path)] = meta
                else:
                    current[target_path] = meta

                updated["ok"] = True

            await self.plugin.cache_service.update_index(updater)
            if not updated["ok"]:
                return web.json_response({"error": updated["error"] or "Update failed"}, status=404)
            return web.json_response({"success": True})

        except Exception as e:
            logger.error(f"Failed to update image: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_batch_delete(self, request):
        """批量删除"""
        try:
            data = await request.json()
            hashes = data.get("hashes", [])
            if not hashes:
                return web.json_response({"success": True, "count": 0})

            index_copy = dict(self.plugin.cache_service.get_cache("index_cache") or {})
            deleted_count = 0

            paths_to_delete = []
            hash_set = set(hashes)

            for path_str, meta in index_copy.items():
                if meta.get("hash") in hash_set:
                    paths_to_delete.append(path_str)

            for path_str in paths_to_delete:
                try:
                    if os.path.exists(path_str):
                        os.remove(path_str)
                    if path_str in index_copy:
                        del index_copy[path_str]
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {path_str}: {e}")

            if deleted_count > 0:
                await self.plugin.cache_service.update_index(
                    lambda current: (current.clear(), current.update(index_copy))
                )

            return web.json_response({"success": True, "count": deleted_count})
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_batch_move(self, request):
        """批量移动"""
        try:
            data = await request.json()
            hashes = data.get("hashes", [])
            target_category = data.get("category")

            if not hashes or not target_category:
                return web.json_response(
                    {"error": "Missing hashes or category"}, status=400
                )

            index_copy = dict(self.plugin.cache_service.get_cache("index_cache") or {})
            moved_count = 0

            target_dir = self.data_dir / "categories" / target_category
            target_dir.mkdir(parents=True, exist_ok=True)

            items_to_move = []
            hash_set = set(hashes)

            for path_str, meta in index_copy.items():
                if meta.get("hash") in hash_set:
                    if meta.get("category") != target_category:
                        items_to_move.append((path_str, meta))

            for old_path_str, meta in items_to_move:
                try:
                    old_path = Path(old_path_str)
                    if not old_path.exists():
                        continue

                    new_path = target_dir / old_path.name

                    shutil.move(str(old_path), str(new_path))

                    del index_copy[old_path_str]
                    meta["path"] = str(new_path)
                    meta["category"] = target_category
                    index_copy[str(new_path)] = meta

                    moved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to move {old_path_str}: {e}")

            if moved_count > 0:
                await self.plugin.cache_service.update_index(
                    lambda current: (current.clear(), current.update(index_copy))
                )

                # 失效缓存
                if hasattr(self.plugin, "image_processor_service"):
                    for h in hashes:
                        self.plugin.image_processor_service.invalidate_cache(h)

            return web.json_response({"success": True, "count": moved_count})
        except Exception as e:
            logger.error(f"Batch move failed: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_stats(self, request):
        try:
            index = self.plugin.cache_service.get_cache("index_cache") or {}

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
            if hasattr(self.plugin, "config_service"):
                categories_count = len(self.plugin.config_service.categories)

            # 前端期望的数据结构是 { stats: { total, categories, today } }
            return web.json_response(
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
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_config(self, request):
        return web.json_response({"version": "1.0.0", "plugin_version": "0.1.0"})

    async def handle_health_check(self, request):
        return web.json_response({"status": "ok", "service": "emoji-manager-webui"})

    async def handle_auth_info(self, request):
        expected = self._get_expected_secret()
        return web.json_response(
            {
                "requires_auth": bool(expected),
                "session_timeout": self._get_session_timeout(),
            }
        )

    async def handle_auth_login(self, request):
        expected = self._get_expected_secret()
        if not expected:
            return web.json_response({"success": True, "requires_auth": False})

        try:
            payload = await request.json()
        except Exception:
            payload = {}

        provided = str((payload or {}).get("password", "") or "").strip()
        if not provided or not hmac.compare_digest(provided, expected):
            return web.json_response({"success": False, "error": "Unauthorized"}, status=401)

        timeout = self._get_session_timeout()
        sid = uuid.uuid4().hex
        exp = time.time() + float(timeout)
        self._sessions[sid] = exp

        resp = web.json_response({"success": True, "expires_at": int(exp)})
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
        resp = web.json_response({"success": True})
        resp.del_cookie(self._cookie_name, path="/")
        return resp

    async def handle_upload_image(self, request):
        try:
            data = await request.post()

            if "file" not in data:
                return web.json_response({"error": "没有上传文件"}, status=400)

            uploaded_file = data["file"]
            if not uploaded_file.filename:
                return web.json_response({"error": "文件名无效"}, status=400)

            category = data.get("emotion", "").strip()
            if not category:
                return web.json_response({"error": "请选择情绪分类"}, status=400)
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
                return web.json_response(
                    {"error": f"不支持的文件类型，允许: {', '.join(allowed_exts)}"},
                    status=400,
                )

            file_content = uploaded_file.file.read()
            file_hash = hashlib.md5(file_content).hexdigest()

            timestamp = int(datetime.now().timestamp())
            unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"

            category_dir = Path(self.data_dir) / "categories" / category
            category_dir.mkdir(parents=True, exist_ok=True)

            file_path = category_dir / unique_filename

            with open(file_path, "wb") as f:
                f.write(file_content)

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

            return web.json_response(
                {
                    "success": True,
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
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_categories(self, request):
        try:
            index = dict(self.plugin.cache_service.get_cache("index_cache") or {})
            categories = {}
            for meta in index.values():
                if isinstance(meta, dict):
                    cat = meta.get("category", "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
            return web.json_response({"categories": categories})
        except Exception as e:
            logger.error(f"获取分类列表失败: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_update_categories(self, request):
        try:
            data = await request.json()
            new_categories_data = data.get("categories", [])

            if not isinstance(new_categories_data, list) or not new_categories_data:
                return web.json_response({"error": "分类列表无效"}, status=400)

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
                self.plugin.config_service.update_config_from_dict(
                    {"categories": category_keys}
                )
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = category_keys

            self.plugin.config_service.category_info.update(category_info)
            self.plugin.config_service.save_category_info()

            return web.json_response({"success": True, "categories": category_keys})
        except Exception as e:
            logger.error(f"更新分类列表失败: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_delete_category(self, request):
        try:
            category_key = (request.match_info.get("key") or "").strip()
            if not category_key:
                return web.json_response({"error": "分类Key无效"}, status=400)

            if (
                not hasattr(self.plugin, "config_service")
                or not self.plugin.config_service
            ):
                return web.json_response({"error": "配置服务不可用"}, status=500)

            current_categories = list(self.plugin.config_service.categories or [])
            if category_key not in current_categories:
                return web.json_response({"error": "分类不存在"}, status=404)

            if len(current_categories) <= 1:
                return web.json_response({"error": "至少需要保留1个分类"}, status=400)

            updated_categories = [c for c in current_categories if c != category_key]

            index_copy = dict(self.plugin.cache_service.get_cache("index_cache") or {})

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
                    old_path.unlink()
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

            await self.plugin.cache_service.update_index(
                lambda current: (current.clear(), current.update(index_copy))
            )

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

                        shutil.rmtree(category_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"删除空分类目录失败: {category_dir}, 错误: {e}")

            if category_key in getattr(self.plugin.config_service, "category_info", {}):
                del self.plugin.config_service.category_info[category_key]
                self.plugin.config_service.save_category_info()

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": updated_categories})
            else:
                self.plugin.config_service.update_config_from_dict(
                    {"categories": updated_categories}
                )
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = updated_categories

            return web.json_response(
                {
                    "success": True,
                    "deleted": category_key,
                    "categories": updated_categories,
                    "deleted_files": deleted_file_count,
                    "pruned_missing": deleted_missing_count,
                }
            )
        except Exception as e:
            logger.error(f"删除分类失败: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_get_emotions(self, request):
        try:
            category_info = self.plugin.config_service.get_category_info()
            return web.json_response({"emotions": category_info})
        except Exception as e:
            logger.error(f"获取情绪分类失败: {e}")
            return web.json_response({"error": str(e)}, status=500)
