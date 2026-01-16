import logging
import os
from pathlib import Path

from aiohttp import web

logger = logging.getLogger("astrbot")


class WebServer:
    def __init__(self, plugin, host: str = "0.0.0.0", port: int = 8899):
        self.plugin = plugin
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self._started = False

        # Web UI 静态文件目录 (插件目录下的 web 文件夹)
        self.static_dir = Path(__file__).parent / "web"

        # 图片数据目录 (数据目录下的 plugin_data/...)
        self.data_dir = self.plugin.base_dir

        self._setup_routes()

    def _setup_routes(self):
        # API 路由
        self.app.router.add_get("/api/images", self.handle_list_images)
        self.app.router.add_delete("/api/images/{hash}", self.handle_delete_image)
        self.app.router.add_get("/api/stats", self.handle_get_stats)
        self.app.router.add_get("/api/config", self.handle_get_config)
        self.app.router.add_get("/api/health", self.handle_health_check)  # 健康检查

        # 静态文件路由
        # 1. 前端页面 - 首页
        self.app.router.add_get("/", self.handle_index)
        # 某些客户端/代理在遇到 FileResponse 异常时会报 "HTTP/0.9"，提供显式入口便于排障
        self.app.router.add_get("/index.html", self.handle_index)

        # 2. 静态资源
        # 插件 web/index.html 如果引用了本地资源（js/css/img），这里提供静态托管。
        # 兼容直接打包在 web/ 目录下的资源结构。
        if self.static_dir.exists():
            self.app.router.add_static("/web", self.static_dir, show_index=False)

        # 3. 图片资源 (映射到实际存储路径)
        # 这里用插件 base_dir 作为根目录，URL 通过 /images/<relative_path> 访问。
        # 仅在目录存在时启用。
        if self.data_dir and Path(self.data_dir).exists():
            logger.debug(f"Mounting static images dir: {self.data_dir}")
            self.app.router.add_static("/images", str(self.data_dir), show_index=False)
        else:
            logger.warning(f"Images data dir not found: {self.data_dir}")

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
                logger.info(f"Emoji Manager WebUI started - listening on all interfaces (0.0.0.0:{self.port})")
                logger.info(f"  → Local access: {protocol}://127.0.0.1:{self.port}")
            else:
                logger.info(f"Emoji Manager WebUI started at {protocol}://{self.host}:{self.port}")

            return True

        except OSError as e:
            if "Address already in use" in str(e) or e.errno == 98 or e.errno == 10048:
                logger.error(f"WebUI 端口 {self.port} 已被占用，请更换端口或关闭占用该端口的程序")
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
                    status=404
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
            include_meta = request.query.get("meta", "0") in ("1", "true", "yes")

            # 获取所有图片数据
            # 优先从 ImageProcessorService 的缓存或索引中获取
            # index 结构: path -> {hash, category, tags, desc, ...}
            # 但我们需要列表形式

            # 从 cache_service 获取持久化索引
            index = self.plugin.cache_service.get_cache("index_cache") or {}

            images = []
            for path_str, meta in index.items():
                # 转换路径为 web 可访问的 URL
                # 假设 path_str 是绝对路径，我们需要将其转换为相对于 data_dir 的路径
                try:
                    abs_path = Path(path_str)
                    rel_path = abs_path.relative_to(self.data_dir)
                    # aiohttp的add_static会自动处理URL编码/解码，我们只需要提供正常的路径
                    # 使用正斜杠作为路径分隔符（Web标准）
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
                        "created_at": meta.get("created_at", 0)
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

                    # 过滤逻辑
                    if category_filter and item["category"] != category_filter:
                        continue

                    if search_query:
                        # 简单的搜索匹配
                        in_tags = any(search_query in t.lower() for t in item["tags"])
                        in_desc = search_query in item["desc"].lower()
                        if not (in_tags or in_desc):
                            continue

                    images.append(item)
                except ValueError:
                    # 路径不在 data_dir 下，可能是旧数据或异常
                    continue

            # 排序（按时间倒序）
            images.sort(key=lambda x: x["created_at"], reverse=True)

            # 分页
            total = len(images)
            start = (page - 1) * page_size
            end = start + page_size
            paged_images = images[start:end]

            return web.json_response({
                "total": total,
                "page": page,
                "size": page_size,
                "images": paged_images
            })
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_delete_image(self, request):
        """删除图片"""
        image_hash = request.match_info["hash"]
        # 实现删除逻辑 -> 调用 plugin 的安全删除方法
        # 需要反向查找 hash 对应的文件路径
        index = self.plugin.cache_service.get_cache("index_cache") or {}
        target_path = None

        for path_str, meta in index.items():
            if meta.get("hash") == image_hash:
                target_path = path_str
                break

        if target_path:
            # 调用插件的删除逻辑
            # 注意：这需要 main.py 中暴露相关方法，或者我们自己实现
            try:
                # 1. 删除文件
                if os.path.exists(target_path):
                    os.remove(target_path)

                # 2. 更新索引
                if target_path in index:
                    del index[target_path]
                    # Fix: Use set_cache (sync) instead of save_cache (non-existent/awaited)
                    self.plugin.cache_service.set_cache("index_cache", index, persist=True)

                # 3. 清理空目录 (可选)

                return web.json_response({"success": True})
            except Exception as e:
                logger.error(f"Failed to delete image {target_path}: {e}")
                return web.json_response({"error": str(e)}, status=500)

        return web.json_response({"error": "Image not found"}, status=404)

    async def handle_get_stats(self, request):
        index = self.plugin.cache_service.get_cache("index_cache") or {}
        categories = {}
        for meta in index.values():
            cat = meta.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        return web.json_response({
            "total_images": len(index),
            "categories": categories
        })

    async def handle_get_config(self, request):
        return web.json_response({
            "version": "1.0.0",
            "plugin_version": "0.1.0"
        })

    async def handle_health_check(self, request):
        """健康检查端点 - 用于反向代理和监控"""
        return web.json_response({
            "status": "ok",
            "service": "emoji-manager-webui"
        })
