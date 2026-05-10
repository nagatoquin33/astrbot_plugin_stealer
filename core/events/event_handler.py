import asyncio
import os
import random
import threading
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image, Plain

from .platform_detector import PlatformDetector
from .image_download_service import ImageDownloadService


class EventHandler:
    """事件处理服务类，负责处理所有与插件相关的事件操作。"""

    HTTP_TIMEOUT_SECONDS = 30
    HTTP_CONNECTOR_LIMIT = 10
    HTTP_CONNECTOR_LIMIT_PER_HOST = 5
    HTTP_DNS_CACHE_SECONDS = 300

    def __init__(self, plugin_instance: Any):
        """初始化事件处理服务。

        Args:
            plugin_instance: Main 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self._cleaned = False  # 清理标志位

        # 图片处理节流相关
        self._last_process_time: float = 0.0  # 上次处理时间（用于interval和cooldown模式）

        # 强制捕获窗口（需要锁保护）
        self._force_capture_windows: dict[str, dict[str, object]] = {}
        self._force_capture_lock = threading.RLock()  # 可重入锁，保护并发访问

        # 子服务
        self._platform_detector = PlatformDetector(plugin_instance)
        self._image_download_service = ImageDownloadService(plugin_instance)

    # ===== 门面委托：子服务方法 =====

    def _normalize_str(self, value: object) -> str:
        """规范化字符串值，委托给 PlatformDetector。"""
        return self._platform_detector._normalize_str(value)

    def _get_event_platform_name(self, event: AstrMessageEvent | None = None) -> str:
        """获取事件平台名（小写），失败时返回空字符串。"""
        return self._platform_detector.get_platform_name(event)

    def _is_telegram_event(self, event: AstrMessageEvent | None = None) -> bool:
        """判断事件是否来自 Telegram 平台。"""
        return self._platform_detector.is_telegram_event(event)

    def _check_platform_emoji_metadata(self, *args, **kwargs) -> bool:
        """检查图片元信息，判断是否为平台标记的表情包。"""
        return self._platform_detector.check_platform_emoji_metadata(*args, **kwargs)

    def _extract_store_emoji_urls(self, event: AstrMessageEvent) -> list[str]:
        """从 OneBot raw_message 里提取 QQ 商城表情的可下载 URL。"""
        return self._platform_detector.extract_store_emoji_urls(event)

    async def _get_aiohttp_session(self) -> None:
        """获取或创建共享的 aiohttp session（已迁移到 ImageDownloadService）。"""
        return await self._image_download_service.get_session()

    async def close_aiohttp_session(self) -> None:
        """关闭共享的 aiohttp session（已迁移到 ImageDownloadService）。"""
        await self._image_download_service.close()

    def _build_download_headers(self) -> dict[str, str]:
        """构建下载请求头（已迁移到 ImageDownloadService）。"""
        return self._image_download_service.build_headers()

    @staticmethod
    def _detect_download_file_type(content_type: str, content: bytes) -> tuple[str, bool]:
        """检测下载文件类型（已迁移到 ImageDownloadService）。"""
        return ImageDownloadService.detect_file_type(content_type, content)

    async def _download_to_temp(
        self, url: str, *, log_download: bool = False
    ) -> tuple[str | None, bool]:
        """从 URL 下载文件到临时文件（已迁移到 ImageDownloadService）。"""
        return await self._image_download_service.download_to_temp(url, log_download=log_download)

    async def _download_original_image(self, img: Image) -> tuple[str | None, bool]:
        """下载原始图片文件（已迁移到 ImageDownloadService）。"""
        return await self._image_download_service.download_original_image(img)

    async def _download_url_to_temp(self, url: str) -> tuple[str | None, bool]:
        """从 URL 下载文件到临时文件（已迁移到 ImageDownloadService）。"""
        return await self._image_download_service.download_url_to_temp(url)

    # ===== EventHandler 核心逻辑 =====
    def _should_process_image(self) -> bool:
        """根据偷图模式（概率/冷却）判断是否应该处理图片。

        - probability 模式：每次收到图片按 steal_chance 概率决定
        - cooldown 模式：两次偷取之间至少间隔 image_processing_cooldown 秒
        """
        steal_mode = self.plugin.steal_mode

        if steal_mode == "cooldown":
            return self._check_cooldown()
        else:
            return self._check_probability()

    def _check_cooldown(self) -> bool:
        """冷却模式：两次处理之间至少间隔 N 秒。"""
        cooldown = self.plugin.image_processing_cooldown
        try:
            cooldown = int(cooldown)
        except Exception:
            cooldown = 10

        current_time = time.time()
        time_since_last = current_time - self._last_process_time

        if time_since_last < cooldown:
            logger.debug(f"冷却检查：跳过（冷却={cooldown}秒，距上次={time_since_last:.1f}秒）")
            return False

        self._last_process_time = current_time
        logger.debug(f"冷却检查：通过（冷却={cooldown}秒，距上次={time_since_last:.1f}秒）")
        return True

    def _check_probability(self) -> bool:
        """概率模式：每次按 steal_chance 概率决定是否偷取。"""
        steal_chance = self.plugin.steal_chance
        try:
            steal_chance = float(steal_chance)
        except Exception:
            steal_chance = 0.6

        if steal_chance <= 0:
            logger.debug("偷图概率为0，跳过偷取")
            return False
        if steal_chance >= 1.0:
            logger.debug("偷图概率为1.0，直接通过")
            return True
        if random.random() >= steal_chance:
            logger.debug(f"概率检查：未通过（概率={steal_chance}）")
            return False

        logger.debug(f"概率检查：通过（概率={steal_chance}）")
        return True

    def _select_items_for_removal(self, image_index: dict) -> list[tuple[str, int]]:
        """从索引中选出需要移除的条目（按创建时间从旧到新排序后取最旧的）。

        Returns:
            需要移除的 (file_path, created_at) 列表；若无需移除则返回空列表。
        """
        try:
            max_reg = int(self.plugin.max_reg_num)
        except (TypeError, ValueError):
            max_reg = 500  # 默认值

        if max_reg <= 0:
            logger.warning(f"容量控制上限无效: max_reg_num={max_reg}，跳过")
            return []

        if len(image_index) <= max_reg:
            return []

        image_items: list[tuple[str, int]] = []
        for file_path, image_info in image_index.items():
            created_at = int(image_info.get("created_at", 0)) if isinstance(image_info, dict) else 0
            image_items.append((file_path, created_at))

        if not image_items:
            return []

        image_items.sort(key=lambda x: x[1])
        remove_count = min(len(image_items) - max_reg, len(image_items))
        return image_items[:remove_count]

    async def _enforce_capacity(self, image_index: dict) -> list[str]:
        """容量控制，删除索引条目并返回需要删除的文件路径。

        Args:
            image_index: 索引字典

        Returns:
            list[str]: 需要删除的文件路径列表
        """
        files_to_delete = []

        try:
            items_to_remove = self._select_items_for_removal(image_index)
            if not items_to_remove:
                return files_to_delete

            logger.info(f"[容量控制-索引] 将删除 {len(items_to_remove)} 个最旧条目")

            for remove_path, _ in items_to_remove:
                if remove_path in image_index:
                    files_to_delete.append(remove_path)

                    if isinstance(image_index[remove_path], dict):
                        category = image_index[remove_path].get("category", "")
                        if category and self.plugin.base_dir:
                            file_name = os.path.basename(remove_path)
                            category_file_path = os.path.join(
                                self.plugin.base_dir, "categories", category, file_name
                            )
                            files_to_delete.append(category_file_path)

                    del image_index[remove_path]

        except Exception as e:
            logger.error(f"同步容量控制失败: {e}")

        return files_to_delete

    def _get_force_capture_key(self, event) -> str:
        """获取强制捕获的唯一键。

        Args:
            event: 消息事件对象

        Returns:
            str: 唯一键
        """
        if hasattr(event, "get_session_id"):
            try:
                session_id = event.get_session_id()
                if session_id:
                    return str(session_id)
            except Exception:
                pass

        if hasattr(event, "unified_msg_origin"):
            try:
                return str(event.unified_msg_origin)
            except Exception:
                pass

        return "global"

    def _cleanup_expired_capture_windows(self) -> int:
        """清理所有过期的强制捕获窗口。

        Returns:
            int: 清理的过期窗口数量
        """
        now = time.time()
        expired_keys = [
            key
            for key, entry in self._force_capture_windows.items()
            if isinstance(entry, dict) and float(entry.get("until", 0)) < now
        ]
        for key in expired_keys:
            self._force_capture_windows.pop(key, None)
        return len(expired_keys)

    def _get_force_capture_sender_id(self, event) -> str | None:
        """获取发送者ID。

        Args:
            event: 消息事件对象

        Returns:
            str | None: 发送者ID
        """
        # 优先使用框架官方 API
        if hasattr(event, "get_sender_id"):
            try:
                sid = event.get_sender_id()
                if sid:
                    return str(sid)
            except Exception:
                pass

        # 兜底：防御性遍历属性
        for attr in ("sender_id", "user_id"):
            value = getattr(event, attr, None)
            if value:
                return str(value)

        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            sender = getattr(message_obj, "sender", None)
            if sender is not None:
                uid = getattr(sender, "user_id", None)
                if uid:
                    return str(uid)

        return None

    def begin_force_capture(self, event, seconds: int) -> None:
        """开始强制捕获窗口。

        Args:
            event: 消息事件对象
            seconds: 捕获窗口持续时间（秒）
        """
        key = self._get_force_capture_key(event)
        sender_id = self._get_force_capture_sender_id(event)
        until = time.time() + max(1, int(seconds))
        with self._force_capture_lock:
            self._force_capture_windows[key] = {"until": until, "sender_id": sender_id}

    def get_force_capture_entry(self, event) -> dict[str, object] | None:
        """获取强制捕获条目。

        Args:
            event: 消息事件对象

        Returns:
            dict | None: 捕获条目，如果不存在或已过期则返回None
        """
        # 先清理所有过期的捕获窗口
        self._cleanup_expired_capture_windows()

        key = self._get_force_capture_key(event)
        with self._force_capture_lock:
            entry = self._force_capture_windows.get(key)
            if not entry:
                return None

            try:
                until = float(entry.get("until", 0))
            except Exception:
                self._force_capture_windows.pop(key, None)
                return None

            if time.time() > until:
                self._force_capture_windows.pop(key, None)
                return None

            expected_sender_id = entry.get("sender_id")
            if expected_sender_id:
                current_sender_id = self._get_force_capture_sender_id(event)
                if current_sender_id and str(current_sender_id) != str(expected_sender_id):
                    return None

            return entry

    def consume_force_capture(self, event) -> None:
        """消费强制捕获条目。

        Args:
            event: 消息事件对象
        """
        key = self._get_force_capture_key(event)
        with self._force_capture_lock:
            self._force_capture_windows.pop(key, None)

    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        if self._cleaned or self.plugin is None:
            return
        if not hasattr(event, "get_messages"):
            return
        plugin_instance = self.plugin
        force_entry = None
        try:
            force_entry = plugin_instance.get_force_capture_entry(event)
        except (AttributeError, KeyError):
            force_entry = None
        force_active = force_entry is not None
        try:
            if not force_active and not plugin_instance.is_steal_enabled_for_event(event):
                return
        except (AttributeError, TypeError, KeyError):
            return
        if not plugin_instance.steal_emoji and not force_active:
            return
        imgs: list[Image] = [comp for comp in event.get_messages() if isinstance(comp, Image)]
        store_urls = self._extract_store_emoji_urls(event)
        if not imgs and not store_urls:
            return
        if force_active:
            await self._handle_force_capture(event, plugin_instance, imgs, store_urls)
            return
        if not self._should_process_image():
            return
        logger.debug(f"开始处理 {len(imgs)} 个表情")
        raw_image_segments: list[dict] = []
        raw_image_file_map: dict[str, dict] = {}
        try:
            raw_event = getattr(getattr(event, "message_obj", None), "raw_message", None)
            logger.debug(f"raw_event type: {type(raw_event).__name__}, is_dict: {isinstance(raw_event, dict)}")
            # raw_event 可能是 dict（aiocqhttp）或对象，需要兼容两种类型
            raw_message = None
            if isinstance(raw_event, dict):
                raw_message = raw_event.get("message")
            elif raw_event is not None:
                raw_message = getattr(raw_event, "message", None)
            logger.debug(f"raw_message type: {type(raw_message).__name__ if raw_message else 'None'}")
            if isinstance(raw_message, list):
                raw_image_segments = [
                    seg
                    for seg in raw_message
                    if isinstance(seg, dict) and seg.get("type") == "image"
                ]
                for seg in raw_image_segments:
                    data = seg.get("data", {}) or {}
                    if not isinstance(data, dict):
                        continue
                    seg_file = self._normalize_str(data.get("file", ""))
                    if seg_file and seg_file not in raw_image_file_map:
                        raw_image_file_map[seg_file] = data
                logger.debug(f"提取到 {len(raw_image_segments)} 个原始图片段, {len(raw_image_file_map)} 个文件映射")
        except Exception as e:
            logger.debug(f"提取原始图片段失败: {e}")
            raw_image_segments = []
            raw_image_file_map = {}
        origin_target_str = ""
        try:
            cfg = getattr(plugin_instance, "plugin_config", None)
            if cfg:
                scope, target_id = cfg.get_event_target(event)
                if scope and target_id:
                    origin_target_str = f"{scope}:{target_id}"
        except (AttributeError, KeyError) as e:
            logger.debug(f"提取来源群信息失败: {e}")
        merged_idx: dict[str, Any] = {}

        def merge_result_idx(result_idx: object) -> None:
            if isinstance(result_idx, dict):
                merged_idx.update(result_idx)

        imgs_to_process: list[tuple[int, Image, dict]] = []
        for i, img in enumerate(imgs):
            try:
                is_platform_emoji = self._check_platform_emoji_metadata(
                    img,
                    event,
                    img_index=i,
                    image_segments=raw_image_segments,
                    image_file_map=raw_image_file_map,
                )
                if not is_platform_emoji:
                    sub_type_value = getattr(img, "subType", "unknown")
                    logger.debug(f"跳过非表情包图片 (subType={sub_type_value})")
                    continue
                logger.info("检测到表情包，准备偷走它！")
                extra_meta = None
                try:
                    seg = raw_image_segments[i] if 0 <= i < len(raw_image_segments) else None
                    data = seg.get("data", {}) if isinstance(seg, dict) else {}
                    if isinstance(data, dict) and (
                        data.get("emoji_id") or data.get("emoji_package_id")
                    ):
                        extra_meta = {
                            "source": "qq_store",
                            "qq_emoji_id": str(data.get("emoji_id") or ""),
                            "qq_emoji_package_id": str(data.get("emoji_package_id") or ""),
                            "origin_url": self._normalize_str(data.get("url", "")),
                            "qq_key": self._normalize_str(data.get("key", "")),
                        }
                except Exception:
                    extra_meta = None
                if origin_target_str:
                    if extra_meta is None:
                        extra_meta = {}
                    extra_meta["origin_target"] = origin_target_str
                imgs_to_process.append((i, img, extra_meta or {}))
            except Exception as e:
                logger.error(f"收集图片信息失败: {e}")
        if imgs_to_process:
            logger.debug(f"开始并行下载 {len(imgs_to_process)} 张图片")
            download_results = await asyncio.gather(
                *[self._download_original_image(item[1]) for item in imgs_to_process],
                return_exceptions=True,
            )
            process_tasks = []
            for (i, img, extra_meta), result in zip(imgs_to_process, download_results):
                if isinstance(result, Exception):
                    logger.error(f"下载图片异常: {result}")
                    continue
                if isinstance(result, tuple):
                    temp_path, _is_gif = result
                else:
                    temp_path = result
                if not temp_path or not os.path.exists(str(temp_path)):
                    logger.warning(f"临时文件不存在: {temp_path}")
                    continue
                process_tasks.append(
                    plugin_instance._process_image(
                        event,
                        temp_path,
                        is_temp=True,
                        is_platform_emoji=True,
                        extra_meta=extra_meta,
                    )
                )
            if process_tasks:
                logger.debug(f"开始并行处理 {len(process_tasks)} 张图片")
                process_results = await asyncio.gather(*process_tasks, return_exceptions=True)
                for result in process_results:
                    if isinstance(result, Exception):
                        logger.error(f"处理图片异常: {result}")
                        continue
                    if not isinstance(result, tuple) or len(result) != 2:
                        continue
                    success, idx = result
                    if success and isinstance(idx, dict):
                        merge_result_idx(idx)
        if store_urls:
            logger.debug(f"开始并行下载 {min(len(store_urls), 3)} 个商城表情")

            async def download_store_url(url: str) -> tuple[str | None, str]:
                try:
                    temp_path, _ = await self._download_url_to_temp(url)
                    return temp_path, url
                except Exception as e:
                    logger.error(f"下载商城表情失败: {e}")
                    return None, url

            store_download_results = await asyncio.gather(
                *[download_store_url(url) for url in store_urls[:3]], return_exceptions=True
            )
            store_process_tasks = []
            for result in store_download_results:
                if isinstance(result, Exception):
                    continue
                temp_path, url = result
                if not temp_path or not os.path.exists(temp_path):
                    continue
                extra_meta = {"source": "qq_store", "origin_url": self._normalize_str(url)}
                if origin_target_str:
                    extra_meta["origin_target"] = origin_target_str
                store_process_tasks.append(
                    plugin_instance._process_image(
                        event,
                        temp_path,
                        is_temp=True,
                        is_platform_emoji=True,
                        extra_meta=extra_meta,
                    )
                )
            if store_process_tasks:
                store_results = await asyncio.gather(*store_process_tasks, return_exceptions=True)
                for result in store_results:
                    if isinstance(result, Exception):
                        continue
                    if isinstance(result, tuple) and len(result) == 2:
                        success, idx = result
                        if success and isinstance(idx, dict):
                            merge_result_idx(idx)
        if merged_idx:
            await plugin_instance._save_index(merged_idx)

    async def _handle_force_capture(
        self,
        event: AstrMessageEvent,
        plugin_instance,
        imgs: list,
        store_urls: list[str],
    ) -> None:
        """处理强制捕获模式：下载并处理单张图片后直接返回。"""
        try:
            temp_path: str | None = None
            is_gif = False

            if imgs:
                img = imgs[0]
                result = await self._download_original_image(img)
                if isinstance(result, tuple):
                    temp_path, is_gif = result
                else:
                    temp_path = result
                if not temp_path:
                    temp_path = await img.convert_to_file_path()
            elif store_urls:
                result = await self._download_url_to_temp(store_urls[0])
                if isinstance(result, tuple):
                    temp_path, is_gif = result
                else:
                    temp_path = result

            if not temp_path or not os.path.exists(str(temp_path)):
                await event.send(MessageChain([Plain(text="❌ 收录失败：图片临时文件不存在")]))
            else:
                success, idx = await plugin_instance._process_image(
                    event, temp_path, is_temp=True, is_platform_emoji=True
                )
                if success and isinstance(idx, dict):
                    await plugin_instance._save_index(idx)
                    await event.send(MessageChain([Plain(text="✅ 已收录并自动分类入库")]))
                else:
                    await event.send(
                        MessageChain(
                            [
                                Plain(
                                    text="❌ 未收录（可能被判定为非表情包/审核不通过/重复或处理失败）"
                                )
                            ]
                        )
                    )
        except Exception as e:
            await event.send(MessageChain([Plain(text=f"❌ 收录失败：{e}")]))
        finally:
            try:
                plugin_instance.consume_force_capture(event)
            except Exception:
                pass

    async def _clean_raw_directory(self) -> int:
        """清理 raw 目录中的所有临时文件。"""
        deleted = 0
        try:
            raw_dir = getattr(self.plugin, "raw_dir", None)
            if raw_dir and raw_dir.exists():
                for f in raw_dir.iterdir():
                    try:
                        if f.is_file():
                            f.unlink()
                            deleted += 1
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"清理 raw 目录失败: {e}")
        if deleted:
            logger.debug(f"raw 目录清理: 删除了 {deleted} 个文件")
        return deleted

    async def cleanup_async(self) -> None:
        """异步清理资源。"""
        await self.close_aiohttp_session()

    def cleanup(self):
        """清理资源。"""
        if self._cleaned:
            return
        self._cleaned = True
        # 清理强制捕获窗口
        if hasattr(self, "_force_capture_windows"):
            self._force_capture_windows.clear()
        # 清理插件引用
        self.plugin = None
        logger.debug("EventHandler 资源已清理")
