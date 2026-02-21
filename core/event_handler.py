import asyncio
import os
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image, Plain


class EventHandler:
    """事件处理服务类，负责处理所有与插件相关的事件操作。"""

    def __init__(self, plugin_instance):
        """初始化事件处理服务。

        Args:
            plugin_instance: Main 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self._scanner_task: asyncio.Task | None = None

        # 图片处理节流相关
        self._last_process_time = 0  # 上次处理时间（用于interval和cooldown模式）
        self._process_count = 0  # 处理计数（用于interval模式）

        # 强制捕获窗口
        self._force_capture_windows: dict[str, dict[str, object]] = {}

    def _should_process_image(self) -> bool:
        """根据配置的节流模式判断是否应该处理图片。

        简化为单一冷却模式。
        """
        current_time = time.time()

        # 冷却模式：两次处理之间至少间隔N秒
        cooldown = getattr(self.plugin, "image_processing_cooldown", 10)
        try:
            cooldown = int(cooldown)
        except Exception:
            cooldown = 10

        time_since_last = current_time - self._last_process_time

        if time_since_last >= cooldown:
            self._last_process_time = current_time
            logger.debug(
                f"节流检查：通过（冷却={cooldown}秒，距上次={time_since_last:.1f}秒）"
            )
            return True
        else:
            logger.debug(
                f"节流检查：跳过（冷却={cooldown}秒，距上次={time_since_last:.1f}秒）"
            )
            return False

    def _check_platform_emoji_metadata(
        self,
        img: Image,
        event: AstrMessageEvent | None = None,
        img_index: int | None = None,
        image_segments: list[dict] | None = None,
        image_file_map: dict[str, dict] | None = None,
    ) -> bool:
        """检查图片元信息，判断是否为平台标记的表情包。

        支持的平台特征：
        - NapCat/OneBot: subType=1 或 sub_type=1 表示表情包
        - QQ: summary包含"表情"关键词

        Args:
            img: 图片组件
            event: 消息事件对象（可选），用于访问原始消息数据

        Returns:
            bool: 是否为平台标记的表情包
        """
        try:

            def normalize_str(value: object) -> str:
                if value is None:
                    return ""
                try:
                    s = str(value)
                except Exception:
                    return ""
                s = s.strip()
                if s.startswith("`") and s.endswith("`") and len(s) >= 2:
                    s = s[1:-1].strip()
                return s

            def is_emoji_summary(summary: object) -> bool:
                s = normalize_str(summary)
                if not s:
                    return False
                s_lower = s.lower()
                return "表情" in s or "emoji" in s_lower or "sticker" in s_lower

            def is_sub_type_emoji(sub_type: object) -> bool:
                if sub_type is None:
                    return False
                if sub_type == 1 or sub_type == "1":
                    return True
                try:
                    return int(sub_type) == 1
                except Exception:
                    return False

            # 方式0: 从原始事件中查找 sub_type (最可靠的方法)
            if (
                image_segments is None
                and event
                and hasattr(event, "message_obj")
                and hasattr(event.message_obj, "raw_message")
            ):
                raw_event = event.message_obj.raw_message
                if hasattr(raw_event, "message") and isinstance(
                    raw_event.message, list
                ):
                    image_segments = [
                        seg
                        for seg in raw_event.message
                        if isinstance(seg, dict) and seg.get("type") == "image"
                    ]

            if image_segments:
                matched_data: dict[str, object] | None = None

                if (
                    img_index is not None
                    and 0 <= img_index < len(image_segments)
                    and isinstance(image_segments[img_index], dict)
                ):
                    matched_data = image_segments[img_index].get("data", {}) or {}
                else:
                    img_file = normalize_str(getattr(img, "file", ""))
                    img_url = normalize_str(getattr(img, "url", ""))
                    img_file_unique = normalize_str(getattr(img, "file_unique", ""))

                    if image_file_map and img_file:
                        matched_data = image_file_map.get(img_file)
                        if matched_data is None and img_file_unique:
                            matched_data = image_file_map.get(img_file_unique)

                    if matched_data is None:
                        for seg in image_segments:
                            if not isinstance(seg, dict):
                                continue
                            data = seg.get("data", {}) or {}
                            if not isinstance(data, dict):
                                continue
                            seg_file = normalize_str(data.get("file", ""))
                            seg_url = normalize_str(data.get("url", ""))

                            if seg_file and (
                                seg_file == img_file
                                or (img_file_unique and seg_file == img_file_unique)
                                or (img_url and seg_file in img_url)
                                or (img_file and seg_file in img_file)
                            ):
                                matched_data = data
                                break

                            if seg_url and (
                                (img_url and seg_url == img_url)
                                or (img_file and seg_url in img_file)
                            ):
                                matched_data = data
                                break

                if matched_data is not None:
                    sub_type = matched_data.get("sub_type")
                    if is_sub_type_emoji(sub_type):
                        logger.debug(
                            f"检测到表情包标记: sub_type={sub_type} (从原始事件)"
                        )
                        return True

                    summary = matched_data.get("summary", "")
                    if is_emoji_summary(summary):
                        logger.debug(
                            f"检测到表情包标记: summary='{summary}' (从原始事件)"
                        )
                        return True

            # 方式1: 检查 Image 对象的 subType 字段
            if hasattr(img, "subType") and img.subType:
                if is_sub_type_emoji(img.subType):
                    logger.debug(f"检测到表情包标记: subType={img.subType}")
                    return True

            # 方式2: 检查 __dict__ 中的 sub_type
            if hasattr(img, "__dict__"):
                img_dict = img.__dict__
                sub_type_underscore = img_dict.get("sub_type")
                if is_sub_type_emoji(sub_type_underscore):
                    logger.debug(
                        f"检测到表情包标记: sub_type={sub_type_underscore} (从__dict__)"
                    )
                    return True

            # 方式3: 通过 toDict() 检查
            try:
                raw_data = img.toDict()
                if isinstance(raw_data, dict) and "data" in raw_data:
                    data = raw_data["data"]

                    sub_type = data.get("sub_type") or data.get("subType")
                    if is_sub_type_emoji(sub_type):
                        logger.debug(
                            f"检测到表情包标记: sub_type={sub_type} (从toDict)"
                        )
                        return True

                    summary = data.get("summary", "")
                    if is_emoji_summary(summary):
                        logger.debug(f"检测到表情包标记: summary='{summary}'")
                        return True

                    img_type = (
                        data.get("type")
                        or data.get("imageType")
                        or data.get("image_type")
                    )
                    if img_type in ["emoji", "sticker", "face", "meme"]:
                        logger.debug(f"检测到表情包标记: type='{img_type}'")
                        return True
            except Exception as e:
                logger.debug(f"无法获取图片字典数据: {e}")

            return False

        except Exception as e:
            logger.debug(f"检查平台表情包元信息失败: {e}")
            return False

    # 注意：这个方法不需要装饰器，因为在Main类中已经使用了装饰器
    # @event_message_type(EventMessageType.ALL)
    # @platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 调试信息
        logger.debug(f"EventHandler.on_message called with event type: {type(event)}")

        # 检查event对象是否正确
        if not hasattr(event, "get_messages"):
            logger.error(
                f"Event object does not have get_messages method. Type: {type(event)}"
            )
            logger.error(f"Event attributes: {dir(event)}")
            return

        plugin_instance = self.plugin
        try:
            if hasattr(
                plugin_instance, "is_meme_enabled_for_event"
            ) and not plugin_instance.is_meme_enabled_for_event(event):
                return
        except Exception:
            return

        force_entry = None
        if hasattr(plugin_instance, "get_force_capture_entry"):
            try:
                force_entry = plugin_instance.get_force_capture_entry(event)
            except Exception:
                force_entry = None

        force_active = force_entry is not None

        if not plugin_instance.steal_emoji and not force_active:
            return

        # 收集所有图片组件
        imgs: list[Image] = [
            comp for comp in event.get_messages() if isinstance(comp, Image)
        ]

        # 如果没有图片，直接返回
        if not imgs:
            return

        if force_active:
            img = imgs[0]
            try:
                temp_path: str = await img.convert_to_file_path()
                if not os.path.exists(temp_path):
                    await event.send(
                        MessageChain([Plain(text="❌ 收录失败：图片临时文件不存在")])
                    )
                else:
                    success, idx = await plugin_instance._process_image(
                        event, temp_path, is_temp=True, is_platform_emoji=True
                    )
                    if success and idx:
                        await plugin_instance._save_index(idx)
                        await event.send(
                            MessageChain([Plain(text="✅ 已收录并自动分类入库")])
                        )
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
                if hasattr(plugin_instance, "consume_force_capture"):
                    try:
                        plugin_instance.consume_force_capture(event)
                    except Exception:
                        pass
            return

        # 检查是否应该处理图片（节流控制）
        if not self._should_process_image():
            logger.debug(f"跳过处理 {len(imgs)} 张图片（节流控制）")
            return

        logger.debug(f"开始处理 {len(imgs)} 张图片")

        raw_image_segments: list[dict] = []
        raw_image_file_map: dict[str, dict] = {}
        try:
            raw_event = getattr(
                getattr(event, "message_obj", None), "raw_message", None
            )
            raw_message = getattr(raw_event, "message", None)
            if isinstance(raw_message, list):
                raw_image_segments = [
                    seg
                    for seg in raw_message
                    if isinstance(seg, dict) and seg.get("type") == "image"
                ]

                def normalize_str(value: object) -> str:
                    if value is None:
                        return ""
                    try:
                        s = str(value)
                    except Exception:
                        return ""
                    s = s.strip()
                    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
                        s = s[1:-1].strip()
                    return s

                for seg in raw_image_segments:
                    data = seg.get("data", {}) or {}
                    if not isinstance(data, dict):
                        continue
                    seg_file = normalize_str(data.get("file", ""))
                    if seg_file and seg_file not in raw_image_file_map:
                        raw_image_file_map[seg_file] = data
        except Exception:
            raw_image_segments = []
            raw_image_file_map = {}

        for i, img in enumerate(imgs):
            try:
                # 检查图片元信息，只处理平台标记的表情包
                is_platform_emoji = self._check_platform_emoji_metadata(
                    img,
                    event,
                    img_index=i,
                    image_segments=raw_image_segments,
                    image_file_map=raw_image_file_map,
                )

                if not is_platform_emoji:
                    # 获取 subType 值用于调试日志
                    sub_type_value = getattr(img, "subType", "unknown")
                    logger.debug(f"跳过非表情包图片 (subType={sub_type_value})")
                    continue

                logger.info("检测到平台标记的表情包，开始处理")

                # 转换图片到临时文件路径
                temp_path: str = await img.convert_to_file_path()

                # 临时文件由框架创建，无需安全检查
                # 安全检查会在 process_image 中处理最终存储路径时进行

                # 确保临时文件存在且可访问
                if not os.path.exists(temp_path):
                    logger.warning(f"临时文件不存在: {temp_path}")
                    continue

                # 使用统一的图片处理方法
                # 传递平台元信息标记，用于优化处理流程
                success, idx = await plugin_instance._process_image(
                    event, temp_path, is_temp=True, is_platform_emoji=is_platform_emoji
                )
                if success and idx:
                    await plugin_instance._save_index(idx)
            except FileNotFoundError as e:
                logger.error(f"图片文件不存在: {e}")
            except PermissionError as e:
                logger.error(f"图片文件权限错误: {e}")
            except asyncio.TimeoutError as e:
                logger.error(f"图片处理超时: {e}")
            except ValueError as e:
                logger.error(f"图片处理参数错误: {e}")
            except Exception as e:
                logger.error(f"处理图片失败: {e}", exc_info=True)

    async def _clean_raw_directory(self) -> int:
        """按时间定时清理raw目录中的原始图片。"""
        try:
            # 使用简化的清理逻辑
            return await self._clean_raw_directory_legacy()

        except Exception as e:
            logger.error(f"清理raw目录失败: {e}")
            return 0

    async def _clean_raw_directory_legacy(self) -> int:
        """简化的raw目录清理逻辑：清理所有raw文件"""
        try:
            total_deleted = 0

            # 清理raw目录中的所有文件
            if self.plugin.base_dir:
                raw_dir = self.plugin.base_dir / "raw"
                if raw_dir.exists():
                    logger.debug(f"开始清理raw目录: {raw_dir}")

                    # 获取raw目录中的所有文件
                    files = list(raw_dir.iterdir())
                    if not files:
                        logger.info(f"raw目录已为空: {raw_dir}")
                    else:
                        # 清理所有文件（因为成功分类的文件已经被立即删除了）
                        deleted_count = 0
                        for file_path in files:
                            try:
                                if file_path.is_file():
                                    if await self.plugin._safe_remove_file(
                                        str(file_path)
                                    ):
                                        deleted_count += 1
                                        logger.debug(f"已删除raw文件: {file_path}")
                                    else:
                                        logger.error(f"删除raw文件失败: {file_path}")
                            except Exception as e:
                                logger.error(
                                    f"处理raw文件时发生错误: {file_path}, 错误: {e}"
                                )

                        logger.info(f"清理raw目录完成，共删除 {deleted_count} 个文件")
                        total_deleted += deleted_count
                else:
                    logger.info(f"raw目录不存在: {raw_dir}")
            else:
                logger.warning("插件base_dir未设置，无法清理raw目录")

            logger.info(f"清理完成，总计删除 {total_deleted} 个文件")
            return total_deleted
        except Exception as e:
            logger.error(f"清理目录时发生错误: {e}", exc_info=True)
            return 0

    async def _enforce_capacity(self, image_index: dict):
        """执行容量控制，删除最旧的图片。"""
        try:
            # 使用简化的容量控制逻辑
            await self._enforce_capacity_legacy(image_index)

        except Exception as e:
            logger.error(f"执行容量控制失败: {e}")

    async def _enforce_capacity_legacy(self, image_index: dict):
        """原有的容量控制逻辑（保持向后兼容）"""
        try:
            max_reg = int(self.plugin.max_reg_num)
            if max_reg <= 0:
                logger.warning(f"容量控制上限无效: max_reg_num={max_reg}，跳过容量控制")
                return

            if len(image_index) <= max_reg:
                return
            if not self.plugin.do_replace:
                return

            image_items = []
            for file_path, image_info in image_index.items():
                created_at = (
                    int(image_info.get("created_at", 0))
                    if isinstance(image_info, dict)
                    else 0
                )
                image_items.append((file_path, created_at))

            if not image_items:
                return

            image_items.sort(key=lambda x: x[1])

            remove_count = len(image_items) - max_reg
            remove_count = max(0, remove_count)  # 确保不为负数
            safe_remove_count = min(remove_count, len(image_items))  # 确保不超出范围

            logger.info(
                f"容量控制: 当前 {len(image_items)} 个，上限 {max_reg}，将删除 {safe_remove_count} 个最旧的"
            )

            for i in range(safe_remove_count):
                remove_path = image_items[i][0]
                try:
                    if os.path.exists(remove_path):
                        await self.plugin._safe_remove_file(remove_path)

                    if remove_path in image_index and isinstance(
                        image_index[remove_path], dict
                    ):
                        category = image_index[remove_path].get("category", "")
                        if category and self.plugin.base_dir:
                            file_name = os.path.basename(remove_path)
                            category_file_path = os.path.join(
                                self.plugin.base_dir, "categories", category, file_name
                            )
                            if os.path.exists(category_file_path):
                                await self.plugin._safe_remove_file(category_file_path)
                except (FileNotFoundError, PermissionError) as e:
                    logger.error(f"删除文件时文件操作错误: {remove_path}, 错误: {e}")
                except OSError as e:
                    logger.error(f"删除文件时系统错误: {remove_path}, 错误: {e}")
                except Exception as e:
                    logger.error(
                        f"删除文件时发生未预期错误: {remove_path}, 错误: {e}",
                        exc_info=True,
                    )

                if remove_path in image_index:
                    del image_index[remove_path]

            logger.info(
                f"容量控制完成，删除了 {remove_count} 个表情包，当前数量: {len(image_index)}"
            )
        except ValueError as e:
            logger.error(f"执行容量控制时配置值错误: {e}")
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"执行容量控制时文件操作错误: {e}")
        except OSError as e:
            logger.error(f"执行容量控制时系统错误: {e}")
        except Exception as e:
            logger.error(f"执行容量控制时发生未预期错误: {e}", exc_info=True)

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

    def _get_force_capture_sender_id(self, event) -> str | None:
        """获取发送者ID。

        Args:
            event: 消息事件对象

        Returns:
            str | None: 发送者ID
        """
        for attr in ("sender_id", "user_id"):
            value = getattr(event, attr, None)
            if value:
                return str(value)

        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            for attr in ("sender_id", "user_id"):
                value = getattr(message_obj, attr, None)
                if value:
                    return str(value)

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
        self._force_capture_windows[key] = {"until": until, "sender_id": sender_id}

    def get_force_capture_entry(self, event) -> dict[str, object] | None:
        """获取强制捕获条目。

        Args:
            event: 消息事件对象

        Returns:
            dict | None: 捕获条目，如果不存在或已过期则返回None
        """
        key = self._get_force_capture_key(event)
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
        self._force_capture_windows.pop(key, None)

    def cleanup(self):
        """清理资源。"""
        # 取消扫描任务
        if self._scanner_task and not self._scanner_task.done():
            self._scanner_task.cancel()
        # 清理插件引用
        self.plugin = None
        logger.debug("EventHandler 资源已清理")
