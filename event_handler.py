import asyncio
import os
import random
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image


class EventHandler:
    """事件处理服务类，负责处理所有与插件相关的事件操作。"""

    def __init__(self, plugin_instance):
        """初始化事件处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self._scanner_task: asyncio.Task | None = None

        # 图片处理节流相关
        self._last_process_time = 0  # 上次处理时间（用于interval和cooldown模式）
        self._process_count = 0  # 处理计数（用于interval模式）

    def _should_process_image(self) -> bool:
        """根据配置的节流模式判断是否应该处理图片。

        Returns:
            bool: 是否应该处理图片
        """
        import time

        mode = self.plugin.image_processing_mode
        current_time = time.time()

        if mode == "always":
            # 总是处理
            return True

        elif mode == "probability":
            # 概率处理
            probability = float(self.plugin.image_processing_probability)
            should_process = random.random() < probability
            if should_process:
                logger.debug(f"概率模式：通过（概率={probability}）")
            else:
                logger.debug(f"概率模式：跳过（概率={probability}）")
            return should_process

        elif mode == "interval":
            # 间隔处理：每N秒只处理一次
            interval = int(self.plugin.image_processing_interval)
            time_since_last = current_time - self._last_process_time

            if time_since_last >= interval:
                self._last_process_time = current_time
                logger.debug(
                    f"间隔模式：通过（间隔={interval}秒，距上次={time_since_last:.1f}秒）"
                )
                return True
            else:
                logger.debug(
                    f"间隔模式：跳过（间隔={interval}秒，距上次={time_since_last:.1f}秒）"
                )
                return False

        elif mode == "cooldown":
            # 冷却模式：两次处理之间至少间隔N秒
            cooldown = int(self.plugin.image_processing_cooldown)
            time_since_last = current_time - self._last_process_time

            if time_since_last >= cooldown:
                self._last_process_time = current_time
                logger.debug(
                    f"冷却模式：通过（冷却={cooldown}秒，距上次={time_since_last:.1f}秒）"
                )
                return True
            else:
                logger.debug(
                    f"冷却模式：跳过（冷却={cooldown}秒，距上次={time_since_last:.1f}秒）"
                )
                return False

        # 默认不处理
        logger.warning(f"未知的图片处理模式: {mode}，跳过处理")
        return False

    # 注意：这个方法不需要装饰器，因为在StealerPlugin类中已经使用了装饰器
    # @event_message_type(EventMessageType.ALL)
    # @platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs):
        """消息监听：偷取消息中的图片并分类存储。"""
        plugin_instance = self.plugin
        message_event = event

        if not plugin_instance.steal_emoji:
            return

        # 收集所有图片组件
        imgs: list[Image] = [
            comp
            for comp in message_event.message_obj.message
            if isinstance(comp, Image)
        ]

        # 如果没有图片，直接返回
        if not imgs:
            return

        # 检查是否应该处理图片（节流控制）
        if not self._should_process_image():
            logger.debug(f"跳过处理 {len(imgs)} 张图片（节流控制）")
            return

        logger.debug(f"开始处理 {len(imgs)} 张图片")

        for img in imgs:
            try:
                # 转换图片到临时文件路径
                temp_path: str = await img.convert_to_file_path()

                # 临时文件由框架创建，无需安全检查
                # 安全检查会在 process_image 中处理最终存储路径时进行

                # 确保临时文件存在且可访问
                if not os.path.exists(temp_path):
                    logger.error(f"临时文件不存在: {temp_path}")
                    continue

                # 使用统一的图片处理方法
                success, idx = await plugin_instance._process_image(
                    message_event, temp_path, is_temp=True
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

    async def _clean_raw_directory(self):
        """按时间定时清理raw目录中的原始图片。"""
        try:
            # 设置清理期限：保留配置指定时间内的文件，超过则删除
            retention_minutes = int(self.plugin.raw_retention_minutes)
            current_time = time.time()
            cutoff_time = current_time - (retention_minutes * 60)

            total_deleted = 0

            # 清理raw目录
            if self.plugin.base_dir:
                raw_dir = self.plugin.base_dir / "raw"
            if raw_dir.exists():
                logger.debug(
                    f"开始清理raw目录: {raw_dir}, 保留期限: {retention_minutes}分钟, 当前时间: {current_time}, 截止时间: {cutoff_time}"
                )

                # 获取raw目录中的所有文件
                files = list(raw_dir.iterdir())
                if not files:
                    logger.info(f"raw目录已为空: {raw_dir}")
                else:
                    # 清理过期文件
                    deleted_count = 0
                    for file_path in files:
                        try:
                            if file_path.is_file():
                                # 获取文件修改时间
                                file_time = file_path.stat().st_mtime
                                logger.debug(
                                    f"检查文件: {file_path}, 修改时间: {file_time}, 是否过期: {file_time < cutoff_time}"
                                )

                                if file_time < cutoff_time:
                                    if await self.plugin._safe_remove_file(
                                        str(file_path)
                                    ):
                                        deleted_count += 1
                                        logger.debug(f"已删除过期文件: {file_path}")
                                    else:
                                        logger.error(f"删除过期文件失败: {file_path}")
                        except (FileNotFoundError, PermissionError) as e:
                            logger.error(
                                f"处理raw文件时文件操作错误: {file_path}, 错误: {e}"
                            )
                        except OSError as e:
                            logger.error(
                                f"处理raw文件时系统错误: {file_path}, 错误: {e}"
                            )
                        except Exception as e:
                            logger.error(
                                f"处理raw文件时发生未预期错误: {file_path}, 错误: {e}",
                                exc_info=True,
                            )

                    logger.info(f"清理raw目录完成，共删除 {deleted_count} 个过期文件")
                    total_deleted += deleted_count
            else:
                logger.info(f"raw目录不存在: {raw_dir}")

            logger.info(f"清理完成，总计删除 {total_deleted} 个过期文件")
        except ValueError as e:
            logger.error(f"清理目录时配置值错误: {e}")
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"清理目录时文件操作错误: {e}")
        except OSError as e:
            logger.error(f"清理目录时系统错误: {e}")
        except Exception as e:
            logger.error(f"清理目录时发生未预期错误: {e}", exc_info=True)

    async def _enforce_capacity(self, image_index: dict):
        """执行容量控制，删除最不常用的图片。"""
        try:
            if len(image_index) <= int(self.plugin.max_reg_num):
                return
            if not self.plugin.do_replace:
                return
            image_items = []
            for file_path, image_info in image_index.items():
                usage_count = (
                    int(image_info.get("usage_count", 0))
                    if isinstance(image_info, dict)
                    else 0
                )
                created_at = (
                    int(image_info.get("created_at", 0))
                    if isinstance(image_info, dict)
                    else 0
                )
                image_items.append((file_path, usage_count, created_at))
            image_items.sort(key=lambda x: (x[1], x[2]))

            # 计算需要删除的数量：当达到上限时一次性删除10个最旧的图片
            # 如果超出数量少于10个，则只删除超出部分
            remove_count = len(image_index) - int(self.plugin.max_reg_num)
            remove_count = min(10, remove_count)  # 最多删除10个

            for i in range(remove_count):
                remove_path = image_items[i][0]
                try:
                    # 删除raw目录中的原始文件
                    if os.path.exists(remove_path):
                        await self.plugin._safe_remove_file(remove_path)

                    # 删除categories目录中的对应副本
                    # 从索引中获取分类信息
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

                # 从索引中删除对应的条目
                if remove_path in image_index:
                    del image_index[remove_path]
        except ValueError as e:
            logger.error(f"执行容量控制时配置值错误: {e}")
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"执行容量控制时文件操作错误: {e}")
        except OSError as e:
            logger.error(f"执行容量控制时系统错误: {e}")
        except Exception as e:
            logger.error(f"执行容量控制时发生未预期错误: {e}", exc_info=True)

    def cleanup(self):
        """清理资源。"""
        # 取消扫描任务
        if self._scanner_task and not self._scanner_task.done():
            self._scanner_task.cancel()
        # 清理插件引用
        self.plugin = None
        logger.debug("EventHandler 资源已清理")
