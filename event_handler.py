import asyncio
import os
import random
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event.filter import (
    on_decorating_result,
)
from astrbot.api.message_components import Image, Plain


class EventHandler:
    """事件处理服务类，负责处理所有与插件相关的事件操作。"""

    def __init__(self, plugin_instance):
        """初始化事件处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self._scanner_task: asyncio.Task | None = None

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

        for img in imgs:
            try:
                # 转换图片到临时文件路径
                temp_path: str = await img.convert_to_file_path()

                # 检查路径安全性
                is_safe, safe_path = plugin_instance._is_safe_path(temp_path)
                if not is_safe:
                    logger.warning(f"不安全的图片路径: {temp_path}")
                    continue

                temp_path = safe_path

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

    @on_decorating_result()
    async def before_send(self, event: AstrMessageEvent, *args, **kwargs):
        """发送消息前的处理：根据文本内容匹配并添加表情包。"""
        if not self.plugin.auto_send or not self.plugin.base_dir:
            return
        result = event.get_result()
        # 只在有文本结果时尝试匹配表情包
        if result is None:
            return

        # 检查result是否具有必要的属性
        if not hasattr(result, "chain") or not hasattr(result, "get_plain_text"):
            logger.error("before_send: 结果对象缺少必要属性，无法处理")
            return

        # 文本仅用于本地规则提取情绪关键字，不再请求额外的 LLM
        text: str = result.get_plain_text() or event.get_message_str() or ""
        if not text.strip():
            logger.debug("没有可处理的文本内容，未触发图片发送")
            return

        emotions, cleaned_text = await self.plugin._extract_emotions_from_text(
            event, text
        )

        # 先执行标签清理，无论是否发送表情包都需要清理标签
        if cleaned_text != text:
            # 创建新的结果对象并更新内容
            new_result = event.make_result().set_result_content_type(
                result.result_content_type
            )

            # 添加除了Plain文本外的其他组件
            for comp in result.chain:
                if not isinstance(comp, Plain):
                    new_result.chain.append(comp)

            # 添加清除标签后的文本
            if cleaned_text.strip():
                new_result.message(cleaned_text.strip())

            # 设置新的结果对象
            event.set_result(new_result)

            # 更新result和text变量，使用清理后的结果
            result = new_result
            text = cleaned_text

        # 如果没有情绪标签，不需要继续处理图片发送
        if not emotions:
            logger.debug("未从文本中提取到情绪关键词，未触发图片发送")
            return

        # 只有在有情绪标签时才检查发送概率
        try:
            chance = float(self.plugin.emoji_chance)
            # 兜底保护，防止配置错误导致永远/从不触发
            if chance <= 0:
                logger.debug("表情包自动发送概率为0，未触发图片发送")
                return
            if chance > 1:
                chance = 1.0
            if random.random() >= chance:
                logger.debug(f"表情包自动发送概率检查未通过 ({chance}), 未触发图片发送")
                return
        except (ValueError, TypeError) as e:
            logger.error(f"解析表情包自动发送概率配置失败: {e}，未触发图片发送")
            return

        logger.debug("表情包自动发送概率检查通过，开始处理图片发送")

        logger.debug(f"提取到情绪关键词: {emotions}")

        # 目前只取第一个识别到的情绪类别
        category = emotions[0]
        if self.plugin.base_dir:
            cat_dir = self.plugin.base_dir / "categories" / category
        if not cat_dir.exists():
            logger.debug(f"情绪'{category}'对应的图片目录不存在，未触发图片发送")
            return

        try:
            files = [p for p in cat_dir.iterdir() if p.is_file()]
            if not files:
                logger.debug(f"情绪'{category}'对应的图片目录为空，未触发图片发送")
                return

            logger.debug(f"从'{category}'目录中找到 {len(files)} 张图片")
            pick = random.choice(files)
            idx = await self.plugin._load_index()
            rec = idx.get(pick.as_posix())
            if isinstance(rec, dict):
                rec["usage_count"] = int(rec.get("usage_count", 0)) + 1
                rec["last_used"] = int(asyncio.get_event_loop().time())
                idx[pick.as_posix()] = rec
                await self.plugin._save_index(idx)
        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"访问情绪图片目录失败: {e}", exc_info=True)
            return
        # 创建新的结果对象并更新内容
        new_result = event.make_result().set_result_content_type(
            result.result_content_type
        )

        # 添加除了Plain文本外的其他组件
        for comp in result.chain:
            if not isinstance(comp, Plain):
                new_result.chain.append(comp)

        # 添加清除标签后的文本
        if cleaned_text.strip():
            new_result.message(cleaned_text.strip())

        # 添加图片
        b64 = await self.plugin._file_to_base64(pick.as_posix())
        new_result.base64_image(b64)

        # 设置新的结果对象
        event.set_result(new_result)

    async def _scanner_loop(self):
        """扫描循环，处理定期维护任务。"""
        # 初始化raw目录清理计时器
        last_raw_clean_time = time.time()

        while True:
            try:
                await asyncio.sleep(max(1, int(self.plugin.maintenance_interval)) * 60)

                # 只有当偷图功能开启时，才执行容量清理和raw目录清理
                if self.plugin.steal_emoji:
                    # 执行容量清理
                    image_index = await self.plugin._load_index()
                    await self._enforce_capacity(image_index)
                    await self.plugin._save_index(image_index)

                    # 检查是否需要清理raw目录（根据raw_clean_interval配置）
                    current_time = time.time()
                    clean_interval_seconds = (
                        max(1, int(self.plugin.raw_clean_interval)) * 60
                    )
                    if current_time - last_raw_clean_time >= clean_interval_seconds:
                        await self._clean_raw_directory()
                        last_raw_clean_time = current_time

            except (FileNotFoundError, PermissionError) as e:
                logger.error(f"扫描循环文件操作错误: {e}")
                # 对于文件操作错误，继续循环
                continue
            except asyncio.TimeoutError as e:
                logger.error(f"扫描循环超时错误: {e}")
                # 超时错误也继续循环
                continue
            except ValueError as e:
                logger.error(f"扫描循环值错误: {e}")
                # 值错误继续循环
                continue
            except Exception as e:
                logger.error(f"扫描循环发生未预期错误: {e}", exc_info=True)
                # 记录详细异常信息，便于调试
                continue

    async def _clean_raw_directory(self):
        """按时间定时清理raw目录中的原始图片。"""
        try:
            # 设置清理期限：保留配置指定时间内的文件，超过则删除
            retention_hours = int(self.plugin.raw_retention_hours)
            current_time = time.time()
            cutoff_time = current_time - (retention_hours * 3600)

            total_deleted = 0

            # 清理raw目录
            if self.plugin.base_dir:
                raw_dir = self.plugin.base_dir / "raw"
            if raw_dir.exists():
                logger.debug(
                    f"开始清理raw目录: {raw_dir}, 保留期限: {retention_hours}小时, 当前时间: {current_time}, 截止时间: {cutoff_time}"
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
                                        file_path.as_posix()
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
                usage_count = int(image_info.get("usage_count", 0)) if isinstance(image_info, dict) else 0
                created_at = int(image_info.get("created_at", 0)) if isinstance(image_info, dict) else 0
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
                    if remove_path in image_index and isinstance(image_index[remove_path], dict):
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
                        f"删除文件时发生未预期错误: {remove_path}, 错误: {e}", exc_info=True
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
