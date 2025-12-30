import os
import random
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image


class CommandHandler:
    """命令处理服务类，负责处理所有与插件相关的命令操作。"""

    def __init__(self, plugin_instance):
        """初始化命令处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance

    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        self.plugin.steal_emoji = True
        self.plugin._persist_config()
        yield event.plain_result("已开启偷表情包")

    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        self.plugin.steal_emoji = False
        self.plugin._persist_config()
        yield event.plain_result("已关闭偷表情包")

    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        self.plugin.auto_send = True
        self.plugin._persist_config()
        yield event.plain_result("已开启自动发送")

    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        self.plugin.auto_send = False
        self.plugin._persist_config()
        yield event.plain_result("已关闭自动发送")

    async def set_vision(self, event: AstrMessageEvent, provider_id: str = ""):
        """设置视觉模型。"""
        if not provider_id:
            yield event.plain_result("请提供视觉模型的 provider_id")
            return
        self.plugin.vision_provider_id = provider_id
        self.plugin._persist_config()
        yield event.plain_result(f"已设置视觉模型: {provider_id}")

    async def show_providers(self, event: AstrMessageEvent):
        """显示当前视觉模型。"""
        vision_provider = self.plugin.vision_provider_id or "当前会话"
        yield event.plain_result(f"视觉模型: {vision_provider}")

    async def status(self, event: AstrMessageEvent):
        """显示插件状态和详细的表情包统计信息。"""
        stealing_status = "开启" if self.plugin.steal_emoji else "关闭"
        auto_send_status = "开启" if self.plugin.auto_send else "关闭"

        image_index = await self.plugin._load_index()
        total_count = len(image_index)
        
        # 添加视觉模型信息
        vision_model = (
            self.plugin.vision_provider_id or "未设置（将使用当前会话默认模型）"
        )
        
        # 基础状态信息
        status_text = "🔧 插件状态:\n"
        status_text += f"偷取: {stealing_status}\n"
        status_text += f"自动发送: {auto_send_status}\n"
        status_text += f"概率: {self.plugin.emoji_chance}\n"
        status_text += f"替换: {self.plugin.do_replace}\n"
        status_text += f"审核: {self.plugin.content_filtration}\n"
        status_text += f"视觉模型: {vision_model}\n\n"
        
        # 后台任务状态
        status_text += "⚙️ 后台任务:\n"
        status_text += f"Raw清理: {'启用' if self.plugin.enable_raw_cleanup else '禁用'} ({self.plugin.raw_cleanup_interval}min)\n"
        status_text += f"容量控制: {'启用' if self.plugin.enable_capacity_control else '禁用'} ({self.plugin.capacity_control_interval}min)\n\n"
        
        # 表情包统计信息
        if total_count == 0:
            status_text += "📊 表情包统计:\n暂无表情包数据"
        else:
            # 按分类统计
            category_stats = {}
            
            for img_path, img_info in image_index.items():
                if isinstance(img_info, dict):
                    # 统计分类
                    category = img_info.get('category', '未分类')
                    category_stats[category] = category_stats.get(category, 0) + 1
            
            # 构建统计信息
            status_text += "📊 表情包统计:\n"
            status_text += f"总数量: {total_count}/{self.plugin.max_reg_num} ({total_count/self.plugin.max_reg_num*100:.1f}%)\n\n"
            
            # 分类统计 - 只显示前5个最多的分类
            status_text += "📂 分类统计 (前5):\n"
            sorted_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories[:5]:
                percentage = count / total_count * 100
                status_text += f"  {category}: {count}张 ({percentage:.1f}%)\n"
            
            if len(sorted_categories) > 5:
                status_text += f"  ...还有{len(sorted_categories)-5}个分类\n"
            
            # 存储统计
            raw_count = len(list(self.plugin.raw_dir.glob("*"))) if self.plugin.raw_dir.exists() else 0
            status_text += f"\n💾 存储信息:\n"
            status_text += f"  原始图片: {raw_count}张 | 分类图片: {total_count}张"
        
        yield event.plain_result(status_text)


    async def push(self, event: AstrMessageEvent, category: str = "", alias: str = ""):
        """手动推送指定分类的表情包。支持使用分类名称或别名。"""
        if not self.plugin.base_dir:
            yield event.plain_result("插件未正确配置，缺少图片存储目录")
            return

        # 初始化目标分类变量
        target_category = None

        # 如果提供了别名，优先使用别名查找实际分类
        if alias:
            aliases = await self.plugin._load_aliases()
            if alias in aliases:
                # 别名存在，映射到实际分类名称
                target_category = aliases[alias]
            else:
                yield event.plain_result("未找到指定的别名")
                return

        # 如果没有提供别名或别名不存在，使用分类参数
        # 如果分类参数也为空，则使用默认分类
        target_category = (
            target_category
            or category
            or (self.plugin.categories[0] if self.plugin.categories else "happy")
        )

        # 将目标分类赋值给cat变量，保持后续代码兼容性
        cat = target_category
        cat_dir = self.plugin.base_dir / "categories" / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            yield event.plain_result(f"分类 {cat} 不存在")
            return
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            yield event.plain_result("该分类暂无表情包")
            return
        pick = random.choice(files)
        b64 = await self.plugin.image_processor_service._file_to_base64(pick.as_posix())
        chain = event.make_result().base64_image(b64).message_chain
        yield event.result_with_message_chain(chain)

    async def debug_image(self, event: AstrMessageEvent):
        """调试命令：处理当前消息中的图片并显示详细信息。"""
        # 收集所有图片组件
        image_components = [
            comp for comp in event.message_obj.message if isinstance(comp, Image)
        ]

        if not image_components:
            yield event.plain_result("当前消息中没有图片")
            return

        # 处理第一张图片
        first_image = image_components[0]
        try:
            # 转换图片到临时文件路径
            temp_file_path = await first_image.convert_to_file_path()

            # 临时文件由框架创建，无需安全检查
            # 安全检查会在 process_image 中处理最终存储路径时进行

            # 确保临时文件存在且可访问
            if not Path(temp_file_path).exists():
                yield event.plain_result("临时文件不存在")
                return

            # 开始调试处理
            result_message = "=== 图片调试信息 ===\n"

            # 1. 基本信息
            image_path = Path(temp_file_path)
            file_size = image_path.stat().st_size
            result_message += f"文件大小: {file_size / 1024:.2f} KB\n"

            # 2. 元数据过滤结果
            # 直接使用plugin中的PILImage引用
            if self.plugin.PILImage is not None:
                try:
                    with self.plugin.PILImage.open(temp_file_path) as image:
                        width, height = image.size
                    result_message += f"分辨率: {width}x{height}\n"
                    aspect_ratio = (
                        max(width, height) / min(width, height)
                        if min(width, height) > 0
                        else 0
                    )
                    result_message += f"宽高比: {aspect_ratio:.2f}\n"
                except Exception as e:
                    result_message += f"获取图片信息失败: {e}\n"

            # 3. 多模态分析结果
            result_message += "\n=== 多模态分析结果 ===\n"

            # 处理图片
            success, image_index = await self.plugin._process_image(
                event, temp_file_path, is_temp=True, idx=None
            )
            if success and image_index:
                for processed_file_path, image_info in image_index.items():
                    if isinstance(image_info, dict):
                        result_message += (
                            f"分类: {image_info.get('category', '未知')}\n"
                        )
                        result_message += f"情绪: {image_info.get('emotion', '未知')}\n"
                        result_message += f"标签: {image_info.get('tags', [])}\n"
                        result_message += f"描述: {image_info.get('desc', '无')}\n"
            else:
                result_message += "图片处理失败\n"

            yield event.plain_result(result_message)

        except Exception as e:
            logger.error(f"调试图片失败: {e}")
            yield event.plain_result(f"调试失败: {str(e)}")

    async def clean(self, event: AstrMessageEvent, mode: str = ""):
        """手动触发清理操作，清理raw目录中的原始图片文件，不影响已分类的表情包。
        
        Args:
            event: 消息事件
            mode: 清理模式，空字符串=清理所有，"expired"=只清理过期文件
        """
        try:
            if mode.lower() == "expired":
                # 只清理过期文件（按保留期限）
                deleted_count = await self._clean_raw_directory_with_count()
                yield event.plain_result(f"✅ 已清理过期文件 {deleted_count} 张（保留期限: {self.plugin.raw_retention_minutes}分钟）")
            else:
                # 默认清理所有raw文件
                deleted_count = await self._force_clean_raw_directory()
                yield event.plain_result(f"✅ raw目录清理完成，共删除 {deleted_count} 张原始图片")
        except Exception as e:
            logger.error(f"手动清理失败: {e}")
            yield event.plain_result(f"❌ 清理失败: {str(e)}")
    
    async def _clean_raw_directory_with_count(self) -> int:
        """按保留期限清理raw目录，返回删除的文件数量。"""
        try:
            if not self.plugin.base_dir:
                logger.warning("插件base_dir未设置，无法清理raw目录")
                return 0
                
            raw_dir = self.plugin.base_dir / "raw"
            if not raw_dir.exists():
                logger.info(f"raw目录不存在: {raw_dir}")
                return 0
                
            # 获取raw目录中的所有文件
            files = list(raw_dir.iterdir())
            if not files:
                logger.info(f"raw目录已为空: {raw_dir}")
                return 0
                
            # 设置清理期限
            import time
            retention_minutes = int(self.plugin.raw_retention_minutes)
            current_time = time.time()
            cutoff_time = current_time - (retention_minutes * 60)
                
            # 删除过期文件
            deleted_count = 0
            for file_path in files:
                try:
                    if file_path.is_file():
                        # 获取文件修改时间
                        file_time = file_path.stat().st_mtime
                        
                        if file_time < cutoff_time:
                            if await self.plugin._safe_remove_file(str(file_path)):
                                deleted_count += 1
                                logger.debug(f"已删除过期文件: {file_path}")
                            else:
                                logger.error(f"删除过期文件失败: {file_path}")
                except Exception as e:
                    logger.error(f"处理raw文件时发生错误: {file_path}, 错误: {e}")
                    
            logger.info(f"按期限清理raw目录完成，共删除 {deleted_count} 个过期文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"按期限清理raw目录失败: {e}")
            raise
    
    async def _force_clean_raw_directory(self) -> int:
        """强制清理raw目录中的所有文件（忽略保留期限），返回删除的文件数量。"""
        try:
            if not self.plugin.base_dir:
                logger.warning("插件base_dir未设置，无法清理raw目录")
                return 0
                
            raw_dir = self.plugin.base_dir / "raw"
            if not raw_dir.exists():
                logger.info(f"raw目录不存在: {raw_dir}")
                return 0
                
            # 获取raw目录中的所有文件
            files = list(raw_dir.iterdir())
            if not files:
                logger.info(f"raw目录已为空: {raw_dir}")
                return 0
                
            # 删除所有文件
            deleted_count = 0
            for file_path in files:
                try:
                    if file_path.is_file():
                        if await self.plugin._safe_remove_file(str(file_path)):
                            deleted_count += 1
                            logger.debug(f"已强制删除文件: {file_path}")
                        else:
                            logger.error(f"强制删除文件失败: {file_path}")
                except Exception as e:
                    logger.error(f"处理raw文件时发生错误: {file_path}, 错误: {e}")
                    
            logger.info(f"强制清理raw目录完成，共删除 {deleted_count} 个文件")
            return deleted_count
            
        except Exception as e:
            logger.error(f"强制清理raw目录失败: {e}")
            raise

    async def enforce_capacity(self, event: AstrMessageEvent):
        """手动执行容量控制，删除最旧的表情包以控制总数量。"""
        try:
            # 加载图片索引
            image_index = await self.plugin._load_index()
            
            current_count = len(image_index)
            max_count = self.plugin.max_reg_num
            
            if current_count <= max_count:
                yield event.plain_result(f"当前表情包数量 {current_count} 未超过限制 {max_count}，无需清理")
                return
            
            # 执行容量控制
            await self.plugin._enforce_capacity(image_index)
            await self.plugin._save_index(image_index)
            
            # 重新统计
            new_count = len(image_index)
            removed_count = current_count - new_count
            
            yield event.plain_result(
                f"容量控制完成\n"
                f"删除了 {removed_count} 个最旧的表情包\n"
                f"当前数量: {new_count}/{max_count}"
            )
        except Exception as e:
            logger.error(f"容量控制失败: {e}")
            yield event.plain_result(f"容量控制失败: {str(e)}")

    async def task_status(self, event: AstrMessageEvent):
        """显示后台任务状态。"""
        status_text = "后台任务状态:\n\n"

        # Raw清理任务
        raw_cleanup_status = "启用" if self.plugin.enable_raw_cleanup else "禁用"
        status_text += "📁 Raw目录清理:\n"
        status_text += f"  状态: {raw_cleanup_status}\n"
        status_text += f"  周期: {self.plugin.raw_cleanup_interval}分钟\n"
        status_text += f"  保留期限: {self.plugin.raw_retention_minutes}分钟\n\n"

        # 容量控制任务
        capacity_status = "启用" if self.plugin.enable_capacity_control else "禁用"
        status_text += "📊 容量控制:\n"
        status_text += f"  状态: {capacity_status}\n"
        status_text += f"  周期: {self.plugin.capacity_control_interval}分钟\n"
        status_text += f"  上限: {self.plugin.max_reg_num}张\n"
        status_text += f"  替换: {'是' if self.plugin.do_replace else '否'}\n\n"

        # 任务运行状态
        raw_task_running = self.plugin.task_scheduler.is_task_running(
            "raw_cleanup_loop"
        )
        capacity_task_running = self.plugin.task_scheduler.is_task_running(
            "capacity_control_loop"
        )

        status_text += "运行状态:\n"
        status_text += f"  Raw清理任务: {'运行中' if raw_task_running else '已停止'}\n"
        status_text += (
            f"  容量控制任务: {'运行中' if capacity_task_running else '已停止'}"
        )

        yield event.plain_result(status_text)

    async def toggle_raw_cleanup(self, event: AstrMessageEvent, action: str = ""):
        """启用/禁用raw目录清理任务。"""
        if action not in ["on", "off"]:
            yield event.plain_result("用法: /meme raw_cleanup <on|off>")
            return

        if action == "on":
            self.plugin.enable_raw_cleanup = True
            # 如果任务未运行，启动它
            if not self.plugin.task_scheduler.is_task_running("raw_cleanup_loop"):
                self.plugin.task_scheduler.create_task(
                    "raw_cleanup_loop", self.plugin._raw_cleanup_loop()
                )
            yield event.plain_result("已启用raw目录清理任务")
        else:
            self.plugin.enable_raw_cleanup = False
            # 停止任务
            self.plugin.task_scheduler.cancel_task("raw_cleanup_loop")
            yield event.plain_result("已禁用raw目录清理任务")

        self.plugin._persist_config()

    async def toggle_capacity_control(self, event: AstrMessageEvent, action: str = ""):
        """启用/禁用容量控制任务。"""
        if action not in ["on", "off"]:
            yield event.plain_result("用法: /meme capacity_control <on|off>")
            return

        if action == "on":
            self.plugin.enable_capacity_control = True
            # 如果任务未运行，启动它
            if not self.plugin.task_scheduler.is_task_running("capacity_control_loop"):
                self.plugin.task_scheduler.create_task(
                    "capacity_control_loop", self.plugin._capacity_control_loop()
                )
            yield event.plain_result("已启用容量控制任务")
        else:
            self.plugin.enable_capacity_control = False
            # 停止任务
            self.plugin.task_scheduler.cancel_task("capacity_control_loop")
            yield event.plain_result("已禁用容量控制任务")

        self.plugin._persist_config()

    async def set_raw_cleanup_interval(
        self, event: AstrMessageEvent, interval: str = ""
    ):
        """设置raw清理周期。"""
        if not interval:
            yield event.plain_result(
                "用法: /meme raw_cleanup_interval <分钟>\n例如: /meme raw_cleanup_interval 30"
            )
            return

        try:
            minutes = int(interval)
            if minutes < 1:
                yield event.plain_result("清理周期必须至少为1分钟")
                return

            self.plugin.raw_cleanup_interval = minutes
            self.plugin._persist_config()
            yield event.plain_result(f"已设置raw清理周期为: {minutes}分钟")
        except ValueError:
            yield event.plain_result("无效的周期值，请输入正整数")

    async def set_capacity_control_interval(
        self, event: AstrMessageEvent, interval: str = ""
    ):
        """设置容量控制周期。"""
        if not interval:
            yield event.plain_result(
                "用法: /meme capacity_interval <分钟>\n例如: /meme capacity_interval 60"
            )
            return

        try:
            minutes = int(interval)
            if minutes < 1:
                yield event.plain_result("控制周期必须至少为1分钟")
                return

            self.plugin.capacity_control_interval = minutes
            self.plugin._persist_config()
            yield event.plain_result(f"已设置容量控制周期为: {minutes}分钟")
        except ValueError:
            yield event.plain_result("无效的周期值，请输入正整数")

    async def throttle_status(self, event: AstrMessageEvent):
        """显示图片处理节流状态。"""
        mode = self.plugin.image_processing_mode
        mode_names = {
            "always": "总是处理",
            "probability": "概率处理",
            "interval": "间隔处理",
            "cooldown": "冷却处理",
        }

        status_text = "图片处理节流状态:\n"
        status_text += f"当前模式: {mode_names.get(mode, mode)}\n"

        if mode == "probability":
            status_text += (
                f"处理概率: {self.plugin.image_processing_probability * 100:.0f}%\n"
            )
        elif mode == "interval":
            status_text += f"处理间隔: {self.plugin.image_processing_interval}秒\n"
        elif mode == "cooldown":
            status_text += f"冷却时间: {self.plugin.image_processing_cooldown}秒\n"

        status_text += "\n说明:\n"
        status_text += "- always: 每张图片都处理（消耗API最多）\n"
        status_text += "- probability: 按概率随机处理\n"
        status_text += "- interval: 每N秒只处理一次\n"
        status_text += "- cooldown: 两次处理间隔至少N秒"

        yield event.plain_result(status_text)

    async def set_throttle_mode(self, event: AstrMessageEvent, mode: str = ""):
        """设置图片处理节流模式。"""
        valid_modes = ["always", "probability", "interval", "cooldown"]

        if not mode or mode not in valid_modes:
            yield event.plain_result(
                f"用法: /meme throttle_mode <模式>\n"
                f"可用模式: {', '.join(valid_modes)}\n"
                f"- always: 总是处理\n"
                f"- probability: 概率处理\n"
                f"- interval: 间隔处理\n"
                f"- cooldown: 冷却处理"
            )
            return

        self.plugin.image_processing_mode = mode
        self.plugin._persist_config()

        mode_names = {
            "always": "总是处理",
            "probability": "概率处理",
            "interval": "间隔处理",
            "cooldown": "冷却处理",
        }

        yield event.plain_result(f"已设置图片处理模式为: {mode_names[mode]}")

    async def set_throttle_probability(
        self, event: AstrMessageEvent, probability: str = ""
    ):
        """设置概率模式的处理概率。"""
        if not probability:
            yield event.plain_result(
                "用法: /meme throttle_probability <概率>\n概率范围: 0.0-1.0（例如 0.3 表示30%）"
            )
            return

        try:
            prob = float(probability)
            if not (0.0 <= prob <= 1.0):
                yield event.plain_result("概率必须在 0.0-1.0 之间")
                return

            self.plugin.image_processing_probability = prob
            self.plugin._persist_config()
            yield event.plain_result(f"已设置处理概率为: {prob * 100:.0f}%")
        except ValueError:
            yield event.plain_result("无效的概率值，请输入 0.0-1.0 之间的数字")

    async def set_throttle_interval(self, event: AstrMessageEvent, interval: str = ""):
        """设置间隔模式的处理间隔。"""
        if not interval:
            yield event.plain_result(
                "用法: /meme throttle_interval <秒数>\n例如: /meme throttle_interval 60"
            )
            return

        try:
            seconds = int(interval)
            if seconds < 1:
                yield event.plain_result("间隔必须至少为1秒")
                return

            self.plugin.image_processing_interval = seconds
            self.plugin._persist_config()
            yield event.plain_result(f"已设置处理间隔为: {seconds}秒")
        except ValueError:
            yield event.plain_result("无效的间隔值，请输入正整数")

    async def set_throttle_cooldown(self, event: AstrMessageEvent, cooldown: str = ""):
        """设置冷却模式的冷却时间。"""
        if not cooldown:
            yield event.plain_result(
                "用法: /meme throttle_cooldown <秒数>\n例如: /meme throttle_cooldown 30"
            )
            return

        try:
            seconds = int(cooldown)
            if seconds < 1:
                yield event.plain_result("冷却时间必须至少为1秒")
                return

            self.plugin.image_processing_cooldown = seconds
            self.plugin._persist_config()
            yield event.plain_result(f"已设置冷却时间为: {seconds}秒")
        except ValueError:
            yield event.plain_result("无效的冷却时间，请输入正整数")

    async def migrate_legacy_data(self, event: AstrMessageEvent):
        """手动迁移旧版本数据。"""
        try:
            yield event.plain_result("开始检查和迁移旧版本数据...")
            
            # 强制重新迁移数据
            migrated_data = await self.plugin._migrate_legacy_data()
            
            if migrated_data:
                yield event.plain_result(f"✅ 成功迁移 {len(migrated_data)} 条记录")
                
                # 显示迁移的分类统计
                category_stats = {}
                for record in migrated_data.values():
                    if isinstance(record, dict):
                        category = record.get('category', '未分类')
                        category_stats[category] = category_stats.get(category, 0) + 1
                
                if category_stats:
                    stats_text = "迁移的分类统计:\n"
                    for category, count in sorted(category_stats.items()):
                        stats_text += f"  {category}: {count}张\n"
                    yield event.plain_result(stats_text)
            else:
                yield event.plain_result("ℹ️ 未发现需要迁移的数据")
                
        except Exception as e:
            logger.error(f"手动迁移失败: {e}")
            yield event.plain_result(f"❌ 迁移失败: {str(e)}")

    def cleanup(self):
        """清理资源。"""
        # CommandHandler 主要是无状态的，清理插件引用即可
        self.plugin = None
        logger.debug("CommandHandler 资源已清理")

    async def list_images(self, event: AstrMessageEvent, category: str = "", limit: str = "10"):
        """列出表情包，支持按分类筛选。
        
        Args:
            event: 消息事件
            category: 可选的分类筛选
            limit: 显示数量限制，默认10张
        """
        try:
            max_limit = int(limit)
            if max_limit < 1:
                max_limit = 10
        except ValueError:
            max_limit = 10

        image_index = await self.plugin._load_index()
        
        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        # 筛选图片
        filtered_images = []
        for img_path, img_info in image_index.items():
            if isinstance(img_info, dict):
                img_category = img_info.get('category', '未分类')
                
                # 如果指定了分类，只显示该分类的图片
                if category and img_category != category:
                    continue
                
                # 检查文件是否存在
                if not Path(img_path).exists():
                    continue
                
                filtered_images.append({
                    'path': img_path,
                    'name': Path(img_path).name,
                    'category': img_category,
                    'created_at': img_info.get('created_at', 0)
                })

        if not filtered_images:
            if category:
                yield event.plain_result(f"分类 '{category}' 中暂无表情包")
            else:
                yield event.plain_result("暂无有效的表情包文件")
            return

        # 按创建时间排序（最新的在前）
        filtered_images.sort(key=lambda x: x['created_at'], reverse=True)
        
        # 限制显示数量
        display_images = filtered_images[:max_limit]
        
        # 构建标题信息
        title = f"📋 表情包列表 ({len(display_images)}/{len(filtered_images)})"
        if category:
            title += f" - 分类: {category}"
        
        # 先发送标题
        yield event.plain_result(title + "\n💡 使用 /meme delete <序号> 删除指定图片")
        
        # 逐个发送图片和信息
        for i, img in enumerate(display_images, 1):
            try:
                # 读取图片并转换为base64
                b64 = await self.plugin.image_processor_service._file_to_base64(img['path'])
                
                # 构建图片信息
                info_text = f"{i:2d}. {img['name'][:20]}{'...' if len(img['name']) > 20 else ''}\n"
                info_text += f"分类: {img['category']}"
                
                # 发送图片和信息
                result = event.make_result().base64_image(b64).message(info_text)
                yield result
                
            except Exception as e:
                # 如果图片读取失败，只发送文本信息
                logger.warning(f"读取图片失败 {img['path']}: {e}")
                info_text = f"{i:2d}. {img['name']} [图片读取失败]\n"
                info_text += f"分类: {img['category']}"
                yield event.plain_result(info_text)
        
        if len(filtered_images) > max_limit:
            yield event.plain_result(f"...还有 {len(filtered_images) - max_limit} 张图片")

    async def list_images_text_only(self, event: AstrMessageEvent, category: str = "", limit: str = "10"):
        """列出表情包（仅文本模式），支持按分类筛选。
        
        Args:
            event: 消息事件
            category: 可选的分类筛选
            limit: 显示数量限制，默认10张
        """
        try:
            max_limit = int(limit)
            if max_limit < 1:
                max_limit = 10
        except ValueError:
            max_limit = 10

        image_index = await self.plugin._load_index()
        
        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        # 筛选图片
        filtered_images = []
        for img_path, img_info in image_index.items():
            if isinstance(img_info, dict):
                img_category = img_info.get('category', '未分类')
                
                # 如果指定了分类，只显示该分类的图片
                if category and img_category != category:
                    continue
                
                # 检查文件是否存在
                if not Path(img_path).exists():
                    continue
                
                filtered_images.append({
                    'path': img_path,
                    'name': Path(img_path).name,
                    'category': img_category,
                    'created_at': img_info.get('created_at', 0)
                })

        if not filtered_images:
            if category:
                yield event.plain_result(f"分类 '{category}' 中暂无表情包")
            else:
                yield event.plain_result("暂无有效的表情包文件")
            return

        # 按创建时间排序（最新的在前）
        filtered_images.sort(key=lambda x: x['created_at'], reverse=True)
        
        # 限制显示数量
        display_images = filtered_images[:max_limit]
        
        # 构建显示文本
        title = f"📋 表情包列表 ({len(display_images)}/{len(filtered_images)})"
        if category:
            title += f" - 分类: {category}"
        
        result_text = title + "\n\n"
        
        for i, img in enumerate(display_images, 1):
            name = img['name']
            # 截断过长的文件名
            if len(name) > 20:
                name = name[:17] + "..."
            
            result_text += f"{i:2d}. {name}\n"
            result_text += f"    分类: {img['category']}\n"
        
        if len(filtered_images) > max_limit:
            result_text += f"\n...还有 {len(filtered_images) - max_limit} 张图片"
        
        result_text += f"\n\n💡 使用 /meme delete <序号> 删除指定图片"
        
        yield event.plain_result(result_text)

    async def delete_image(self, event: AstrMessageEvent, identifier: str = ""):
        """删除指定的表情包。
        
        Args:
            event: 消息事件
            identifier: 图片标识符，可以是序号、文件名或路径
        """
        if not identifier:
            yield event.plain_result(
                "用法: /meme delete <序号|文件名>\n"
                "先使用 /meme list 查看图片列表获取序号"
            )
            return

        image_index = await self.plugin._load_index()
        
        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        # 获取所有有效图片
        valid_images = []
        for img_path, img_info in image_index.items():
            if isinstance(img_info, dict) and Path(img_path).exists():
                valid_images.append({
                    'path': img_path,
                    'name': Path(img_path).name,
                    'category': img_info.get('category', '未分类'),
                    'created_at': img_info.get('created_at', 0)
                })

        # 按创建时间排序（与list命令保持一致，最新的在前）
        valid_images.sort(key=lambda x: x['created_at'], reverse=True)

        target_image = None

        # 尝试按序号查找
        try:
            index = int(identifier) - 1  # 转换为0基索引
            if 0 <= index < len(valid_images):
                target_image = valid_images[index]
        except ValueError:
            # 不是数字，尝试按文件名查找
            for img in valid_images:
                if img['name'] == identifier or img['name'].startswith(identifier):
                    target_image = img
                    break

        if not target_image:
            yield event.plain_result(
                f"未找到图片: {identifier}\n"
                "请使用 /meme list 查看可用的图片列表"
            )
            return

        # 执行删除操作
        success = await self._delete_image_files(target_image['path'])
        
        if success:
            # 从索引中移除
            if target_image['path'] in image_index:
                del image_index[target_image['path']]
                await self.plugin._save_index(image_index)
            
            # 如果使用增强存储系统，同时更新数据库
            if (hasattr(self.plugin, 'lifecycle_manager') and 
                self.plugin.lifecycle_manager):
                try:
                    await self._delete_from_enhanced_storage(target_image['path'])
                except Exception as e:
                    logger.warning(f"更新增强存储系统失败: {e}")
            
            yield event.plain_result(
                f"✅ 已删除表情包:\n"
                f"文件: {target_image['name']}\n"
                f"分类: {target_image['category']}"
            )
        else:
            yield event.plain_result(f"❌ 删除失败: {target_image['name']}")

    async def _delete_image_files(self, img_path: str) -> bool:
        """删除图片文件（raw目录和categories目录）。
        
        Args:
            img_path: 图片路径
            
        Returns:
            bool: 是否删除成功
        """
        try:
            deleted_files = []
            
            # 删除主文件（通常在raw目录）
            if Path(img_path).exists():
                Path(img_path).unlink()
                deleted_files.append(img_path)
                logger.info(f"已删除主文件: {img_path}")
            
            # 查找并删除categories目录中的对应文件
            if hasattr(self.plugin, 'categories_dir') and self.plugin.categories_dir:
                img_name = Path(img_path).name
                
                # 遍历所有分类目录
                for category_dir in self.plugin.categories_dir.iterdir():
                    if category_dir.is_dir():
                        category_file = category_dir / img_name
                        if category_file.exists():
                            category_file.unlink()
                            deleted_files.append(str(category_file))
                            logger.info(f"已删除分类文件: {category_file}")
            
            logger.info(f"删除操作完成，共删除 {len(deleted_files)} 个文件")
            return len(deleted_files) > 0
            
        except Exception as e:
            logger.error(f"删除图片文件失败: {e}")
            return False

    async def _delete_from_enhanced_storage(self, img_path: str):
        """从增强存储系统中删除记录。
        
        Args:
            img_path: 图片路径
        """
        try:
            if not (hasattr(self.plugin, 'lifecycle_manager') and 
                   self.plugin.lifecycle_manager):
                return
            
            # 查找对应的生命周期记录
            records = await self.plugin.lifecycle_manager.get_files_by_path(img_path)
            
            for record in records:
                # 标记为删除状态
                from .storage.models import ProcessingStatus
                await self.plugin.lifecycle_manager.update_processing_status(
                    record.record_id, 
                    ProcessingStatus.MARKED_FOR_DELETION,
                    failure_reason="用户手动删除"
                )
                
                # 记录删除事件
                if (hasattr(self.plugin, 'statistics_tracker') and 
                    self.plugin.statistics_tracker):
                    from .storage.models import ProcessingEventType
                    await self.plugin.statistics_tracker.record_processing_event(
                        ProcessingEventType.IMAGE_DELETED,
                        metadata={"file_path": img_path, "deletion_type": "manual"}
                    )
            
            logger.info(f"已更新增强存储系统记录: {img_path}")
            
        except Exception as e:
            logger.error(f"更新增强存储系统失败: {e}")
            # 不抛出异常，避免影响主删除流程
