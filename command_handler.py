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
        yield event.plain_result("已开启偷表情包")

    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        self.plugin.steal_emoji = False
        yield event.plain_result("已关闭偷表情包")

    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        self.plugin.auto_send = True
        yield event.plain_result("已开启自动发送")

    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        self.plugin.auto_send = False
        yield event.plain_result("已关闭自动发送")

    async def set_emotion_provider(self, event: AstrMessageEvent, provider_id: str = ""):
        """设置情绪分析模型。"""
        if not provider_id:
            yield event.plain_result("请提供情绪分析模型的 provider_id")
            return
        # 同时更新实例属性和配置服务中的值，确保同步
        self.plugin.emotion_analysis_provider_id = provider_id
        self.plugin.config_service.emotion_analysis_provider_id = provider_id
        yield event.plain_result(f"已设置情绪分析模型: {provider_id}")

    async def toggle_natural_analysis(self, event: AstrMessageEvent, action: str = ""):
        """启用/禁用自然语言情绪分析。"""
        if action not in ["on", "off"]:
            current_status = "启用" if self.plugin.enable_natural_emotion_analysis else "禁用"
            yield event.plain_result(f"当前自然语言分析状态: {current_status}\n用法: /meme natural_analysis <on|off>")
            return

        if action == "on":
            self.plugin.enable_natural_emotion_analysis = True
            yield event.plain_result("✅ 已启用自然语言情绪分析（LLM模式）\n\n💡 提示：如果之前使用被动标签模式，建议使用 /reset 清除AI对话上下文，避免继续输出 &&emotion&& 标签")
        else:
            self.plugin.enable_natural_emotion_analysis = False
            yield event.plain_result("❌ 已禁用自然语言情绪分析（被动标签模式）\n\n💡 提示：LLM现在会在回复开头插入 &&emotion&& 标签，插件会自动清理这些标签")

    async def emotion_analysis_stats(self, event: AstrMessageEvent):
        """显示情绪分析统计信息。"""
        try:
            # 显示当前模式
            mode = "智能模式" if self.plugin.enable_natural_emotion_analysis else "被动模式"
            
            status_text = f"🧠 情绪分析模式: {mode}\n\n"
            
            if self.plugin.enable_natural_emotion_analysis:
                # 智能模式：显示轻量模型分析统计
                stats = self.plugin.smart_emotion_matcher.get_analyzer_stats()
                
                if "message" in stats:
                    status_text += f"轻量模型分析: {stats['message']}\n"
                else:
                    status_text += "📊 轻量模型分析统计:\n"
                    status_text += f"总分析次数: {stats['total_analyses']}\n"
                    status_text += f"缓存命中率: {stats['cache_hit_rate']}\n"
                    status_text += f"成功率: {stats['success_rate']}\n"
                    status_text += f"平均响应时间: {stats['avg_response_time']}\n"
                    status_text += f"缓存大小: {stats['cache_size']}\n"
                
                status_text += "\n💡 智能模式说明:\n"
                status_text += "- 不向LLM注入提示词\n"
                status_text += "- 使用轻量模型分析回复语义\n"
                status_text += "- 自动识别情绪并发送表情包\n"
            else:
                # 被动模式：显示标签识别说明
                status_text += "📋 被动模式说明:\n"
                status_text += "- 向LLM注入情绪选择提示词\n"
                status_text += "- LLM在回复中插入 &&情绪&& 标签\n"
                status_text += "- 插件识别标签并发送表情包\n"
                status_text += "- 依赖LLM遵循格式要求\n"
            
            status_text += "\n⚙️ 配置状态:\n"
            status_text += f"自动发送: {'启用' if self.plugin.auto_send else '禁用'}\n"
            status_text += f"分析模型: {self.plugin.emotion_analysis_provider_id or '使用当前会话模型'}\n"
            
            yield event.plain_result(status_text)
        except Exception as e:
            yield event.plain_result(f"获取统计信息失败: {e}")

    async def clear_emotion_cache(self, event: AstrMessageEvent):
        """清空情绪分析缓存。"""
        try:
            self.plugin.smart_emotion_matcher.clear_cache()
            yield event.plain_result("✅ 情绪分析缓存已清空")
        except Exception as e:
            yield event.plain_result(f"❌ 清空缓存失败: {e}")

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
                    category = img_info.get("category", "未分类")
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
            status_text += "\n💾 存储信息:\n"
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
        result = event.make_result().base64_image(b64)
        yield result

    async def clean(self, event: AstrMessageEvent, mode: str = ""):
        """手动触发清理操作，清理raw目录中的原始图片文件，不影响已分类的表情包。

        Args:
            event: 消息事件
            mode: 清理模式，现在只支持清理所有raw文件
        """
        try:
            # 清理所有raw文件（因为成功分类的文件已经被立即删除了）
            deleted_count = await self._force_clean_raw_directory()
            yield event.plain_result(f"✅ raw目录清理完成，共删除 {deleted_count} 张原始图片")
        except Exception as e:
            logger.error(f"手动清理失败: {e}")
            yield event.plain_result(f"❌ 清理失败: {str(e)}")

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
            await self.plugin.task_scheduler.cancel_task("raw_cleanup_loop")
            yield event.plain_result("已禁用raw目录清理任务")

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
            await self.plugin.task_scheduler.cancel_task("capacity_control_loop")
            yield event.plain_result("已禁用容量控制任务")

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
                        category = record.get("category", "未分类")
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

    async def list_images(self, event: AstrMessageEvent, category: str = "", limit: str = "10", show_images: bool = True):
        """列出表情包，支持按分类筛选。

        Args:
            event: 消息事件
            category: 可选的分类筛选
            limit: 显示数量限制，默认10张
            show_images: 是否显示图片，默认True
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
                img_category = img_info.get("category", "未分类")

                # 如果指定了分类，只显示该分类的图片
                if category and img_category != category:
                    continue

                # 检查文件是否存在
                if not Path(img_path).exists():
                    continue

                filtered_images.append({
                    "path": img_path,
                    "name": Path(img_path).name,
                    "category": img_category,
                    "created_at": img_info.get("created_at", 0)
                })

        if not filtered_images:
            if category:
                yield event.plain_result(f"分类 '{category}' 中暂无表情包")
            else:
                yield event.plain_result("暂无有效的表情包文件")
            return

        # 按创建时间排序（最新的在前）
        filtered_images.sort(key=lambda x: x["created_at"], reverse=True)

        # 限制显示数量
        display_images = filtered_images[:max_limit]

        if show_images:
            # 显示图片模式
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
                    b64 = await self.plugin.image_processor_service._file_to_base64(img["path"])

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
        else:
            # 纯文本模式
            # 构建标题信息
            title = f"📋 表情包列表 ({len(display_images)}/{len(filtered_images)})"
            if category:
                title += f" - 分类: {category}"

            result_text = title + "\n\n"

            for i, img in enumerate(display_images, 1):
                name = img["name"]
                # 截断过长的文件名
                if len(name) > 20:
                    name = name[:17] + "..."

                result_text += f"{i:2d}. {name}\n"
                result_text += f"    分类: {img['category']}\n"

            if len(filtered_images) > max_limit:
                result_text += f"\n...还有 {len(filtered_images) - max_limit} 张图片"

            result_text += "\n\n💡 使用 /meme delete <序号> 删除指定图片"

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
                    "path": img_path,
                    "name": Path(img_path).name,
                    "category": img_info.get("category", "未分类"),
                    "created_at": img_info.get("created_at", 0)
                })

        # 按创建时间排序（与list命令保持一致，最新的在前）
        valid_images.sort(key=lambda x: x["created_at"], reverse=True)

        target_image = None

        # 尝试按序号查找
        try:
            index = int(identifier) - 1  # 转换为0基索引
            if 0 <= index < len(valid_images):
                target_image = valid_images[index]
        except ValueError:
            # 不是数字，尝试按文件名查找
            for img in valid_images:
                if img["name"] == identifier or img["name"].startswith(identifier):
                    target_image = img
                    break

        if not target_image:
            yield event.plain_result(
                f"未找到图片: {identifier}\n"
                "请使用 /meme list 查看可用的图片列表"
            )
            return

        # 执行删除操作
        success = await self._delete_image_files(target_image["path"])

        if success:
            # 从索引中移除
            if target_image["path"] in image_index:
                del image_index[target_image["path"]]
                await self.plugin._save_index(image_index)

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
            if hasattr(self.plugin, "categories_dir") and self.plugin.categories_dir:
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

    async def rebuild_index(self, event: AstrMessageEvent):
        """重建索引命令，用于从旧版本迁移或修复索引。

        扫描 categories 目录中的所有图片文件，重新构建索引。
        """
        try:
            yield event.plain_result("🔄 开始重建索引，请稍候...")

            # 调用插件的重建索引方法
            rebuilt_index = await self.plugin._rebuild_index_from_files()

            if not rebuilt_index:
                yield event.plain_result(
                    "⚠️ 未找到可重建的图片文件。\n"
                    f"请确保 categories 目录中存在图片文件:\n"
                    f"{self.plugin.categories_dir}"
                )
                return

            # 获取旧索引进行对比（创建独立副本）
            old_index = await self.plugin._load_index()
            old_count = len(old_index)

            # 尝试加载旧版本遗留文件（Legacy Data）- 独立存储，不修改 old_index
            import json
            legacy_metadata_count = 0
            legacy_data_map = {}  # 独立存储 legacy 数据
            possible_legacy_paths = [
                self.plugin.base_dir / "index.json",
                self.plugin.base_dir / "image_index.json",
                self.plugin.base_dir / "cache" / "index.json",
                # 其他可能的路径
                Path("data/plugin_data/astrbot_plugin_stealer/index.json"),
                Path("data/plugin_data/astrbot_plugin_stealer/image_index.json"),
            ]

            for legacy_path in possible_legacy_paths:
                if legacy_path.exists():
                    try:
                        with open(legacy_path, encoding="utf-8") as f:
                            legacy_data = json.load(f)
                            if isinstance(legacy_data, dict):
                                legacy_data_map.update(legacy_data)
                                legacy_metadata_count += len(legacy_data)
                    except Exception:
                        pass

            # --- 智能合并逻辑开始 ---
            # 1. 建立哈希查找表，用于处理文件路径变更的情况
            # 合并 old_index 和 legacy_data_map 用于查找
            combined_index = {**old_index, **legacy_data_map}
            
            old_hash_map = {}
            for k, v in combined_index.items():
                if isinstance(v, dict) and v.get("hash"):
                    old_hash_map[v["hash"]] = v
            # 同时也建立文件名->数据映射（处理哈希可能变化但文件名没变的情况）
            old_name_map = {}
            for k, v in combined_index.items():
                if isinstance(v, dict):
                     path_obj = Path(k)
                     old_name_map[path_obj.name] = v
                     # 同时也用纯文件名（不带扩展名）建立映射
                     old_name_map[path_obj.stem] = v

            recovered_count = 0

            # 2. 遍历重建的索引，尝试恢复元数据
            for new_path, new_data in rebuilt_index.items():
                old_data = None
                new_path_obj = Path(new_path)

                # 优先级1: 路径直接匹配
                if new_path in combined_index:
                    old_data = combined_index[new_path]
                # 优先级2: 哈希匹配（最可靠）
                elif new_data.get("hash") in old_hash_map:
                    old_data = old_hash_map[new_data["hash"]]
                # 优先级3: 文件名匹配（尝试多种格式）
                elif new_path_obj.name in old_name_map:
                    old_data = old_name_map[new_path_obj.name]
                elif new_path_obj.stem in old_name_map:
                    old_data = old_name_map[new_path_obj.stem]
                # 优先级4: 尝试从路径中提取文件名后匹配
                else:
                    for old_path, old_val in combined_index.items():
                        if isinstance(old_val, dict):
                            old_path_obj = Path(old_path)
                            # 比较文件名（忽略大小写扩展名）
                            if old_path_obj.stem.lower() == new_path_obj.stem.lower():
                                old_data = old_val
                                break

                # 如果找到了旧数据，恢复关键元数据
                if old_data and isinstance(old_data, dict):
                    # 恢复描述和标签
                    if old_data.get("desc"):
                        new_data["desc"] = old_data["desc"]
                    if old_data.get("tags"):
                        new_data["tags"] = old_data["tags"]
                    # 兼容可能存在的其他字段
                    if "source_message" in old_data:
                        new_data["source_message"] = old_data["source_message"]

                    recovered_count += 1

            # 3. 使用新的索引作为最终索引（自动清理了不存在的文件记录）
            final_index = rebuilt_index
            # --- 智能合并逻辑结束 ---

            # 保存合并后的索引
            await self.plugin._save_index(final_index)

            # 统计信息
            new_count = len(final_index)

            # 按分类统计
            category_stats = {}
            for img_info in final_index.values():
                if isinstance(img_info, dict):
                    cat = img_info.get("category", "未分类")
                    category_stats[cat] = category_stats.get(cat, 0) + 1

            # 构建结果消息
            result_msg = "✅ 索引重建完成！\n\n"
            result_msg += "📊 统计信息:\n"
            result_msg += f"  当前索引数量: {old_count}\n"
            if legacy_metadata_count > 0:
                result_msg += f"  旧版备份数据: {legacy_metadata_count} 条\n"
            result_msg += f"  现有文件数: {new_count}\n"
            result_msg += f"  已恢复元数据: {recovered_count} 条\n"

            if category_stats:
                result_msg += "\n📂 分类统计:\n"
                for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
                    result_msg += f"  {cat}: {count}张\n"

            yield event.plain_result(result_msg)

        except Exception as e:
            logger.error(f"重建索引失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 重建索引失败: {str(e)}")



