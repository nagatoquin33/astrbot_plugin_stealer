from collections import Counter
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class CommandHandler:
    """命令处理服务类，负责处理所有与插件相关的命令操作。"""

    def __init__(self, plugin_instance: Any):
        """初始化命令处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self._cleaned = False  # 清理标志位

    def _apply_config_updates(self, updates: dict) -> None:
        self.plugin._update_config_from_dict(updates)

    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        self._apply_config_updates({"steal_meme": True})
        yield event.plain_result("已开启偷表情包")

    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        self._apply_config_updates({"steal_meme": False})
        yield event.plain_result("已关闭偷表情包")

    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        self._apply_config_updates({"auto_send_meme": True})
        yield event.plain_result("已开启自动发送")

    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        self._apply_config_updates({"auto_send_meme": False})
        yield event.plain_result("已关闭自动发送")

    # ===== 门面委托：子服务方法 =====

    async def group_filter(self, event, scope="", list_name="", action="", target="", target_id=""):
        """目标过滤白/黑名单 CRUD（已迁移到 TargetFilterCommand）。"""
        from .target_filter_command import TargetFilterCommand

        async for result in TargetFilterCommand(self.plugin).group_filter(
            event, scope, list_name, action, target, target_id
        ):
            yield result

    async def list_images(self, event, category="", limit="10", page="1"):
        """列出表情包（已迁移到 ImageManagementCommand）。"""
        from .image_mgmt_command import ImageManagementCommand

        async for result in ImageManagementCommand(self.plugin).list_images(
            event, category, limit, page
        ):
            yield result

    async def delete_image(self, event, identifier=""):
        """删除表情包（已迁移到 ImageManagementCommand）。"""
        from .image_mgmt_command import ImageManagementCommand

        async for result in ImageManagementCommand(self.plugin).delete_image(event, identifier):
            yield result

    async def blacklist_image(self, event, identifier=""):
        """拉黑表情包（已迁移到 ImageManagementCommand）。"""
        from .image_mgmt_command import ImageManagementCommand

        async for result in ImageManagementCommand(self.plugin).blacklist_image(event, identifier):
            yield result

    async def set_image_scope(self, event, identifier="", scope_mode=""):
        """设置表情包作用域（已迁移到 ImageManagementCommand）。"""
        from .image_mgmt_command import ImageManagementCommand

        async for result in ImageManagementCommand(self.plugin).set_image_scope(
            event, identifier, scope_mode
        ):
            yield result

    async def rebuild_index(self, event):
        """重建索引（已迁移到 IndexRebuildCommand）。"""
        from .index_rebuild_command import IndexRebuildCommand

        async for result in IndexRebuildCommand(self.plugin).rebuild_index(event):
            yield result

    # ===== CommandHandler 核心逻辑 =====

    async def capture(self, event: AstrMessageEvent):
        window_seconds = 30

        if hasattr(self.plugin, "begin_force_capture"):
            self.plugin.begin_force_capture(event, window_seconds)
            yield event.plain_result(
                f"✅ 已进入强制接收窗口：{window_seconds} 秒内发送 1 张图片将自动分类并入库"
            )
            return

        yield event.plain_result("❌ 插件未初始化强制接收能力")

    async def toggle_natural_analysis(self, event: AstrMessageEvent, action: str = ""):
        """启用/禁用自然语言情绪分析。"""
        if action not in ["on", "off"]:
            current_status = "启用" if self.plugin.enable_natural_emotion_analysis else "禁用"
            yield event.plain_result(
                f"当前自然语言分析状态: {current_status}\n用法: /meme natural_analysis <on|off>"
            )
            return

        if action == "on":
            self._apply_config_updates({"enable_natural_emotion_analysis": True})
            yield event.plain_result(
                "✅ 已启用自然语言情绪分析（LLM模式）\n\n💡 提示：如果之前使用被动标签模式，建议使用 /reset 清除AI对话上下文，避免继续输出 &&emotion&& 标签"
            )
        else:
            self._apply_config_updates({"enable_natural_emotion_analysis": False})
            yield event.plain_result(
                "❌ 已禁用自然语言情绪分析（被动标签模式）\n\n💡 提示：LLM现在会在回复开头插入 &&emotion&& 标签，插件会自动清理这些标签"
            )

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
            status_text += f"自动发送: {'启用' if self.plugin.auto_send_meme else '禁用'}\n"
            status_text += (
                f"分析模型: {self.plugin.emotion_analysis_provider_id or '使用当前会话模型'}\n"
            )

            yield event.plain_result(status_text)
        except Exception as e:
            yield event.plain_result(f"获取统计信息失败: {e}")

    async def clear_emotion_cache(self, event: AstrMessageEvent):
        """清空情绪分析缓存。"""
        try:
            await self.plugin.smart_emotion_matcher.clear_cache()
            yield event.plain_result("✅ 情绪分析缓存已清空")
        except Exception as e:
            yield event.plain_result(f"❌ 清空缓存失败: {e}")

    async def status(self, event: AstrMessageEvent):
        """显示插件状态和详细的表情包统计信息。"""
        stealing_status = "开启" if self.plugin.steal_meme else "关闭"
        auto_send_meme_status = "开启" if self.plugin.auto_send_meme else "关闭"

        image_index = await self.plugin._load_index()
        total_count = len(image_index)

        # 添加视觉模型信息
        vision_model = self.plugin.vision_provider_id or "未设置（将使用当前会话默认模型）"

        # 基础状态信息
        steal_mode = self.plugin.steal_mode
        if steal_mode == "probability":
            mode_desc = f"概率模式 (概率={self.plugin.steal_chance})"
        else:
            mode_desc = f"冷却模式 (冷却={self.plugin.image_processing_cooldown}秒)"

        status_text = "🔧 插件状态:\n"
        status_text += f"偷取: {stealing_status}\n"
        status_text += f"偷图模式: {mode_desc}\n"
        status_text += f"自动发送: {auto_send_meme_status}\n"
        status_text += f"发送概率: {self.plugin.meme_chance}\n"
        status_text += f"审核: {self.plugin.content_filtration}\n"
        status_text += f"视觉模型: {vision_model}\n\n"

        # 后台任务状态
        status_text += "⚙️ 后台任务:\n"
        status_text += "Raw清理: 自动 (30min)\n"
        status_text += "容量控制: 自动 (60min)\n\n"

        # 表情包统计信息
        if total_count == 0:
            status_text += "📊 表情包统计:\n暂无表情包数据"
        else:
            # 按分类统计
            category_stats = Counter(
                img_info.get("category", "未分类")
                for img_info in image_index.values()
                if isinstance(img_info, dict)
            )

            # 构建统计信息
            status_text += "📊 表情包统计:\n"
            status_text += f"总数量: {total_count}/{self.plugin.max_reg_num} ({total_count / self.plugin.max_reg_num * 100:.1f}%)\n\n"

            # 分类统计 - 只显示前5个最多的分类
            status_text += "📂 分类统计 (前5):\n"
            sorted_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories[:5]:
                percentage = count / total_count * 100
                status_text += f"  {category}: {count}张 ({percentage:.1f}%)\n"

            if len(sorted_categories) > 5:
                status_text += f"  ...还有{len(sorted_categories) - 5}个分类\n"

            # 存储统计
            raw_count = (
                len(list(self.plugin.raw_dir.glob("*"))) if self.plugin.raw_dir.exists() else 0
            )
            status_text += "\n💾 存储信息:\n"
            status_text += f"  原始图片: {raw_count}张 | 分类图片: {total_count}张"

        yield event.plain_result(status_text)

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
        event_handler = self.plugin.event_handler
        if event_handler is not None:
            return await event_handler._clean_raw_directory()
        return 0

    async def enforce_capacity(self, event: AstrMessageEvent):
        """手动执行容量控制，删除最旧的表情包以控制总数量。"""
        try:
            # 加载图片索引
            image_index = await self.plugin._load_index()

            current_count = len(image_index)
            max_count = self.plugin.max_reg_num

            if current_count <= max_count:
                yield event.plain_result(
                    f"当前表情包数量 {current_count} 未超过限制 {max_count}，无需清理"
                )
                return

            # 执行容量控制
            await self.plugin.event_handler._enforce_capacity(image_index)
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

    def cleanup(self):
        """清理资源。"""
        if self._cleaned:
            return
        self._cleaned = True
        # CommandHandler 主要是无状态的，清理插件引用即可
        self.plugin = None
