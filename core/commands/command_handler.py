import asyncio
from typing import Any

from astrbot.api import logger

from astrbot.api.event import AstrMessageEvent

from ..maintenance.retention import library_counts


class CommandHandler:
    """命令处理服务类，负责处理所有与插件相关的命令操作。"""

    def __init__(self, plugin_instance: Any):
        """初始化命令处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self._cleaned = False  # 清理标志位

    def _apply_config_updates(self, updates: dict) -> bool:
        try:
            return bool(self.plugin.update_config(updates))
        except Exception as exc:
            logger.error(f"指令更新配置失败: {exc}")
            return False

    async def set_switch(self, event, key: str, enabled: bool, label: str):
        saved = self._apply_config_updates({key: enabled})
        action = "开启" if enabled else "关闭"
        yield event.plain_result(f"已{action}{label}" if saved else f"❌ {action}{label}失败：配置未保存")

    def meme_on(self, event):
        return self.set_switch(event, "steal_meme", True, "偷表情包")

    def meme_off(self, event):
        return self.set_switch(event, "steal_meme", False, "偷表情包")

    def auto_on(self, event):
        return self.set_switch(event, "auto_send_meme", True, "自动发送")

    def auto_off(self, event):
        return self.set_switch(event, "auto_send_meme", False, "自动发送")

    async def capture(self, event: AstrMessageEvent):
        window_seconds = 30

        self.plugin.begin_force_capture(event, window_seconds)
        yield event.plain_result(
            f"✅ 已进入强制接收窗口：{window_seconds} 秒内发送 1 张图片将自动分类并入库"
        )

    async def toggle_natural_analysis(self, event: AstrMessageEvent, action: str = ""):
        """启用/禁用自然语言情绪分析。"""
        action = str(action or "").strip().lower()
        if action not in ["on", "off"]:
            current_status = "启用" if self.plugin.plugin_config.enable_natural_emotion_analysis else "禁用"
            yield event.plain_result(
                f"当前智能检索状态: {current_status}\n用法: /meme natural_analysis <on|off>"
            )
            return

        enabled = action == "on"
        async for result in self.set_switch(event, "enable_natural_emotion_analysis", enabled, "智能检索"):
            yield result

    async def emotion_analysis_stats(self, event: AstrMessageEvent):
        """显示当前匹配模式和轻量模型的关键统计。"""
        cfg = self.plugin.plugin_config
        lines = [f"表情匹配：{'智能检索' if cfg.enable_natural_emotion_analysis else '原文检索'}",
                 f"自动发送：{'开启' if cfg.auto_send_meme else '关闭'}"]
        if cfg.enable_natural_emotion_analysis:
            stats = self.plugin.emotion_analyzer.get_stats()
            if "message" in stats:
                lines.append(str(stats["message"]))
            else:
                lines.extend(f"{label}：{stats[key]}" for label, key in (
                    ("分析次数", "total_analyses"), ("缓存命中", "cache_hit_rate"),
                    ("成功率", "success_rate"), ("平均响应", "avg_response_time"),
                ))
        yield event.plain_result("\n".join(lines))

    async def clear_emotion_cache(self, event: AstrMessageEvent):
        """清空情绪分析缓存。"""
        try:
            await self.plugin.emotion_analyzer.clear_cache()
            yield event.plain_result("✅ 情绪分析缓存已清空")
        except Exception as e:
            yield event.plain_result(f"❌ 清空缓存失败: {e}")

    async def status(self, event: AstrMessageEvent):
        """显示运行状态和数据库聚合统计。"""
        cfg = self.plugin.plugin_config
        categories, counts = await asyncio.gather(
            asyncio.to_thread(self.plugin.db_service.get_category_counts),
            asyncio.to_thread(self.plugin.db_service.get_library_counts),
        )
        total = sum(categories.values())
        lines = [
            f"偷图：{'开启' if cfg.steal_meme else '关闭'}｜自动发送：{'开启' if cfg.auto_send_meme else '关闭'}",
            f"表情：{total}｜收藏：{counts['favorites']}｜角色：{counts['characters']}",
            f"自动淘汰计数：{counts['automatic']}/{cfg.max_reg_num}",
        ]
        if categories:
            top = sorted(categories.items(), key=lambda item: item[1], reverse=True)[:5]
            lines.append("分类：" + "、".join(f"{name} {count}" for name, count in top))
        else:
            lines.append("暂无表情包数据")
        yield event.plain_result("\n".join(lines))

    async def tag_stats(self, event: AstrMessageEvent, limit: str = ""):
        """标签/场景统计：高频标签、低频标签、零标签条目。

        用法: /meme tag_stats [N]   （N 为 Top 数量，默认 15）
        """
        try:
            db = getattr(self.plugin, "db_service", None)
            if db is None or not hasattr(db, "get_tag_stats"):
                yield event.plain_result("❌ 数据库服务不可用")
                return

            top_n = 15
            try:
                parsed = int(str(limit or "").strip())
                if 1 <= parsed <= 50:
                    top_n = parsed
            except ValueError:
                pass

            # 在线程池执行同步 SQLite 查询，避免阻塞事件循环。
            stats = await asyncio.to_thread(db.get_tag_stats, top_n)

            lines: list[str] = []
            lines.append("🏷️ 标签统计（打标质量体检）:")
            lines.append(
                f"总表情: {stats['total_emojis']} | 有标签: {stats['total_with_tags']} | "
                f"无标签: {stats['zero_tag_count']}"
            )
            lines.append(f"\n📈 高频标签 Top {top_n}:")
            if stats["top_tags"]:
                for i, item in enumerate(stats["top_tags"], 1):
                    lines.append(f"  {i}. {item['tag']} ×{item['count']}")
            else:
                lines.append("  （暂无标签）")

            if stats["single_use_tags"]:
                lines.append("\n🔍 低频标签（仅出现 1 次，疑似噪声/同义词，可考虑合并）:")
                shown = "、".join(stats["single_use_tags"][:10])
                more = f" 等共 {len(stats['single_use_tags'])} 个" if len(stats["single_use_tags"]) > 10 else ""
                lines.append(f"  {shown}{more}")

            if stats["top_scenes"]:
                lines.append(f"\n🎬 高频场景 Top {min(5, len(stats['top_scenes']))}:")
                for i, item in enumerate(stats["top_scenes"][:5], 1):
                    lines.append(f"  {i}. {item['scene']} ×{item['count']}")

            lines.append("\n💡 提示: 低频标签多说明 VLM 输出不稳定，可在审核时统一措辞")

            yield event.plain_result("\n".join(lines))
        except Exception as e:
            logger.error(f"标签统计失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 标签统计失败: {e}")

    async def clean(self, event: AstrMessageEvent, mode: str = ""):
        """手动触发清理操作，清理raw目录中的原始图片文件，不影响已分类的表情包。

        Args:
            event: 消息事件
            mode: 留空或 raw，均只清理 raw 原始图片
        """
        if str(mode or "").strip().lower() not in {"", "raw"}:
            yield event.plain_result("用法: /meme clean [raw]；仅清理 raw 原始图片")
            return
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
        """手动执行容量控制，按加权评分控制通用库数量。"""
        try:
            # 加载图片索引
            image_index = await self.plugin.index_manager.load_index()

            current_count = library_counts(image_index)["automatic"]
            max_count = self.plugin.plugin_config.max_reg_num

            if current_count <= max_count:
                yield event.plain_result(
                    f"当前通用库自动淘汰计数 {current_count} 未超过限制 {max_count}，无需清理"
                )
                return

            # 执行容量控制
            await self.plugin.event_handler._enforce_capacity(image_index)
            await self.plugin.index_manager.save_index(image_index)

            # 重新统计
            new_count = library_counts(image_index)["automatic"]
            removed_count = max(0, current_count - new_count)

            yield event.plain_result(
                f"容量控制完成\n"
                f"删除了 {removed_count} 个通用表情包\n"
                f"通用库自动淘汰计数: {new_count}/{max_count}（收藏和角色库另行管理）"
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
