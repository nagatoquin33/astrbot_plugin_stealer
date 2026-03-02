import json
import os
import random
from collections import Counter
from pathlib import Path
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

    def _apply_config_updates(self, updates: dict) -> None:
        self.plugin._update_config_from_dict(updates)

    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        self._apply_config_updates({"steal_emoji": True})
        yield event.plain_result("已开启偷表情包")

    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        self._apply_config_updates({"steal_emoji": False})
        yield event.plain_result("已关闭偷表情包")

    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        self._apply_config_updates({"auto_send": True})
        yield event.plain_result("已开启自动发送")

    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        self._apply_config_updates({"auto_send": False})
        yield event.plain_result("已关闭自动发送")

    async def group_filter(
        self,
        event: AstrMessageEvent,
        list_name: str = "",
        action: str = "",
        group_id: str = "",
    ):
        """管理群聊黑白名单。

        用法:
        - /meme group show
        - /meme group wl add [群号]
        - /meme group wl del [群号]
        - /meme group wl clear
        - /meme group bl add [群号]
        - /meme group bl del [群号]
        - /meme group bl clear
        """

        def normalize_gid(value: object) -> str:
            s = str(value).strip()
            return s

        def get_event_group_id() -> str:
            try:
                gid = event.get_group_id()
            except Exception:
                gid = ""
            return normalize_gid(gid)

        def format_id_list(ids: list[str], *, max_items: int = 30) -> str:
            if not ids:
                return "（空）"
            shown = ids[:max_items]
            suffix = (
                f" … 还有 {len(ids) - max_items} 个" if len(ids) > max_items else ""
            )
            return ", ".join(shown) + suffix

        raw_list_name = (list_name or "").strip().lower()
        raw_action = (action or "").strip().lower()

        if raw_list_name in {"show", "list", "ls", "status"} and not raw_action:
            raw_action = raw_list_name
            raw_list_name = ""

        if raw_action in {"", "help", "h"}:
            yield event.plain_result(
                "用法:\n"
                "/meme group show\n"
                "/meme group wl add [群号]\n"
                "/meme group wl del [群号]\n"
                "/meme group wl clear\n"
                "/meme group bl add [群号]\n"
                "/meme group bl del [群号]\n"
                "/meme group bl clear\n\n"
                "说明:\n"
                "- whitelist( wl ) 非空时优先生效：只有在白名单内的群才会偷/发\n"
                "- whitelist 为空时使用 blacklist( bl )：黑名单内的群不会偷/发\n"
                "- 未提供群号时默认使用当前群号"
            )
            return

        if raw_action in {"show", "list", "ls", "status"}:
            cfg = self.plugin.plugin_config
            if cfg is None:
                yield event.plain_result("❌ 配置服务不可用")
                return
            whitelist = list(cfg.group_whitelist or [])
            blacklist = list(cfg.group_blacklist or [])
            mode = "白名单优先" if whitelist else ("黑名单" if blacklist else "未启用")
            yield event.plain_result(
                "群聊过滤状态:\n"
                f"- 模式: {mode}\n"
                f"- 白名单({len(whitelist)}): {format_id_list(whitelist)}\n"
                f"- 黑名单({len(blacklist)}): {format_id_list(blacklist)}"
            )
            return

        if raw_list_name in {"wl", "white", "whitelist", "白", "白名单"}:
            list_key = "group_whitelist"
            list_title = "白名单"
        elif raw_list_name in {"bl", "black", "blacklist", "黑", "黑名单"}:
            list_key = "group_blacklist"
            list_title = "黑名单"
        else:
            yield event.plain_result(
                "❌ 参数错误：请使用 wl 或 bl，或使用 /meme group help"
            )
            return

        cfg = self.plugin.plugin_config
        if cfg is None:
            yield event.plain_result("❌ 配置服务不可用")
            return

        current: list[str] = list(getattr(cfg, list_key, []) or [])
        current_set = set(current)

        if raw_action in {"clear", "reset"}:
            updated = []
            ok = bool(cfg.update_config({list_key: updated}))
            if ok:
                yield event.plain_result(f"✅ 已清空群聊{list_title}")
            else:
                yield event.plain_result(f"❌ 清空群聊{list_title}失败")
            return

        gid = normalize_gid(group_id) if group_id else get_event_group_id()
        if not gid:
            yield event.plain_result("❌ 未获取到群号，请在群聊中执行或手动提供群号")
            return

        if raw_action in {"add", "a", "append", "+"}:
            if gid in current_set:
                yield event.plain_result(f"ℹ️ 群 {gid} 已在{list_title}中")
                return
            updated = current + [gid]
            ok = bool(cfg.update_config({list_key: updated}))
            if ok:
                yield event.plain_result(f"✅ 已添加群 {gid} 到{list_title}")
            else:
                yield event.plain_result(f"❌ 添加群 {gid} 到{list_title}失败")
            return

        if raw_action in {"del", "delete", "rm", "remove", "-"}:
            if gid not in current_set:
                yield event.plain_result(f"ℹ️ 群 {gid} 不在{list_title}中")
                return
            updated = [x for x in current if x != gid]
            ok = bool(cfg.update_config({list_key: updated}))
            if ok:
                yield event.plain_result(f"✅ 已从{list_title}移除群 {gid}")
            else:
                yield event.plain_result(f"❌ 从{list_title}移除群 {gid}失败")
            return

        yield event.plain_result(
            "❌ 参数错误：请使用 add/del/clear/show，或 /meme group help"
        )

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
            current_status = (
                "启用" if self.plugin.enable_natural_emotion_analysis else "禁用"
            )
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
            mode = (
                "智能模式"
                if self.plugin.enable_natural_emotion_analysis
                else "被动模式"
            )

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
            await self.plugin.smart_emotion_matcher.clear_cache()
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
        steal_mode = self.plugin.steal_mode
        if steal_mode == "probability":
            mode_desc = f"概率模式 (概率={self.plugin.steal_chance})"
        else:
            mode_desc = f"冷却模式 (冷却={self.plugin.image_processing_cooldown}秒)"

        status_text = "🔧 插件状态:\n"
        status_text += f"偷取: {stealing_status}\n"
        status_text += f"偷图模式: {mode_desc}\n"
        status_text += f"自动发送: {auto_send_status}\n"
        status_text += f"发送概率: {self.plugin.emoji_chance}\n"
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
            sorted_categories = sorted(
                category_stats.items(), key=lambda x: x[1], reverse=True
            )
            for category, count in sorted_categories[:5]:
                percentage = count / total_count * 100
                status_text += f"  {category}: {count}张 ({percentage:.1f}%)\n"

            if len(sorted_categories) > 5:
                status_text += f"  ...还有{len(sorted_categories) - 5}个分类\n"

            # 存储统计
            raw_count = (
                len(list(self.plugin.raw_dir.glob("*")))
                if self.plugin.raw_dir.exists()
                else 0
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
            yield event.plain_result(
                f"✅ raw目录清理完成，共删除 {deleted_count} 张原始图片"
            )
        except Exception as e:
            logger.error(f"手动清理失败: {e}")
            yield event.plain_result(f"❌ 清理失败: {str(e)}")

    async def _force_clean_raw_directory(self) -> int:
        """强制清理raw目录中的所有文件（忽略保留期限），返回删除的文件数量。"""
        if hasattr(self.plugin, "_clean_raw_directory"):
            return await self.plugin._clean_raw_directory()
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

    def cleanup(self):
        """清理资源。"""
        # CommandHandler 主要是无状态的，清理插件引用即可
        self.plugin = None
        logger.debug("CommandHandler 资源已清理")

    async def list_images(
        self,
        event: AstrMessageEvent,
        category: str = "",
        limit: str = "10",
        show_images: bool = True,
    ):
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

                filtered_images.append(
                    {
                        "path": img_path,
                        "name": Path(img_path).name,
                        "category": img_category,
                        "created_at": img_info.get("created_at", 0),
                    }
                )

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
            yield event.plain_result(
                title + "\n💡 使用 /meme delete <序号> 删除指定图片"
            )

            # 逐个发送图片和信息
            for i, img in enumerate(display_images, 1):
                try:
                    # 读取图片并转换为base64
                    b64 = await self.plugin.image_processor_service._file_to_gif_base64(
                        img["path"]
                    )

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
                yield event.plain_result(
                    f"...还有 {len(filtered_images) - max_limit} 张图片"
                )
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
                valid_images.append(
                    {
                        "path": img_path,
                        "name": Path(img_path).name,
                        "category": img_info.get("category", "未分类"),
                        "created_at": img_info.get("created_at", 0),
                    }
                )

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
                f"未找到图片: {identifier}\n请使用 /meme list 查看可用的图片列表"
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
                await self.plugin._safe_remove_file(img_path)
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
                            await self.plugin._safe_remove_file(str(category_file))
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
            legacy_metadata_count = 0
            legacy_data_map = {}  # 独立存储 legacy 数据
            possible_legacy_paths = []
            if self.plugin.base_dir:
                possible_legacy_paths.extend(
                    [
                        self.plugin.base_dir / "index.json",
                        self.plugin.base_dir / "image_index.json",
                        self.plugin.base_dir / "cache" / "index.json",
                    ]
                )

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
            category_stats = Counter(
                img_info.get("category", "未分类")
                for img_info in final_index.values()
                if isinstance(img_info, dict)
            )

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
                for cat, count in sorted(
                    category_stats.items(), key=lambda x: x[1], reverse=True
                ):
                    result_msg += f"  {cat}: {count}张\n"

            yield event.plain_result(result_msg)

        except Exception as e:
            logger.error(f"重建索引失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 重建索引失败: {str(e)}")

    async def get_emoji_count(self) -> int:
        """获取表情包总数。

        Returns:
            int: 表情包数量
        """
        idx = await self.plugin.cache_service.load_index()
        return len(idx)

    async def get_emoji_info(self) -> dict:
        """获取表情包信息。

        Returns:
            dict: 包含当前数量、最大数量等信息
        """
        idx = await self.plugin.cache_service.load_index()
        return {
            "current_count": len(idx),
            "max_count": self.plugin.max_reg_num,
            "available_emojis": len(idx),
        }

    async def get_available_emotions(self) -> list[str]:
        """获取所有可用的情绪分类。

        Returns:
            list[str]: 情绪分类列表
        """
        idx = await self.plugin.cache_service.load_index()
        s = set()
        for v in idx.values():
            if isinstance(v, dict):
                # 使用 category 字段（新版本索引结构）
                cat = v.get("category")
                if isinstance(cat, str) and cat:
                    s.add(cat)
        return sorted(s)

    async def get_all_descriptions(self) -> list[str]:
        """获取所有表情包描述。

        Returns:
            list[str]: 描述列表
        """
        image_index = await self.plugin.cache_service.load_index()
        descriptions = []
        for record in image_index.values():
            if isinstance(record, dict):
                description = record.get("desc")
                if isinstance(description, str) and description:
                    descriptions.append(description)
        return descriptions

    async def load_all_emoji_records(self) -> list[tuple[str, dict]]:
        """加载所有有效的表情包记录。

        Returns:
            list[tuple[str, dict]]: (路径, 记录字典) 的列表
        """
        idx = await self.plugin.cache_service.load_index()
        return [
            (k, v) for k, v in idx.items() if isinstance(v, dict) and os.path.exists(k)
        ]

    async def get_random_emojis(
        self, count: int | None = 1
    ) -> list[tuple[str, str, str]]:
        """获取随机表情包。

        Args:
            count: 数量

        Returns:
            list[tuple[str, str, str]]: (路径, 描述, 情绪) 的列表
        """
        all_records = await self.load_all_emoji_records()
        if not all_records:
            return []
        sample_count = max(1, int(count or 1))
        picked_records = random.sample(all_records, min(sample_count, len(all_records)))
        return [self._record_to_tuple(path, rec) for path, rec in picked_records]

    async def get_emoji_by_emotion(self, emotion: str) -> tuple[str, str, str] | None:
        """根据情绪获取表情包。

        Args:
            emotion: 情绪标签

        Returns:
            tuple[str, str, str] | None: (路径, 描述, 情绪) 或 None
        """
        all_records = await self.load_all_emoji_records()
        candidates = []
        for image_path, record_dict in all_records:
            record_emotion = str(
                record_dict.get("emotion", record_dict.get("category", ""))
            )
            record_tags = record_dict.get("tags", [])
            if emotion and (
                emotion == record_emotion
                or (
                    isinstance(record_tags, list)
                    and emotion in [str(tag) for tag in record_tags]
                )
            ):
                candidates.append((image_path, record_dict))
        if not candidates:
            return None
        picked_path, picked_record = random.choice(candidates)
        return self._record_to_tuple(picked_path, picked_record)

    async def get_emoji_by_description(
        self, description: str
    ) -> tuple[str, str, str] | None:
        """根据描述获取表情包。

        Args:
            description: 描述文本

        Returns:
            tuple[str, str, str] | None: (路径, 描述, 情绪) 或 None
        """
        all_records = await self.load_all_emoji_records()
        candidates = []
        for image_path, record_dict in all_records:
            desc_text = str(record_dict.get("desc", ""))
            if description and description in desc_text:
                candidates.append((image_path, record_dict))
        if not candidates:
            for image_path, record_dict in all_records:
                record_tags = record_dict.get("tags", [])
                if isinstance(record_tags, list):
                    if any(str(description) in str(tag) for tag in record_tags):
                        candidates.append((image_path, record_dict))
        if not candidates:
            return None
        picked_path, picked_record = random.choice(candidates)
        return self._record_to_tuple(picked_path, picked_record)

    def _record_to_tuple(
        self, image_path: str, record_dict: dict
    ) -> tuple[str, str, str]:
        """将索引记录转换为 (路径, 描述, 情绪) 元组。"""
        description = str(record_dict.get("desc", ""))
        emotion = str(
            record_dict.get(
                "emotion",
                record_dict.get(
                    "category",
                    self.plugin.categories[0] if self.plugin.categories else "开心",
                ),
            )
        )
        return (image_path, description, emotion)
