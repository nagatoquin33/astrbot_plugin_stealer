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
        self._cleaned = False  # 清理标志位

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
        scope: str = "",
        list_name: str = "",
        action: str = "",
        target: str = "",
        target_id: str = "",
    ):
        cfg = self.plugin.plugin_config
        if cfg is None:
            yield event.plain_result("配置服务不可用")
            return

        def format_items(items: list[str], *, max_items: int = 30) -> str:
            if not items:
                return "（空）"
            shown = items[:max_items]
            suffix = f" ... 还有 {len(items) - max_items} 项" if len(items) > max_items else ""
            return ", ".join(shown) + suffix

        def resolve_target(raw_target: str, raw_target_id: str) -> str:
            if raw_target and raw_target_id:
                lowered = raw_target.lower()
                if lowered in {"group", "g"}:
                    return cfg.normalize_target_entry(raw_target_id, "group")
                if lowered in {"user", "u", "qq"}:
                    return cfg.normalize_target_entry(raw_target_id, "user")

            combined = str(raw_target or raw_target_id or "").strip()
            if combined:
                return cfg.normalize_target_entry(combined, "group")

            event_scope, event_id = cfg.get_event_target(event)
            if event_scope and event_id:
                return f"{event_scope}:{event_id}"
            return ""

        raw_scope = (scope or "").strip().lower()
        raw_list_name = (list_name or "").strip().lower()
        raw_action = (action or "").strip().lower()

        if raw_scope in {"show", "list", "ls", "status"} and not raw_list_name:
            raw_action = raw_scope
            raw_scope = ""
        elif raw_scope in {"wl", "white", "whitelist", "bl", "black", "blacklist"} and not raw_action:
            raw_action = raw_list_name
            raw_list_name = raw_scope
            raw_scope = "send"

        if raw_action in {"", "help", "h"}:
            help_text = (
                "用法：\n"
                "/meme group show\n"
                "/meme group <send|steal> show\n"
                "/meme group <send|steal> <wl|bl> <add|del|clear> [group:ID|user:QQ]\n"
                "/meme group <send|steal> <wl|bl> <add|del> <group|user> <ID>\n\n"
                "说明：\n"
                "- send 控制发表情\n"
                "- steal 控制偷表情\n"
                "- 白名单非空时优先生效\n"
                "- 支持 group:123456 和 user:123456\n"
                "- 不填目标时默认使用当前会话目标"
            )
            yield event.plain_result(help_text)
            return

        if raw_action in {"show", "list", "ls", "status"}:
            if raw_scope in {"", "all"}:
                sections = []
                for action_name, title in (("send", "发表情"), ("steal", "偷表情")):
                    whitelist, blacklist = cfg._get_action_lists(action_name)
                    mode = "白名单" if whitelist else ("黑名单" if blacklist else "未启用")
                    sections.append(
                        f"{title}:\n"
                        f"- 模式：{mode}\n"
                        f"- 白名单({len(whitelist)})：{format_items(whitelist)}\n"
                        f"- 黑名单({len(blacklist)})：{format_items(blacklist)}"
                    )
                yield event.plain_result("\n\n".join(sections))
                return

            if raw_scope not in {"send", "steal"}:
                yield event.plain_result("目标类型无效，请使用 send 或 steal")
                return

            whitelist, blacklist = cfg._get_action_lists(raw_scope)
            mode = "白名单" if whitelist else ("黑名单" if blacklist else "未启用")
            title = "发表情" if raw_scope == "send" else "偷表情"
            yield event.plain_result(
                f"{title}:\n"
                f"- 模式：{mode}\n"
                f"- 白名单({len(whitelist)})：{format_items(whitelist)}\n"
                f"- 黑名单({len(blacklist)})：{format_items(blacklist)}"
            )
            return

        if raw_scope not in {"send", "steal"}:
            yield event.plain_result("目标类型无效，请使用 send 或 steal")
            return

        if raw_list_name in {"wl", "white", "whitelist"}:
            list_key = f"{raw_scope}_target_whitelist"
            list_title = "白名单"
        elif raw_list_name in {"bl", "black", "blacklist"}:
            list_key = f"{raw_scope}_target_blacklist"
            list_title = "黑名单"
        else:
            yield event.plain_result("名单类型无效，请使用 wl 或 bl")
            return

        current: list[str] = list(getattr(cfg, list_key, []) or [])
        current_set = set(current)

        if raw_action in {"clear", "reset"}:
            ok = bool(cfg.update_config({list_key: []}))
            scope_title = "发表情" if raw_scope == "send" else "偷表情"
            msg = f"已清空{scope_title}{list_title}" if ok else f"清空{scope_title}{list_title}失败"
            yield event.plain_result(msg)
            return

        normalized_target = resolve_target(target, target_id)
        if not normalized_target:
            yield event.plain_result("缺少目标，请使用 group:群号 或 user:QQ号")
            return

        if raw_action in {"add", "a", "append", "+"}:
            if normalized_target in current_set:
                yield event.plain_result(f"{normalized_target} 已在{list_title}中")
                return
            updated = current + [normalized_target]
            ok = bool(cfg.update_config({list_key: updated}))
            msg = f"已将 {normalized_target} 加入{list_title}" if ok else f"加入{list_title}失败：{normalized_target}"
            yield event.plain_result(msg)
            return

        if raw_action in {"del", "delete", "rm", "remove", "-"}:
            if normalized_target not in current_set:
                yield event.plain_result(f"{normalized_target} 不在{list_title}中")
                return
            updated = [item for item in current if item != normalized_target]
            ok = bool(cfg.update_config({list_key: updated}))
            msg = f"已将 {normalized_target} 移出{list_title}" if ok else f"移出{list_title}失败：{normalized_target}"
            yield event.plain_result(msg)
            return

        yield event.plain_result("操作无效，请使用 add / del / clear / show")

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
        if self._cleaned:
            return
        self._cleaned = True
        # CommandHandler 主要是无状态的，清理插件引用即可
        self.plugin = None
        logger.debug("CommandHandler 资源已清理")

    async def list_images(
        self,
        event: AstrMessageEvent,
        category: str = "",
        limit: str = "10",
        page: str = "1",
        show_images: bool = True,
    ):
        """列出表情包，支持按分类筛选。

        Args:
            event: 消息事件
            category: 可选的分类筛选
            limit: 显示数量限制，默认10张
            show_images: 是否显示图片，默认True
        """
        # 参数解析目标：
        # - /meme list            -> page=1, per_page=默认
        # - /meme list 2          -> page=2 (默认每页数量)
        # - /meme list happy 2    -> 分类=happy, page=2
        # - /meme list 20 2       -> per_page=20, page=2
        # - /meme list happy 20 2 -> 分类=happy, per_page=20, page=2
        category = str(category or "").strip()
        limit = str(limit or "").strip()
        page = str(page or "").strip()

        # 仅提供一个数字：视为翻页
        if category.isdigit() and (not limit or limit == "10") and (not page or page == "1"):
            page, category = category, ""
            limit = "10"
        # /meme list happy 2 -> 分类 + 页码
        elif category and (not category.isdigit()) and limit.isdigit() and (not page or page == "1"):
            page, limit = limit, "10"
        # /meme list 20 2 -> 每页数量 + 页码
        elif category.isdigit() and limit.isdigit() and (not page or page == "1"):
            page, limit, category = limit, category, ""

        # 解析每页数量
        try:
            per_page = int(limit)
        except Exception:
            per_page = 10
        per_page = max(1, min(20, per_page))

        # 解析页码
        try:
            page_num = int(page)
        except Exception:
            page_num = 1
        page_num = max(1, page_num)

        image_index = await self.plugin._load_index()

        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        # 收集所有有效图片（先不做分类过滤，保证序号与 /meme delete 的全局序号一致）
        all_images = []
        for img_path, img_info in image_index.items():
            if isinstance(img_info, dict):
                img_category = img_info.get("category", "未分类")
                img_desc = img_info.get("desc", "")
                img_source = img_info.get("source", "")
                img_pkg = img_info.get("qq_emoji_package_id", "")

                # 检查文件是否存在
                if not Path(img_path).exists():
                    continue

                all_images.append(
                    {
                        "path": img_path,
                        "name": Path(img_path).name,
                        "category": img_category,
                        "desc": str(img_desc or ""),
                        "source": str(img_source or ""),
                        "qq_emoji_package_id": str(img_pkg or ""),
                        "created_at": img_info.get("created_at", 0),
                    }
                )

        if not all_images:
            if category:
                yield event.plain_result(f"分类 '{category}' 中暂无表情包")
            else:
                yield event.plain_result("暂无有效的表情包文件")
            return

        # 按创建时间排序（最新的在前），并分配全局序号（与 delete 的序号一致）
        all_images.sort(key=lambda x: x["created_at"], reverse=True)
        for i, img in enumerate(all_images, 1):
            img["index"] = i

        # 分类过滤只影响展示与分页，不影响序号
        filtered_images = [
            img for img in all_images if (not category or img.get("category") == category)
        ]
        if not filtered_images:
            yield event.plain_result(f"分类 '{category}' 中暂无表情包")
            return

        total_filtered = len(filtered_images)
        total_all = len(all_images)
        total_pages = max(1, (total_filtered + per_page - 1) // per_page)
        if page_num > total_pages:
            page_num = total_pages

        start = (page_num - 1) * per_page
        display_images = filtered_images[start : start + per_page]

        if show_images and getattr(self.plugin, "image_processor_service", None):
            # 优先使用 AstrBot 内置 html-to-pic（网络 t2i），失败再回退到本地 PIL 渲染
            file_path = await self.plugin.image_processor_service.render_emoji_list_page_file(
                items=display_images,
                page=page_num,
                total_pages=total_pages,
                total_filtered=total_filtered,
                total_all=total_all,
                category=category,
                per_page=per_page,
            )
            if file_path:
                if event.get_platform_name() == "aiocqhttp":
                    # aiocqhttp 下尽量不要把图片交给 AstrBot 的 Image -> base64:// 转换链路，
                    # 否则很容易出现 1200/timeout 并在 respond.stage 打满堆栈。
                    if await self._try_send_onebot_image_auto(event, file_path):
                        yield event.make_result().stop_event()
                        return
                else:
                    yield event.make_result().file_image(file_path).stop_event()
                    return

            # 非 aiocqhttp 平台：回退到 base64 渲染
            if event.get_platform_name() != "aiocqhttp":
                b64 = await self.plugin.image_processor_service.render_emoji_list_page_base64(
                    items=display_images,
                    page=page_num,
                    total_pages=total_pages,
                    total_filtered=total_filtered,
                    total_all=total_all,
                    category=category,
                    per_page=per_page,
                )
                if b64:
                    yield event.make_result().base64_image(b64).stop_event()
                    return

        # 纯文本 fallback（或渲染失败）
        title = f"表情包列表 ({page_num}/{total_pages}) ({len(display_images)}/{total_filtered}) (总 {total_all})"
        if category:
            title += f" - 分类: {category}"

        result_text = title + "\n\n"
        for img in display_images:
            idx = int(img.get("index", 0) or 0)
            desc = str(img.get("desc", "") or "").strip()
            if not desc:
                desc = str(img.get("name", "") or "")
            if len(desc) > 28:
                desc = desc[:25] + "..."
            result_text += f"{idx:4d}. {desc}\n"

        result_text += "\n用法: /meme list [分类] [每页数量] [页码]\n删除: /meme delete <序号>"
        yield event.plain_result(result_text).stop_event()

    async def _try_send_onebot_image_auto(
        self,
        event: AstrMessageEvent,
        image_path: str,
    ) -> bool:
        """aiocqhttp/OneBot 下尽量用 bot.call_action 直发，避免进入 respond.stage 的 Image->base64 发送链路。

        策略：
        1) file:// 物理路径发送（更快，且可绕过 base64 体积/超时问题）
        2) 若路径发送失败，则回退 base64:// 直传
        """
        try:
            if not image_path:
                return False
            if event.get_platform_name() != "aiocqhttp":
                return False
            bot = getattr(event, "bot", None)
            if bot is None or not hasattr(bot, "call_action"):
                return False

            # 尝试修复权限：NapCat 会把源文件 copy 到自身 temp 目录，源文件不可读/目标目录不可写都会触发 EACCES
            try:
                os.chmod(image_path, 0o644)
            except Exception:
                pass

            # 构造 OneBot 认可的 file:// 路径
            file_str = image_path
            if not str(image_path).startswith(("http://", "https://", "base64://", "file://")):
                if os.path.isabs(image_path):
                    if image_path.startswith("/"):
                        file_str = f"file://{image_path}"
                    else:
                        file_str = f"file:///{image_path}"
                else:
                    file_str = f"file:///{os.path.abspath(image_path)}"

            message = [{"type": "image", "data": {"file": file_str}}]
            session_id = event.get_session_id()
            if not session_id or not str(session_id).isdigit():
                return False

            async def _send_once() -> None:
                if event.is_private_chat():
                    await bot.call_action(
                        "send_private_msg",
                        user_id=int(session_id),
                        message=message,
                    )
                else:
                    await bot.call_action(
                        "send_group_msg",
                        group_id=int(session_id),
                        message=message,
                    )

            try:
                await _send_once()
                return True
            except Exception as e:
                err = str(e)
                # 权限错误：尝试给 NapCat temp 目录放宽权限后重试一次
                if ("EACCES" in err or "permission denied" in err.lower()) and "/app/.config/QQ/NapCat/temp" in err:
                    try:
                        os.chmod("/app/.config/QQ/NapCat/temp", 0o777)
                    except Exception:
                        pass
                    try:
                        await _send_once()
                        return True
                    except Exception as e2:
                        logger.debug(f"OneBot file:// 重试仍失败: {e2}")
                        # fallthrough to base64 fallback
                else:
                    logger.debug(f"OneBot file:// 发送图片失败: {e}")

            # file:// 失败，回退 base64:// 直传（仍然用 call_action，避免 respond.stage 堆栈）
            try:
                with open(image_path, "rb") as f:
                    raw = f.read()
                b64 = base64.b64encode(raw).decode("utf-8")
                msg2 = [{"type": "image", "data": {"file": f"base64://{b64}"}}]
                if event.is_private_chat():
                    await bot.call_action(
                        "send_private_msg",
                        user_id=int(session_id),
                        message=msg2,
                    )
                else:
                    await bot.call_action(
                        "send_group_msg",
                        group_id=int(session_id),
                        message=msg2,
                    )
                return True
            except Exception as e3:
                logger.debug(f"OneBot base64:// 直传也失败: {e3}")
                return False
        except Exception as e:
            logger.debug(f"OneBot 直发图片失败: {e}")
            return False

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
