"""目标过滤命令：负责白/黑名单的 CRUD 操作。"""

from typing import Any

from astrbot.api.event import AstrMessageEvent


class TargetFilterCommand:
    """负责目标过滤白/黑名单的增删改查。"""

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance

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
            lowered = str(raw_target or "").strip().lower()
            if lowered in {"group", "g", "user", "u", "qq"}:
                if not str(raw_target_id or "").strip():
                    return ""
                scope = "group" if lowered in {"group", "g"} else "user"
                return f"{scope}:{str(raw_target_id).strip()}"

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
        elif raw_scope in {"wl", "white", "whitelist", "bl", "black", "blacklist"}:
            # /meme group wl add [target] 是 send 名单的简写。
            target_id = target
            target = action
            raw_action = raw_list_name
            raw_list_name = raw_scope
            raw_scope = "send"
        elif raw_scope in {"send", "steal"} and raw_list_name in {
            "show", "list", "ls", "status"
        } and not raw_action:
            raw_action = raw_list_name
            raw_list_name = ""

        if raw_action in {"help", "h"} or (
            not raw_action and raw_list_name not in {"priority", "prio", "order"}
        ):
            help_text = (
                "用法：\n"
                "/meme group show\n"
                "/meme group <send|steal> show\n"
                "/meme group <send|steal> priority <wl|bl>\n"
                "/meme group <send|steal> <wl|bl> <add|del|clear> [group:ID|user:QQ]\n"
                "/meme group <send|steal> <wl|bl> <add|del> <group|user> <ID>\n\n"
                "说明：\n"
                "- send 控制发表情\n"
                "- steal 控制偷表情\n"
                "- 支持同时设置白名单和黑名单\n"
                "- priority wl = 白名单优先；priority bl = 黑名单优先\n"
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
                    priority = cfg._get_action_filter_mode(action_name)
                    mode = (
                        "同时启用"
                        if whitelist and blacklist
                        else "白名单"
                        if whitelist
                        else "黑名单"
                        if blacklist
                        else "未启用"
                    )
                    priority_label = "黑名单优先" if priority == "blacklist_first" else "白名单优先"
                    sections.append(
                        f"{title}:\n"
                        f"- 模式：{mode}\n"
                        f"- 优先级：{priority_label}\n"
                        f"- 白名单({len(whitelist)})：{format_items(whitelist)}\n"
                        f"- 黑名单({len(blacklist)})：{format_items(blacklist)}"
                    )
                yield event.plain_result("\n\n".join(sections))
                return

            if raw_scope not in {"send", "steal"}:
                yield event.plain_result("目标类型无效，请使用 send 或 steal")
                return

            whitelist, blacklist = cfg._get_action_lists(raw_scope)
            priority = cfg._get_action_filter_mode(raw_scope)
            mode = (
                "同时启用"
                if whitelist and blacklist
                else "白名单"
                if whitelist
                else "黑名单"
                if blacklist
                else "未启用"
            )
            priority_label = "黑名单优先" if priority == "blacklist_first" else "白名单优先"
            title = "发表情" if raw_scope == "send" else "偷表情"
            yield event.plain_result(
                f"{title}:\n"
                f"- 模式：{mode}\n"
                f"- 优先级：{priority_label}\n"
                f"- 白名单({len(whitelist)})：{format_items(whitelist)}\n"
                f"- 黑名单({len(blacklist)})：{format_items(blacklist)}"
            )
            return

        if raw_scope not in {"send", "steal"}:
            yield event.plain_result("目标类型无效，请使用 send 或 steal")
            return

        if raw_list_name in {"priority", "prio", "order"}:
            current_mode = cfg._get_action_filter_mode(raw_scope)
            if raw_action in {"", "show", "list", "ls", "status"}:
                label = "黑名单优先" if current_mode == "blacklist_first" else "白名单优先"
                yield event.plain_result(
                    f"{'发表情' if raw_scope == 'send' else '偷表情'} 当前优先级：{label}"
                )
                return

            if raw_action in {"wl", "white", "whitelist"}:
                updated_mode = "whitelist_first"
                label = "白名单优先"
            elif raw_action in {"bl", "black", "blacklist"}:
                updated_mode = "blacklist_first"
                label = "黑名单优先"
            else:
                yield event.plain_result("优先级无效，请使用 wl 或 bl")
                return

            list_key = f"{raw_scope}_target_filter_mode"
            ok = bool(cfg.update_config({list_key: updated_mode}))
            scope_title = "发表情" if raw_scope == "send" else "偷表情"
            msg = f"已将{scope_title}优先级设置为{label}" if ok else f"设置{scope_title}优先级失败"
            yield event.plain_result(msg)
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
            msg = (
                f"已将 {normalized_target} 加入{list_title}"
                if ok
                else f"加入{list_title}失败：{normalized_target}"
            )
            yield event.plain_result(msg)
            return

        if raw_action in {"del", "delete", "rm", "remove", "-"}:
            if normalized_target not in current_set:
                yield event.plain_result(f"{normalized_target} 不在{list_title}中")
                return
            updated = [item for item in current if item != normalized_target]
            ok = bool(cfg.update_config({list_key: updated}))
            msg = (
                f"已将 {normalized_target} 移出{list_title}"
                if ok
                else f"移出{list_title}失败：{normalized_target}"
            )
            yield event.plain_result(msg)
            return

        yield event.plain_result("操作无效，请使用 add / del / clear / show")
