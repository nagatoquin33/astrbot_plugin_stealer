"""插件功能面板及随机试发。"""

import asyncio
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ..events.event_context import get_event_platform_name, get_event_session_key
from ..events.persona_library import prepare_persona_scope
from ..search.meme_scope_service import MemeScopeService
from ..util.normalization import normalize_character_key
from .qq_menu import build_keyboard, panel_content, panel_rows, send_keyboard
from .panel_actions import PANEL_ACTIONS
from .qq_panel_interactions import QQPanelInteractions
from .qq_callback_event import identity_tag, source_sender


@dataclass
class PanelSession:
    owner: tuple[str, str]
    expires_at: float
    event: Any
    expiry: asyncio.TimerHandle | None = None
    busy: bool = False
    sent: set[str] = field(default_factory=set)
    handled: set[str] = field(default_factory=set)


class MemeMenuCommand:
    """功能按钮通过后台回调执行，随机试发独立于 LLM 选图状态。"""

    TTL_SECONDS = 300

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self._scope_service = MemeScopeService(plugin)
        self._sessions: dict[str, PanelSession] = {}
        self._owners: dict[tuple[str, str], str] = {}
        self.interactions = QQPanelInteractions(self)

    def command_prefix(self, event: Any) -> str:
        prefixes = self.plugin.context.get_config(umo=get_event_session_key(event)).get("wake_prefix", ["/"])
        return prefixes[0] if prefixes else ""

    def current_admin(self, event: Any) -> bool:
        config = self.plugin.context.get_config(umo=get_event_session_key(event))
        return (
            str(event.get_sender_id()) in config.get("admins_id", [])
            and event.get_extra("_api_key_allow_admin_role") is not False
        )

    @staticmethod
    def supports_buttons(event: Any) -> bool:
        return get_event_platform_name(event) in {"qq_official", "qq_official_webhook"}

    @staticmethod
    def _owner(event: Any) -> tuple[str, str]:
        return get_event_session_key(event), str(event.get_sender_id())

    def _discard(self, token: str) -> None:
        session = self._sessions.pop(token, None)
        if session is not None:
            if session.expiry:
                session.expiry.cancel()
            if self._owners.get(session.owner) == token:
                self._owners.pop(session.owner, None)

    def close(self) -> None:
        """卸载时释放面板状态和到期计时器。"""
        self.interactions.close()
        for token in list(self._sessions):
            self._discard(token)

    async def aclose(self) -> None:
        tasks = list(self.interactions.tasks)
        self.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _session_for(self, event: Any) -> str:
        owner = self._owner(event)
        token = self._owners.get(owner)
        now = asyncio.get_running_loop().time()
        if token and self._sessions[token].expires_at > now:
            return token
        if token:
            self._discard(token)
        token = secrets.token_hex(6)
        session = PanelSession(owner, now + self.TTL_SECONDS, getattr(event, "_stealer_original_event", event))
        session.expiry = asyncio.get_running_loop().call_later(self.TTL_SECONDS, self._discard, token)
        self._sessions[token] = session
        self._owners[owner] = token
        return token

    async def _lookup(self, event: Any, token: str) -> PanelSession | None:
        session = self._sessions.get(token)
        if session is None or session.expires_at <= asyncio.get_running_loop().time():
            self._discard(token)
            await event.send(event.plain_result(f"面板已过期或已收起，请使用 {self.command_prefix(event)}meme menu 重新打开。"))
            return None
        if session.owner != self._owner(event):
            await event.send(event.plain_result(f"这个面板属于其他用户或会话，请使用 {self.command_prefix(event)}meme menu 打开自己的面板。"))
            return None
        return session

    async def show_panel(self, event: Any, section: str = "overview") -> None:
        section = str(section or "overview").strip().lower()
        if section not in {"overview", "settings", "maintenance"}:
            await event.send(event.plain_result(f"用法：{self.command_prefix(event)}meme menu [overview|settings|maintenance]"))
            return
        admin = event.is_admin()
        if section != "overview" and not admin:
            await event.send(event.plain_result("运行设置和维护工具仅限 AstrBot 管理员使用。"))
            return
        token = self._session_for(event)
        rows = panel_rows(section, admin)
        prefix = self.command_prefix(event)
        content = panel_content(section, admin, self.plugin.plugin_config)
        if self.supports_buttons(event):
            try:
                self.interactions.bind(getattr(event, "_stealer_original_event", event).bot)
                sender, sender_field = source_sender(event)
                await send_keyboard(event, content, build_keyboard(rows, sender, token))
                logger.info(
                    f"[MemeMenu] 面板已发出 panel={identity_tag(token)} section={section} "
                    f"qq_user={identity_tag(sender)} user_field={sender_field} "
                    f"framework_user={identity_tag(event.get_sender_id())} "
                    f"group={identity_tag(getattr(event.message_obj.raw_message, 'group_openid', None))}"
                )
                return
            except Exception as exc:
                logger.warning(f"[MemeMenu] QQ 面板发送失败，回退文字指令: {exc}")
        commands = [
            f"{button.label}：{prefix}{PANEL_ACTIONS[button.action].command}"
            + (f" {token}" if button.action in {"random", "close"} else "")
            for row in rows for button in row
        ]
        await event.send(event.plain_result(content + "\n\n功能指令：\n" + "\n".join(commands)))

    async def _visible_index(self, event: Any) -> dict:
        await prepare_persona_scope(self.plugin, event)
        index = await asyncio.to_thread(self.plugin.db_service.get_index_cache_readonly)

        def visible():
            return {
                path: data for path, data in index.items()
                if Path(path).is_file() and self._scope_service._is_entry_allowed_for_event(data, event)
            }

        return await asyncio.to_thread(visible)

    async def random_image(self, event: Any, token: str = "") -> None:
        """从当前会话可发送的表情中随机试发，同一面板内不重复。"""
        session = await self._lookup(event, token or self._session_for(event))
        if session is None:
            return
        if session.busy:
            await event.send(event.plain_result("正在发送表情，请稍等。"))
            return
        if not self.plugin.is_send_enabled_for_event(event):
            await event.send(event.plain_result("当前会话已禁用表情包发送。"))
            return
        session.busy = True
        try:
            await self._random_send(event, session)
        finally:
            session.busy = False

    async def _random_send(self, event: Any, session: PanelSession) -> None:
        index = await self._visible_index(event)
        paths = [path for path in index if path not in session.sent]
        if not paths:
            await event.send(event.plain_result("当前会话暂无未试发的可用表情，稍后重开面板可再次随机。"))
            return
        character = self._scope_service.active_character(event)
        if character:
            preferred = [path for path in paths if normalize_character_key(index[path].get("character")) == character]
            paths = preferred or paths
        await self._send_path(event, session, secrets.choice(paths))

    async def execute_action(self, event: Any, token: str, action: str, session: PanelSession) -> None:
        spec = PANEL_ACTIONS[action]
        if spec.admin and not self.current_admin(event):
            await event.send(event.plain_result("当前账号已无管理员权限，操作未执行。"))
            return
        if action in {"overview", "settings", "maintenance"}:
            await self.show_panel(event, action)
        elif action == "random":
            await self._random_send(event, session)
        elif action == "close":
            await self.dismiss(event, token)
        else:
            handler = getattr(getattr(self.plugin, spec.service), spec.method)
            async for result in handler(event, *spec.args):
                await event.send(result)

    async def _send_path(self, event: Any, session: PanelSession, path: str) -> None:
        """发送前检查实时限制，成功后更新使用记录和自动发送冷却。"""
        if not self.plugin.is_send_enabled_for_event(event):
            await event.send(event.plain_result("当前会话已禁用表情包发送。"))
            return
        await prepare_persona_scope(self.plugin, event)
        if not Path(path).is_file() or not self.plugin.meme_selector.is_path_allowed_for_event(path, event):
            await event.send(event.plain_result("这张表情已移除或不允许在当前会话发送，请重新随机。"))
            return
        try:
            mode = await self.plugin.meme_selector.send_emoji_message(event, path)
            if not mode:
                await event.send(event.plain_result("随机发送失败，可以点击重试。"))
                return
        except Exception as exc:
            logger.error(f"[MemeMenu] 随机发送失败: {exc}", exc_info=True)
            await event.send(event.plain_result("随机发送失败，可以点击重试。"))
            return
        session.sent.add(path)
        try:
            await self.plugin.meme_selector.record_emoji_usage(path, trigger="menu")
            await self.plugin._emoji_sender_engine.mark_auto_emoji_sent(event)
        except Exception as exc:
            logger.error(f"[MemeMenu] 图片已发送，使用记录更新失败: {exc}", exc_info=True)

    async def dismiss(self, event: Any, token: str) -> None:
        if await self._lookup(event, token) is not None:
            self._discard(token)
            await event.send(event.plain_result("面板已收起，原面板按钮已失效。"))
