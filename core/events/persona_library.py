"""会话生效人格与角色表情库的关联。"""

from astrbot.api import logger

from .event_context import get_event_platform_name, unwrap_event
from ..util.normalization import normalize_character_key


async def resolve_persona_character(plugin, event):
    """获取框架当前生效的人格，并映射到已有角色键。"""
    event = unwrap_event(event)
    cfg = getattr(plugin, "plugin_config", None)
    if event is None or not getattr(cfg, "persona_character_isolation", False):
        return ""
    try:
        context = plugin.context
        origin = event.unified_msg_origin
        manager = context.conversation_manager
        conversation_id = await manager.get_curr_conversation_id(origin)
        conversation = await manager.get_conversation(origin, conversation_id) if conversation_id else None
        selected, _persona, _forced, _webchat = await context.persona_manager.resolve_selected_persona(
            umo=origin,
            conversation_persona_id=getattr(conversation, "persona_id", None),
            platform_name=get_event_platform_name(event),
        )
        if not selected or selected == "[%None]" or (_persona is None and not _webchat):
            return ""
        mapping = getattr(cfg, "persona_character_map", {}) or {}
        return normalize_character_key(mapping.get(selected, selected))
    except Exception as exc:
        logger.warning(f"[人格表情库] 获取当前人格失败，仅使用通用表情库: {exc}")
        return ""


async def prepare_persona_scope(plugin, event):
    """在本次事件上保存选图所需的人格，供同步筛选链路读取。"""
    event = unwrap_event(event)
    if event is not None and getattr(getattr(plugin, "plugin_config", None), "persona_character_isolation", False):
        setattr(event, "_stealer_persona_character", await resolve_persona_character(plugin, event))
    return event
