"""AstrBot 消息事件的容错读取工具。"""

from typing import Any


def normalize_event_value(value: object) -> str:
    """把适配器返回的标量安全转换为字符串。"""
    if value is None:
        return ""
    try:
        normalized = str(value).strip()
    except Exception:
        return ""
    if normalized.startswith("`") and normalized.endswith("`") and len(normalized) >= 2:
        normalized = normalized[1:-1].strip()
    return normalized


def get_event_platform_name(event: Any | None) -> str:
    """读取小写平台名，兼容 ``get_platform_name`` / ``get_platform_id``。"""
    if event is None:
        return ""
    for getter_name in ("get_platform_name", "get_platform_id"):
        getter = getattr(event, getter_name, None)
        if not callable(getter):
            continue
        try:
            platform_name = normalize_event_value(getter()).lower()
        except Exception:
            continue
        if platform_name:
            return platform_name
    return ""


def get_event_session_key(event: Any | None, *, default: str = "global") -> str:
    """读取稳定会话键，供冷却、后台任务和强制捕获窗口共用。"""
    if event is None:
        return default

    getter = getattr(event, "get_session_id", None)
    if callable(getter):
        try:
            session_id = normalize_event_value(getter())
        except Exception:
            session_id = ""
        if session_id:
            return session_id

    try:
        unified_msg_origin = normalize_event_value(
            getattr(event, "unified_msg_origin", "")
        )
    except Exception:
        unified_msg_origin = ""
    return unified_msg_origin or default


def unwrap_event(event: Any) -> Any:
    """从 AstrBot v4.26+ 的工具上下文中取出真实消息事件。"""
    try:
        from astrbot.core.agent.run_context import ContextWrapper
    except ImportError:
        return event
    if isinstance(event, ContextWrapper):
        return event.context.event
    return event
