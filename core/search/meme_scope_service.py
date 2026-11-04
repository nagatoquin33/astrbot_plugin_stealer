"""表情包作用域检查服务。"""

from typing import Any

from astrbot.api.event import AstrMessageEvent

from ..util.normalization import canonicalize_path, normalize_character_key, normalize_scope_mode


class MemeScopeService:
    """负责检查表情包是否对当前事件可用。"""

    def __init__(self, plugin_instance: Any = None) -> None:
        self.plugin = plugin_instance

    def active_character(self, event: AstrMessageEvent | None) -> str:
        cfg = getattr(self.plugin, "plugin_config", None)
        if event is None or not getattr(cfg, "persona_character_isolation", False):
            return ""
        return normalize_character_key(getattr(event, "_stealer_persona_character", ""))

    def _get_event_target_entry(self, event: AstrMessageEvent | None) -> str:
        if event is None:
            return ""

        cfg = getattr(self.plugin, "plugin_config", None)
        if not cfg:
            return ""

        try:
            scope, target_id = cfg.get_event_target(event)
        except Exception:
            return ""

        if not scope or not target_id:
            return ""
        return f"{scope}:{target_id}"

    def _is_entry_allowed_for_event(
        self, data: dict | None, event: AstrMessageEvent | None
    ) -> bool:
        if not isinstance(data, dict) or event is None:
            return True

        cfg = getattr(self.plugin, "plugin_config", None)
        if getattr(cfg, "persona_character_isolation", False):
            character = normalize_character_key(data.get("character"))
            active = self.active_character(event)
            if character and character != active:
                return False

        if normalize_scope_mode(data.get("scope_mode")) != "local":
            return True

        origin_target = str(data.get("origin_target", "") or "").strip()
        current_target = self._get_event_target_entry(event)
        if not origin_target or not current_target:
            return False
        return origin_target == current_target

    def is_path_allowed_for_event(self, path: str, event: AstrMessageEvent | None) -> bool:
        if not path:
            return False

        if event is None:
            return True

        # 优先使用数据库服务
        db_service = getattr(self.plugin, "db_service", None)
        if db_service:
            data = db_service.get_emoji(path)
            if data is None:
                # 尝试规范化路径匹配
                target_path = canonicalize_path(path)
                all_paths = db_service.get_all_paths()
                for stored_path in all_paths:
                    if canonicalize_path(stored_path) == target_path:
                        data = db_service.get_emoji(stored_path)
                        break
            if data is None:
                return False
            return self._is_entry_allowed_for_event(data, event)

        # 数据库不可用时拒绝访问该图片。
        return False
