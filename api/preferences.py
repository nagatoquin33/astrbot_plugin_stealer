"""Dashboard preferences routes."""

import inspect
from typing import Any

from quart import jsonify, request

from astrbot.api import logger


class PreferenceRoutes:
    def _normalize_theme(self, raw: Any) -> str:
        value = str(raw or "").strip()
        return value if value in self.VALID_THEMES else "auto"

    @staticmethod
    def _normalize_view(raw: Any) -> str:
        return "list" if str(raw or "").strip() == "list" else "grid"

    @staticmethod
    def _normalize_sidebar(raw: Any) -> str:
        return "collapsed" if str(raw or "").strip() == "collapsed" else "expanded"

    def _config_default_theme(self) -> str:
        cfg = getattr(self.plugin, "plugin_config", None)
        if cfg is None:
            return "auto"
        # 优先读 AstrBotConfig 实时值（_data 随配置保存立即更新），
        # PluginConfig 是插件初始化时的快照，改配置后不会自动刷新。
        data = getattr(cfg, "_data", None)
        if data is not None and hasattr(data, "get"):
            raw = data.get("webui_theme")
            if raw:
                return self._normalize_theme(raw)
        return self._normalize_theme(getattr(cfg, "webui_theme", "auto"))

    async def _await_maybe(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _read_dashboard_prefs(self) -> dict[str, str]:
        stored: Any = None
        getter = getattr(self.plugin, "get_kv_data", None)
        if callable(getter):
            try:
                stored = await self._await_maybe(getter(self.DASHBOARD_PREFS_KEY, {}))
            except Exception as e:
                logger.debug(f"读取 WebUI 偏好失败: {e}")
                stored = None
        memory_prefs = getattr(self.plugin, "_dashboard_prefs", {}) or {}
        if not isinstance(stored, dict) or (not stored and isinstance(memory_prefs, dict)):
            stored = memory_prefs
        if not isinstance(stored, dict):
            stored = {}
        return dict(stored)

    def _resolve_dashboard_prefs(
        self,
        stored: dict[str, str],
        config_theme: str | None = None,
    ) -> dict[str, str]:
        config_theme = config_theme or self._config_default_theme()
        stored_theme = str(stored.get("theme") or "").strip()
        theme = stored_theme if stored_theme in self.VALID_THEMES else config_theme
        return {
            "theme": self._normalize_theme(theme),
            "view": self._normalize_view(stored.get("view")),
            "sidebar": self._normalize_sidebar(stored.get("sidebar")),
        }

    async def _load_dashboard_prefs(self) -> dict[str, str]:
        stored = await self._read_dashboard_prefs()
        return self._resolve_dashboard_prefs(stored)

    async def _save_dashboard_prefs(self, prefs: dict[str, str]) -> None:
        prefs = dict(prefs)
        setattr(self.plugin, "_dashboard_prefs", prefs)
        setter = getattr(self.plugin, "put_kv_data", None)
        if not callable(setter):
            return
        try:
            await self._await_maybe(setter(self.DASHBOARD_PREFS_KEY, prefs))
        except Exception as e:
            logger.warning(f"保存 WebUI 偏好失败: {e}")

    async def _update_dashboard_prefs(self, payload: dict[str, Any]) -> dict[str, str]:
        stored = await self._read_dashboard_prefs()
        config_theme = self._config_default_theme()

        if "theme" in payload:
            theme = str(payload.get("theme") or "").strip()
            if theme in self.VALID_THEMES:
                stored["theme"] = theme
        if "view" in payload:
            stored["view"] = self._normalize_view(payload.get("view"))
        if "sidebar" in payload:
            stored["sidebar"] = self._normalize_sidebar(payload.get("sidebar"))

        await self._save_dashboard_prefs(stored)
        return self._resolve_dashboard_prefs(stored, config_theme)

    async def handle_prefs(self):
        """WebUI 主题、视图和侧栏偏好：KV 持久化。"""
        if request.method == "GET":
            prefs = await self._load_dashboard_prefs()
            return jsonify({"success": True, **prefs})

        payload = {}
        try:
            payload = await request.get_json() or {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        current = await self._update_dashboard_prefs(payload)
        return jsonify({"success": True, **current})
