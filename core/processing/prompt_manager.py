"""提示词管理器：负责加载、缓存和渲染 VLM 分类提示词模板。"""

from typing import Any


class PromptManager:
    """管理表情包分类所需的 VLM 提示词。"""

    _PROMPT_PLACEHOLDER = "{emotion_list}"

    _FALLBACK_PROMPT = (
        "分析表情包：从 `{emotion_list}` 中选择情绪分类，每个分类机会均等。"
        '返回JSON格式：{"category": "分类名", "tags": [], '
        '"description": "画面描述", "overlay_text": "", "scenes": []}'
        "tags 没有则 []。"
    )
    _FALLBACK_FILTER_PROMPT = (
        '审核图片是否含不当内容，不当则返回{"approved": false, "reason": "审核不通过"}。'
        "否则从 `{emotion_list}` 中选择情绪分类，每个分类机会均等。"
        '返回JSON格式：{"approved": true, "category": "分类名", "tags": [], '
        '"description": "画面描述", "overlay_text": "", "scenes": []}'
        "tags 没有则 []。"
    )

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance
        self.plugin_config = getattr(plugin_instance, "plugin_config", None)

        self.emoji_classification_prompt = getattr(
            plugin_instance, "EMOJI_CLASSIFICATION_PROMPT", self._FALLBACK_PROMPT
        )
        self.emoji_classification_with_filter_prompt = getattr(
            plugin_instance,
            "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT",
            self._FALLBACK_FILTER_PROMPT,
        )
        self.categories = list(self.plugin_config.categories or []) if self.plugin_config else []

    def update_config(
        self,
        categories=None,
        emoji_classification_prompt=None,
        emoji_classification_with_filter_prompt=None,
    ) -> None:
        if categories is not None:
            self.categories = categories
        if emoji_classification_prompt is not None:
            self.emoji_classification_prompt = emoji_classification_prompt
        if emoji_classification_with_filter_prompt is not None:
            self.emoji_classification_with_filter_prompt = emoji_classification_with_filter_prompt

    def build_classification_prompt(
        self, *, use_filter: bool = False, categories: list[str] | None = None
    ) -> str:
        """根据当前配置构建完整的 VLM 分类提示词。"""
        if categories is None and self.plugin_config and hasattr(
            self.plugin_config, "get_vlm_categories"
        ):
            categories = self.plugin_config.get_vlm_categories()
        elif categories:
            categories = [item for item in categories if item != "other"]
        emotion_list = self._build_emotion_list_str(categories)
        template = (
            self.emoji_classification_with_filter_prompt
            if use_filter
            else self.emoji_classification_prompt
        )
        return self._render_prompt_template(template, emotion_list)

    def _build_emotion_list_str(self, categories: list[str] | None = None) -> str:
        categories = categories if categories is not None else (self.categories or [])
        categories = [c for c in categories if isinstance(c, str) and c.strip()]
        info_map = getattr(self.plugin_config, "category_info", None) or {}

        lines = []
        for raw_key in categories:
            key = raw_key.strip()
            info = info_map.get(key)
            if isinstance(info, dict):
                name = str(info.get("name", "")).strip()
                desc = str(info.get("desc", "")).strip()
            else:
                name = ""
                desc = ""

            if name and name != key:
                if desc:
                    lines.append(f"{key} - {name}：{desc}")
                else:
                    lines.append(f"{key} - {name}")
            else:
                if desc:
                    lines.append(f"{key}：{desc}")
                else:
                    lines.append(key)

        if lines:
            return "\n".join(lines)
        return ", ".join(categories)

    @staticmethod
    def _render_prompt_template(template: str, emotion_list: str) -> str:
        """仅替换 emotion_list 占位符，保留 JSON 花括号原样输出。"""
        if not template:
            return emotion_list
        return template.replace(PromptManager._PROMPT_PLACEHOLDER, emotion_list)
