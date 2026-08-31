"""分类结果解析器：负责解析 VLM 的 JSON/文本响应。"""

from typing import Any

import json
import re

from astrbot.api import logger

from ..util.normalization import normalize_label_list as normalize_values
from .semantic_schema import (
    MAX_DESC_CHARS,
    MAX_EMOTIONS,
    MAX_OVERLAY_CHARS,
    MAX_SCENES,
    MAX_TAGS,
    clip_chars,
)


class ClassificationParser:
    """负责解析 VLM 的分类响应。"""

    CATEGORY_FILTERED = "过滤不通过"

    def __init__(self, plugin_instance=None) -> None:
        self.plugin = plugin_instance

    @staticmethod
    def normalize_label_list(
        values: Any,
        max_count: int,
        *,
        allow_duplicates: bool = False,
    ) -> list[str]:
        """规范化标签/场景列表：拆分字符串、去空白、去空项、保序去重、截断到上限。

        Args:
            values: VLM 输出的 tags/scenes（list 或逗号分隔字符串）
            max_count: 最多保留数量（超出截断）
            allow_duplicates: 是否允许重复（默认去重）
        """
        return normalize_values(
            values,
            max_count,
            allow_duplicates=allow_duplicates,
        )

    def sanitize_scenes(self, values: Any, overlay_text: str = "") -> list[str]:
        """scenes 保留适用对话句，最多 40 字；优先放入图上文字。"""
        overlay_scene = clip_chars(str(overlay_text or "").strip(), 40)
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in self.normalize_label_list(values, MAX_SCENES + 2):
            if len(item) > 40:
                continue
            if item in seen:
                continue
            seen.add(item)
            cleaned.append(item)
            if len(cleaned) >= MAX_SCENES:
                break
        if overlay_scene and overlay_scene not in seen:
            cleaned.insert(0, overlay_scene)
            cleaned = cleaned[:MAX_SCENES]
        return cleaned

    def _normalize_category(self, raw: str, *, fallback_other: bool = True) -> str:
        """将 VLM 返回的分类文本规范化为有效分类名（委托到 ImageProcessorService）。"""
        if self.plugin and hasattr(self.plugin, "image_processor_service"):
            normalizer = getattr(
                self.plugin.image_processor_service, "_normalize_category", None
            )
            if callable(normalizer):
                try:
                    return normalizer(raw, fallback_other=fallback_other)
                except TypeError:
                    result = normalizer(raw)
                    if result:
                        return result
                    return self._fallback_category() if fallback_other else ""
        text = str(raw or "").strip().lower()
        if not text:
            return self._fallback_category() if fallback_other else ""
        return text

    def _fallback_category(self) -> str:
        cfg = getattr(self.plugin, "plugin_config", None) if self.plugin else None
        if cfg and hasattr(cfg, "closest_category"):
            return cfg.closest_category("")
        return "confused"

    def _parse_classification_response(
        self, response: str, file_path: str
    ) -> tuple[str, list[str], str, str, list[str], str, list[str]]:
        """Parse the classification payload returned by the VLM.

        Returns:
            (category, tags, description, emotion, scenes, overlay_text, emotions)
        """
        response = response.strip()

        data = self._extract_json_payload(response)
        if data is None:
            logger.debug(f"JSON parse failed, fallback to legacy format: {response[:100]}")
            return self._parse_legacy_format(response)

        approved = data.get("approved")
        reason = str(data.get("reason", ""))
        if (
            approved is False
            or str(approved).strip().lower() in {"false", "0", "no", "rejected"}
            or "\u5ba1\u6838\u4e0d\u901a\u8fc7" in reason
        ):
            logger.warning(f"Image moderation rejected: {file_path}")
            return self.CATEGORY_FILTERED, [], "", self.CATEGORY_FILTERED, [], "", []

        category = data.get("category", "")
        tags = data.get("tags", [])
        description = clip_chars(
            self._sanitize_model_scalar(data.get("description", "emoji")) or "emoji",
            MAX_DESC_CHARS,
        )
        scenes = data.get("scenes", [])
        overlay_text = clip_chars(
            self._sanitize_model_scalar(data.get("overlay_text", "")),
            MAX_OVERLAY_CHARS,
        )

        normalized_category = self._normalize_category(category, fallback_other=True)
        tags = self.normalize_label_list(tags, MAX_TAGS)
        scenes = self.sanitize_scenes(scenes, overlay_text)

        extra_emotions = self.normalize_label_list(
            data.get("emotions", data.get("emotion_labels", [])),
            MAX_EMOTIONS,
        )
        emotions: list[str] = []
        seen: set[str] = set()
        if normalized_category and normalized_category != self.CATEGORY_FILTERED:
            emotions.append(normalized_category)
            seen.add(normalized_category)
        for item in extra_emotions:
            mapped = self._normalize_category(item, fallback_other=False)
            if not mapped or mapped in seen or mapped == self.CATEGORY_FILTERED:
                continue
            seen.add(mapped)
            emotions.append(mapped)
            if len(emotions) >= MAX_EMOTIONS:
                break

        return (
            normalized_category,
            tags,
            description,
            normalized_category,
            scenes,
            overlay_text,
            emotions,
        )

    def _sanitize_model_scalar(self, value: Any) -> str:
        """Normalize single-value model outputs before category matching."""
        text = str(value or "").strip()
        text = text.strip("`")
        text = text.strip(" \t\r\n\"'")
        text = re.sub(r"^[\[\(\{<]+|[\]\)\}>]+$", "", text)
        text = text.rstrip("\u3002\uff01\uff0c\u3001\uff1b;\uff1a:")
        return text.strip()

    def _extract_json_payload(self, response: str) -> dict[str, Any] | None:
        """从 VLM 响应中提取第一个合法 JSON 对象。"""
        candidates: list[str] = []

        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", response, flags=re.DOTALL)
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())
        candidates.append(response.strip())

        for candidate in candidates:
            parsed = self._try_parse_json_candidate(candidate)
            if parsed is not None:
                return parsed

        return None

    def _try_parse_json_candidate(self, text: str) -> dict[str, Any] | None:
        """解析候选文本中的 JSON 对象，兼容前后缀说明文字。"""
        decoder = json.JSONDecoder()

        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    def _parse_legacy_format(
        self, response: str
    ) -> tuple[str, list[str], str, str, list[str], str, list[str]]:
        """兼容旧格式：管道符分隔的响应。"""
        if self.CATEGORY_FILTERED in response or "审核不通过" in response:
            return self.CATEGORY_FILTERED, [], "", self.CATEGORY_FILTERED, [], "", []

        parts = [p.strip() for p in response.strip().split("|")]
        emotion_result = parts[0] if parts else ""
        tags_str = parts[1] if len(parts) > 1 else ""
        tags_result = self.normalize_label_list(tags_str, MAX_TAGS)
        desc_result = clip_chars(parts[2] if len(parts) > 2 else "表情包", MAX_DESC_CHARS)
        scenes_str = parts[3] if len(parts) > 3 else ""
        scenes_result = normalize_values(scenes_str)
        overlay_text = clip_chars(parts[4] if len(parts) > 4 else "", MAX_OVERLAY_CHARS)

        category = self._normalize_category(emotion_result, fallback_other=True)
        emotions = [category] if category and category != self.CATEGORY_FILTERED else []
        return (
            category,
            tags_result,
            desc_result,
            category,
            self.sanitize_scenes(scenes_result, overlay_text),
            overlay_text,
            emotions,
        )
