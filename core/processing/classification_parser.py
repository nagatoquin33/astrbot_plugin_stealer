"""分类结果解析器：负责解析 VLM 的 JSON/文本响应。"""

import json
import re
from typing import Any

from astrbot.api import logger


class ClassificationParser:
    """负责解析 VLM 的分类响应。"""

    CATEGORY_FILTERED = "过滤不通过"

    def __init__(self, plugin_instance=None) -> None:
        self.plugin = plugin_instance

    def _normalize_category(self, raw: str) -> str:
        """将 VLM 返回的分类文本规范化为有效分类名（委托到 ImageProcessorService）。"""
        if self.plugin and hasattr(self.plugin, "image_processor_service"):
            return self.plugin.image_processor_service._normalize_category(raw)
        return str(raw or "").strip().lower()

    def _parse_classification_response(
        self, response: str, file_path: str
    ) -> tuple[str, list[str], str, str, list[str]]:
        """Parse the classification payload returned by the VLM."""
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
            return self.CATEGORY_FILTERED, [], "", self.CATEGORY_FILTERED, []

        category = data.get("category", "")
        tags = data.get("tags", [])
        description = self._sanitize_model_scalar(data.get("description", "emoji")) or "emoji"
        scenes = data.get("scenes", [])

        normalized_category = self._normalize_category(category)

        if isinstance(tags, str):
            tags = [t.strip() for t in tags.replace(chr(65292), ",").split(",") if t.strip()]
        elif not isinstance(tags, list):
            tags = []

        if isinstance(scenes, str):
            scenes = [s.strip() for s in scenes.replace(chr(65292), ",").split(",") if s.strip()]
        elif not isinstance(scenes, list):
            scenes = []

        return normalized_category, tags, description, normalized_category, scenes

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

    def _parse_legacy_format(self, response: str) -> tuple[str, list[str], str, str, list[str]]:
        """兼容旧格式：管道符分隔的响应。"""
        # 处理审核不通过
        if self.CATEGORY_FILTERED in response or "审核不通过" in response:
            return self.CATEGORY_FILTERED, [], "", self.CATEGORY_FILTERED, []

        # 兼容旧格式：情绪分类|语义标签|画面描述|场景标签
        parts = [p.strip() for p in response.strip().split("|")]
        emotion_result = parts[0] if parts else ""
        tags_str = parts[1] if len(parts) > 1 else ""
        tags_result = [
            t.strip()
            for t in tags_str.replace("，", ",").replace("、", ",").split(",")
            if t.strip()
        ]
        desc_result = parts[2] if len(parts) > 2 else "表情包"
        scenes_str = parts[3] if len(parts) > 3 else ""
        scenes_result = [
            s.strip()
            for s in scenes_str.replace("，", ",").replace("、", ",").replace("；", ",").split(",")
            if s.strip()
        ]

        category = self._normalize_category(emotion_result)
        return category, tags_result, desc_result, category, scenes_result
