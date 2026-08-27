"""表情包语义字段约定：入库文档拼接、数量上限、嵌入语料版本。

2C2G 约束：不引入本地 CLIP / 视觉向量。检索只使用已有远程文本 Embedding
与 BM25；本模块只决定「把什么短文本送进这些索引」。
"""

from __future__ import annotations

from typing import Any

CATEGORY_OTHER = "other"
EMBEDDING_TEXT_VERSION = "v2"

MAX_TAGS = 6
MAX_SCENES = 3
MAX_EMOTIONS = 3
MAX_OVERLAY_CHARS = 80
MAX_DESC_CHARS = 80


def as_label_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return []


def clip_chars(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars > 0 and len(value) > max_chars:
        return value[:max_chars].rstrip()
    return value


def build_meme_search_text(
    entry: dict[str, Any],
    *,
    category_info: dict[str, Any] | None = None,
    character_info: dict[str, Any] | None = None,
) -> str:
    """拼一条给文本嵌入 / BM25 用的检索文档。

    图上文字和使用句权重大于英文分类名，以便对话查询对得上中文梗。
    角色名只来自用户归档，不由 VLM 填写。
    """
    overlay = clip_chars(str(entry.get("overlay_text") or ""), MAX_OVERLAY_CHARS)
    desc = clip_chars(str(entry.get("desc") or ""), MAX_DESC_CHARS)
    category = str(entry.get("category") or "").strip()
    character = str(entry.get("character") or "").strip()
    tags = as_label_list(entry.get("tags"))
    scenes = as_label_list(entry.get("scenes"))
    emotions = as_label_list(entry.get("emotions"))
    if not emotions and category:
        emotions = [category]

    info_map = category_info or {}
    emotion_bits: list[str] = []
    for key in [category, *emotions]:
        if not key:
            continue
        emotion_bits.append(key)
        info = info_map.get(key)
        if isinstance(info, dict):
            name = str(info.get("name") or "").strip()
            desc_text = str(info.get("desc") or "").strip()
            if name:
                emotion_bits.append(name)
            if desc_text:
                emotion_bits.append(desc_text)

    character_bits: list[str] = []
    if character:
        character_bits.append(character)
        char_meta = (character_info or {}).get(character)
        if isinstance(char_meta, dict):
            char_name = str(char_meta.get("name") or "").strip()
            if char_name:
                character_bits.append(char_name)

    parts: list[str] = []
    if overlay:
        parts.append(overlay)
    parts.extend(character_bits)
    if desc:
        parts.append(desc)
    parts.extend(tags)
    parts.extend(scenes)
    parts.extend(emotion_bits)

    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        token = str(part).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return " ".join(ordered)
