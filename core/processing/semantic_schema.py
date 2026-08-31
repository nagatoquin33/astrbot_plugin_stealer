"""表情包语义字段约定：入库文档拼接、数量上限、嵌入语料版本。

2C2G 约束：不引入本地 CLIP / 视觉向量。检索只使用已有远程文本 Embedding
与 BM25；本模块只决定「把什么短文本送进这些索引」。
"""

from __future__ import annotations

from typing import Any

from ..util.normalization import normalize_label_list

EMBEDDING_TEXT_VERSION = "v2"

MAX_TAGS = 3
MAX_SCENES = 2
MAX_EMOTIONS = 3
MAX_OVERLAY_CHARS = 80
MAX_DESC_CHARS = 80


def as_label_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return normalize_label_list(value, allow_duplicates=True)
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
    bm25: bool = False,
) -> str:
    """拼一条给文本嵌入 / BM25 用的检索文档。

    图上文字和使用句权重大于英文分类名，以便对话查询对得上中文梗。
    角色名只来自用户归档，不由 VLM 填写。

    bm25=True 时用于 BM25 底层兜底：重复叠加图上文字和角色名，让 BM25
    更偏重画面/原文信号；分类只保留 key 一次，不把分类描述当主要匹配依据。
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

    if bm25:
        # 底层兜底：图上文字/角色名/适用对话重复出现以提高 BM25 权重；分类只保留 key 一次。
        weighted: list[str] = []
        if overlay:
            weighted.extend([overlay] * 3)
        weighted.extend(character_bits * 2)
        if desc:
            weighted.append(desc)
        weighted.extend(tags)
        weighted.extend(scenes * 2)
        if category:
            weighted.append(category)
        return " ".join(str(part).strip() for part in weighted if str(part).strip())

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
