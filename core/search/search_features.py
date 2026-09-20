"""检索共享的纯数据转换；不依赖选择器或服务实例。"""

from functools import lru_cache
from typing import Any

from .text_similarity import _extract_words


def entry_category(data: dict | None) -> str:
    """从数据字典中获取小写的分类名。

    Args:
        data: 图片元数据字典

    Returns:
        str: 小写的分类名，如果不存在则返回空字符串
    """
    if not isinstance(data, dict):
        return ""
    return str(data.get("category", "")).lower()


def parse_tags(raw_tags: Any) -> list[str]:
    """安全解析 tags 字段，兼容字符串和列表类型。"""
    if isinstance(raw_tags, str):
        return [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
    if isinstance(raw_tags, list):
        return [str(t).lower() for t in raw_tags if t]
    return []


@lru_cache(maxsize=4096)
def collect_phrase_words(items: tuple[str, ...]) -> frozenset[str]:
    words = set()
    for item in items:
        words.update(_extract_words(item))
    return frozenset(words)


@lru_cache(maxsize=4096)
def prepare_entry_text_features(
    category: str,
    desc: str,
    tags: tuple[str, ...],
    scenes: tuple[str, ...] = (),
    overlay: str = "",
    character: str = "",
) -> tuple[str, frozenset[str], frozenset[str], frozenset[str], str]:
    desc_lower = str(desc or "").lower()
    tag_words = collect_phrase_words(tags)
    scene_words = collect_phrase_words(scenes)
    all_text = " ".join(
        part
        for part in [
            str(category or ""),
            desc_lower,
            " ".join(tags),
            " ".join(scenes),
            str(overlay or ""),
            str(character or ""),
        ]
        if part
    )
    all_words = _extract_words(all_text)
    return desc_lower, tag_words, scene_words, all_words, all_text


def normalize_category(plugin, category: str) -> str:
    """归一化分类名称，返回有效分类或空字符串。"""
    if not category:
        return ""
    cfg = getattr(plugin, "plugin_config", None)
    if not cfg:
        return ""
    try:
        result = cfg.normalize_category_strict(category)
        return result or ""
    except Exception:
        return ""
