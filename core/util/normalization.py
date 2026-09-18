"""跨模块共享的值规范化工具。

这些函数只做纯数据转换，不依赖插件实例，避免 WebAPI、命令、数据库和
检索服务各自维护一套略有差异的别名与去重规则。
"""

import posixpath
import re
from typing import Any


_METADATA_SEPARATOR_RE = re.compile(r"[,，、;；\n\t]+")
_CSV_SEPARATOR_RE = re.compile(r",")
_CATEGORY_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
)
_PLUGIN_RESERVED_CATEGORY_KEYS = frozenset({"other", "unknown"})

_PUBLIC_SCOPE_ALIASES = frozenset({"public", "global", "all"})
_LOCAL_SCOPE_ALIASES = frozenset({"local", "private", "scoped"})


def canonicalize_path(path: object) -> str:
    """生成用于比较/去重的稳定路径键。"""
    raw = str(path or "").replace("\\", "/")
    try:
        normalized = posixpath.normpath(raw)
    except Exception:
        normalized = raw
    return normalized.casefold()


def normalize_scope_mode(
    value: object,
    *,
    default: str | None = "public",
) -> str | None:
    """把作用域及其历史别名统一为 ``public`` / ``local``。

    ``default=None`` 适合需要区分“无效输入”的命令参数校验；持久化与读取
    场景使用默认的 ``public``，保持旧数据的安全回退语义。
    """
    raw = str(value or "").strip().lower()
    if raw in _PUBLIC_SCOPE_ALIASES:
        return "public"
    if raw in _LOCAL_SCOPE_ALIASES:
        return "local"
    return default


def normalize_character_key(value: object) -> str:
    """规范化角色键。"""
    return str(value or "").strip().lower()


def normalize_category_key(value: object) -> str:
    """Return a safe category directory key or raise ``ValueError``.

    Category keys are persisted as directory names and model labels, so keep
    them portable and stable across Windows/Linux and case-insensitive stores.
    """
    key = str(value or "").strip().lower()
    if not key:
        raise ValueError("分类 key 不能为空")
    if not _CATEGORY_KEY_RE.fullmatch(key):
        raise ValueError("分类 key 需以英文字母开头，仅含小写字母、数字、_、-，最长48字符")
    if key in _WINDOWS_RESERVED_NAMES:
        raise ValueError("分类 key 是系统保留名称，请更换")
    if key in _PLUGIN_RESERVED_CATEGORY_KEYS:
        raise ValueError("分类 key 是插件保留名称，请更换")
    return key


def normalize_label_list(
    values: Any,
    max_count: int | None = None,
    *,
    allow_duplicates: bool = False,
    csv_only: bool = False,
) -> list[str]:
    """规范化标签、场景或情绪列表。

    字符串默认支持中英文逗号、顿号、分号与换行；``csv_only`` 用于保留
    只把英文逗号视为分隔符的旧接口语义。
    """
    if max_count is not None and max_count <= 0:
        return []

    if isinstance(values, str):
        splitter = _CSV_SEPARATOR_RE if csv_only else _METADATA_SEPARATOR_RE
        items = [item.strip() for item in splitter.split(values) if item.strip()]
    elif isinstance(values, (list, tuple)):
        items = [
            str(item).strip()
            for item in values
            if item is not None and str(item).strip()
        ]
    else:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not allow_duplicates and item in seen:
            continue
        seen.add(item)
        result.append(item)
        if max_count is not None and len(result) >= max_count:
            break
    return result
