"""Human-readable labels for command-rendered meme listings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote


_EXTERNAL_CHANNELS = {
    "meme_pack": "资源包",
    "github": "GitHub 仓库",
    "http_json": "JSON API",
}

_SOURCE_LABELS = {
    "auto": "自动收录",
    "automatic": "自动收录",
    "manual": "手动添加",
    "llm": "LLM 工具",
    "api": "API 导入",
    "qq_store": "QQ 商城",
}


def image_source_label(source: Any, add_method: Any = "") -> str:
    """Translate stored provenance markers into concise user-facing labels."""
    value = str(source or "").strip()
    method = str(add_method or "").strip()
    if value.startswith("external:") or method == "external_import":
        channel = value.partition(":")[2].strip().lower()
        channel_name = _EXTERNAL_CHANNELS.get(channel)
        if not channel_name:
            channel_name = channel.replace("_", " ").strip() or "其他渠道"
        return f"外部导入 · {channel_name}"
    return _SOURCE_LABELS.get(value.lower(), value)


def _readable_filename(value: Any, *, hide_opaque: bool) -> str:
    filename = str(value or "").strip().replace("\\", "/")
    if not filename:
        return ""
    filename = unquote(filename.rsplit("/", 1)[-1]).strip()
    stem = Path(filename).stem.strip()
    if not stem or "\ufffd" in stem or any(ord(char) < 32 for char in stem):
        return ""

    # Imported files are stored as ext_<hash>_<original-stem>.<ext>.
    generated = re.match(r"^ext_[0-9a-f]{12}(?:_(.*))?$", stem, flags=re.IGNORECASE)
    if generated:
        stem = str(generated.group(1) or "").strip()
        if not stem:
            return ""
    if hide_opaque and re.fullmatch(r"[0-9a-f]{12,}", stem, flags=re.IGNORECASE):
        return ""
    return stem[:120]


def image_display_title(item: dict[str, Any]) -> str:
    """Prefer content text and original names; hide opaque import hashes."""
    external = str(item.get("source") or "").startswith("external:") or str(
        item.get("add_method") or ""
    ) == "external_import"
    for field in ("desc", "overlay_text"):
        value = " ".join(str(item.get(field) or "").split()).strip()
        opaque_hash = external and bool(
            re.fullmatch(r"[0-9a-f]{12,}", value, flags=re.IGNORECASE)
        )
        if value and "\ufffd" not in value and not opaque_hash:
            return value[:120]

    for field in ("original_name", "name"):
        value = _readable_filename(item.get(field), hide_opaque=external)
        if value:
            return value

    return "未命名表情"
