"""Data contracts shared by external source readers and the importer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ExternalSourceError(RuntimeError):
    """A source is malformed or cannot be read."""


class ExternalSourceSecurityError(ExternalSourceError):
    """A source failed a safety check (path, URL, size, or image type)."""


@dataclass(slots=True)
class SourceItem:
    """One image plus normalized metadata from a source catalog."""

    external_id: str
    relative_path: str = ""
    category: str = ""
    description: str = ""
    caption: str = ""
    visible_text: str = ""
    tags: list[str] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)
    filename: str = ""
    source_url: str = ""
    license: str = ""
    attribution: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    size: int = 0
    # Pack readers use this local reference.  HTTP readers leave it empty and
    # the importer downloads ``source_url`` instead.
    local_ref: str = ""

    @property
    def text_description(self) -> str:
        return str(self.description or self.caption or "").strip()


@dataclass(slots=True)
class SourceInspection:
    source_type: str
    source_id: str
    name: str
    endpoint: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)
    items: list[SourceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_bytes: int = 0
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def categories(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in self.items:
            key = str(item.category or "").strip()
            if key and key not in seen:
                seen.add(key)
                result.append(key)
        return result

    def as_dict(self, *, preview_limit: int = 20) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "name": self.name,
            "endpoint": self.endpoint,
            "manifest": self.manifest,
            "item_count": len(self.items),
            "total_bytes": self.total_bytes,
            "categories": self.categories,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "truncated": self.truncated,
            "items": [
                {
                    "id": item.external_id,
                    "path": item.relative_path,
                    "category": item.category,
                    "filename": item.filename,
                    "description": item.text_description,
                    "tags": list(item.tags),
                    "source_url": item.source_url,
                    "license": item.license,
                }
                for item in self.items[: max(0, int(preview_limit))]
            ],
        }


@dataclass(slots=True)
class PackInspection(SourceInspection):
    source_path: Path | None = None
    is_archive: bool = False
