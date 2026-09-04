"""Reader for AstrBot Meme Pack exports and Meme Manager pack directories.

Only the public pack layout is consumed.  The reader never imports another
plugin, executes files, or extracts an archive blindly; every member is
validated before it can be read.
"""

from __future__ import annotations

import json
import posixpath
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .models import ExternalSourceError, ExternalSourceSecurityError, PackInspection, SourceItem


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_ID_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def safe_member_path(value: str) -> str:
    """Return a normalized relative archive path or raise a security error."""

    raw = str(value or "").replace("\\", "/")
    if not raw or "\x00" in raw:
        raise ExternalSourceSecurityError("pack member path is empty or contains NUL")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise ExternalSourceSecurityError(f"absolute pack member path: {value}")
    normalized = posixpath.normpath(raw)
    if normalized in {"", "."} or normalized == ".." or normalized.startswith("../"):
        raise ExternalSourceSecurityError(f"path traversal in pack member: {value}")
    return normalized


def _coerce_list(value: Any, *, limit: int = 16) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[,，;；\n\t]+", value)
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value_item in values:
        text = str(value_item or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _safe_id(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = _ID_RE.sub("_", text).strip("._")
    return text[:180] or fallback


def _json_bytes(raw: bytes, label: str) -> dict[str, Any] | list[Any] | None:
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ExternalSourceError(f"{label} exceeds the metadata size limit")
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExternalSourceError(f"invalid JSON in {label}") from exc
    if not isinstance(value, (dict, list)):
        raise ExternalSourceError(f"{label} must contain an object or array")
    return value


def _iter_metadata(value: Any, prefix: str = ""):
    """Yield path/id keyed metadata dictionaries from several pack versions."""

    if isinstance(value, dict):
        path_keys = ("relative_path", "path", "file", "file_path", "filename", "name")
        path_value = next((value.get(key) for key in path_keys if value.get(key)), None)
        if path_value:
            yield str(path_value), value
        for key, nested in value.items():
            if isinstance(nested, dict):
                # A dict keyed by the relative path is common in legacy exports.
                if ("/" in str(key) or "\\" in str(key) or Path(str(key)).suffix.lower() in IMAGE_EXTENSIONS) and not any(
                    value.get(candidate) for candidate in path_keys
                ):
                    yield str(key), nested
                yield from _iter_metadata(nested, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(nested, list):
                yield from _iter_metadata(nested, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _iter_metadata(nested, f"{prefix}[{index}]")


def _metadata_index(values: list[Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for value in values:
        for raw_path, metadata in _iter_metadata(value):
            try:
                path = safe_member_path(raw_path)
            except ExternalSourceSecurityError:
                continue
            index[path] = dict(metadata)
            index.setdefault(PurePosixPath(path).name, dict(metadata))
            if path.startswith("memes/"):
                index.setdefault(path[6:], dict(metadata))
    return index


def _manifest_payload(value: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    for nested_key in ("manifest", "pack"):
        nested = value.get(nested_key)
        if isinstance(nested, dict):
            merged = dict(value)
            merged.pop(nested_key, None)
            merged.update(nested)
            return merged
    return dict(value)


class PackSource:
    """Inspect and read a pack directory or ZIP export."""

    def __init__(
        self,
        source_path: str | Path,
        *,
        max_items: int = 2000,
        max_archive_bytes: int = 1024 * 1024 * 1024,
        max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024,
        max_file_bytes: int = 1024 * 1024 * 1024,
        max_members: int = 20_000,
        member_prefix: str = "",
    ) -> None:
        self.source_path = Path(source_path).expanduser().resolve()
        self.max_items = max(1, int(max_items))
        self.max_archive_bytes = max(1, int(max_archive_bytes))
        self.max_uncompressed_bytes = max(1, int(max_uncompressed_bytes))
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.max_members = max(1, int(max_members))
        raw_prefix = str(member_prefix or "").replace("\\", "/").strip("/")
        self.member_prefix = safe_member_path(raw_prefix).rstrip("/") if raw_prefix else ""
        self._inspection: PackInspection | None = None
        self._zip_members: dict[str, zipfile.ZipInfo] = {}

    @property
    def is_archive(self) -> bool:
        return self.source_path.is_file() and self.source_path.suffix.lower() in {".zip", ".meme-pack"}

    def inspect(self) -> PackInspection:
        if self._inspection is not None:
            return self._inspection
        if not self.source_path.exists():
            raise ExternalSourceError(f"source path does not exist: {self.source_path}")
        if self.is_archive:
            try:
                inspection = self._inspect_archive()
            except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError) as exc:
                raise ExternalSourceError(f"invalid or unsupported pack archive: {exc}") from exc
        elif self.source_path.is_dir():
            inspection = self._inspect_directory()
        else:
            raise ExternalSourceError("source must be a directory or ZIP archive")
        self._inspection = inspection
        return inspection

    def read_item(self, item: SourceItem) -> bytes:
        """Read one already-inspected item while applying the byte limit again."""

        if self.is_archive:
            member = self._zip_members.get(item.local_ref)
            if member is None:
                raise ExternalSourceError(f"pack item no longer exists: {item.local_ref}")
            with zipfile.ZipFile(self.source_path) as archive:
                raw = archive.read(member)
        else:
            root = self.source_path.resolve()
            path = (root / item.local_ref).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ExternalSourceSecurityError("pack item escapes source directory") from exc
            if path.is_symlink() or not path.is_file():
                raise ExternalSourceError(f"pack item is not a regular file: {item.local_ref}")
            raw = path.read_bytes()
        if len(raw) > self.max_file_bytes:
            raise ExternalSourceSecurityError(f"pack item exceeds the per-file byte limit: {item.relative_path}")
        return raw

    @staticmethod
    def detect_archive_root(
        source_path: str | Path,
        *,
        max_members: int | None = None,
    ) -> str:
        """Return a single top-level directory used by GitHub archive ZIPs."""
        try:
            with zipfile.ZipFile(source_path) as archive:
                infos = archive.infolist()
                if max_members is not None and len(infos) > max_members:
                    raise ExternalSourceSecurityError("pack archive contains too many members")
                names = [
                    safe_member_path(info.filename)
                    for info in infos
                    if not info.is_dir()
                ]
        except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, NotImplementedError, OSError) as exc:
            raise ExternalSourceError(f"invalid or unsupported pack archive: {exc}") from exc
        if not names:
            return ""
        first_part = names[0].split("/", 1)[0]
        if all(name == first_part or name.startswith(f"{first_part}/") for name in names):
            return first_part
        return ""

    def _logical_member_name(self, member_name: str) -> str | None:
        if not self.member_prefix:
            return member_name
        prefix = f"{self.member_prefix}/"
        if member_name == self.member_prefix:
            return ""
        if not member_name.startswith(prefix):
            return None
        return member_name[len(prefix) :]

    def _inspect_archive(self) -> PackInspection:
        archive_bytes = self.source_path.stat().st_size
        if archive_bytes > self.max_archive_bytes:
            raise ExternalSourceSecurityError("pack archive exceeds the compressed byte limit")
        warnings: list[str] = []
        errors: list[str] = []
        total_uncompressed = 0
        json_values: list[Any] = []
        image_members: list[tuple[str, zipfile.ZipInfo]] = []
        with zipfile.ZipFile(self.source_path) as archive:
            members = archive.infolist()
            if len(members) > self.max_members:
                raise ExternalSourceSecurityError("pack archive contains too many members")
            for info in members:
                try:
                    member_name = safe_member_path(info.filename)
                except ExternalSourceSecurityError as exc:
                    errors.append(str(exc))
                    continue
                if info.is_dir():
                    continue
                logical_name = self._logical_member_name(member_name)
                if logical_name is None or not logical_name:
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    errors.append(f"symlink member is not allowed: {logical_name}")
                    continue
                if info.file_size > self.max_file_bytes:
                    errors.append(f"member exceeds the per-file byte limit: {logical_name}")
                    continue
                total_uncompressed += int(info.file_size)
                if total_uncompressed > self.max_uncompressed_bytes:
                    errors.append("pack archive exceeds the uncompressed byte limit")
                    break
                if logical_name in self._zip_members:
                    errors.append(f"duplicate pack member is not allowed: {logical_name}")
                    continue
                self._zip_members[logical_name] = info
                lower = logical_name.lower()
                if lower.endswith(
                    (
                        "meme_pack_export.json",
                        "manifest.json",
                        "memes_data.json",
                        "semantic_index.json",
                        "semantic_metadata.json",
                    )
                ):
                    if info.file_size > _MAX_MANIFEST_BYTES:
                        errors.append(f"{logical_name} exceeds the metadata size limit")
                        continue
                    try:
                        json_values.append(_json_bytes(archive.read(info), logical_name))
                    except ExternalSourceError as exc:
                        errors.append(str(exc))
                if Path(logical_name).suffix.lower() in IMAGE_EXTENSIONS and not self._is_preview(logical_name):
                    image_members.append((logical_name, info))
        manifest = self._find_manifest(json_values)
        items = self._build_items(image_members, manifest, json_values, warnings)
        if not items:
            errors.append("pack contains no supported images")
        if len(image_members) > self.max_items:
            warnings.append(f"source contains more than {self.max_items} images; preview is truncated")
        name = str(manifest.get("name") or manifest.get("id") or self.source_path.stem)
        return PackInspection(
            source_type="meme_pack",
            source_id=self._source_id(manifest),
            name=name[:160],
            endpoint=str(self.source_path),
            manifest=manifest,
            items=items,
            warnings=warnings,
            errors=errors,
            total_bytes=total_uncompressed,
            truncated=len(image_members) > self.max_items,
            source_path=self.source_path,
            is_archive=True,
        )

    def _inspect_directory(self) -> PackInspection:
        root = self.source_path
        warnings: list[str] = []
        errors: list[str] = []
        manifest_paths = [root / "manifest.json", root / "meme_pack_export.json"]
        if not any(path.is_file() for path in manifest_paths):
            for candidate in sorted(root.glob("*/manifest.json")):
                if candidate.is_file():
                    manifest_paths.append(candidate)
                    break
        json_values: list[Any] = []
        for candidate in manifest_paths:
            if not candidate.is_file():
                continue
            try:
                if candidate.stat().st_size > _MAX_MANIFEST_BYTES:
                    raise ExternalSourceError(
                        f"{candidate.relative_to(root)} exceeds the metadata size limit"
                    )
                json_values.append(_json_bytes(candidate.read_bytes(), str(candidate.relative_to(root))))
            except (OSError, ValueError, ExternalSourceError) as exc:
                errors.append(str(exc))
        for candidate in (
            root / "memes_data.json",
            root / "semantic_index.json",
            root / "semantic_metadata.json",
        ):
            if candidate.is_file():
                try:
                    if candidate.stat().st_size > _MAX_MANIFEST_BYTES:
                        raise ExternalSourceError(
                            f"{candidate.relative_to(root)} exceeds the metadata size limit"
                        )
                    json_values.append(_json_bytes(candidate.read_bytes(), str(candidate.relative_to(root))))
                except (OSError, ValueError, ExternalSourceError) as exc:
                    errors.append(str(exc))
        image_members: list[tuple[str, zipfile.ZipInfo | None]] = []
        member_count = 0
        total_uncompressed = 0
        for path in root.rglob("*"):
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                member_count += 1
                if member_count > self.max_members:
                    errors.append("pack directory contains too many files")
                    break
                relative = safe_member_path(str(path.relative_to(root)))
                size = path.stat().st_size
                if size > self.max_file_bytes:
                    errors.append(f"member exceeds the per-file byte limit: {relative}")
                    continue
                total_uncompressed += int(size)
                if total_uncompressed > self.max_uncompressed_bytes:
                    errors.append("pack directory exceeds the total byte limit")
                    break
                if path.suffix.lower() not in IMAGE_EXTENSIONS or self._is_preview(relative):
                    continue
                image_members.append((relative, None))
            except (OSError, ValueError, ExternalSourceSecurityError) as exc:
                errors.append(str(exc))
        manifest = self._find_manifest(json_values)
        items = self._build_items(image_members, manifest, json_values, warnings)
        if not items:
            errors.append("pack contains no supported images")
        if len(image_members) > self.max_items:
            warnings.append(f"source contains more than {self.max_items} images; preview is truncated")
        name = str(manifest.get("name") or manifest.get("id") or root.name)
        return PackInspection(
            source_type="meme_pack",
            source_id=self._source_id(manifest),
            name=name[:160],
            endpoint=str(root),
            manifest=manifest,
            items=items,
            warnings=warnings,
            errors=errors,
            total_bytes=total_uncompressed,
            truncated=len(image_members) > self.max_items,
            source_path=root,
            is_archive=False,
        )

    @staticmethod
    def _is_preview(path: str) -> bool:
        parts = {part.lower() for part in PurePosixPath(path).parts}
        return bool(parts & {"preview", "previews", "thumbnail", "thumbnails", "thumbs"})

    @staticmethod
    def _find_manifest(values: list[Any]) -> dict[str, Any]:
        for value in values:
            payload = _manifest_payload(value)
            if any(key in payload for key in ("id", "name", "categories")):
                return payload
        return {}

    def _build_items(
        self,
        members: list[tuple[str, zipfile.ZipInfo | None]],
        manifest: dict[str, Any],
        json_values: list[Any],
        warnings: list[str],
    ) -> list[SourceItem]:
        metadata = _metadata_index(json_values)
        default_license = str(manifest.get("license") or "").strip()
        items: list[SourceItem] = []
        for relative, info in sorted(members, key=lambda value: value[0].lower())[: self.max_items]:
            metadata_path = relative
            if "/memes/" in relative:
                # ZIP exports often wrap the documented layout in one archive
                # root directory.  Semantic metadata still points at memes/.
                metadata_path = f"memes/{relative.split('/memes/', 1)[1]}"
            lookup = (
                metadata.get(relative)
                or metadata.get(metadata_path)
                or metadata.get(relative[6:] if relative.startswith("memes/") else "")
                or metadata.get(Path(relative).name, {})
            )
            lookup = lookup if isinstance(lookup, dict) else {}
            category = str(lookup.get("category") or self._category_from_path(relative)).strip()
            source_url = str(lookup.get("source_url") or lookup.get("url") or "").strip()
            item_id = _safe_id(
                lookup.get("id") or lookup.get("external_id") or lookup.get("entry_id"),
                relative,
            )
            item = SourceItem(
                external_id=item_id,
                relative_path=relative,
                category=category,
                description=str(lookup.get("description") or "").strip(),
                caption=str(lookup.get("caption") or lookup.get("desc") or "").strip(),
                visible_text=str(lookup.get("visible_text") or lookup.get("overlay_text") or "").strip(),
                tags=_coerce_list(lookup.get("tags")),
                scenes=_coerce_list(lookup.get("scenes") or lookup.get("scene")),
                emotions=_coerce_list(lookup.get("emotions")),
                filename=Path(relative).name,
                source_url=source_url,
                license=str(lookup.get("license") or default_license).strip(),
                attribution=str(lookup.get("attribution") or manifest.get("author") or "").strip(),
                metadata=dict(lookup),
                size=int(info.file_size) if info is not None else self._directory_size(relative),
                local_ref=relative,
            )
            items.append(item)
        return items

    def _directory_size(self, relative: str) -> int:
        try:
            return int((self.source_path / relative).stat().st_size)
        except OSError:
            return 0

    @staticmethod
    def _category_from_path(relative: str) -> str:
        parts = PurePosixPath(relative).parts
        if "memes" in parts:
            index = parts.index("memes")
            if index + 1 < len(parts) - 1:
                return parts[index + 1]
        return parts[-2] if len(parts) > 1 else ""

    def _source_id(self, manifest: dict[str, Any]) -> str:
        value = _safe_id(manifest.get("id"), self.source_path.stem)
        return f"pack:{value}"[:190]
