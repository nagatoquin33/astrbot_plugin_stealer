"""Pure metadata matching used when rebuilding the image index."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def has_meaningful_metadata(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("tags") or metadata.get("desc") or metadata.get("scenes"))


def _build_lookup_maps(
    index_map: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    hash_map: dict[str, dict[str, Any]] = {}
    name_map: dict[str, dict[str, Any]] = {}
    for path_key, metadata in index_map.items():
        if not isinstance(metadata, dict):
            continue

        if metadata.get("hash"):
            hash_value = str(metadata["hash"])
            existing = hash_map.get(hash_value)
            if existing is None or (
                not has_meaningful_metadata(existing)
                and has_meaningful_metadata(metadata)
            ):
                hash_map[hash_value] = metadata

        path = Path(path_key)
        for name_key in (path.name, path.stem):
            existing = name_map.get(name_key)
            if existing is None or (
                not has_meaningful_metadata(existing)
                and has_meaningful_metadata(metadata)
            ):
                name_map[name_key] = metadata
    return hash_map, name_map


def _first_casefold_stem_match(
    new_path: Path, *source_indexes: dict[str, Any]
) -> dict[str, Any] | None:
    needle = new_path.stem.lower()
    for source_index in source_indexes:
        for old_path, old_value in source_index.items():
            if not isinstance(old_value, dict):
                continue
            if Path(old_path).stem.lower() == needle:
                return old_value
    return None


def _restore_metadata(
    target: dict[str, Any], source: dict[str, Any] | None
) -> None:
    if not isinstance(source, dict):
        return
    if source.get("desc"):
        target["desc"] = source["desc"]
    if source.get("tags"):
        target["tags"] = source["tags"]

    for key in (
        "is_favorite",
        "character",
        "retention_class",
        "use_count",
        "last_used_at",
        "created_at",
        "source_message",
        "source",
        "origin_target",
        "scope_mode",
        "qq_emoji_id",
        "qq_emoji_package_id",
        "origin_url",
        "qq_key",
        "scenes",
        "scene",
    ):
        if key in source:
            target[key] = source[key]


def restore_rebuilt_metadata(
    rebuilt_index: dict[str, Any],
    database_index: dict[str, Any],
    legacy_index: dict[str, Any],
) -> None:
    """Restore metadata while retaining the established source-match order."""
    database_hashes, database_names = _build_lookup_maps(database_index)
    legacy_hashes, legacy_names = _build_lookup_maps(legacy_index)

    for new_path, new_metadata in rebuilt_index.items():
        if not isinstance(new_metadata, dict):
            continue
        new_path_obj = Path(new_path)
        new_hash = new_metadata.get("hash")
        candidates = (
            database_index.get(new_path),
            database_hashes.get(new_hash),
            legacy_index.get(new_path),
            legacy_hashes.get(new_hash),
            database_names.get(new_path_obj.name),
            database_names.get(new_path_obj.stem),
            legacy_names.get(new_path_obj.name),
            legacy_names.get(new_path_obj.stem),
        )
        old_metadata = next(
            (candidate for candidate in candidates if isinstance(candidate, dict)),
            None,
        )
        if old_metadata is None:
            old_metadata = _first_casefold_stem_match(
                new_path_obj, database_index, legacy_index
            )
        _restore_metadata(new_metadata, old_metadata)
