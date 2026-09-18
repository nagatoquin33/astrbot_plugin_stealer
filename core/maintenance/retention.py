"""Library ownership and deterministic weighted capacity eviction."""

import math


def library_group(meta: dict) -> str:
    """Favorites take precedence without changing the stored character."""
    if meta.get("is_favorite"):
        return "favorites"
    if str(meta.get("character") or "").strip():
        return "characters"
    return "general"


def is_auto_evictable(meta: object) -> bool:
    return (
        isinstance(meta, dict)
        and library_group(meta) == "general"
        and str(meta.get("retention_class") or "native").strip()
        not in {"external", "pinned"}
    )


def library_counts(index: dict) -> dict[str, int]:
    counts = dict(general=0, favorites=0, characters=0, automatic=0)
    for meta in index.values():
        if isinstance(meta, dict):
            counts[library_group(meta)] += 1
            counts["automatic"] += int(is_auto_evictable(meta))
    return counts


def nonnegative_number(value: object, default: float = 0) -> float:
    try:
        number = float(value)
        return max(0, number) if math.isfinite(number) else default
    except (ValueError, TypeError, OverflowError):
        return default


def eviction_candidates(index: dict, limit: int, usage_weight: float = 0.7) -> list[tuple[str, int]]:
    """Score low usage and old creation time on [0, 1]; largest score first.

    Only eligible entries contribute to either normalization or the quota.
    Equal scores prefer lower usage, then older creation, then stable path.
    """
    if limit <= 0:
        return []
    items = [
        (path, nonnegative_number(meta.get("use_count")),
         int(nonnegative_number(meta.get("created_at"))))
        for path, meta in index.items() if is_auto_evictable(meta)
    ]
    overflow = len(items) - limit
    if overflow <= 0:
        return []
    weight = min(1, nonnegative_number(usage_weight, 0.7))
    min_usage = min(item[1] for item in items)
    usage_span = max(item[1] for item in items) - min_usage
    min_created = min(item[2] for item in items)
    age_span = max(item[2] for item in items) - min_created

    def sort_key(item):
        path, usage, created = item
        low_usage = 1 - (usage - min_usage) / usage_span if usage_span else 0
        old_age = 1 - (created - min_created) / age_span if age_span else 0
        score = weight * low_usage + (1 - weight) * old_age
        return -score, usage, created, path

    items.sort(key=sort_key)
    return [(path, created) for path, _, created in items[:overflow]]
