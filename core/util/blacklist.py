"""表情包黑名单的统一写入入口。"""

import time
from typing import Any

from astrbot.api import logger


async def add_blacklist_hash(
    plugin: Any,
    image_hash: object,
    *,
    timestamp: int | None = None,
) -> bool:
    """优先写数据库，数据库不可用时回退旧缓存。"""
    normalized_hash = str(image_hash or "").strip()
    if not normalized_hash:
        return False

    created_at = int(time.time()) if timestamp is None else int(timestamp)
    try:
        db = getattr(plugin, "db_service", None)
        if db is not None and hasattr(db, "add_blacklist"):
            await db.add_blacklist(normalized_hash, created_at)
            return True

        cache = getattr(plugin, "cache_service", None)
        if cache is not None:
            await cache.set(
                "blacklist_cache",
                normalized_hash,
                created_at,
                persist=True,
            )
            return True
    except Exception as exc:
        logger.error(f"写入黑名单失败: {exc}", exc_info=True)
    return False
