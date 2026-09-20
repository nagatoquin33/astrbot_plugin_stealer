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
    """写入唯一持久化来源；数据库不可用时明确返回失败。"""
    normalized_hash = str(image_hash or "").strip()
    if not normalized_hash:
        return False

    created_at = int(time.time()) if timestamp is None else int(timestamp)
    try:
        db = getattr(plugin, "db_service", None)
        if db is not None and hasattr(db, "add_blacklist"):
            await db.add_blacklist(normalized_hash, created_at)
            return True

    except Exception as exc:
        logger.error(f"写入黑名单失败: {exc}", exc_info=True)
    return False
