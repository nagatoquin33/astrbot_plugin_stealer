"""待审核图片发布服务，供 HTTP 审核和后台识图共用。"""

import asyncio
import os

from astrbot.api import logger

from ..db.index_manager import refresh_search_entry
from ..util.safe_io import commit_image_file, unique_image_path


async def promote_pending(plugin, pending_id: int) -> tuple[bool, str]:
    """发布待审核图片并提交分类、标签、场景等元数据，成功后刷新检索索引。

    文件发布与元数据提交通过补偿流程处理失败。
    """
    db = plugin.db_service
    config = plugin.plugin_config
    row = await asyncio.to_thread(db.get_pending, pending_id)
    if not row:
        return False, "pending not found"

    src_path = str(row.get("path", "") or "")
    if not src_path or not os.path.isfile(src_path):
        # 文件已丢失：清理孤儿 pending 行
        await asyncio.to_thread(db.delete_pending, pending_id)
        return False, "pending file missing"

    category = str(row.get("category", "") or "").strip()
    known = set(config.get_vlm_categories() if hasattr(config, "get_vlm_categories") else (config.categories or []))
    if not category or category == "other" or category not in known:
        category = config.closest_category(category) if hasattr(config, "closest_category") else "confused"

    async with db.file_lock:
        if category not in config.categories:
            return False, "分类已被移除"
        cat_dir = config.ensure_category_dir(category)
        cat_path = str(unique_image_path(cat_dir, os.path.basename(src_path)))
        try:
            async def commit_approval(_target):
                if category not in config.categories:
                    raise ValueError("分类已被移除")
                if not await db.approve_pending(pending_id, cat_path, category):
                    raise RuntimeError("pending metadata promotion failed")
            await commit_image_file(src_path, cat_path, commit_approval)
            await refresh_search_entry(plugin, cat_path)
            return True, ""
        except Exception as exc:
            logger.error(f"pending approval failed id={pending_id}: {exc}", exc_info=True)
            return False, str(exc)
