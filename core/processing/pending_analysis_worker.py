"""Drain approved images using durable pending rows and the existing scheduler."""

import asyncio
import json
from pathlib import Path

from astrbot.api import logger

from .pending_promotion import promote_pending


class PendingAnalysisWorker:
    def __init__(self, plugin):
        self.plugin = plugin
        self.wakeup = asyncio.Event()

    async def start(self):
        existing = self.plugin.task_scheduler._tasks.get('pending_analysis')
        if existing is not None and not existing.done():
            return
        await asyncio.to_thread(self.plugin.db_service.recover_pending_analysis)
        self.plugin.task_scheduler.create_task('pending_analysis', self.run(), replace_existing=False)

    async def stop(self):
        await self.plugin.task_scheduler.cancel_task('pending_analysis')

    async def process_one(self) -> bool:
        db = self.plugin.db_service
        row = await db.claim_pending_analysis()
        if row is None:
            return False
        pending_id = row['id']
        try:
            if not Path(row['path']).is_file():
                raise FileNotFoundError('待审核原图已丢失')
            if row['analysis_state'] != 'ready':
                proc = self.plugin.image_processor_service
                category, tags, desc, emotion, scenes, text, emotions = await proc.classify_image(None, row['path'])
                if category in (proc.CATEGORY_FILTERED, proc.CATEGORY_NOT_EMOJI) or emotion in (proc.CATEGORY_FILTERED, proc.CATEGORY_NOT_EMOJI):
                    raise ValueError(category or emotion)
                if not category:
                    raise ValueError('识图失败，请检查视觉模型后重试')
                cfg = self.plugin.plugin_config
                if category not in cfg.categories:
                    category = cfg.closest_category(category)
                fields = dict(category=category, tags=tags, desc=desc, scenes=scenes, overlay_text=text, emotions=emotions)
                manual = set(json.loads(row['manual_fields']))
                fields = {key: value for key, value in fields.items() if key not in manual}
                if not fields:
                    fields = {'category': row['category'] or category}
                if not await db.update_pending(pending_id, fields, from_analysis=True):
                    raise RuntimeError('识图结果保存失败')
            # 文件发布与元数据提升由独立业务服务处理。
            ok, error = await promote_pending(self.plugin, pending_id)
            if not ok:
                raise RuntimeError(error)
        except asyncio.CancelledError:
            await db.finish_pending_analysis(pending_id)
            raise
        except Exception as exc:
            await db.finish_pending_analysis(pending_id, str(exc))
            logger.warning(f'审核图片后台处理失败 id={pending_id}: {exc}')
        return True

    async def run(self):
        while True:
            self.wakeup.clear()
            try:
                if await self.process_one():
                    continue
            except Exception as exc:
                logger.warning(f'审核图片后台队列异常: {exc}')
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
