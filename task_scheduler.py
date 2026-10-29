import asyncio
from collections.abc import Coroutine
from typing import Any

from astrbot.api import logger


class TaskScheduler:
    """任务调度器类，负责管理后台任务的创建、执行和取消。"""

    def __init__(self):
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False

    @staticmethod
    def _task_done_callback(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"后台任务 '{task.get_name()}' 异常退出: {exc!r}",
                exc_info=exc,
            )

    @classmethod
    def create_detached_task(
        cls,
        coro: Coroutine[Any, Any, Any],
        *,
        name: str = "",
    ) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name or None)
        task.add_done_callback(cls._task_done_callback)
        return task

    def create_task(
        self, name: str, coro: Coroutine[Any, Any, Any], replace_existing: bool = True, *, retain: bool = True
    ) -> asyncio.Task[Any] | None:
        if self._closed:
            coro.close()
            return None
        if name in self._tasks:
            if replace_existing:
                old_task = self._tasks.pop(name)
                if not old_task.done():
                    old_task.cancel()
            else:
                return None

        try:
            task = self.create_detached_task(coro, name=name)
            self._tasks[name] = task
            if not retain:
                def forget(completed):
                    if self._tasks.get(name) is completed:
                        self._tasks.pop(name)
                task.add_done_callback(forget)
            logger.info(f"创建任务: {name}")
            return task
        except Exception as e:
            logger.error(f"创建任务 {name} 失败: {e}")
            return None

    async def cancel_task(self, name: str) -> bool:
        if name not in self._tasks:
            return False

        try:
            task = self._tasks[name]
            if not task.done():
                try:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"任务 {name} 取消时出错: {e}")
            if self._tasks.get(name) is task:
                self._tasks.pop(name)
            logger.info(f"取消任务: {name}")
            return True
        except Exception as e:
            logger.error(f"取消任务 {name} 失败: {e}")
            return False

    async def cancel_all_tasks(self) -> None:
        for name in list(self._tasks.keys()):
            await self.cancel_task(name)

    async def cleanup(self):
        self._closed = True
        await self.cancel_all_tasks()
        logger.info("任务调度器已关闭")
