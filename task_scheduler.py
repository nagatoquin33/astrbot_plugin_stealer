import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from astrbot.api import logger


class TaskScheduler:
    """任务调度器类，负责管理后台任务的创建、执行和取消。"""

    def __init__(self):
        """初始化任务调度器。"""
        self._tasks: dict[str, asyncio.Task[Any]] = {}  # 任务字典，key为任务名称
        self._task_callbacks: dict[str, Callable[..., Any]] = {}  # 任务回调函数

    @staticmethod
    def _task_done_callback(task: asyncio.Task[Any]) -> None:
        """任务完成回调，记录未处理的异常。"""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                f"后台任务 '{task.get_name()}' 异常退出: {exc!r}",
                exc_info=exc,
            )

    def create_task(
        self, name: str, coro: Coroutine[Any, Any, Any], replace_existing: bool = True
    ) -> asyncio.Task[Any] | None:
        """创建一个新的后台任务。

        Args:
            name: 任务名称
            coro: 协程对象
            replace_existing: 如果任务已存在，是否替换

        Returns:
            创建的任务对象，如果任务已存在且不替换则返回None
        """
        if name in self._tasks:
            if replace_existing:
                old_task = self._tasks.pop(name)
                if not old_task.done():
                    old_task.cancel()
            else:
                return None

        try:
            task = asyncio.create_task(coro, name=name)
            task.add_done_callback(self._task_done_callback)
            self._tasks[name] = task
            logger.info(f"创建任务: {name}")
            return task
        except Exception as e:
            logger.error(f"创建任务 {name} 失败: {e}")
            return None

    async def cancel_task(self, name: str) -> bool:
        """取消指定名称的任务。

        Args:
            name: 任务名称

        Returns:
            任务是否成功取消
        """
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
            del self._tasks[name]
            logger.info(f"取消任务: {name}")
            return True
        except Exception as e:
            logger.error(f"取消任务 {name} 失败: {e}")
            return False

    async def cancel_all_tasks(self) -> None:
        """取消所有任务。"""
        for name in list(self._tasks.keys()):
            await self.cancel_task(name)

    def schedule_interval_task(
        self,
        name: str,
        callback: Callable[..., Coroutine[Any, Any, Any]],
        interval: float,
        replace_existing: bool = True,
    ) -> asyncio.Task[Any] | None:
        """调度一个定期执行的任务。

        Args:
            name: 任务名称
            callback: 定期执行的异步回调函数
            interval: 执行间隔（秒）
            replace_existing: 如果任务已存在，是否替换

        Returns:
            创建的任务对象，如果任务已存在且不替换则返回None
        """
        # 保存回调函数
        self._task_callbacks[name] = callback

        # 定义循环执行的协程
        async def interval_task():
            while True:
                try:
                    await asyncio.shield(callback())
                except asyncio.CancelledError:
                    logger.info(f"任务 {name} 被取消，退出循环")
                    raise
                except Exception as e:
                    logger.error(f"执行定期任务 {name} 失败: {e}", exc_info=True)

                try:
                    await asyncio.sleep(max(1.0, interval))
                except asyncio.CancelledError:
                    logger.info(f"任务 {name} 在睡眠时被取消")
                    raise

        # 创建任务
        return self.create_task(name, interval_task(), replace_existing)

    def schedule_interval_task_minutes(
        self,
        name: str,
        callback: Callable,
        interval_minutes: float,
        replace_existing: bool = True,
    ) -> asyncio.Task | None:
        """调度一个定期执行的任务（以分钟为单位）。

        Args:
            name: 任务名称
            callback: 定期执行的回调函数
            interval_minutes: 执行间隔（分钟）
            replace_existing: 如果任务已存在，是否替换

        Returns:
            创建的任务对象，如果任务已存在且不替换则返回None
        """
        return self.schedule_interval_task(
            name, callback, interval_minutes * 60, replace_existing
        )

    def get_task(self, name: str) -> asyncio.Task | None:
        """获取指定名称的任务。

        Args:
            name: 任务名称

        Returns:
            任务对象，如果不存在则返回None
        """
        return self._tasks.get(name)

    def get_active_tasks(self) -> set[str]:
        """获取所有活动任务的名称。

        Returns:
            活动任务名称集合
        """
        return set(self._tasks.keys())

    def is_task_running(self, name: str) -> bool:
        """检查任务是否正在运行。

        Args:
            name: 任务名称

        Returns:
            任务是否正在运行
        """
        task = self.get_task(name)
        return task is not None and not task.done()

    async def shutdown(self):
        """关闭任务调度器，取消所有任务。"""
        await self.cancel_all_tasks()
        logger.info("任务调度器已关闭")

    async def cleanup(self):
        """清理资源。"""
        await self.shutdown()
