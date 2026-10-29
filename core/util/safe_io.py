"""提供文件安全删除、同名路径分配及图片入库时的文件发布和失败补偿。
"""

import asyncio
import os
import shutil
import tempfile
from functools import wraps
from contextlib import AsyncExitStack, contextmanager
from collections import Counter
from pathlib import Path

from astrbot.api import logger


_active_files = Counter()


def file_in_use(path) -> bool:
    return bool(_active_files[str(Path(path).resolve()).casefold()])


@contextmanager
def protect_file(path):
    """在模型读取期间保留原文件，允许维护操作跳过正在使用的文件。"""
    key = str(Path(path).resolve()).casefold()
    _active_files[key] += 1
    try:
        yield
    finally:
        _active_files[key] -= 1
        if not _active_files[key]:
            del _active_files[key]


def storage_write(handler):
    """串行化 WebUI 清理和分类删除，与图片发布使用同一把锁。"""
    @wraps(handler)
    async def wrapped(self, *args, **kwargs):
        async with self._db.file_lock:
            return await handler(self, *args, **kwargs)
    return wrapped


def unique_image_path(directory: Path, filename: str) -> Path:
    """为同名文件追加递增序号，返回未被占用的目标路径。"""
    candidate = directory / filename
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = directory / f"{Path(filename).stem}_{counter}{Path(filename).suffix}"
    return candidate


async def safe_remove_file(file_path: str) -> bool:
    """安全删除文件。

    - 路径不存在视为成功（幂等）
    - FileNotFoundError 视为成功（其他进程可能已删）
    - 其他异常返回 False 并记录日志

    Args:
        file_path: 要删除的文件路径

    Returns:
        bool: 是否成功（或幂等成功）
    """
    try:
        if os.path.exists(file_path):
            await asyncio.to_thread(os.remove, file_path)
            logger.debug(f"已删除文件: {file_path}")
            return True
        logger.debug(f"文件不存在，无需删除: {file_path}")
        return True
    except FileNotFoundError:
        logger.debug(f"文件已被删除: {file_path}")
        return True
    except PermissionError as e:
        logger.warning(f"删除文件权限不足: {file_path}, 错误: {e}")
        return False
    except Exception as e:
        logger.error(f"删除文件失败: {file_path}, 错误: {e}")
        return False


async def _finish_operation(operation):
    """Wait for an in-flight file/DB operation before propagating caller cancellation."""
    task = asyncio.ensure_future(operation)
    cancelled = False
    while True:
        try:
            return await asyncio.shield(task), cancelled
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True


def _publish_image(source: Path, target: Path, consume: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if consume:
        try:
            os.link(source, target)
            return
        except FileExistsError:
            raise
        except OSError:
            pass  # Cross-volume or no hardlink support: stage a copy.
    with tempfile.TemporaryDirectory(prefix=".meme-", dir=target.parent) as directory:
        staging = Path(directory) / "asset"
        shutil.copy2(source, staging)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        try:
            os.replace(staging, target)
        except BaseException:
            target.unlink(missing_ok=True)
            raise


async def commit_image_file(source, target, commit, *, consume: bool = True, lock=None):
    if lock is not None:
        async with lock:
            return await commit_image_file(source, target, commit, consume=consume)
    return await _commit_image_file(source, target, commit, consume=consume)


async def _commit_image_file(source, target, commit, *, consume: bool = True):
    """Plugin ingestion: publish without overwrite, commit metadata, compensate on failure."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        result, cancelled = await _finish_operation(commit(target))
    else:
        _, cancelled = await _finish_operation(asyncio.to_thread(_publish_image, source, target, consume))
        async with AsyncExitStack() as rollback:
            rollback.push_async_callback(safe_remove_file, str(target))
            if cancelled:
                raise asyncio.CancelledError
            result, cancelled = await _finish_operation(commit(target))
            rollback.pop_all()
        if consume:
            _, cleanup_cancelled = await _finish_operation(safe_remove_file(str(source)))
            cancelled |= cleanup_cancelled
    if cancelled:
        raise asyncio.CancelledError
    return result


async def write_image_file(content: bytes, target, commit, *, lock=None):
    """Stage upload bytes using tempfile, then share the ingestion compensation path."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".meme-upload-", dir=target.parent) as directory:
        source = Path(directory) / "asset"
        _, cancelled = await _finish_operation(asyncio.to_thread(source.write_bytes, content))
        if cancelled:
            raise asyncio.CancelledError
        return await commit_image_file(source, target, commit, lock=lock)
