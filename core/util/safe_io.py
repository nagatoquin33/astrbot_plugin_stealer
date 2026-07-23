"""安全文件 IO 工具。

从 `Main._safe_remove_file` / `ImageProcessorService.safe_remove_file` 抽出来的独立工具，
方便插件各模块按需使用，不再依赖 `self.plugin._safe_remove_file` 这种穿透调用。
"""

import asyncio
import os

from astrbot.api import logger


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


class SafeFileOps:
    """文件 IO 工具类，封装常用操作。"""

    remove = staticmethod(safe_remove_file)