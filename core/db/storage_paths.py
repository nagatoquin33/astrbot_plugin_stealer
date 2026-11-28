"""SQLite 与检索缓存共用的图库路径转换。"""

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


_IMAGE_DIRS = {"categories", "pending"}


def store_image_path(path: str | None, root: Path | None) -> str | None:
    """仅将插件目录内的正式和待审图片写为相对路径。"""
    if not path or root is None:
        return path
    absolute = Path(os.path.normpath(path))
    if absolute.is_relative_to(root):
        relative = absolute.relative_to(root)
        if relative.parts and relative.parts[0] in _IMAGE_DIRS:
            return relative.as_posix()
    return path


def resolve_image_path(path: str | None, root: Path | None) -> str | None:
    if root is not None and path and path.startswith(("categories/", "pending/")):
        return str(root / path)
    return path


def legacy_image_path(path: str, root: Path) -> str:
    """兼容从另一套安装目录或另一系统复制来的旧库。"""
    local = store_image_path(path, root)
    if local != path:
        return local
    normalized = PurePosixPath(path.replace("\\", "/"))
    if normalized.is_absolute() or PureWindowsPath(path).is_absolute():
        parts = normalized.parts
        for index in range(len(parts) - 2, 0, -1):
            if parts[index - 1].casefold() == root.name.casefold() and parts[index] in _IMAGE_DIRS:
                relative = parts[index:]
                if ".." not in relative:
                    return "/".join(relative)
    return path
