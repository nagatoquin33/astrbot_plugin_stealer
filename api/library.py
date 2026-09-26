"""Image delivery, library listing, and statistics routes."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from quart import jsonify, request

try:
    from quart import send_file
except ImportError:
    send_file = None  # type: ignore[assignment]

from astrbot.api import logger

from ..core.maintenance.retention import library_counts, library_group

class LibraryRoutes:
    # ── Image serving ─────────────────────────────────────────

    async def handle_serve_image(self):
        """直接服务图片文件（用于页面展示）。"""
        if send_file is None:
            return jsonify({"success": False, "error": "send_file 不可用"}), 500
        file_path = request.args.get("path", "")
        if not file_path or not os.path.isfile(file_path):
            return jsonify({"success": False, "error": "文件不存在"}), 404
        try:
            Path(file_path).resolve().relative_to(self._data_dir.resolve())
        except ValueError:
            return jsonify({"success": False, "error": "路径非法"}), 403
        return await send_file(file_path)

    async def handle_image_data(self):
        """返回图片的 base64 data URL。"""
        image_hash = request.args.get("hash", "").strip()
        if not image_hash:
            return jsonify({"success": False, "error": "缺少 hash"})
        file_path = None
        for path_str, meta in self._get_index().items():
            if isinstance(meta, dict) and meta.get("hash") == image_hash:
                file_path = path_str
                break

        if not file_path:
            db = self._db
            if db and hasattr(db, "get_pending_by_hash"):
                try:
                    pending_row = db.get_pending_by_hash(image_hash)
                except Exception:
                    pending_row = None
                if pending_row:
                    file_path = pending_row.get("path")

        if file_path and os.path.isfile(file_path):
            try:
                data_url = self._file_base64(file_path)
                return jsonify({"success": True, "hash": image_hash, "url": data_url})
            except Exception as e:
                logger.warning(f"读取图片失败: {e}")
        return jsonify({"success": False, "error": "图片未找到"})

    async def _get_or_create_thumbnail(
        self, img_hash: str, file_path: str, max_size: int = 300
    ) -> str:
        """生成或返回缓存的缩略图路径。"""
        thumb_dir = self._data_dir / "thumb_cache"
        thumb_path = thumb_dir / f"{img_hash}_{max_size}.jpg"

        if thumb_path.exists():
            return str(thumb_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        thumb_dir.mkdir(parents=True, exist_ok=True)

        def _create() -> str:
            with Image.open(file_path) as img:
                try:
                    img.seek(0)
                except EOFError:
                    pass
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                if img.mode in ("RGBA", "LA"):
                    canvas = Image.new("RGB", img.size, (16, 16, 16))
                    canvas.paste(img, mask=img.split()[-1])
                    img = canvas
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(thumb_path, "JPEG", quality=82, optimize=True)
            return str(thumb_path)

        return await asyncio.to_thread(_create)

    async def handle_thumbnail(self):
        """返回缩略图的 base64 data URL，用于列表展示。"""
        img_hash = request.args.get("hash", "").strip()
        max_size = request.args.get("size", 300, type=int)
        try:
            max_size = max(32, min(int(max_size or 300), 400))
        except (TypeError, ValueError):
            max_size = 300

        if not img_hash:
            return jsonify({"success": False, "error": "缺少 hash"}), 400

        file_path = None
        for p, m in self._get_index().items():
            if isinstance(m, dict) and m.get("hash") == img_hash:
                file_path = p
                break

        # 回退：正式库未命中时查待审核池（审核区缩略图走此路径）
        if not file_path:
            db = self._db
            if db and hasattr(db, "get_pending_by_hash"):
                try:
                    pending_row = db.get_pending_by_hash(img_hash)
                except Exception:
                    pending_row = None
                if pending_row:
                    file_path = pending_row.get("path")

        if not file_path or not os.path.isfile(file_path):
            return jsonify({"success": False, "error": "图片未找到"}), 404

        try:
            thumb_path = await self._get_or_create_thumbnail(
                img_hash, file_path, max_size
            )
            data_url = self._file_base64(thumb_path)
            return jsonify({"success": True, "hash": img_hash, "url": data_url})
        except Exception as e:
            logger.warning(f"生成缩略图失败: {e}")
            return jsonify({"success": False, "error": "缩略图生成失败"}), 500

    # ── List / Stats / Health ─────────────────────────────────

    def _build_image_list_payload(
        self,
        *,
        total: int,
        page: int,
        page_size: int,
        images: list[dict],
        category_counts: dict[str, int],
        character_counts: dict[str, int],
        libraries: dict[str, int],
    ) -> dict:
        return {
            "success": True,
            "total": total,
            "page": page,
            "size": page_size,
            "images": images,
            "categories": self._build_categories_list(category_counts),
            "characters": self._build_characters_list(character_counts),
            "unassigned_character_count": int(character_counts.get("", 0) or 0),
            "favorite_count": self._count_favorites(),
            "libraries": libraries,
            "automatic_limit": self.plugin.plugin_config.max_reg_num,
        }

    def _list_images_from_database(
        self,
        *,
        db: Any,
        page: int,
        page_size: int,
        category: str | None,
        search: str,
        sort_order: str,
        favorite_only: bool,
        character: str,
        library: str,
    ) -> dict | None:
        """Query the paginated SQLite view when it contains live rows."""
        get_paginated = getattr(db, "get_emojis_paginated", None) if db else None
        if not db or not callable(get_paginated) or db.count_total() <= 0:
            return None

        raw, total, category_counts = get_paginated(
            page=page,
            page_size=page_size,
            category=category,
            sort_order=sort_order,
            search_query=search or None,
            favorite_only=favorite_only,
            character=character or None,
            **({"library": library} if library else {}),
        )
        images = [
            item for item in (self._build_image_item(row["path"], row) for row in raw) if item
        ]
        character_counts = (
            db.get_character_counts(exclude_favorites=True)
            if library == "characters"
            else db.get_character_counts()
        )
        return self._build_image_list_payload(
            total=total,
            page=page,
            page_size=page_size,
            images=images,
            category_counts=category_counts,
            character_counts=character_counts,
            libraries=db.get_library_counts(),
        )

    def _list_images_from_index(
        self,
        *,
        page: int,
        page_size: int,
        category: str | None,
        search: str,
        sort_order: str,
        favorite_only: bool,
        character: str,
        library: str,
    ) -> dict:
        """Build the same response shape from legacy index-only data."""
        index = self._get_index()
        images: list[dict] = []
        category_counts: dict[str, int] = {}

        for path_str, metadata in index.items():
            if not Path(path_str).exists():
                continue
            item = self._build_image_item(path_str, metadata)
            if not item:
                continue
            if search and not (
                any(search in str(tag).lower() for tag in item["tags"])
                or search in item["desc"].lower()
                or any(search in str(scene).lower() for scene in item.get("scenes", []))
            ):
                continue
            if library in {"general", "favorites", "characters"} and library_group(item) != library:
                continue

            category_key = item["category"]
            category_counts[category_key] = category_counts.get(category_key, 0) + 1
            if category and item["category"] != category:
                continue
            if favorite_only and not item.get("is_favorite"):
                continue
            item_character = str(item.get("character", "") or "")
            if character == "__none__" and item_character:
                continue
            if character and character != "__none__" and item_character != character:
                continue
            images.append(item)

        sort_fields = {
            "least_used": ("use_count", "created_at"),
            "most_used": ("use_count", "last_used_at"),
            "last_used": ("last_used_at", "use_count"),
        }.get(sort_order, ("created_at",))
        images.sort(
            key=lambda item: tuple(int(item.get(field, 0) or 0) for field in sort_fields)
            + (str(item.get("hash", "")),),
            reverse=sort_order not in {"oldest", "least_used"},
        )

        total = len(images)
        start = (page - 1) * page_size
        character_counts: dict[str, int] = {}
        for metadata in index.values():
            if not isinstance(metadata, dict):
                continue
            if library == "characters" and metadata.get("is_favorite"):
                continue
            key = str(metadata.get("character", "") or "")
            character_counts[key] = character_counts.get(key, 0) + 1

        return self._build_image_list_payload(
            total=total,
            page=page,
            page_size=page_size,
            images=images[start : start + page_size],
            category_counts=category_counts,
            character_counts=character_counts,
            libraries=library_counts(index),
        )

    async def handle_list_images(self):
        """Return a paginated image list and category counts."""
        try:
            page = request.args.get("page", 1, type=int)
            page_size = request.args.get("size", 50, type=int)
            category = request.args.get("category", None)
            search = str(request.args.get("q", "")).lower()
            sort_order = request.args.get("sort", "newest")
            favorite_only = request.args.get("favorite_only", "false").lower() == "true"
            character = str(request.args.get("character", "") or "")
            library = str(request.args.get("library", "") or "")

            payload = self._list_images_from_database(
                db=self._db,
                page=page,
                page_size=page_size,
                category=category,
                search=search,
                sort_order=sort_order,
                favorite_only=favorite_only,
                character=character,
                library=library,
            )
            if payload is None:
                payload = self._list_images_from_index(
                    page=page,
                    page_size=page_size,
                    category=category,
                    search=search,
                    sort_order=sort_order,
                    favorite_only=favorite_only,
                    character=character,
                    library=library,
                )
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error listing images: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_get_stats(self):
        try:
            index = self._get_index()
            today_start = (
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            )
            today_count = sum(
                1
                for m in index.values()
                if isinstance(m, dict) and m.get("created_at", 0) >= today_start
            )
            cat_count = len(self._cfg.categories) if hasattr(self.plugin, "plugin_config") else 0
            return jsonify(
                {
                    "success": True,
                    "stats": {"total": len(index), "categories": cat_count, "today": today_count},
                }
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify({"success": False, "error": str(e)})

    async def handle_health_check(self):
        return jsonify({"success": True, "status": "ok", "service": "emoji-manager-webui"})
