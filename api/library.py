"""Image delivery, library listing, and statistics routes."""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
from quart import jsonify, request

try:
    from quart import send_file
except ImportError:
    send_file = None  # type: ignore[assignment]

from astrbot.api import logger


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
        file_path = await asyncio.to_thread(self._find_image_path_by_hash, image_hash)

        if file_path and os.path.isfile(file_path):
            try:
                data_url = self._file_base64(file_path)
                return jsonify({"success": True, "hash": image_hash, "url": data_url})
            except Exception as e:
                logger.warning(f"读取图片失败: {e}")
        return jsonify({"success": False, "error": "图片未找到"})

    def _find_image_path_by_hash(self, image_hash: str) -> str | None:
        """预览优先使用正式图库中的记录，未命中时查询待审核池。"""
        found = self._find_index_entry_by_hash(image_hash)
        if found:
            return found[0]
        try:
            pending = self._db.get_pending_by_hash(image_hash)
        except Exception:
            pending = None
        return pending.get("path") if pending else None

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

        file_path = await asyncio.to_thread(self._find_image_path_by_hash, img_hash)

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
        page: int,
        page_size: int,
        category: str | None,
        search: str,
        sort_order: str,
        favorite_only: bool,
        character: str,
        library: str,
    ) -> dict:
        """从 SQLite 分页读取图库，包括空图库的分类和角色统计。"""
        db = self._db
        raw, total, category_counts = db.get_emojis_paginated(
            page=page,
            page_size=page_size,
            category=category,
            sort_order=sort_order,
            search_query=search or None,
            favorite_only=favorite_only,
            character=character or None,
            library=library,
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

            payload = await asyncio.to_thread(
                self._list_images_from_database,
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
        cfg = getattr(self.plugin, "plugin_config", None)
        setup = None if cfg is None else {
            "capture_enabled": bool(getattr(cfg, "steal_meme", False)),
            "auto_send_enabled": bool(getattr(cfg, "auto_send_meme", False)),
            "context_enabled": bool(getattr(cfg, "enable_natural_emotion_analysis", False)),
            "embedding_enabled": bool(getattr(cfg, "enable_embedding_search", False)),
        }
        return jsonify({"success": True, "status": "ok", "service": "emoji-manager-webui", "setup": setup})
