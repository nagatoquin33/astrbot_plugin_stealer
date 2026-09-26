"""TaxonomyRoutes methods for PluginAPI."""

import asyncio
import shutil
from pathlib import Path

from quart import jsonify, request

from astrbot.api import logger

from ..core.db.index_manager import delete_index_paths, invalidate_search
from ..core.util.normalization import (
    normalize_category_key,
)
from ..core.util.safe_io import safe_remove_file

class TaxonomyRoutes:
    # ── Categories ────────────────────────────────────────────

    async def handle_categories(self):
        if request.method == "POST":
            return await self._categories_update()
        return await self._categories_list()

    async def _categories_list(self):
        try:
            cats = {key: 0 for key in self._get_category_keys()}
            for meta in self._get_index().values():
                if isinstance(meta, dict):
                    c = str(meta.get("category", "unknown"))
                    cats[c] = cats.get(c, 0) + 1
            return jsonify({"success": True, "categories": cats})
        except Exception as e:
            logger.error(f"获取分类失败: {e}")
            return jsonify({"success": False, "error": str(e)})

    async def _categories_update(self):
        try:
            data = await request.get_json() or {}
            items = data.get("categories", [])
            if not isinstance(items, list) or not items:
                return jsonify({"success": False, "error": "分类列表无效"})

            keys: list[str] = []
            info: dict[str, dict] = {}
            seen: set[str] = set()
            seen_names: dict[str, str] = {}
            for item in items:
                if isinstance(item, dict) and item.get("key"):
                    try:
                        key = normalize_category_key(item["key"])
                    except ValueError as exc:
                        return jsonify({"success": False, "error": str(exc)}), 400
                    if key in seen:
                        return jsonify({"success": False, "error": f"分类 key 重复: {key}"}), 400
                    seen.add(key)
                    keys.append(key)
                    name = str(item.get("name", "")).strip()[:40]
                    desc = str(item.get("desc", "")).strip()[:200]
                    name_key = name.casefold()
                    if name_key and name_key in seen_names:
                        return jsonify(
                            {
                                "success": False,
                                "error": f"分类显示名称重复: {name}（{seen_names[name_key]} / {key}）",
                            }
                        ), 400
                    if name_key:
                        seen_names[name_key] = key
                    if name or desc:
                        info[key] = {"name": name, "desc": desc}
                elif isinstance(item, str):
                    try:
                        key = normalize_category_key(item)
                    except ValueError as exc:
                        return jsonify({"success": False, "error": str(exc)}), 400
                    if key in seen:
                        return jsonify({"success": False, "error": f"分类 key 重复: {key}"}), 400
                    seen.add(key)
                    keys.append(key)

            if not keys:
                return jsonify({"success": False, "error": "分类列表无效"})

            self.plugin.update_config({"categories": keys})

            cur_info = dict(getattr(self._cfg, "category_info", {}) or {})
            self._cfg.category_info = {k: cur_info.get(k, {}) for k in keys}
            self._cfg.category_info.update(info)
            self._cfg.ensure_category_dirs(keys)
            self._cfg.save_category_info()
            return jsonify({"success": True, "categories": keys})
        except Exception as e:
            logger.error(f"更新分类失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_delete_category(self):
        try:
            data = await request.get_json() or {}
            try:
                key = normalize_category_key(data.get("key", ""))
            except ValueError as exc:
                return jsonify({"success": False, "error": str(exc)}), 400

            cur_cats = list(self._cfg.categories or [])
            if key not in cur_cats:
                return jsonify({"success": False, "error": "分类不存在"})
            if len(cur_cats) <= 1:
                return jsonify({"success": False, "error": "至少需要保留1个分类"})

            updated = [c for c in cur_cats if c != key]
            deleted = 0

            index = self._get_index()
            deleted_paths: list[str] = []
            for p, m in list(index.items()):
                if not isinstance(m, dict) or m.get("category") != key:
                    continue
                old = Path(p)
                try:
                    if not old.exists() or await safe_remove_file(str(old)):
                        deleted_paths.append(p)
                        deleted += 1
                        h = m.get("hash")
                        if h and hasattr(self.plugin, "image_processor_service"):
                            self.plugin.image_processor_service.invalidate_cache(h)
                except Exception as ex:
                    logger.warning(f"删除分类文件失败: {old}, {ex}")

            if deleted_paths:
                await delete_index_paths(self.plugin, deleted_paths)

            cat_dir = self._data_dir / "categories" / key
            try:
                if cat_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, cat_dir, True)
            except Exception as e:
                logger.warning(f"删除分类目录失败: {cat_dir}, {e}")

            if key in getattr(self._cfg, "category_info", {}):
                del self._cfg.category_info[key]
                self._cfg.save_category_info()

            self.plugin.update_config({"categories": updated})

            return jsonify(
                {"success": True, "deleted": key, "categories": updated, "deleted_files": deleted}
            )
        except Exception as e:
            logger.error(f"删除分类失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_characters(self):
        if request.method == "POST":
            return await self._characters_update()
        return await self._characters_list()

    async def _characters_list(self):
        try:
            counts = self._db.get_character_counts() if self._db and hasattr(self._db, "get_character_counts") else {}
            return jsonify(
                {
                    "success": True,
                    "characters": self._build_characters_list(counts),
                    "unassigned": int(counts.get("", 0) or 0),
                }
            )
        except Exception as e:
            logger.error(f"获取角色列表失败: {e}")
            return jsonify({"success": False, "error": str(e)})

    async def _characters_update(self):
        try:
            data = await request.get_json() or {}
            items = data.get("characters", [])
            if not isinstance(items, list):
                return jsonify({"success": False, "error": "角色列表无效"})
            keys: list[str] = []
            info: dict[str, dict] = {}
            seen: set[str] = set()
            for item in items:
                if isinstance(item, dict) and item.get("key"):
                    key = self._normalize_character_key(str(item.get("key") or ""))
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    keys.append(key)
                    name = str(item.get("name", "") or "").strip()
                    desc = str(item.get("desc", "") or "").strip()
                    info[key] = {"name": name or key, "desc": desc}
                elif isinstance(item, str):
                    key = self._normalize_character_key(item)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    keys.append(key)
            self._cfg.characters = keys
            self._cfg.character_info = info
            self._cfg.save_characters()
            self._cfg.save_character_info()
            counts = self._db.get_character_counts() if self._db and hasattr(self._db, "get_character_counts") else {}
            return jsonify(
                {
                    "success": True,
                    "characters": self._build_characters_list(counts),
                }
            )
        except Exception as e:
            logger.error(f"更新角色列表失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_delete_character(self):
        try:
            data = await request.get_json() or {}
            key = self._normalize_character_key(str(data.get("key") or ""))
            if not key:
                return jsonify({"success": False, "error": "缺少 key"})
            updated = [item for item in self._cfg.get_characters() if item != key]
            info = dict(self._cfg.character_info or {})
            info.pop(key, None)
            if self._db and hasattr(self._db, "clear_character"):
                self._db.clear_character(key)
            self._cfg.characters = updated
            self._cfg.character_info = info
            self._cfg.save_characters()
            self._cfg.save_character_info()
            invalidate_search(self.plugin)
            return jsonify({"success": True, "deleted": key, "characters": updated})
        except Exception as e:
            logger.error(f"删除角色失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_get_emotions(self):
        try:
            info = self._cfg.get_category_info()
            return jsonify({"success": True, "emotions": info})
        except Exception as e:
            logger.error(f"获取情绪分类失败: {e}")
            return jsonify({"success": False, "error": str(e)})
