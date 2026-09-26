"""MaintenanceRoutes methods for PluginAPI."""

import os
from pathlib import Path

from quart import jsonify, request

from astrbot.api import logger

from ..core.db.index_manager import delete_index_paths

class MaintenanceRoutes:
    # ── Repair / Storage maintenance ─────────────────────────

    async def handle_scope_repair(self):
        try:
            data = await request.get_json() or {}
            origin_target = str(data.get("origin_target", "") or "").strip()
            if not origin_target:
                return jsonify({"success": False, "error": "缺少 origin_target"})

            hashes_raw = data.get("hashes", [])
            hashes = {str(h).strip() for h in hashes_raw if str(h).strip()}
            scope_mode = self._norm_scope(data.get("scope_mode", "local"))
            only_missing = str(data.get("only_missing", "true")).lower() != "false"

            updated = 0
            skipped = 0
            index = self._get_index()
            for path, meta in index.items():
                if not isinstance(meta, dict):
                    continue
                img_hash = str(meta.get("hash", "") or "")
                if hashes and img_hash not in hashes:
                    continue
                if only_missing and str(meta.get("origin_target", "") or "").strip():
                    skipped += 1
                    continue
                updates = {"origin_target": origin_target}
                if scope_mode:
                    updates["scope_mode"] = scope_mode
                if await self._update_index_path(path, updates):
                    updated += 1

            return jsonify({"success": True, "count": updated, "skipped": skipped})
        except Exception as e:
            logger.error(f"作用域来源修复失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_storage_scan(self):
        try:
            return jsonify(self._build_storage_report())
        except Exception as e:
            logger.error(f"存储扫描失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_storage_cleanup(self):
        try:
            data = await request.get_json() or {}
            report = self._build_storage_report(include_items=True)
            sections_raw = data.get("sections")
            if isinstance(sections_raw, list):
                sections = {str(section) for section in sections_raw}
            else:
                sections = {
                    key
                    for key in ("stale_index", "orphan_files", "thumb_cache", "temp_files", "raw_files")
                    if str(data.get(key, "false")).lower() == "true"
                }

            if not sections:
                strategy = str(
                    data.get(
                        "strategy",
                        getattr(self.plugin, "storage_cleanup_strategy", "balanced"),
                    )
                    or "balanced"
                )
                if strategy == "conservative":
                    sections = {"stale_index", "temp_files"}
                elif strategy == "aggressive":
                    sections = {
                        "stale_index",
                        "orphan_files",
                        "thumb_cache",
                        "temp_files",
                        "raw_files",
                    }
                else:
                    sections = {"stale_index", "orphan_files", "thumb_cache", "temp_files"}

            removed: dict[str, int] = {}
            if "stale_index" in sections:
                index = self._get_index()
                stale_paths: list[str] = []
                for path in index.keys():
                    if not isinstance(path, str):
                        continue
                    # stale_index 语义：物理文件已丢失但索引条目仍存在。
                    # 只删索引条目，绝不删文件——文件已经不在了。
                    try:
                        file_exists = Path(path).resolve().is_file()
                    except Exception:
                        file_exists = Path(path).is_file()
                    if not file_exists:
                        stale_paths.append(path)
                if stale_paths:
                    await delete_index_paths(self.plugin, stale_paths)
                removed["stale_index"] = len(stale_paths)

            for key in ("orphan_files", "thumb_cache", "temp_files", "raw_files"):
                if key not in sections:
                    continue
                removed[key] = await self._remove_report_files(
                    report.get(key, {}).get("items", [])
                )

            return jsonify(
                {
                    "success": True,
                    "removed": removed,
                    "report": self._build_storage_report(),
                }
            )
        except Exception as e:
            logger.error(f"存储清理失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    # ── VLM Analyze ───────────────────────────────────────────

    async def handle_analyze_image(self):
        tmp_file_to_cleanup = None
        try:
            proc = getattr(self.plugin, "image_processor_service", None)
            if not proc:
                return jsonify({"success": False, "error": "图片处理服务不可用"})

            data = await request.get_json() or {}
            img_hash = (data.get("hash", "") or "").strip()
            img_base64 = (data.get("base64", "") or "").strip()

            file_path = None

            # 优先通过 hash 从索引查找文件路径
            if img_hash:
                found = self._find_index_entry_by_hash(img_hash)
                file_path = found[0] if found else None
                if not file_path or not os.path.isfile(file_path):
                    file_path = None

            # hash 查不到或未提供 hash 时，回退到 base64 方式
            if not file_path and img_base64:
                import tempfile

                file_content = self._decode_base64_payload(img_base64)
                if file_content is None:
                    return jsonify({"success": False, "error": "图片数据无效"})
                ext = ".png"
                if img_base64.startswith("data:image/jpeg") or img_base64.startswith("data:image/jpg"):
                    ext = ".jpg"
                elif img_base64.startswith("data:image/gif"):
                    ext = ".gif"
                elif img_base64.startswith("data:image/webp"):
                    ext = ".webp"

                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(file_content)
                tmp.close()
                file_path = tmp.name
                tmp_file_to_cleanup = file_path

            if not file_path:
                return jsonify({"success": False, "error": "缺少 hash 或 base64 图片数据"})

            cat, tags, desc, _, scenes, overlay_text, emotions = await proc.classify_image(
                event=None,
                file_path=file_path,
                categories=list(self._cfg.categories or []),
                content_filtration=False,
            )
            if cat == getattr(proc, "CATEGORY_FILTERED", None):
                return jsonify({"success": False, "error": "图片内容审核不通过"})
            if not cat:
                return jsonify({"success": False, "error": "无法识别图片分类"})

            return jsonify(
                {
                    "success": True,
                    "category": cat,
                    "tags": tags,
                    "description": desc,
                    "scenes": scenes or [],
                    "overlay_text": overlay_text or "",
                    "emotions": emotions or [],
                }
            )
        except Exception as e:
            logger.error(f"VLM分析失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": f"分析失败: {e}"})
        finally:
            if tmp_file_to_cleanup:
                try:
                    os.unlink(tmp_file_to_cleanup)
                except Exception:
                    pass
