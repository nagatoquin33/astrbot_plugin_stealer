"""ExternalSourceRoutes methods for PluginAPI."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

from quart import jsonify, request

from astrbot.api import logger

from ..core.util.safe_io import safe_remove_file
from ..core.sources.models import ExternalSourceError, ExternalSourceSecurityError

class ExternalSourceRoutes:
    # ── External sources (v3) ────────────────────────────────

    async def _source_json(self) -> dict[str, Any]:
        try:
            payload = await request.get_json() or {}
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            raise ExternalSourceError("request body must be a JSON object")
        return dict(payload)

    def _validate_source_pack_path(self, payload: dict[str, Any]) -> None:
        source_type = str(
            payload.get("source_type")
            or payload.get("type")
            or payload.get("kind")
            or ""
        ).lower()
        if not source_type and payload.get("path"):
            source_type = "meme_pack"
        if source_type not in {"pack", "meme_pack", "meme-manager", "meme_manager"}:
            return
        raw_path = payload.get("path") or payload.get("endpoint")
        if not raw_path:
            raise ExternalSourceError("pack source requires path")
        source_path = Path(str(raw_path)).expanduser().resolve()
        # Web requests may read same-instance plugin data or an uploaded pack
        # under this plugin's data directory.  This prevents a dashboard call
        # from turning into an arbitrary local-file reader.
        allowed_roots = [self._data_dir.resolve(), self._data_dir.parent.resolve()]
        if not any(self._path_is_under(source_path, root) for root in allowed_roots):
            raise ExternalSourceSecurityError(
                "pack path must be inside AstrBot plugin data; upload the archive first"
            )

    @staticmethod
    def _path_is_under(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    async def _resolve_source_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        service = self._sources
        if service is None:
            raise ExternalSourceError("source service is unavailable")
        source_id = str(payload.get("source_id") or "").strip()
        has_descriptor = any(
            payload.get(key)
            for key in (
                "source_type",
                "type",
                "kind",
                "path",
                "endpoint",
                "url",
                "repository",
                "repo",
            )
        )
        if source_id and not has_descriptor:
            stored = service.source_spec(source_id)
            if not stored:
                raise ExternalSourceError("source was not found")
            stored.update(payload)
            payload = stored
        self._validate_source_pack_path(payload)
        return payload

    @staticmethod
    def _source_error_response(exc: Exception):
        status = 403 if isinstance(exc, ExternalSourceSecurityError) else 400
        return jsonify({"success": False, "error": str(exc)}), status

    @staticmethod
    def _save_source_upload_limited(upload: Any, target: Path, max_bytes: int) -> int:
        """Stream a FileStorage upload to disk without exceeding the source cap."""

        stream = getattr(upload, "stream", upload)
        read = getattr(stream, "read", None)
        if not callable(read):
            raise ExternalSourceError("archive upload stream is unavailable")
        total = 0
        with target.open("xb") as output:
            while True:
                chunk = read(64 * 1024)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    raise ExternalSourceError("archive upload returned invalid data")
                total += len(chunk)
                if total > max_bytes:
                    raise ExternalSourceSecurityError("pack archive exceeds the byte limit")
                output.write(chunk)
        return total

    async def handle_sources(self):
        """List discovered/registered sources, or register a descriptor."""
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            if request.method == "GET":
                return jsonify(
                    {
                        "success": True,
                        "sources": service.list_sources(),
                        "defaults": {
                            "enabled": bool(
                                getattr(self._cfg, "external_sources_enabled", True)
                            ),
                            "review": bool(
                                getattr(self._cfg, "external_source_default_review", False)
                                or getattr(self._cfg, "content_filtration", False)
                            ),
                            "review_forced": bool(
                                getattr(self._cfg, "content_filtration", False)
                            ),
                            "allow_http": bool(
                                getattr(self._cfg, "external_source_allow_http", False)
                            ),
                            "max_items": int(
                                getattr(self._cfg, "external_source_max_items", 2000)
                            ),
                        },
                    }
                )
            payload = await self._resolve_source_payload(await self._source_json())
            source = await service.register(payload)
            return jsonify({"success": True, "source": source})
        except (ExternalSourceError, OSError, ValueError) as exc:
            return self._source_error_response(exc)

    async def handle_source_inspect(self):
        """Preflight a pack/API without mutating the library."""
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            payload = await self._resolve_source_payload(await self._source_json())
            inspection = await service.inspect_dict(payload)
            return jsonify({"success": True, "inspection": inspection})
        except (ExternalSourceError, OSError, ValueError) as exc:
            return self._source_error_response(exc)

    async def handle_source_import(self):
        """Start a background import after the same preflight used by the UI."""
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            payload = await self._resolve_source_payload(await self._source_json())
            job = service.start_import(payload)
            return jsonify({"success": True, "accepted": True, "job": job}), 202
        except (ExternalSourceError, OSError, ValueError) as exc:
            return self._source_error_response(exc)

    async def handle_source_sync(self):
        """Re-run a registered source using its saved descriptor."""
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            payload = await self._resolve_source_payload(await self._source_json())
            job = service.start_import(payload)
            return jsonify({"success": True, "accepted": True, "job": job}), 202
        except (ExternalSourceError, OSError, ValueError) as exc:
            return self._source_error_response(exc)

    async def handle_source_job(self):
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        job_id = str(request.args.get("job_id", "") or "").strip()
        job = service.get_job(job_id)
        if not job:
            return jsonify({"success": False, "error": "source job not found"}), 404
        return jsonify({"success": True, "job": job})

    async def handle_source_job_cancel(self):
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            payload = await self._source_json()
            cancelled = await service.cancel_job(str(payload.get("job_id") or ""))
            return jsonify({"success": cancelled, "cancelled": cancelled})
        except ExternalSourceError as exc:
            return self._source_error_response(exc)

    async def handle_source_delete(self):
        """Forget a registry entry while preserving already copied images."""
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            payload = await self._source_json()
            source_id = str(payload.get("source_id") or "").strip()
            if not source_id:
                raise ExternalSourceError("source_id is required")
            deleted = await service.delete_source(source_id)
            return jsonify(
                {
                    "success": deleted,
                    "deleted": deleted,
                    "images_preserved": True,
                }
            )
        except ExternalSourceError as exc:
            return self._source_error_response(exc)

    async def handle_source_upload(self):
        """Stage a browser-uploaded Meme Pack archive for safe inspection."""
        service = self._sources
        if service is None:
            return jsonify({"success": False, "error": "source service unavailable"}), 503
        try:
            try:
                await service.cleanup_staged_uploads()
            except Exception as cleanup_error:
                logger.debug(f"[Source] 上传前清理暂存归档失败: {cleanup_error}")
            files = await request.files
            upload = files.get("file") if files is not None else None
            if upload is None:
                raise ExternalSourceError("archive file is required")
            filename = Path(str(getattr(upload, "filename", "") or "pack.zip")).name
            if Path(filename).suffix.lower() not in {".zip", ".meme-pack"}:
                raise ExternalSourceError("only ZIP or .meme-pack archives are accepted")
            max_bytes = int(
                getattr(self._cfg, "external_source_max_archive_bytes", 1024 * 1024 * 1024)
            )
            upload_dir = service.import_dir / "uploads"
            await asyncio.to_thread(upload_dir.mkdir, parents=True, exist_ok=True)
            target = upload_dir / f"{uuid.uuid4().hex}_{filename}"
            try:
                await asyncio.to_thread(
                    self._save_source_upload_limited,
                    upload,
                    target,
                    max_bytes,
                )
                inspection = await service.inspect_dict(
                    {"source_type": "meme_pack", "path": str(target)}
                )
            except Exception:
                await safe_remove_file(str(target))
                raise
            return jsonify(
                {
                    "success": True,
                    "path": str(target),
                    "inspection": inspection,
                }
            )
        except (ExternalSourceError, OSError, ValueError) as exc:
            return self._source_error_response(exc)
