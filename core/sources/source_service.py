"""Unified import service for packs, other plugin data, and HTTP catalogs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PIL import Image, UnidentifiedImageError

from astrbot.api import logger
from ..db.index_manager import invalidate_search, refresh_search_entry

from ..util.normalization import (
    normalize_category_key,
    normalize_character_key,
    normalize_scope_mode,
)
from ..util.safe_io import safe_remove_file
from .github_source import GitHubSource
from .http_source import HTTPSource
from .models import (
    ExternalSourceError,
    ExternalSourceSecurityError,
    SourceInspection,
    SourceItem,
)
from .pack_source import PackSource


_FORMAT_SUFFIX = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "GIF": ".gif",
    "WEBP": ".webp",
    "BMP": ".bmp",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(slots=True)
class _SourceImportPlan:
    source: dict[str, Any]
    category_map: dict[str, Any]
    review: bool
    scope_mode: str
    origin_target: str
    character: str


class SourceService:
    """Coordinates source inspection, persistent provenance, and background jobs."""

    JOB_TTL_SECONDS = 60 * 60
    STAGED_UPLOAD_TTL_SECONDS = 24 * 60 * 60
    MAX_ACTIVE_JOBS = 1
    PUBLIC_METADATA_KEYS = frozenset(
        {
            "id",
            "external_id",
            "relative_path",
            "path",
            "file",
            "file_path",
            "filename",
            "name",
            "category",
            "emotion",
            "description",
            "desc",
            "caption",
            "visible_text",
            "overlay_text",
            "tags",
            "scenes",
            "scene",
            "emotions",
            "source_url",
            "url",
            "license",
            "attribution",
            "author",
            "status",
            "content_sha256",
        }
    )

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self.db = getattr(plugin, "db_service", None)
        self.data_dir = Path(getattr(plugin, "base_dir", Path.cwd())).resolve()
        self.import_dir = self.data_dir / "external_sources"
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False

    async def initialize(self) -> None:
        await asyncio.to_thread(self.import_dir.mkdir, parents=True, exist_ok=True)
        await self.cleanup_staged_uploads()

    async def close(self) -> None:
        self._closed = True
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _cfg(self, name: str, default: Any) -> Any:
        config = getattr(self.plugin, "plugin_config", None)
        return getattr(config, name, default) if config is not None else default

    def _pack_reader(self, path: str | Path) -> PackSource:
        return PackSource(
            path,
            max_items=int(self._cfg("external_source_max_items", 2000)),
            max_archive_bytes=int(
                self._cfg("external_source_max_archive_bytes", 1024 * 1024 * 1024)
            ),
            max_uncompressed_bytes=int(
                self._cfg("external_source_max_uncompressed_bytes", 4 * 1024 * 1024 * 1024)
            ),
            max_file_bytes=int(self._cfg("external_source_max_image_bytes", 32 * 1024 * 1024)),
        )

    def _http_reader(self, spec: dict[str, Any]) -> HTTPSource:
        endpoint = str(spec.get("endpoint") or spec.get("url") or "").strip()
        headers = spec.get("headers") if isinstance(spec.get("headers"), dict) else {}
        return HTTPSource(
            endpoint,
            allow_http=bool(self._cfg("external_source_allow_http", False)),
            max_items=int(self._cfg("external_source_max_items", 2000)),
            max_file_bytes=int(self._cfg("external_source_max_image_bytes", 32 * 1024 * 1024)),
            headers=headers,
        )

    def _github_reader(self, spec: dict[str, Any]) -> GitHubSource:
        return GitHubSource(
            spec,
            cache_dir=self.import_dir / "github_cache",
            max_items=int(self._cfg("external_source_max_items", 2000)),
            max_archive_bytes=int(
                self._cfg("external_source_max_archive_bytes", 1024 * 1024 * 1024)
            ),
            max_uncompressed_bytes=int(
                self._cfg("external_source_max_uncompressed_bytes", 4 * 1024 * 1024 * 1024)
            ),
            max_file_bytes=int(self._cfg("external_source_max_image_bytes", 32 * 1024 * 1024)),
            headers=spec.get("headers") if isinstance(spec.get("headers"), dict) else {},
        )

    async def inspect(self, spec: dict[str, Any]) -> tuple[SourceInspection, Any]:
        if not bool(self._cfg("external_sources_enabled", True)):
            raise ExternalSourceError("external sources are disabled in plugin settings")
        if not isinstance(spec, dict):
            raise ExternalSourceError("source descriptor must be an object")
        source_type = self._infer_source_type(spec)
        if source_type in {"pack", "meme_pack", "meme-manager", "meme_manager"}:
            path = spec.get("path") or spec.get("endpoint")
            if not path:
                raise ExternalSourceError("pack source requires path")
            reader = self._pack_reader(str(path))
            inspection = await asyncio.to_thread(reader.inspect)
            self._check_inspection_ids(inspection)
            return inspection, reader
        if source_type in {"http", "http_json", "api"}:
            reader = self._http_reader(spec)
            try:
                inspection = await reader.inspect(
                    cursor=str(spec.get("cursor") or "") or None
                )
                self._check_inspection_ids(inspection)
                return inspection, reader
            except Exception:
                await reader.close()
                raise
        if source_type in {"github", "github_repo", "github-repo", "repository"}:
            reader = self._github_reader(spec)
            try:
                inspection = await reader.inspect()
                self._check_inspection_ids(inspection)
                return inspection, reader
            except Exception:
                await reader.close()
                raise
        raise ExternalSourceError(f"unsupported source type: {source_type or 'empty'}")

    @staticmethod
    def _infer_source_type(spec: dict[str, Any]) -> str:
        source_type = str(
            spec.get("source_type") or spec.get("type") or spec.get("kind") or ""
        ).strip().lower()
        if source_type:
            return source_type

        github_candidate = str(
            spec.get("repository")
            or spec.get("repo")
            or spec.get("url")
            or spec.get("endpoint")
            or ""
        ).strip()
        github_host = ""
        if github_candidate:
            try:
                github_host = str(urlsplit(github_candidate).hostname or "").lower()
            except ValueError:
                github_host = ""
        if (
            spec.get("repository")
            or spec.get("repo")
            or github_host in {"github.com", "www.github.com"}
        ):
            return "github"
        if spec.get("path"):
            return "meme_pack"
        if spec.get("endpoint") or spec.get("url"):
            return "http_json"
        return ""

    @staticmethod
    def _check_inspection_ids(inspection: SourceInspection) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in inspection.items:
            if item.external_id in seen:
                duplicates.add(item.external_id)
            seen.add(item.external_id)
        if duplicates:
            preview = ", ".join(sorted(duplicates)[:5])
            inspection.errors.append(f"source contains duplicate item IDs: {preview}")

    async def inspect_dict(self, spec: dict[str, Any]) -> dict[str, Any]:
        inspection, reader = await self.inspect(spec)
        try:
            result = inspection.as_dict()
            result["manifest"] = self._json_safe(result.get("manifest", {}))
            result["endpoint"] = self._redact_endpoint(result.get("endpoint"))
            for item in result.get("items", []):
                if isinstance(item, dict) and item.get("source_url"):
                    item["source_url"] = self._redact_endpoint(item["source_url"])
            result["capacity"] = self._capacity_preview(len(inspection.items))
            return result
        finally:
            await self._close_reader(reader)

    def _capacity_preview(self, item_count: int) -> dict[str, Any]:
        current = int(self.db.count_total()) if self.db and hasattr(self.db, "count_total") else 0
        configured = max(0, int(self._cfg("max_reg_num", 0) or 0))
        return {
            "current": current,
            "incoming": max(0, int(item_count)),
            "configured_limit": configured,
            "external_items_are_protected": True,
            "would_exceed_limit": bool(configured and current + item_count > configured),
        }

    async def register(self, spec: dict[str, Any]) -> dict[str, Any]:
        inspection, reader = await self.inspect(spec)
        try:
            source = self._source_record(inspection, spec)
            if self.db and hasattr(self.db, "upsert_source"):
                await self.db.upsert_source(source)
            return self._public_source(source)
        finally:
            await self._close_reader(reader)

    def list_sources(self, *, discover: bool = True) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        known: set[str] = set()
        if self.db and hasattr(self.db, "get_sources"):
            for source in self.db.get_sources():
                public = self._public_source(source)
                sources.append(public)
                known.add(str(public.get("source_id") or ""))
        if discover:
            for source in self.discover_meme_manager_packs():
                if source["source_id"] not in known:
                    sources.append(source)
        return sources

    def discover_meme_manager_packs(self) -> list[dict[str, Any]]:
        """Discover same-instance Meme Manager packs through documented files."""

        data_parent = self.data_dir.parent
        candidates = [
            data_parent / "astrbot_plugin_meme_manager",
            data_parent / "meme_manager",
        ]
        result: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_ids: set[str] = set()
        for candidate in candidates:
            try:
                root = candidate.resolve()
            except OSError:
                continue
            key = os.path.normcase(str(root))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            packs_dir = root / "packs"
            if not packs_dir.is_dir():
                continue
            try:
                pack_dirs = sorted(path for path in packs_dir.iterdir() if path.is_dir())
            except OSError:
                continue
            for pack_dir in pack_dirs:
                manifest = self._read_small_json(pack_dir / "manifest.json")
                pack_id = str(manifest.get("id") or pack_dir.name).strip()
                name = str(manifest.get("name") or pack_id).strip()
                source_id = f"pack:{pack_id}"[:190]
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                result.append(
                    {
                        "source_id": source_id,
                        "source_type": "meme_pack",
                        "name": name[:160],
                        "endpoint": str(pack_dir),
                        "enabled": True,
                        "status": "discovered",
                        "discovered": True,
                        "version": str(manifest.get("version") or ""),
                        "config": {"path": str(pack_dir), "provider": "meme_manager"},
                    }
                )
        return result

    @staticmethod
    def _read_small_json(path: Path) -> dict[str, Any]:
        try:
            if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
                return {}
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return dict(value) if isinstance(value, dict) else {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    async def delete_source(self, source_id: str) -> bool:
        if not self.db or not hasattr(self.db, "delete_source"):
            return False
        normalized_id = str(source_id or "").strip()
        previous_spec = self.source_spec(normalized_id)
        deleted = bool(await self.db.delete_source(normalized_id))
        if deleted and previous_spec:
            # A browser upload is retained while its source is registered so
            # that later syncs remain reproducible.  Once the registry entry is
            # removed, release that archive immediately when no other source
            # references it.
            await self.cleanup_staged_uploads(
                force_paths=self._local_upload_paths(previous_spec)
            )
        return deleted

    async def cleanup_staged_uploads(
        self, *, force_paths: set[Path] | None = None
    ) -> int:
        """Remove expired, unreferenced browser-uploaded archives."""

        return await asyncio.to_thread(
            self._cleanup_staged_uploads_sync,
            {path.resolve() for path in (force_paths or set())},
        )

    def _cleanup_staged_uploads_sync(self, force_paths: set[Path]) -> int:
        upload_dir = (self.import_dir / "uploads").resolve()
        if not upload_dir.is_dir():
            return 0
        referenced = self._registered_upload_paths()
        try:
            ttl = max(
                60,
                int(
                    self._cfg(
                        "external_source_upload_ttl_seconds",
                        self.STAGED_UPLOAD_TTL_SECONDS,
                    )
                ),
            )
        except (TypeError, ValueError):
            ttl = self.STAGED_UPLOAD_TTL_SECONDS
        cutoff = time.time() - ttl
        removed = 0
        try:
            entries = list(upload_dir.iterdir())
        except OSError:
            return 0
        for entry in entries:
            if entry.is_symlink() or not entry.is_file():
                continue
            try:
                resolved = entry.resolve()
                resolved.relative_to(upload_dir)
                stat = entry.stat()
            except (OSError, ValueError):
                continue
            if resolved in referenced:
                continue
            if resolved not in force_paths and stat.st_mtime > cutoff:
                continue
            try:
                entry.unlink()
                removed += 1
            except OSError:
                logger.debug(f"[Source] 清理暂存上传失败: {entry}")
        return removed

    def _registered_upload_paths(self) -> set[Path]:
        if not self.db or not hasattr(self.db, "get_sources"):
            return set()
        paths: set[Path] = set()
        try:
            sources = self.db.get_sources()
        except Exception:
            return paths
        for source in sources if isinstance(sources, list) else []:
            if not isinstance(source, dict):
                continue
            paths.update(self._local_upload_paths(source))
            config = source.get("config")
            if isinstance(config, dict):
                paths.update(self._local_upload_paths(config))
        return paths

    def _local_upload_paths(self, value: Any) -> set[Path]:
        upload_dir = (self.import_dir / "uploads").resolve()
        candidates: list[Any] = []
        if isinstance(value, dict):
            candidates.extend(value.get(key) for key in ("path", "endpoint"))
            config = value.get("config")
            if isinstance(config, dict):
                candidates.extend(config.get(key) for key in ("path", "endpoint"))
        elif isinstance(value, (str, Path)):
            candidates.append(value)
        paths: set[Path] = set()
        for candidate in candidates:
            raw = str(candidate or "").strip()
            if not raw:
                continue
            try:
                parsed = urlsplit(raw)
            except ValueError:
                continue
            if parsed.scheme.lower() in {"http", "https", "ftp", "file"}:
                continue
            try:
                resolved = Path(raw).expanduser().resolve()
                resolved.relative_to(upload_dir)
            except (OSError, RuntimeError, ValueError):
                continue
            paths.add(resolved)
        return paths

    def start_import(self, spec: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise ExternalSourceError("source service is shutting down")
        self._prune_jobs()
        active = sum(
            1 for job in self._jobs.values() if job.get("status") in {"queued", "running"}
        )
        if active >= self.MAX_ACTIVE_JOBS:
            raise ExternalSourceError("too many external source jobs are already running")
        job_id = uuid.uuid4().hex
        now = time.time()
        job = {
            "job_id": job_id,
            "status": "queued",
            "total": 0,
            "processed": 0,
            "imported": 0,
            "duplicates": 0,
            "pending": 0,
            "stale": 0,
            "failed": 0,
            "errors": [],
            "created_at": now,
            "updated_at": now,
        }
        self._jobs[job_id] = job
        task = asyncio.create_task(self._run_import(job_id, dict(spec)), name=f"source_import_{job_id[:8]}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda _task, value=job_id: self._tasks.pop(value, None))
        return dict(job)

    async def import_now(self, spec: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = time.time()
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "total": 0,
            "processed": 0,
            "imported": 0,
            "duplicates": 0,
            "pending": 0,
            "stale": 0,
            "failed": 0,
            "errors": [],
            "created_at": now,
            "updated_at": now,
        }
        await self._run_import(job_id, dict(spec))
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        self._prune_jobs()
        job = self._jobs.get(str(job_id or ""))
        return copy.deepcopy(job) if job else None

    async def cancel_job(self, job_id: str) -> bool:
        task = self._tasks.get(str(job_id or ""))
        if task is None or task.done():
            return False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    def _prune_jobs(self) -> None:
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.get("status") not in {"queued", "running"}
            and now - float(job.get("completed_at") or job.get("updated_at") or now)
            > self.JOB_TTL_SECONDS
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    async def _run_import(self, job_id: str, spec: dict[str, Any]) -> None:
        job = self._jobs[job_id]
        source_id = ""
        reader: Any = None
        try:
            job.update(status="running", updated_at=time.time())
            inspection, reader = await self.inspect(spec)
            if not inspection.ok:
                raise ExternalSourceError("; ".join(inspection.errors))
            job["total"] = len(inspection.items)
            plan = self._build_import_plan(inspection, spec)
            source_id = plan.source["source_id"]
            job["source_id"] = source_id
            job["source_name"] = plan.source["name"]
            if self.db and hasattr(self.db, "upsert_source"):
                await self.db.upsert_source(plan.source)
                await self.db.update_source_status(source_id, status="syncing")
            await self._import_inspected_items(job, reader, inspection.items, plan)
            await self._complete_source_import(job, inspection, source_id)
        except asyncio.CancelledError:
            job.update(status="cancelled", completed_at=time.time(), updated_at=time.time())
            if source_id and self.db and hasattr(self.db, "update_source_status"):
                await self.db.update_source_status(source_id, status="cancelled")
        except Exception as exc:
            message = str(exc)
            job.update(
                status="failed",
                error=message,
                completed_at=time.time(),
                updated_at=time.time(),
            )
            if source_id and self.db and hasattr(self.db, "update_source_status"):
                await self.db.update_source_status(source_id, status="error", last_error=message)
            logger.error(f"[Source] 外部源导入失败: {exc}", exc_info=True)
        finally:
            await self._close_reader(reader)

    def _build_import_plan(
        self, inspection: SourceInspection, spec: dict[str, Any]
    ) -> _SourceImportPlan:
        category_map = (
            spec.get("category_map") if isinstance(spec.get("category_map"), dict) else {}
        )
        review = bool(
            spec.get("review", self._cfg("external_source_default_review", False))
        ) or bool(self._cfg("content_filtration", False))
        scope_mode = normalize_scope_mode(spec.get("scope_mode")) or "public"
        origin_target = str(spec.get("origin_target") or "").strip()
        if scope_mode == "local" and not (
            origin_target.startswith("group:") or origin_target.startswith("user:")
        ):
            raise ExternalSourceError(
                "local-scope import requires origin_target group:<id> or user:<id>"
            )
        if review and self.db and hasattr(self.db, "count_pending"):
            pending_capacity = int(self._cfg("steal_pool_capacity", 200))
            if self.db.count_pending() + len(inspection.items) > pending_capacity:
                raise ExternalSourceError(
                    "external import would exceed the pending pool capacity"
                )

        # Resolve/create the optional role only after import-wide validation.
        character = self._prepare_character(spec)
        if character:
            spec["character"] = character
        return _SourceImportPlan(
            source=self._source_record(inspection, spec),
            category_map=category_map,
            review=review,
            scope_mode=scope_mode,
            origin_target=origin_target,
            character=character,
        )

    async def _import_inspected_items(
        self,
        job: dict[str, Any],
        reader: Any,
        items: list[SourceItem],
        plan: _SourceImportPlan,
    ) -> None:
        for item in items:
            raw: bytes | None = None
            try:
                raw = await self._read_source_item(reader, item)
                outcome = await self._import_item(
                    source=plan.source,
                    item=item,
                    raw=raw,
                    category_map=plan.category_map,
                    review=plan.review,
                    scope_mode=plan.scope_mode,
                    origin_target=plan.origin_target,
                    character=plan.character,
                )
                job[outcome] = int(job.get(outcome, 0)) + 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                job["failed"] += 1
                errors = job["errors"]
                if len(errors) < 20:
                    errors.append(f"{item.external_id}: {exc}")
                logger.warning(f"[Source] 导入条目失败 {item.external_id}: {exc}")
            finally:
                raw = None
                job["processed"] += 1
                job["updated_at"] = time.time()

    async def _complete_source_import(
        self,
        job: dict[str, Any],
        inspection: SourceInspection,
        source_id: str,
    ) -> None:
        if self.db and hasattr(self.db, "reconcile_source_items"):
            await self.db.reconcile_source_items(
                source_id,
                [item.external_id for item in inspection.items],
            )
        job.update(status="completed", completed_at=time.time(), updated_at=time.time())
        if self.db and hasattr(self.db, "count_stale_source_items"):
            job["stale"] = self.db.count_stale_source_items(source_id)
        if self.db and hasattr(self.db, "update_source_status"):
            await self.db.update_source_status(
                source_id,
                status="ready",
                item_count=len(inspection.items),
                last_sync_at=int(time.time()),
            )
        invalidate_search(self.plugin)

    @staticmethod
    async def _close_reader(reader: Any) -> None:
        close = getattr(reader, "close", None)
        if not callable(close):
            return
        try:
            value = close()
            if asyncio.iscoroutine(value):
                await value
        except Exception as exc:
            logger.debug(f"[Source] 关闭源读取器失败: {exc}")

    async def _read_source_item(self, reader: Any, item: SourceItem) -> bytes:
        method = reader.read_item
        if asyncio.iscoroutinefunction(method):
            return await method(item)
        return await asyncio.to_thread(method, item)

    async def _import_item(
        self,
        *,
        source: dict[str, Any],
        item: SourceItem,
        raw: bytes,
        category_map: dict[str, Any],
        review: bool,
        scope_mode: str,
        origin_target: str,
        character: str,
    ) -> str:
        image_hash = hashlib.sha256(raw).hexdigest()
        if self.db and hasattr(self.db, "blacklisted_hashes"):
            if image_hash in self.db.blacklisted_hashes():
                raise ExternalSourceError("image hash is blacklisted")
        width, height, image_format, suffix = self._validate_image(raw)
        existing = self.db.get_emoji_by_hash(image_hash) if self.db else None
        if existing:
            if (
                character
                and self.db
                and hasattr(self.db, "update_path")
                and not str((existing[1] or {}).get("character") or "").strip()
            ):
                await self.db.update_path(existing[0], {"character": character})
                await refresh_search_entry(self.plugin, existing[0])
            await self._link_item(source, item, image_hash, existing[0])
            return "duplicates"
        if self.db and hasattr(self.db, "get_pending_by_hash"):
            pending = self.db.get_pending_by_hash(image_hash)
            if pending:
                if (
                    character
                    and not str(pending.get("character") or "").strip()
                    and pending.get("id")
                    and hasattr(self.db, "update_pending")
                ):
                    await self.db.update_pending(
                        int(pending["id"]),
                        {"character": character},
                    )
                await self._link_item(
                    source,
                    item,
                    image_hash,
                    None,
                    pending_path=str(pending.get("path") or ""),
                )
                return "duplicates"
        category = self._map_category(item.category, category_map)
        now = int(time.time())
        metadata = {
            "hash": image_hash,
            "category": category,
            "tags": self._normalize_labels(item.tags),
            "scenes": self._normalize_labels(item.scenes),
            "desc": item.text_description[:1000],
            "overlay_text": str(item.visible_text or "")[:500],
            "emotions": self._normalize_labels(item.emotions),
            "source": f"external:{source['source_type']}",
            "source_url": str(item.source_url or "")[:2000],
            "original_name": str(item.filename or Path(item.relative_path).name)[:255],
            "width": width,
            "height": height,
            "format": image_format.lower(),
            "bytes": len(raw),
            "add_method": "external_import",
            "scope_mode": scope_mode,
            "origin_target": origin_target,
            "character": character,
            "created_at": now,
            "retention_class": "external",
        }
        if review:
            config = getattr(self.plugin, "plugin_config", None)
            target_dir = Path(getattr(config, "pending_dir", self.data_dir / "pending"))
            target_dir.mkdir(parents=True, exist_ok=True)
            target = self._unique_target(target_dir, item, image_hash, suffix)
            await self._write_atomic(target, raw)
            metadata["path"] = str(target)
            pending_id = await self.db.insert_pending(metadata) if self.db else None
            if not pending_id:
                await safe_remove_file(str(target))
                raise ExternalSourceError("failed to insert pending metadata")
            await self._link_item(source, item, image_hash, None, pending_path=str(target))
            return "pending"
        target_dir = self._category_dir(category)
        target = self._unique_target(target_dir, item, image_hash, suffix)
        await self._write_atomic(target, raw)
        metadata["path"] = str(target)
        inserted = await self.db.insert_batch([metadata]) if self.db else 0
        if inserted <= 0:
            await safe_remove_file(str(target))
            raise ExternalSourceError("failed to insert image metadata")
        await self._link_item(source, item, image_hash, str(target))
        await refresh_search_entry(self.plugin, str(target), metadata)
        return "imported"

    def _prepare_character(self, spec: dict[str, Any]) -> str:
        """Resolve the optional role assignment and create a new role once."""

        raw = str(spec.get("character") or "").strip()
        if not raw:
            return ""
        if len(raw) > 80 or any(ord(character) < 32 for character in raw):
            raise ExternalSourceSecurityError("character key is invalid")
        config = getattr(self.plugin, "plugin_config", None)
        normalizer = getattr(config, "normalize_character_key", normalize_character_key)
        character = str(normalizer(raw) or "").strip()[:80]
        if not character:
            return ""
        getter = getattr(config, "get_characters", None) if config else None
        known = (
            list(getter())
            if callable(getter)
            else list(getattr(config, "characters", []) or [])
        )
        normalized_known = {
            str(normalizer(value) or "").strip(): str(value) for value in known
        }
        if character in normalized_known:
            return normalized_known[character]
        if not bool(spec.get("create_character", False)):
            raise ExternalSourceError(
                f"character is not registered: {character}; enable create_character to add it"
            )
        if config is None or not hasattr(config, "characters"):
            return character
        config.characters = [*known, character]
        info = dict(getattr(config, "character_info", {}) or {})
        info.setdefault(character, {"name": raw[:80], "desc": ""})
        config.character_info = info
        save_characters = getattr(config, "save_characters", None)
        save_info = getattr(config, "save_character_info", None)
        if callable(save_characters):
            save_characters()
        if callable(save_info):
            save_info()
        return character

    async def _link_item(
        self,
        source: dict[str, Any],
        item: SourceItem,
        image_hash: str,
        path: str | None,
        *,
        pending_path: str = "",
    ) -> None:
        if not self.db or not hasattr(self.db, "link_source_item"):
            return
        metadata = self._public_item_metadata(item)
        if pending_path:
            metadata["pending_path"] = pending_path
        remote_hash = str(item.metadata.get("content_sha256") or "").strip().lower()
        if not _SHA256_RE.fullmatch(remote_hash):
            remote_hash = image_hash
        await self.db.link_source_item(
            {
                "source_id": source["source_id"],
                "external_id": item.external_id,
                "path": path,
                "source_category": item.category,
                "source_url": self._redact_endpoint(item.source_url),
                "license": item.license,
                "attribution": item.attribution,
                "remote_hash": remote_hash,
                "metadata": metadata,
            }
        )

    def _validate_image(self, raw: bytes) -> tuple[int, int, str, str]:
        max_bytes = int(self._cfg("external_source_max_image_bytes", 32 * 1024 * 1024))
        if not raw or len(raw) > max_bytes:
            raise ExternalSourceSecurityError("image exceeds the configured byte limit")
        max_pixels = int(self._cfg("external_source_max_pixels", 40_000_000))
        try:
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
                image_format = str(image.format or "").upper()
                if width <= 0 or height <= 0 or width * height > max_pixels:
                    raise ExternalSourceSecurityError("image exceeds the configured pixel limit")
                if image_format not in _FORMAT_SUFFIX:
                    raise ExternalSourceSecurityError(f"unsupported image format: {image_format}")
                image.verify()
        except ExternalSourceSecurityError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise ExternalSourceError("invalid or corrupted image") from exc
        return width, height, image_format, _FORMAT_SUFFIX[image_format]

    @classmethod
    def _public_item_metadata(cls, item: SourceItem) -> dict[str, Any]:
        """Keep useful provenance while excluding arbitrary catalog secrets."""

        metadata: dict[str, Any] = {}
        for key, value in item.metadata.items():
            key_text = str(key)
            if key_text.lower() not in cls.PUBLIC_METADATA_KEYS:
                continue
            if key_text.lower() in {"source_url", "url"}:
                value = cls._redact_endpoint(value)
            metadata[key_text[:80]] = cls._json_safe(value)
        if item.source_url:
            metadata["source_url"] = cls._redact_endpoint(item.source_url)
        metadata.setdefault("relative_path", str(item.relative_path or "")[:1000])
        return metadata

    @classmethod
    def _json_safe(cls, value: Any, depth: int = 0) -> Any:
        if depth >= 3:
            return str(value)[:1000]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:1000]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, nested in list(value.items())[:32]:
                key_text = str(key)
                key_lower = key_text.lower()
                if any(
                    marker in key_lower
                    for marker in (
                        "token",
                        "secret",
                        "password",
                        "authorization",
                        "api_key",
                        "apikey",
                    )
                ):
                    continue
                safe_value = cls._json_safe(nested, depth + 1)
                if key_lower in {"url", "source_url", "endpoint", "repository"}:
                    safe_value = cls._redact_endpoint(safe_value)
                result[key_text[:80]] = safe_value
            return result
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(nested, depth + 1) for nested in list(value)[:32]]
        return str(value)[:1000]

    def _map_category(self, raw: str, category_map: dict[str, Any]) -> str:
        source_key = str(raw or "").strip()
        mapped = str(category_map.get(source_key, source_key) or "").strip()
        config = getattr(self.plugin, "plugin_config", None)
        known = list(config.get_categories()) if config and hasattr(config, "get_categories") else []
        if mapped in known and self._is_safe_category(mapped):
            return mapped
        if config and hasattr(config, "normalize_category_strict"):
            strict = config.normalize_category_strict(mapped)
            if strict and strict in known and self._is_safe_category(strict):
                return strict
        if config and hasattr(config, "closest_category"):
            closest = str(config.closest_category(mapped or source_key) or "").strip()
            if closest and self._is_safe_category(closest):
                return closest
        fallback = "confused" if "confused" in known else (known[0] if known else "unknown")
        if not self._is_safe_category(fallback):
            raise ExternalSourceSecurityError("no safe target category is available")
        return fallback

    @staticmethod
    def _is_safe_category(category: str) -> bool:
        try:
            return normalize_category_key(category) == str(category or "").strip().lower()
        except ValueError:
            return False

    def _category_dir(self, category: str) -> Path:
        if not self._is_safe_category(category):
            raise ExternalSourceSecurityError(f"unsafe target category: {category}")
        config = getattr(self.plugin, "plugin_config", None)
        if config and hasattr(config, "ensure_category_dir"):
            target = Path(config.ensure_category_dir(category)).resolve()
        else:
            target = (self.data_dir / "categories" / category).resolve()
            target.mkdir(parents=True, exist_ok=True)
        categories_root = (self.data_dir / "categories").resolve()
        try:
            target.relative_to(categories_root)
        except ValueError as exc:
            raise ExternalSourceSecurityError("target category escapes storage root") from exc
        return target

    @staticmethod
    def _normalize_labels(values: list[str], limit: int = 16) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for raw in values:
            text = str(raw or "").strip()[:120]
            if text and not text.lower().startswith("category:") and text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _unique_target(target_dir: Path, item: SourceItem, image_hash: str, suffix: str) -> Path:
        raw_stem = Path(str(item.filename or item.external_id or "meme")).stem
        stem = re.sub(r"[^\w.-]+", "_", raw_stem, flags=re.UNICODE).strip("._")[:64]
        stem = stem or "meme"
        base = f"ext_{image_hash[:12]}_{stem}"
        candidate = target_dir / f"{base}{suffix}"
        index = 1
        while candidate.exists():
            candidate = target_dir / f"{base}_{index}{suffix}"
            index += 1
        return candidate

    async def _write_atomic(self, target: Path, raw: bytes) -> None:
        staging = self.import_dir / "staging"
        await asyncio.to_thread(staging.mkdir, parents=True, exist_ok=True)
        temporary = staging / f"{uuid.uuid4().hex}.part"
        try:
            await asyncio.to_thread(temporary.write_bytes, raw)
            await asyncio.to_thread(os.replace, temporary, target)
        finally:
            if temporary.exists():
                await safe_remove_file(str(temporary))

    def _source_record(
        self, inspection: SourceInspection, spec: dict[str, Any]
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "source_type": inspection.source_type,
            "category_map": spec.get("category_map")
            if isinstance(spec.get("category_map"), dict)
            else {},
            "review": bool(
                spec.get("review", self._cfg("external_source_default_review", False))
            ),
            "scope_mode": normalize_scope_mode(spec.get("scope_mode")) or "public",
            "origin_target": str(spec.get("origin_target") or "").strip(),
            "character": str(spec.get("character") or "").strip()[:80],
            "create_character": bool(spec.get("create_character", False)),
        }
        if inspection.source_type in {"meme_pack", "github"}:
            config["path"] = inspection.endpoint
            if inspection.source_type == "github":
                config.pop("path", None)
                if (
                    spec.get("repository")
                    or spec.get("repo")
                    or spec.get("url")
                    or spec.get("endpoint")
                ):
                    config["repository"] = str(
                        spec.get("repository")
                        or spec.get("repo")
                        or spec.get("url")
                        or spec.get("endpoint")
                    )[:2000]
                if spec.get("ref"):
                    config["ref"] = str(spec.get("ref"))[:255]
                if spec.get("subpath"):
                    config["subpath"] = str(spec.get("subpath"))[:1000]
                if isinstance(spec.get("headers"), dict):
                    config["headers"] = {
                        str(key)[:80]: str(value)[:4096]
                        for key, value in list(spec["headers"].items())[:16]
                        if str(key).lower()
                        in {"accept", "authorization", "user-agent", "x-api-key"}
                    }
        else:
            config["endpoint"] = inspection.endpoint
            if isinstance(spec.get("headers"), dict):
                config["headers"] = {
                    str(key)[:80]: str(value)[:4096]
                    for key, value in list(spec["headers"].items())[:16]
                    if str(key).lower()
                    in {"accept", "authorization", "user-agent", "x-api-key"}
                }
        return {
            "source_id": inspection.source_id,
            "source_type": inspection.source_type,
            "name": str(spec.get("name") or inspection.name)[:160],
            "endpoint": inspection.endpoint,
            "config": config,
            "enabled": bool(spec.get("enabled", True)),
            "status": "idle",
            "item_count": len(inspection.items),
        }

    @staticmethod
    def _public_source(source: dict[str, Any]) -> dict[str, Any]:
        result = dict(source)
        result["endpoint"] = SourceService._redact_endpoint(result.get("endpoint"))
        config = result.get("config")
        if isinstance(config, dict):
            config = dict(config)
            for endpoint_key in ("endpoint", "repository"):
                if config.get(endpoint_key):
                    config[endpoint_key] = SourceService._redact_endpoint(config[endpoint_key])
            headers = config.get("headers")
            if isinstance(headers, dict):
                config["headers"] = {str(key): "********" for key in headers}
            result["config"] = config
        result.pop("config_json", None)
        return result

    @staticmethod
    def _redact_endpoint(value: Any) -> str:
        endpoint = str(value or "")
        try:
            parsed = urlsplit(endpoint)
        except ValueError:
            return endpoint
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.query:
            return endpoint
        sensitive = {
            "access_token",
            "api_key",
            "apikey",
            "auth",
            "authorization",
            "key",
            "password",
            "secret",
            "signature",
            "token",
        }
        query = [
            (
                key,
                "********"
                if key.lower() in sensitive
                or any(
                    marker in key.lower()
                    for marker in ("token", "secret", "password", "auth", "apikey", "api_key", "signature")
                )
                else item_value,
            )
            for key, item_value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
        return urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
        )

    def source_spec(self, source_id: str) -> dict[str, Any] | None:
        if not self.db or not hasattr(self.db, "get_sources"):
            return None
        for source in self.db.get_sources():
            if source.get("source_id") != source_id:
                continue
            config = source.get("config") if isinstance(source.get("config"), dict) else {}
            spec = dict(config)
            spec["source_type"] = source.get("source_type")
            spec.setdefault("path", source.get("endpoint"))
            spec.setdefault("endpoint", source.get("endpoint"))
            spec.setdefault("name", source.get("name"))
            return spec
        return None
