"""SQLite source repository operations."""

import asyncio
import json
import time
from typing import Any




class SourceRepository:
    # ── 外部源注册表（v3 additive schema） ──

    @staticmethod
    def _decode_json_object(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        try:
            parsed = json.loads(str(value or "{}"))
        except (TypeError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    def get_sources(self) -> list[dict[str, Any]]:
        """Return registered source descriptors without exposing raw secrets."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM meme_source ORDER BY updated_at DESC, source_id ASC"
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["enabled"] = bool(item.get("enabled", 1))
                item["config"] = self._decode_json_object(item.pop("config_json", None))
                result.append(item)
            return result

    async def upsert_source(self, source: dict[str, Any]) -> bool:
        if not isinstance(source, dict) or not str(source.get("source_id") or "").strip():
            return False
        async with self._write_lock:
            return await asyncio.to_thread(self._upsert_source_sync, source)

    def _upsert_source_sync(self, source: dict[str, Any]) -> bool:
        now = int(time.time())
        source_id = str(source.get("source_id") or "").strip()[:190]
        source_type = str(source.get("source_type") or "").strip()[:40]
        name = str(source.get("name") or source_id).strip()[:160]
        endpoint = str(source.get("endpoint") or "").strip()[:2000]
        config = source.get("config", source.get("config_json", {}))
        config_json = (
            str(config)
            if isinstance(config, str)
            else json.dumps(config if isinstance(config, dict) else {}, ensure_ascii=False)
        )
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO meme_source
                    (source_id, source_type, name, endpoint, config_json, enabled,
                     status, last_error, item_count, last_sync_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    source_type=excluded.source_type,
                    name=excluded.name,
                    endpoint=excluded.endpoint,
                    config_json=excluded.config_json,
                    enabled=excluded.enabled,
                    status=excluded.status,
                    last_error=excluded.last_error,
                    item_count=excluded.item_count,
                    last_sync_at=COALESCE(excluded.last_sync_at, meme_source.last_sync_at),
                    updated_at=excluded.updated_at
                """,
                (
                    source_id,
                    source_type or "unknown",
                    name or source_id,
                    endpoint,
                    config_json,
                    1 if bool(source.get("enabled", True)) else 0,
                    str(source.get("status") or "idle"),
                    str(source.get("last_error") or "")[:1000] or None,
                    int(source.get("item_count") or 0),
                    source.get("last_sync_at"),
                    int(source.get("created_at") or now),
                    now,
                ),
            )
        return True

    async def update_source_status(
        self,
        source_id: str,
        *,
        status: str,
        last_error: str = "",
        item_count: int | None = None,
        last_sync_at: int | None = None,
    ) -> bool:
        if not source_id:
            return False
        async with self._write_lock:
            return await asyncio.to_thread(
                self._update_source_status_sync,
                source_id,
                status,
                last_error,
                item_count,
                last_sync_at,
            )

    def _update_source_status_sync(
        self,
        source_id: str,
        status: str,
        last_error: str,
        item_count: int | None,
        last_sync_at: int | None,
    ) -> bool:
        assignments = ["status = ?", "last_error = ?", "updated_at = ?"]
        values: list[Any] = [str(status or "idle")[:40], str(last_error or "")[:1000] or None, int(time.time())]
        if item_count is not None:
            assignments.append("item_count = ?")
            values.append(max(0, int(item_count)))
        if last_sync_at is not None:
            assignments.append("last_sync_at = ?")
            values.append(int(last_sync_at))
        values.append(source_id)
        with self._get_connection() as conn:
            cur = conn.execute(
                f"UPDATE meme_source SET {', '.join(assignments)} WHERE source_id = ?",
                values,
            )
            return bool(cur.rowcount)

    async def delete_source(self, source_id: str) -> bool:
        if not source_id:
            return False
        async with self._write_lock:
            return await asyncio.to_thread(self._delete_source_sync, source_id)

    def _delete_source_sync(self, source_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM meme_source WHERE source_id = ?", (source_id,))
            return bool(cur.rowcount)

    def get_source_item(self, source_id: str, external_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM meme_source_item WHERE source_id = ? AND external_id = ?",
                (source_id, external_id),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["metadata"] = self._decode_json_object(result.pop("metadata_json", None))
            return result

    async def link_source_item(self, item: dict[str, Any]) -> bool:
        if not isinstance(item, dict) or not item.get("source_id") or not item.get("external_id"):
            return False
        async with self._write_lock:
            return await asyncio.to_thread(self._link_source_item_sync, item)

    def _link_source_item_sync(self, item: dict[str, Any]) -> bool:
        now = int(time.time())
        metadata = item.get("metadata", item.get("metadata_json", {}))
        metadata_json = (
            str(metadata)
            if isinstance(metadata, str)
            else json.dumps(metadata if isinstance(metadata, dict) else {}, ensure_ascii=False)
        )
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO meme_source_item
                    (source_id, external_id, path, source_category, source_url,
                     license, attribution, remote_hash, metadata_json,
                     first_seen_at, last_seen_at, stale)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(source_id, external_id) DO UPDATE SET
                    path=excluded.path,
                    source_category=excluded.source_category,
                    source_url=excluded.source_url,
                    license=excluded.license,
                    attribution=excluded.attribution,
                    remote_hash=excluded.remote_hash,
                    metadata_json=excluded.metadata_json,
                    last_seen_at=excluded.last_seen_at,
                    stale=0
                """,
                (
                    str(item.get("source_id"))[:190],
                    str(item.get("external_id"))[:180],
                    item.get("path"),
                    str(item.get("source_category") or "")[:160],
                    str(item.get("source_url") or "")[:2000],
                    str(item.get("license") or "")[:160],
                    str(item.get("attribution") or "")[:500],
                    str(item.get("remote_hash") or "")[:128],
                    metadata_json,
                    int(item.get("first_seen_at") or now),
                    now,
                ),
            )
        return True

    def get_source_items(self, source_id: str) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM meme_source_item WHERE source_id = ? ORDER BY external_id",
                (source_id,),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["metadata"] = self._decode_json_object(item.pop("metadata_json", None))
                result.append(item)
            return result

    async def mark_source_items_stale(self, source_id: str) -> int:
        if not source_id:
            return 0
        async with self._write_lock:
            return await asyncio.to_thread(self._mark_source_items_stale_sync, source_id)

    def _mark_source_items_stale_sync(self, source_id: str) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE meme_source_item SET stale = 1 WHERE source_id = ?",
                (source_id,),
            )
            return int(cur.rowcount or 0)

    async def reconcile_source_items(
        self,
        source_id: str,
        seen_external_ids: list[str],
    ) -> int:
        """Mark only entries absent from a successfully inspected catalog stale."""

        if not source_id:
            return 0
        seen = {str(value)[:180] for value in seen_external_ids if str(value)}
        async with self._write_lock:
            return await asyncio.to_thread(
                self._reconcile_source_items_sync,
                source_id,
                seen,
            )

    def _reconcile_source_items_sync(self, source_id: str, seen: set[str]) -> int:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT external_id, stale FROM meme_source_item WHERE source_id = ?",
                (source_id,),
            ).fetchall()
            stale_ids = [row["external_id"] for row in rows if row["external_id"] not in seen]
            current_ids = [row["external_id"] for row in rows if row["external_id"] in seen]
            conn.executemany(
                "UPDATE meme_source_item SET stale = 1 WHERE source_id = ? AND external_id = ?",
                [(source_id, external_id) for external_id in stale_ids],
            )
            conn.executemany(
                "UPDATE meme_source_item SET stale = 0 WHERE source_id = ? AND external_id = ?",
                [(source_id, external_id) for external_id in current_ids],
            )
            return len(stale_ids)

    def count_stale_source_items(self, source_id: str) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM meme_source_item "
                "WHERE source_id = ? AND stale = 1",
                (source_id,),
            ).fetchone()
            return int(row["cnt"] if row else 0)

    async def promote_source_pending_path(self, pending_path: str, path: str) -> int:
        """Attach provenance after an externally imported pending image is approved."""
        if not pending_path or not path:
            return 0
        async with self._write_lock:
            return await asyncio.to_thread(
                self._promote_source_pending_path_sync, pending_path, path
            )

    def _promote_source_pending_path_sync(self, pending_path: str, path: str) -> int:
        updated = 0
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT source_id, external_id, metadata_json "
                "FROM meme_source_item WHERE path IS NULL"
            ).fetchall()
            for row in rows:
                metadata = self._decode_json_object(row["metadata_json"])
                if str(metadata.get("pending_path") or "") != pending_path:
                    continue
                metadata.pop("pending_path", None)
                cur = conn.execute(
                    """
                    UPDATE meme_source_item
                    SET path = ?, metadata_json = ?, last_seen_at = ?
                    WHERE source_id = ? AND external_id = ?
                    """,
                    (
                        path,
                        json.dumps(metadata, ensure_ascii=False),
                        int(time.time()),
                        row["source_id"],
                        row["external_id"],
                    ),
                )
                updated += int(cur.rowcount or 0)
        return updated

    def count_external(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM emoji WHERE retention_class = 'external'"
            ).fetchone()
            return int(row["cnt"] if row else 0)
