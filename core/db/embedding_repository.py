"""SQLite embedding repository operations."""

import sqlite3
import time
from typing import Any




class EmbeddingRepository:
    def get_search_refresh_generations(self, paths: list[str]) -> dict[str, int]:
        generations = {}
        with self._get_connection() as conn:
            for start in range(0, len(paths), self._RELATED_FETCH_CHUNK_SIZE):
                batch = [self._store_path(path) for path in paths[start:start + self._RELATED_FETCH_CHUNK_SIZE]]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(f"SELECT path, generation FROM search_refresh WHERE path IN ({placeholders})", batch)
                generations.update({self._resolve_path(row["path"]): row["generation"] for row in rows})
        return generations

    def enqueue_search_refresh(self, paths: list[str]) -> None:
        with self._get_connection() as conn:
            conn.executemany(
                "INSERT INTO search_refresh(path) VALUES (?) ON CONFLICT(path) "
                "DO UPDATE SET generation = generation + 1, retry_at = 0",
                [(path,) for path in dict.fromkeys(map(self._store_path, paths))],
            )

    def clear_all_embeddings(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emoji_embedding")

    def get_search_refresh_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM search_refresh WHERE retry_at <= ? ORDER BY retry_at, path LIMIT ?",
                (int(time.time()), max(1, limit)),
            ).fetchall()
            return [dict(row, path=self._resolve_path(row["path"])) for row in rows]

    def finish_search_refresh(self, path: str, generation: int, error: str = "") -> None:
        """A newer committed change must survive an older worker's completion."""
        path = self._store_path(path)
        with self._get_connection() as conn:
            if error:
                conn.execute(
                    "UPDATE search_refresh SET attempts = attempts + 1, "
                    "retry_at = ? + MIN(300, 5 * (1 << MIN(attempts, 6))), last_error = ? "
                    "WHERE path = ? AND generation = ?",
                    (int(time.time()), error[:500], path, generation),
                )
            else:
                conn.execute("DELETE FROM search_refresh WHERE path = ? AND generation = ?", (path, generation))

    # ── 嵌入向量 (emoji_embedding) CRUD ──

    def upsert_embedding(
        self,
        path: str,
        vector_blob: bytes,
        dim: int,
        model_sig: str,
    ) -> None:
        """写入或更新某 path 的向量。"""
        if not path or not vector_blob:
            return
        now = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO emoji_embedding (path, vector, dim, model_sig, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    model_sig = excluded.model_sig,
                    updated_at = excluded.updated_at
                """,
                (self._store_path(path), sqlite3.Binary(vector_blob), int(dim), model_sig, now),
            )

    def delete_embedding(self, path: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emoji_embedding WHERE path = ?", (self._store_path(path),))

    def load_embeddings_by_sig(self, model_sig: str) -> list[dict[str, Any]]:
        """加载某 model_sig 的所有向量行，用于构建内存索引矩阵。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, vector, dim FROM emoji_embedding WHERE model_sig = ?",
                (model_sig,),
            ).fetchall()
            return [
                {
                    "path": self._resolve_path(r["path"]),
                    "vector": bytes(r["vector"]),
                    "dim": int(r["dim"]),
                }
                for r in rows
            ]

    def count_embeddings_by_sig(self, model_sig: str) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji_embedding WHERE model_sig = ?",
                (model_sig,),
            ).fetchone()
            return int(row["cnt"] if row else 0)

    def get_all_embedding_paths(self, model_sig: str | None = None) -> list[str]:
        """所有已存向量的 path（用于对比 emoji 表，检测缺失/陈旧向量）。"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT path FROM emoji_embedding" if model_sig is None else
                                "SELECT path FROM emoji_embedding WHERE model_sig = ?",
                                () if model_sig is None else (model_sig,)).fetchall()
            return [self._resolve_path(r["path"]) for r in rows]
