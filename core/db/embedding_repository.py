"""SQLite embedding repository operations."""

import sqlite3
import time
from typing import Any




class EmbeddingRepository:
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
                (path, sqlite3.Binary(vector_blob), int(dim), model_sig, now),
            )

    def delete_embedding(self, path: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emoji_embedding WHERE path = ?", (path,))

    def load_embeddings_by_sig(self, model_sig: str) -> list[dict[str, Any]]:
        """加载某 model_sig 的所有向量行，用于构建内存索引矩阵。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, vector, dim FROM emoji_embedding WHERE model_sig = ?",
                (model_sig,),
            ).fetchall()
            return [
                {
                    "path": r["path"],
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

    def get_all_embedding_paths(self) -> list[str]:
        """所有已存向量的 path（用于对比 emoji 表，检测缺失/陈旧向量）。"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT path FROM emoji_embedding").fetchall()
            return [r["path"] for r in rows]
