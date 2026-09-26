"""SQLite library queries operations."""

from typing import Any




class LibraryQueries:
    # ── 分页查询 ──

    # 排序字段白名单（防止SQL注入）
    _VALID_ORDER_FIELDS = {
        "newest": "e.created_at DESC, e.path DESC",
        "oldest": "e.created_at ASC, e.path ASC",
        "least_used": "e.use_count ASC, e.created_at ASC, e.path ASC",
        "most_used": "e.use_count DESC, e.last_used_at DESC, e.path ASC",
        "last_used": "e.last_used_at DESC, e.use_count DESC, e.path ASC",
    }

    _LIBRARY_CLAUSES = {
        "general": "COALESCE(e.is_favorite, 0) = 0 AND TRIM(COALESCE(e.character, '')) = ''",
        "favorites": "e.is_favorite = 1",
        "characters": "COALESCE(e.is_favorite, 0) = 0 AND TRIM(COALESCE(e.character, '')) != ''",
    }

    def get_library_counts(self) -> dict[str, int]:
        with self._get_connection() as conn:
            counts = {
                key: conn.execute(f"SELECT COUNT(*) FROM emoji e WHERE {clause}").fetchone()[0]
                for key, clause in self._LIBRARY_CLAUSES.items()
            }
            counts["automatic"] = conn.execute(
                "SELECT COUNT(*) FROM emoji e WHERE " + self._LIBRARY_CLAUSES["general"]
                + " AND TRIM(COALESCE(e.retention_class, 'native')) NOT IN ('external', 'pinned')"
            ).fetchone()[0]
            return counts

    def get_emojis_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        category: str | None = None,
        sort_order: str = "newest",
        search_query: str | None = None,
        scope_target: str | None = None,
        favorite_only: bool = False,
        character: str | None = None,
        library: str = "",
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        """分页获取表情包列表，支持过滤、搜索和排序。

        Args:
            page: 页码（从1开始）
            page_size: 每页数量
            category: 分类过滤（可选）
            sort_order: 排序方式 - "newest"(最新), "oldest"(最旧), "most_used"(最常用)
            search_query: 搜索关键词（匹配标签、描述、场景）
            scope_target: scope 过滤目标（可选）
            favorite_only: 仅显示收藏的表情包

        Returns:
            tuple: (图片列表, 总数, 分类统计)
        """
        # 白名单验证排序字段（防止SQL注入）
        order_sql = self._VALID_ORDER_FIELDS.get(sort_order, "e.created_at DESC")

        with self._get_connection() as conn:
            # 构建基础查询条件
            where_clauses: list[str] = []
            params: list[Any] = []
            category_count_where_clauses: list[str] = []
            category_count_params: list[Any] = []

            library_clause = self._LIBRARY_CLAUSES.get(library)
            if library_clause:
                where_clauses.append(library_clause)
                category_count_where_clauses.append(library_clause)

            # 分类过滤
            if category:
                where_clauses.append("e.category = ?")
                params.append(category)

            if character == "__none__":
                where_clauses.append("(e.character IS NULL OR e.character = '')")
            elif character:
                where_clauses.append("e.character = ?")
                params.append(character)

            # 收藏过滤
            if favorite_only:
                where_clauses.append("e.is_favorite = 1")
                category_count_where_clauses.append("e.is_favorite = 1")

            # scope 过滤
            if scope_target:
                where_clauses.append("(e.scope_mode = 'public' OR e.origin_target = ?)")
                params.append(scope_target)
                category_count_where_clauses.append(
                    "(e.scope_mode = 'public' OR e.origin_target = ?)"
                )
                category_count_params.append(scope_target)

            # 搜索过滤（描述/标签/场景/分类/hash/来源/文件名/路径）
            if search_query:
                search_pattern = f"%{search_query}%"
                search_clause = (
                    "(e.desc LIKE ? OR e.category LIKE ? OR e.hash LIKE ?"
                    " OR e.origin_target LIKE ? OR e.source LIKE ? OR e.path LIKE ?"
                    " OR e.overlay_text LIKE ? OR e.character LIKE ?"
                    " OR EXISTS("
                    "SELECT 1 FROM emoji_tag t WHERE t.path = e.path AND t.tag LIKE ?"
                    ") OR EXISTS("
                    "SELECT 1 FROM emoji_scene s WHERE s.path = e.path AND s.scene LIKE ?"
                    "))"
                )
                where_clauses.append(search_clause)
                params.extend([search_pattern] * 10)
                category_count_where_clauses.append(search_clause)
                category_count_params.extend([search_pattern] * 10)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            category_count_where_sql = ""
            if category_count_where_clauses:
                category_count_where_sql = "WHERE " + " AND ".join(category_count_where_clauses)

            # 计算总数
            count_sql = f"SELECT COUNT(*) as cnt FROM emoji e {where_sql}"
            total = conn.execute(count_sql, params).fetchone()["cnt"]

            # 分类统计（用于侧边栏显示）
            cat_count_sql = f"""
                SELECT e.category, COUNT(*) as cnt
                FROM emoji e {category_count_where_sql}
                GROUP BY e.category
            """
            cat_rows = conn.execute(cat_count_sql, category_count_params).fetchall()
            category_counts = {r["category"]: r["cnt"] for r in cat_rows}

            # 分页查询
            offset = (page - 1) * page_size
            limit = page_size

            data_sql = f"""
                SELECT e.path, e.hash, e.category, e.desc, e.scope_mode,
                       e.origin_target, e.created_at, e.use_count, e.last_used_at,
                       e.is_favorite, e.reviewed_at, e.source,
                       e.source_url, e.original_name, e.width, e.height,
                       e.format, e.bytes, e.add_method,
                       e.overlay_text, e.emotions_json, e.character, e.retention_class
                FROM emoji e {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(data_sql, params + [limit, offset]).fetchall()

            if not rows:
                return [], total, category_counts

            # 批量获取标签和场景（解决N+1问题）
            paths = [r["path"] for r in rows]
            tags_map = self._load_related_map(
                conn, table="emoji_tag", value_column="tag", paths=paths
            )
            scenes_map = self._load_related_map(
                conn, table="emoji_scene", value_column="scene", paths=paths
            )

            # 构建结果列表
            images: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["tags"] = tags_map.get(row["path"], [])
                item["scenes"] = scenes_map.get(row["path"], [])
                images.append(self._hydrate_entry(item))

            return images, total, category_counts
