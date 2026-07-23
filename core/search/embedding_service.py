"""嵌入向量服务：封装 AstrBot EmbeddingProvider + FaissVecDB / numpy 降级。

参考 astrbot_plugin_livingmemory 的实现模式：
- Provider 获取：context.get_all_embedding_providers() / provider_manager.inst_map
- 向量存储：优先 FaissVecDB，不可用时降级 SQLite + numpy
- 维度校验：provider 切换时自动清理旧索引
- 回填：启动时为已有 emoji 批量补算向量
"""

import json
from typing import Any

import numpy as np

from astrbot.api import logger


class EmbeddingService:
    """嵌入向量服务。

    两层架构（对齐 livingmemory 的 FaissVecDB 模式）：
    1. 优先 FaissVecDB（astrbot.core.db.vec_db.faiss_impl.vec_db）
    2. 降级 SQLite emoji_embedding 表 + numpy 内存矩阵
    """

    # 回填参数
    BACKFILL_BATCH_SIZE = 20

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self._provider: Any | None = None
        self._provider_dim: int = 0

        # Provider 负缓存：同一配置下未找到时不再重复探测/打印日志
        self._provider_not_found: bool = False
        self._last_enable_embedding_search: bool | None = None
        self._last_embedding_provider_id: str | None = None

        # FaissVecDB
        self._faiss_db: Any | None = None
        self._faiss_available: bool | None = None  # None=未检测

        # numpy 降级
        self._fallback_matrix: np.ndarray | None = None
        self._fallback_paths: list[str] = []
        self._fallback_dim: int = 0
        self._fallback_loaded: bool = False

    # ═══════════════════════════════════════════════════
    #  Provider（对齐 livingmemory _initialize_providers）
    # ═══════════════════════════════════════════════════

    def _embedding_enabled(self) -> bool:
        """读取插件配置中的嵌入检索开关。"""
        return getattr(self.plugin, "enable_embedding_search", True)

    def _reset_provider_state_if_changed(self) -> None:
        """配置开关或 provider ID 变化时重置 provider 探测状态。"""
        current_enable = self._embedding_enabled()
        current_id = str(getattr(self.plugin, "embedding_provider_id", "") or "").strip()
        if (
            self._last_enable_embedding_search != current_enable
            or self._last_embedding_provider_id != current_id
        ):
            self._provider = None
            self._provider_dim = 0
            self._provider_not_found = False
            self._faiss_available = None
            self._faiss_db = None
            self.invalidate_cache()
            self._last_enable_embedding_search = current_enable
            self._last_embedding_provider_id = current_id

    def _get_provider(self) -> Any | None:
        """获取 EmbeddingProvider（优先配置 ID，留空取首个）。"""
        # 开关关闭时直接跳过，不触发 provider 探测与日志
        if not self._embedding_enabled():
            return None

        self._reset_provider_state_if_changed()

        if self._provider is not None:
            return self._provider

        # 负缓存：同一配置下已确认无 provider，直接返回避免重复日志
        if self._provider_not_found:
            return None

        provider_id = getattr(self.plugin, "embedding_provider_id", None) or ""

        # 1. 按 ID 查找（对齐 livingmemory _initialize_providers）
        if provider_id:
            provider = self._find_provider_by_id(provider_id)
            if provider is not None:
                # 类型校验：get_provider_by_id 返回的可能是 chat/stt/tts/embedding
                if not self._is_embedding_provider(provider):
                    logger.warning(
                        f"[Embedding] Provider '{provider_id}' 不是 EmbeddingProvider 类型，已忽略"
                    )
                else:
                    self._provider = provider
                    self._provider_dim = self._get_provider_dim(provider)
                    logger.info(
                        f"[Embedding] 使用指定 Provider: {provider_id} (dim={self._provider_dim})"
                    )
                    return self._provider

        # 2. 取框架首个 Embedding Provider
        try:
            providers = self.plugin.context.get_all_embedding_providers()
        except Exception:
            providers = []

        if providers:
            self._provider = providers[0]
            self._provider_dim = self._get_provider_dim(self._provider)
            pid = self._extract_provider_id(self._provider)
            logger.info(f"[Embedding] 自动选择首个 Provider: {pid} (dim={self._provider_dim})")
            return self._provider

        logger.info("[Embedding] 未找到 Embedding Provider，嵌入检索不可用")
        self._provider_not_found = True
        return None

    def _find_provider_by_id(self, provider_id: str) -> Any | None:
        """按 ID 查找 provider（对齐 livingmemory 的静默查找模式）。"""
        # 静默查找：直接读 provider_manager.inst_map，避免 AstrBot 的 warning 日志
        try:
            pm = getattr(self.plugin.context, "provider_manager", None)
            inst_map = getattr(pm, "inst_map", None)
            if isinstance(inst_map, dict):
                p = inst_map.get(provider_id)
                if p is not None:
                    return p
        except Exception:
            pass

        # 回退到公开 API
        try:
            return self.plugin.context.get_provider_by_id(provider_id)
        except Exception:
            pass

        return None

    @staticmethod
    def _is_embedding_provider(provider: Any) -> bool:
        """校验 provider 是否为 EmbeddingProvider 类型（对齐 livingmemory isinstance 检查）。"""
        try:
            from astrbot.core.provider.provider import EmbeddingProvider
            return isinstance(provider, EmbeddingProvider)
        except ImportError:
            # 无法导入时退化为鸭子类型检查：有 get_embedding 方法即可
            return hasattr(provider, "get_embedding")

    @staticmethod
    def _extract_provider_id(provider: Any) -> str:
        """从 provider 实例提取 ID 字符串。"""
        cfg = getattr(provider, "provider_config", {})
        if isinstance(cfg, dict):
            return cfg.get("id", "unknown")
        return getattr(cfg, "id", "unknown")

    @staticmethod
    def _get_provider_dim(provider: Any) -> int:
        """获取 provider 的输出维度。"""
        if hasattr(provider, "get_dim"):
            try:
                return provider.get_dim()
            except Exception:
                pass
        return 0

    # ═══════════════════════════════════════════════════
    #  可用性
    # ═══════════════════════════════════════════════════

    def is_available(self) -> bool:
        """嵌入检索是否可用。"""
        if not getattr(self.plugin, "enable_embedding_search", True):
            return False

        if self._init_faiss():
            return True
        if self._init_fallback():
            return True
        return False

    # ═══════════════════════════════════════════════════
    #  FaissVecDB（对齐 livingmemory _complete_initialization）
    # ═══════════════════════════════════════════════════

    def _init_faiss(self) -> bool:
        """初始化 FaissVecDB。"""
        if self._faiss_available is not None:
            return self._faiss_available

        provider = self._get_provider()
        if provider is None:
            self._faiss_available = False
            return False

        try:
            import faiss  # noqa: F401
            from astrbot.core.db.vec_db.faiss_impl.vec_db import FaissVecDB
        except ImportError:
            logger.info("[Embedding] faiss 未安装，降级 numpy")
            self._faiss_available = False
            return False

        try:
            data_dir = self._resolve_data_dir()
            db_path = f"{data_dir}/emoji_faiss.db"
            index_path = f"{data_dir}/emoji_faiss.index"

            self._faiss_db = FaissVecDB(db_path, index_path, provider)
            self._faiss_available = True
            logger.info(f"[Embedding] FaissVecDB 就绪 (dim={self._provider_dim})")
            return True
        except Exception as e:
            logger.warning(f"[Embedding] FaissVecDB 初始化失败: {e}")
            self._faiss_available = False
            return False

    def _resolve_data_dir(self) -> str:
        """解析插件数据目录。"""
        if hasattr(self.plugin, "plugin_config"):
            return str(self.plugin.plugin_config.data_dir)
        if hasattr(self.plugin, "data_dir"):
            return str(self.plugin.data_dir)
        return "."

    # ═══════════════════════════════════════════════════
    #  numpy 降级（SQLite emoji_embedding 表）
    # ═══════════════════════════════════════════════════

    def _init_fallback(self) -> bool:
        """初始化 numpy 降级路径。"""
        provider = self._get_provider()
        if provider is None:
            return False
        self._load_fallback_matrix()
        return True  # provider 可用即就绪

    def _load_fallback_matrix(self) -> None:
        """从 SQLite emoji_embedding 表加载向量到内存矩阵。"""
        if self._fallback_loaded:
            return

        self._fallback_matrix = None
        self._fallback_paths = []
        self._fallback_dim = 0

        db = getattr(self.plugin, "db_service", None)
        if not db or not hasattr(db, "load_embeddings_by_sig"):
            self._fallback_loaded = True
            return

        rows = None
        for sig in ("fallback",):
            try:
                rows = db.load_embeddings_by_sig(sig)
                if rows:
                    break
            except Exception:
                continue

        if not rows:
            self._fallback_loaded = True
            return

        # 构建 numpy 矩阵（对齐 livingmemory 的向量加载模式）
        vectors = []
        paths = []
        dim = 0
        for r in rows:
            blob = r.get("vector")
            path = r.get("path")
            if not blob or not path:
                continue
            try:
                vec = np.frombuffer(bytes(blob), dtype=np.float32)
                d = int(r.get("dim", 0))
                if d > 0 and len(vec) == d:
                    vectors.append(vec)
                    paths.append(str(path))
                    dim = d
            except Exception:
                continue

        if vectors:
            mat = np.stack(vectors, axis=0)
            # L2 归一化（对齐 livingmemory 的余弦相似度计算）
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._fallback_matrix = mat / norms
            self._fallback_paths = paths
            self._fallback_dim = dim
            logger.info(f"[Embedding] numpy 降级矩阵已加载: {len(paths)} 条 (dim={dim})")
        else:
            logger.debug("[Embedding] numpy 降级矩阵为空（无有效向量）")

        self._fallback_loaded = True

    def _check_fallback_dimension(self) -> bool:
        """对齐 livingmemory _check_and_fix_dimension_mismatch：
        检查降级矩阵维度是否与当前 provider 匹配，不匹配则清空。
        """
        if self._fallback_dim == 0 or self._provider_dim == 0:
            return True  # 无法判断，继续使用
        if self._fallback_dim != self._provider_dim:
            logger.warning(
                f"[Embedding] 维度不匹配: 已存={self._fallback_dim}, "
                f"provider={self._provider_dim}。旧向量将被清除并重建。"
            )
            # 清空 SQLite 向量表
            db = getattr(self.plugin, "db_service", None)
            if db and hasattr(db, "_get_connection"):
                try:
                    with db._get_connection() as conn:
                        conn.execute("DELETE FROM emoji_embedding")
                    logger.info("[Embedding] 已清除旧向量，将在回填中重建")
                except Exception as e:
                    logger.warning(f"[Embedding] 清除旧向量失败: {e}")
            self._fallback_loaded = False
            self._fallback_matrix = None
            self._fallback_paths = []
            self._fallback_dim = 0
            return False
        return True

    # ═══════════════════════════════════════════════════
    #  初始化
    # ═══════════════════════════════════════════════════

    async def initialize(self) -> None:
        """异步初始化（对齐 livingmemory 的 initialize 流程）。"""
        # 开关关闭时完全跳过，不探测 provider、不输出日志
        if not self._embedding_enabled():
            return

        # 1. Provider 就绪检查
        provider = self._get_provider()
        if provider is None:
            logger.info(
                "[Embedding] 未就绪 — 请在 AstrBot 后台配置 Embedding 模型"
            )
            return

        # 2. 尝试 FaissVecDB
        if self._init_faiss() and self._faiss_db is not None:
            try:
                await self._faiss_db.initialize()
                logger.info("[Embedding] FaissVecDB 初始化完成 ✅")
            except Exception as e:
                logger.warning(f"[Embedding] FaissVecDB 初始化失败，降级: {e}")
                self._faiss_available = False

        # 3. 降级 numpy
        if not self._faiss_available or self._faiss_available is None:
            if self._init_fallback():
                self._check_fallback_dimension()
                count = len(self._fallback_paths)
                logger.info(
                    f"[Embedding] numpy 降级方案就绪 ✅ ({count} 条向量, dim={self._fallback_dim})"
                    if count > 0 else
                    "[Embedding] numpy 降级方案就绪 ✅ — 向量库为空，新入库自动填充"
                )

    # ═══════════════════════════════════════════════════
    #  插入 / 删除
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _build_search_text(entry: dict[str, Any]) -> str:
        """拼接嵌入文本：category + desc + tags + scenes。"""
        parts = [
            str(entry.get("category", "") or ""),
            str(entry.get("desc", "") or ""),
        ]
        for field in ("tags", "scenes"):
            vals = entry.get(field, []) or []
            if isinstance(vals, list):
                parts.extend(str(v) for v in vals if v)
        return " ".join(parts).strip()

    async def insert_emoji(self, path: str, entry: dict[str, Any]) -> bool:
        """插入单条 emoji 向量（对齐 FaissVecDB.insert 模式）。

        失败不阻塞入库流程。
        """
        # 开关关闭时跳过写入，避免触发 provider 探测
        if not self._embedding_enabled():
            return False

        text = self._build_search_text(entry)
        if not text:
            return False

        # 截断超长文本（对齐 livingmemory _MAX_CONTENT_CHARS = 4000）
        if len(text) > 4000:
            text = text[:4000]

        # FaissVecDB 路径
        if self._init_faiss() and self._faiss_db is not None:
            try:
                await self._faiss_db.insert(
                    content=text,
                    metadata={"path": path, "category": str(entry.get("category", ""))},
                )
                return True
            except Exception as e:
                logger.debug(f"[Embedding] FaissVecDB 插入失败: {e}")

        # numpy 降级
        return await self._fallback_insert(path, text)

    async def delete_by_path(self, path: str) -> bool:
        """按 path 删除向量。"""
        if self._init_faiss() and self._faiss_db is not None:
            try:
                ds = self._faiss_db.document_storage
                docs = await ds.get_documents(metadata_filters={"path": path}, limit=1)
                if docs and len(docs) > 0:
                    uuid_id = docs[0].get("doc_id")
                    if uuid_id:
                        await self._faiss_db.delete(uuid_id)
                        return True
            except Exception:
                pass

        db = getattr(self.plugin, "db_service", None)
        if db and hasattr(db, "delete_embedding"):
            db.delete_embedding(path)
            self._fallback_loaded = False
            return True
        return False

    async def _fallback_insert(self, path: str, text: str) -> bool:
        """降级：get_embedding → upsert_embedding。"""
        provider = self._get_provider()
        if provider is None:
            return False

        try:
            vec = await provider.get_embedding(text)
            if not vec or len(vec) == 0:
                return False
        except Exception as e:
            logger.debug(f"[Embedding] get_embedding 失败: {e}")
            return False

        db = getattr(self.plugin, "db_service", None)
        if not db or not hasattr(db, "upsert_embedding"):
            return False

        try:
            blob = np.array(vec, dtype=np.float32).tobytes()
            db.upsert_embedding(path, blob, dim=len(vec), model_sig="fallback")
            self._fallback_loaded = False  # 下次 search 时重载
            return True
        except Exception as e:
            logger.debug(f"[Embedding] upsert_embedding 失败: {e}")
            return False

    # ═══════════════════════════════════════════════════
    #  检索
    # ═══════════════════════════════════════════════════

    async def search(self, query: str, k: int = 80) -> list[tuple[str, float]]:
        """向量检索 top-K（对齐 FaissVecDB.retrieve 模式）。

        Returns:
            [(path, similarity_score), ...]  按相似度降序
        """
        if not query or not query.strip():
            return []

        # 截断查询（对齐 livingmemory _MAX_QUERY_CHARS = 2000）
        processed = query[:2000] if len(query) > 2000 else query

        # FaissVecDB 路径
        if self._init_faiss() and self._faiss_db is not None:
            try:
                results = await self._faiss_db.retrieve(
                    query=processed, k=k, fetch_k=k * 2, rerank=False
                )
                out: list[tuple[str, float]] = []
                for r in results:
                    data = getattr(r, "data", None)
                    if data is None:
                        continue
                    # FaissVecDB 返回的 data 是 {"id": int, "text": str, "metadata": dict}
                    if isinstance(data, dict):
                        meta = data.get("metadata", {})
                        if isinstance(meta, dict):
                            p = meta.get("path", "")
                            if p:
                                out.append((p, float(r.similarity)))
                        elif isinstance(meta, str):
                            # metadata 被序列化成了 JSON 字符串
                            try:
                                meta_dict = json.loads(meta)
                                p = meta_dict.get("path", "")
                                if p:
                                    out.append((p, float(r.similarity)))
                            except (json.JSONDecodeError, TypeError):
                                pass
                    elif isinstance(data, str):
                        # data 本身是 JSON 字符串
                        try:
                            data_dict = json.loads(data)
                            meta = data_dict.get("metadata", {})
                            if isinstance(meta, dict):
                                p = meta.get("path", "")
                            elif isinstance(meta, str):
                                meta = json.loads(meta)
                                p = meta.get("path", "") if isinstance(meta, dict) else ""
                            if p:
                                out.append((p, float(r.similarity)))
                        except (json.JSONDecodeError, TypeError):
                            pass
                return out
            except Exception as e:
                logger.warning(f"[Embedding] FaissVecDB 检索失败: {e}")

        # numpy 降级
        return await self._fallback_search(processed, k)

    async def _fallback_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """降级：numpy 余弦相似度检索。"""
        try:
            self._load_fallback_matrix()
            if self._fallback_matrix is None or len(self._fallback_paths) == 0:
                return []

            provider = self._get_provider()
            if provider is None:
                return []

            # 嵌入查询
            try:
                vec = await provider.get_embedding(query)
                if not vec:
                    return []
            except Exception:
                return []

            # 余弦相似度
            qv = np.array(vec, dtype=np.float32)
            q_norm = np.linalg.norm(qv)
            if q_norm == 0:
                return []
            qv = qv / q_norm

            # 维度安全检查：不匹配则清空旧向量（对齐 livingmemory _check_and_fix_dimension_mismatch）
            if self._fallback_matrix.shape[1] != len(qv):
                logger.warning(
                    f"[Embedding] 维度不匹配: matrix={self._fallback_matrix.shape[1]}, "
                    f"query={len(qv)}。清除旧向量，将在下次回填中重建。"
                )
                self._fallback_loaded = False
                self._fallback_matrix = None
                self._fallback_paths = []
                self._fallback_dim = 0
                return []

            scores = self._fallback_matrix @ qv
            top_idx = np.argsort(scores)[::-1][:k]

            results: list[tuple[str, float]] = []
            for idx in top_idx:
                s = float(scores[idx])
                if s < 0.15:  # 低相似度截断
                    continue
                results.append((self._fallback_paths[int(idx)], s))
            return results
        except Exception as e:
            logger.warning(f"[Embedding] numpy 搜索异常: {e}")
            return []

    def invalidate_cache(self) -> None:
        """标记缓存过期。"""
        self._fallback_loaded = False
        self._fallback_matrix = None
        self._fallback_paths = []

    # ═══════════════════════════════════════════════════
    #  回填（对齐 livingmemory 的批量重建模式）
    # ═══════════════════════════════════════════════════

    async def backfill_existing(self, batch_size: int | None = None) -> int:
        """启动时批量回填缺少向量的旧 emoji。

        对齐 livingmemory index_rebuild 的批量处理模式。
        关键：回填到当前活跃的存储后端（FaissVecDB 或 SQLite），不混用。
        """
        # 开关关闭时跳过回填，避免触发 provider 探测与重复日志
        if not self._embedding_enabled():
            return 0

        if batch_size is None:
            batch_size = self.BACKFILL_BATCH_SIZE

        db = getattr(self.plugin, "db_service", None)
        if not db:
            return 0

        # 获取所有 emoji 路径
        try:
            all_paths = db.get_all_paths()
        except Exception as e:
            logger.warning(f"[Embedding] 回填失败 — 无法获取 emoji 列表: {e}")
            return 0
        if not all_paths:
            return 0

        # 判断当前活跃的存储后端，据此检查已有向量
        using_faiss = self._init_faiss() and self._faiss_db is not None

        if using_faiss:
            # FaissVecDB：检查其文档存储中已有的 path
            try:
                ds = self._faiss_db.document_storage
                existing_docs = await ds.get_documents(metadata_filters={}, limit=100000)
                embedded = set()
                for doc in existing_docs:
                    if isinstance(doc, dict):
                        meta = doc.get("metadata", {})
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except (json.JSONDecodeError, TypeError):
                                meta = {}
                        if isinstance(meta, dict):
                            p = meta.get("path", "")
                            if p:
                                embedded.add(p)
            except Exception:
                embedded = set()
        else:
            # SQLite 降级
            try:
                embedded = set(db.get_all_embedding_paths())
            except Exception:
                embedded = set()

        missing = [p for p in all_paths if p not in embedded]
        if not missing:
            backend = "FaissVecDB" if using_faiss else "SQLite"
            logger.info(
                f"[Embedding] 回填跳过 — {backend} 中全部 {len(all_paths)} 条已有向量"
            )
            # 如果 FaissVecDB 空但 SQLite 有数据，提示迁移
            if using_faiss:
                sqlite_embedded = set()
                try:
                    sqlite_embedded = set(db.get_all_embedding_paths())
                except Exception:
                    pass
                if len(sqlite_embedded) > len(embedded):
                    logger.info(
                        f"[Embedding] 检测到 SQLite 中有 {len(sqlite_embedded)} 条旧向量，"
                        f"正在迁移到 FaissVecDB..."
                    )
                    missing = [p for p in all_paths if p not in embedded]
                    # 清除 embedded 判断，强制全量回填到 FaissVecDB
                    embedded = set()
                    missing = [p for p in all_paths if p in sqlite_embedded]
            if not missing:
                return 0

        backend = "FaissVecDB" if using_faiss else "SQLite"
        logger.info(
            f"[Embedding] 回填开始 → {backend}: {len(missing)}/{len(all_paths)} 条缺少向量"
        )

        # 加载索引元数据
        idx = {}
        try:
            db = getattr(self.plugin, "db_service", None)
            if db and db.count_total() > 0:
                idx = db.get_index_cache_readonly()
        except Exception:
            pass

        written = 0
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            batch_written = 0
            for path in batch:
                entry = idx.get(path, {})
                if not entry:
                    try:
                        entry = db.get_emoji(path) or {}
                    except Exception:
                        pass
                text = self._build_search_text(entry)
                if not text:
                    continue
                if using_faiss:
                    # 直接写入 FaissVecDB
                    ok = await self._faiss_insert(path, text, entry)
                else:
                    # 写入 SQLite
                    ok = await self._fallback_insert(path, text)
                if ok:
                    batch_written += 1
            written += batch_written
            if batch_written > 0:
                logger.info(
                    f"[Embedding] 回填进度: {min(i + batch_size, len(missing))}/{len(missing)}, "
                    f"本批 +{batch_written}"
                )

        # 刷新缓存
        if not using_faiss:
            self._fallback_loaded = False
            self._load_fallback_matrix()

        logger.info(
            f"[Embedding] 回填完成 → {backend}: 成功 {written}/{len(missing)}"
        )
        return written

    async def _faiss_insert(self, path: str, text: str, entry: dict[str, Any]) -> bool:
        """向 FaissVecDB 插入一条。"""
        if self._faiss_db is None:
            return False
        try:
            await self._faiss_db.insert(
                content=text,
                metadata={"path": path, "category": str(entry.get("category", ""))},
            )
            return True
        except Exception as e:
            logger.debug(f"[Embedding] FaissVecDB 插入失败 {path}: {e}")
            return False

    # ═══════════════════════════════════════════════════
    #  清理
    # ═══════════════════════════════════════════════════

    async def close(self) -> None:
        """关闭 FaissVecDB 并重置状态。"""
        if self._faiss_db is not None:
            try:
                await self._faiss_db.close()
            except Exception:
                pass
            self._faiss_db = None
            self._faiss_available = False
