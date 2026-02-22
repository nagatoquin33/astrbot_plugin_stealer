import asyncio
import os
import random
import re
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

# 尝试导入文本相似度计算模块
try:
    from .text_similarity import calculate_hybrid_similarity, calculate_simple_similarity as calculate_similarity
except ImportError:
    # 如果导入失败，提供简单的降级实现
    def calculate_similarity(s1: str, s2: str) -> float:
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        return 1.0 if s1 == s2 else 0.0

    def calculate_hybrid_similarity(text1: str, text2: str) -> float:
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        if text2_lower in text1_lower or text1_lower in text2_lower:
            return 0.5
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        if words1 and words2:
            return len(words1 & words2) / max(len(words1), len(words2))
        return 0.0


class EmojiSelector:
    """表情包选择器，负责查找、筛选和选择表情包。"""

    # 正则表达式模式常量
    HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:&&|\\&\\&)")
    INCOMPLETE_HEX_PATTERN = re.compile(
        r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:[|]|\n|$)"
    )
    SINGLE_HEX_PATTERN = re.compile(r"&([^&\s]+?)&")

    # 选择器常量
    MAX_RECENT_USAGE = 10  # 最近使用记录最大数量
    MIN_RECENT_USAGE = 3  # 最近使用记录最小数量

    # 相似度阈值常量
    SIMILARITY_THRESHOLD = 0.65  # 模糊匹配相似度阈值

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self.categories: list[str] = (
            plugin_instance.categories if hasattr(plugin_instance, "categories") else []
        )
        self._selection_lock = asyncio.Lock()

    def _check_group_allowed(self, event: AstrMessageEvent) -> bool:
        """检查当前群组是否允许使用表情包功能。

        Args:
            event: 消息事件对象

        Returns:
            bool: True 表示允许，False 表示不允许
        """
        config = getattr(self.plugin, "plugin_config", None)
        if not config:
            return True
        if not hasattr(config, "is_group_allowed"):
            return True
        group_id = config.get_group_id(event) if hasattr(config, "get_group_id") else None
        return config.is_group_allowed(group_id)

    def _canon_path(self, path: str) -> str:
        """规范化路径用于比较去重。"""
        try:
            return os.path.normcase(os.path.abspath(os.path.normpath(path)))
        except Exception:
            return str(path or "")

    def _get_category_from_data(self, data: dict | None) -> str:
        """从数据字典中获取小写的分类名。

        Args:
            data: 图片元数据字典

        Returns:
            str: 小写的分类名，如果不存在则返回空字符串
        """
        if not isinstance(data, dict):
            return ""
        return str(data.get("category", "")).lower()

    def _get_recent_usage(self, category: str) -> list[str]:
        recent_usage_key = f"recent_usage_{category}"
        recent_usage = getattr(self.plugin, recent_usage_key, [])
        if not isinstance(recent_usage, list):
            recent_usage = []
        normalized = []
        for item in recent_usage:
            if not item:
                continue
            normalized.append(self._canon_path(str(item)))
        return normalized

    def _set_recent_usage(self, category: str, recent_usage: list[str]) -> None:
        recent_usage_key = f"recent_usage_{category}"
        setattr(self.plugin, recent_usage_key, recent_usage)

    def normalize_category(self, category: str) -> str:
        """归一化分类名称，返回有效分类或空字符串。"""
        if not category:
            return ""
        cfg = getattr(self.plugin, "plugin_config", None)
        if not cfg:
            return ""
        try:
            result = cfg.normalize_category_strict(category)
            return result or ""
        except Exception:
            return ""

    async def extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            res: list[str] = []
            seen: set[str] = set()
            cleaned_text = str(text)
            valid_categories = set(self.categories)

            # 1. 处理显式包裹标记
            temp_text = cleaned_text
            for match in self.HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    seen.add(norm_cat)
                    res.append(norm_cat)
                temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            # 2. 处理残缺标记
            temp_text = cleaned_text
            for match in self.INCOMPLETE_HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    logger.debug(f"检测到残缺情绪标签: {emotion} -> {norm_cat}")
                    seen.add(norm_cat)
                    res.append(norm_cat)
                    temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            # 3. 处理单个&包裹
            temp_text = cleaned_text
            for match in self.SINGLE_HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    seen.add(norm_cat)
                    res.append(norm_cat)
                    temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            return res, cleaned_text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text

    async def select_emoji(self, category: str, context_text: str = "") -> str | None:
        """选择表情包（智能或随机）。"""
        async with self._selection_lock:
            use_smart = getattr(self.plugin, "smart_emoji_selection", True)

            if use_smart and context_text and len(context_text.strip()) > 5:
                smart_path = await self._select_emoji_smart_impl(category, context_text)
                if smart_path:
                    return smart_path

            return self._select_emoji_random_impl(category, use_smart=use_smart)

    async def select_emoji_smart(self, category: str, context_text: str) -> str | None:
        """智能选择表情包（强制智能）。"""
        async with self._selection_lock:
            return await self._select_emoji_smart_impl(category, context_text)

    def _select_emoji_random_impl(self, category: str, use_smart: bool) -> str | None:
        categories_dir = None
        if self.plugin.plugin_config and getattr(
            self.plugin.plugin_config, "categories_dir", None
        ):
            categories_dir = self.plugin.plugin_config.categories_dir

        if not categories_dir:
            return None

        cat_dir = Path(categories_dir) / category
        if not cat_dir.exists():
            return None

        try:
            files = [p for p in cat_dir.iterdir() if p.is_file()]
            if not files:
                return None

            recent_usage = self._get_recent_usage(category)
            recent_set = set(recent_usage)
            candidates = [(p, self._canon_path(str(p))) for p in files]

            # 过滤最近使用
            available = [p for p, canon in candidates if canon not in recent_set]
            if not available:
                available = [p for p, _ in candidates]
                recent_usage = []
                recent_set = set()

            picked = random.choice(available)
            picked_path = self._canon_path(str(picked))

            if picked_path in recent_set:
                recent_usage = [p for p in recent_usage if p != picked_path]
            recent_usage.append(picked_path)

            max_recent = min(
                self.MAX_RECENT_USAGE, max(self.MIN_RECENT_USAGE, len(files) // 2)
            )
            if len(recent_usage) > max_recent:
                recent_usage = recent_usage[-max_recent:]

            self._set_recent_usage(category, recent_usage)
            return str(picked)
        except Exception as e:
            logger.error(f"随机选择表情包失败: {e}")
            return None

    async def _select_emoji_smart_impl(
        self, category: str, context_text: str
    ) -> str | None:
        try:
            cache_service = getattr(self.plugin, "cache_service", None)
            if not cache_service:
                return None

            idx = await cache_service.load_index()
            candidates = []

            # 这里重用了 search_images 的核心逻辑，但加入了 context_text 的匹配
            # 为了减少重复，我们可以直接调用 search_images 搜索 context_text
            # 但 context_text 是长文本，search_images 适合短语
            # 所以这里保留独立逻辑，或者复用 calculate_hybrid_similarity

            # 简化版实现：
            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                file_cat = self._get_category_from_data(data)
                if file_cat != category:
                    continue

                desc = str(data.get("desc", ""))

                score = calculate_hybrid_similarity(context_text, desc)
                if score > 0.3:
                    candidates.append((file_path, score))

            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]

            return None

        except Exception as e:
            logger.error(f"智能选择表情包失败: {e}")
            return None

    async def search_images(
        self, query: str, limit: int = 1, idx: dict | None = None
    ) -> list[tuple[str, str, str, str]]:
        """根据查询词搜索图片。"""
        try:
            if idx is None:
                if hasattr(self.plugin, "cache_service"):
                    idx = self.plugin.cache_service.get_index_cache()

            if not idx:
                return []

            query_lower = query.lower()
            query_tokens = [t for t in query_lower.split() if len(t) > 1]
            query_chars = set(query_lower) if len(query_lower) > 1 else set()

            MAX_STR_LENGTH = 20
            top_k = limit * 3
            top_candidates = []

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                score = 0
                desc = str(data.get("desc", "")).lower()
                # 安全处理 tags 字段，兼容字符串和列表类型
                raw_tags = data.get("tags", [])
                if isinstance(raw_tags, str):
                    tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
                elif isinstance(raw_tags, list):
                    tags = [str(t).lower() for t in raw_tags if t]
                else:
                    tags = []
                tags_str = ", ".join(tags)
                category = self._get_category_from_data(data)

                # 快速过滤
                if query_chars:
                    all_text = category + " " + desc + " " + " ".join(tags)
                    if not query_chars.intersection(set(all_text)):
                        continue

                # 精确匹配分类
                if query_lower == category:
                    score = 20
                    top_candidates.append((file_path, desc, category, tags_str, score))
                    continue

                # 包含匹配分类
                if query_lower in category or category in query_lower:
                    score = 10

                # 描述匹配
                if query_lower == desc:
                    score = max(score, 15)
                elif query_lower in desc:
                    score = max(score, 12)
                elif query_tokens:
                    matched = sum(1 for token in query_tokens if token in desc)
                    score = max(score, matched * 3)

                if score >= 15:
                    top_candidates.append((file_path, desc, category, tags_str, score))
                    continue

                # 标签匹配
                if query_lower == tags[0] if tags else False:
                    score = max(score, 12)
                elif tags and query_lower in tags[0]:
                    score = max(score, 8)
                elif query_tokens:
                    for tag in tags:
                        matched = sum(1 for token in query_tokens if token in tag)
                        score = max(score, matched * 2)
                        if score >= 15:
                            break

                if score >= 15:
                    top_candidates.append((file_path, desc, category, tags_str, score))
                    continue

                # 模糊匹配 (降级)
                if score >= 10 and query_tokens:
                    has_fuzzy = False
                    for target in [category, desc] + tags[:2]:
                        if len(target) > 1 and len(query_lower) > 1:
                            sim = calculate_similarity(
                                query_lower[:MAX_STR_LENGTH], target[:MAX_STR_LENGTH]
                            )
                            if sim >= self.SIMILARITY_THRESHOLD:
                                score = max(score, int(4 + (sim - self.SIMILARITY_THRESHOLD) * 12))
                                has_fuzzy = True
                                break
                    if has_fuzzy:
                        top_candidates.append(
                            (file_path, desc, category, tags_str, score)
                        )
                        continue

                if score > 0:
                    top_candidates.append((file_path, desc, category, tags_str, score))

                if len(top_candidates) > top_k:
                    top_candidates.sort(key=lambda x: x[4], reverse=True)
                    top_candidates = top_candidates[:top_k]

            top_candidates.sort(key=lambda x: x[4], reverse=True)
            results = [
                (item[0], item[1], item[2], item[3]) for item in top_candidates[:limit]
            ]

            # 如果没有结果，尝试关键词映射
            if not results:
                cfg = getattr(self.plugin, "plugin_config", None)
                keyword_map = cfg.get_keyword_map() if cfg else {}
                if query in keyword_map:
                    mapped_cat = keyword_map[query]
                    logger.debug(f"尝试映射查询 '{query}' -> 分类 '{mapped_cat}'")
                    # 递归调用（注意避免无限递归，这里只映射一次）
                    # 为了简单，直接在 idx 中找分类匹配
                    for file_path, data in idx.items():
                        if not isinstance(data, dict):
                            continue
                        cat = self._get_category_from_data(data)
                        if cat == mapped_cat:
                            # 安全处理 tags 字段
                            raw_tags = data.get("tags", [])
                            if isinstance(raw_tags, str):
                                tags_str = raw_tags
                            elif isinstance(raw_tags, list):
                                tags_str = ", ".join(str(t) for t in raw_tags)
                            else:
                                tags_str = ""
                            results.append(
                                (file_path, str(data.get("desc", "")), cat, tags_str)
                            )
                            if len(results) >= limit:
                                break

            return results

        except Exception as e:
            logger.error(f"搜索图片失败: {e}")
            return []

    async def smart_search(
        self,
        query: str,
        limit: int = 5,
        idx: dict | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """智能搜索表情包（带多级 fallback）。

        搜索顺序：
        1) 直接用 query 调用 search_images
        2) 关键词映射（如"无语" -> dumb）
        3) 模糊匹配到分类（相似度阈值 0.4）

        Args:
            query: 搜索关键词
            limit: 返回结果数量
            idx: 索引缓存，为 None 时自动加载

        Returns:
            list[tuple[path, desc, emotion, tags]]
        """
        # 1) 直接搜索

        results = await self.search_images(query, limit=limit, idx=idx)
        if results:
            return results

        # 2) 关键词映射
        cfg = getattr(self.plugin, "plugin_config", None)
        keyword_map = cfg.get_keyword_map() if cfg else {}
        if query in keyword_map:
            mapped_category = keyword_map[query]
            results = await self.search_images(mapped_category, limit=limit, idx=idx)
            if results:
                return results

        # 3) 模糊匹配到分类
        best_match = None
        best_score = 0.0
        for category in self.categories:
            score = calculate_hybrid_similarity(query, category)
            if score > best_score and score > 0.4:
                best_score = score
                best_match = category

        if best_match:
            results = await self.search_images(best_match, limit=limit, idx=idx)

        return results

    def check_send_probability(self) -> bool:
        """检查表情包发送概率。"""
        try:
            emoji_chance = getattr(self.plugin, "emoji_chance", 0.4)
            chance = float(emoji_chance)
            if chance <= 0:
                logger.debug("表情包自动发送概率为0，未触发图片发送")
                return False
            if chance > 1:
                chance = 1.0
            if random.random() >= chance:
                logger.debug(f"表情包自动发送概率检查未通过 ({chance}), 未触发图片发送")
                return False

            logger.debug("表情包自动发送概率检查通过")
            return True
        except Exception as e:
            logger.error(f"解析表情包自动发送概率配置失败: {e}")
            return False

    async def send_emoji_with_text(
        self, event: AstrMessageEvent, emoji_path: str, cleaned_text: str
    ) -> None:
        """发送表情包（异步场景下直接发送新消息）。"""
        try:
            # 检查是否已通过 LLM 工具主动发送过表情包
            if event.get_extra("stealer_active_sent"):
                logger.debug("[Stealer] 已主动发送过表情包，跳过自动发送")
                return

            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Image as ImageComponent

            if not self._check_group_allowed(event):
                return

            image_processor = getattr(self.plugin, "image_processor_service", None)

            if not image_processor:
                return

            # 这里假设 _file_to_gif_base64 仍在 ImageProcessorService 中
            # 如果它被移动了，需要相应调整
            if hasattr(image_processor, "_file_to_gif_base64"):
                b64 = await image_processor._file_to_gif_base64(emoji_path)
                await event.send(MessageChain([ImageComponent.fromBase64(b64)]))
                logger.debug(f"[Stealer] 已发送表情包: {emoji_path}")

        except Exception as e:
            logger.error(f"发送表情包失败: {e}", exc_info=True)

    async def send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ) -> None:
        """发送显式指定的表情包列表和文本。"""
        from astrbot.api.message_components import Plain

        try:
            # 获取当前结果
            result = event.get_result()

            # 创建新的结果对象
            new_result = event.make_result().set_result_content_type(
                result.result_content_type
            )

            # 添加除了Plain文本外的其他组件
            for comp in result.chain:
                if not isinstance(comp, Plain):
                    new_result.chain.append(comp)

            # 添加清理后的文本
            if cleaned_text.strip():
                new_result.message(cleaned_text.strip())

            # 依次添加图片
            image_processor = getattr(self.plugin, "image_processor_service", None)
            if not image_processor:
                return

            for path in emoji_paths:
                try:
                    if os.path.exists(path):
                        if hasattr(image_processor, "_file_to_gif_base64"):
                            b64 = await image_processor._file_to_gif_base64(path)
                            new_result.base64_image(b64)
                    else:
                        logger.warning(f"显式表情包文件不存在: {path}")
                except Exception as e:
                    logger.error(f"加载显式表情包失败: {path}, {e}")

            # 设置新的结果对象
            event.set_result(new_result)
        except Exception as e:
            logger.error(f"发送显式表情包失败: {e}", exc_info=True)

    async def try_send_emoji(
        self, event: AstrMessageEvent, emotions: list[str], cleaned_text: str
    ) -> bool:
        """尝试发送表情包。"""
        if not self._check_group_allowed(event):
            return False

        # 如果本轮已经通过 LLM 工具主动发送过表情包，则跳过自动发送

        if event.get_extra("stealer_active_sent"):
            logger.debug("[Stealer] 检测到 stealer_active_sent=True，跳过自动表情发送")
            return False

        # 1. 检查发送概率
        if not self.check_send_probability():
            return False

        # 2. 智能选择表情包（传入上下文）
        emoji_path = await self.select_emoji(emotions[0], cleaned_text)
        if not emoji_path:
            return False

        # 3. 发送表情包
        await self.send_emoji_with_text(event, emoji_path, cleaned_text)

        logger.debug("已发送表情包")
        return True
