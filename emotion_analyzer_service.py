import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

# 导入文本相似度计算模块
try:
    from .text_similarity import calculate_hybrid_similarity
except ImportError:
    # 如果导入失败，提供一个简单的降级实现
    def calculate_hybrid_similarity(text1: str, text2: str) -> float:
        """简单的文本相似度计算（降级实现）"""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        # 简单的包含关系检查
        if text2_lower in text1_lower or text1_lower in text2_lower:
            return 0.5
        # 简单的词重叠检查
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        if words1 and words2:
            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap
        return 0.0


class EmotionAnalyzerService:
    """情感分析服务类，处理文本情绪分析和映射。"""

    # 正则表达式模式常量
    # 1. 标准模式：&&happy&& 或 && happy && (允许内含空格)
    # 2. 容错模式：处理可能被Markdown转义的情况，如 \&\&happy\&\&
    HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:&&|\\&\\&)")

    # 3. 残缺模式：处理模型输出 &&happy| 或 &&happy\n 这种忘记闭合的情况
    # 仅匹配后跟 |、换行符或字符串结束的情况，避免误伤正常文本
    INCOMPLETE_HEX_PATTERN = re.compile(
        r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:[|]|\n|$)"
    )

    # 单个&的匹配太容易误伤（如 URL 参数），仅作为最后的兜底，且要求情绪词必须在列表内
    SINGLE_HEX_PATTERN = re.compile(r"&([^&\s]+?)&")

    def __init__(self, plugin_instance):
        self.plugin_instance = plugin_instance
        self.categories = (
            plugin_instance.categories if hasattr(plugin_instance, "categories") else []
        )

    def normalize_category(self, category: str) -> str:
        """将任意文本归一化到预定义的情绪分类中。"""
        if not category:
            return ""

        category = category.strip().lower()

        # 检查是否直接匹配categories中的类别（LLM输出的标签已经通过提示词限制，直接匹配即可）
        for cat in self.categories:
            if cat.lower() == category:
                return cat

        # 不进行兜底分类，返回空字符串表示无法分类
        return ""

    async def extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            res: list[str] = []
            seen: set[str] = set()
            cleaned_text = str(text)
            valid_categories = set(self.categories)  # 使用self.categories确保一致性

            # 1. 处理显式包裹标记：&&情绪&&
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

            # 2. 处理残缺标记：&&情绪| 或 &&情绪\n
            # 必须严格校验是否在 valid_categories 中，避免误判
            temp_text = cleaned_text
            for match in self.INCOMPLETE_HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    # 只有当它是合法的情绪词时，才认为是标签并移除
                    logger.debug(f"检测到残缺情绪标签: {emotion} -> {norm_cat}")
                    seen.add(norm_cat)
                    res.append(norm_cat)
                    temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            # 3. 处理单个&包裹的标记：&情绪&
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

    def update_config(self, categories=None):
        """更新配置参数"""
        if categories is not None:
            self.categories = categories

    def is_in_parentheses(self, text: str, index: int) -> bool:
        """判断字符串中指定索引位置是否在括号内。

        支持圆括号()和方括号[]。

        Args:
            text: 文本字符串
            index: 索引位置

        Returns:
            bool: 是否在括号内
        """
        parentheses_count = 0
        bracket_count = 0

        for i in range(index):
            if text[i] == "(":
                parentheses_count += 1
            elif text[i] == ")":
                parentheses_count -= 1
            elif text[i] == "[":
                bracket_count += 1
            elif text[i] == "]":
                bracket_count -= 1

        return parentheses_count > 0 or bracket_count > 0

    async def select_emoji(self, category: str, context_text: str = "") -> str | None:
        """选择表情包（智能或随机）。

        Args:
            category: 情绪分类
            context_text: 上下文文本

        Returns:
            str | None: 表情包路径
        """
        import random
        from pathlib import Path

        # 检查是否启用智能选择
        use_smart = getattr(self.plugin_instance, "smart_emoji_selection", True)

        # 如果启用智能选择且提供了上下文，使用智能选择
        if use_smart and context_text and len(context_text.strip()) > 5:
            smart_path = await self.select_emoji_smart(category, context_text)
            if smart_path:
                return smart_path

        # 降级到随机选择（原有逻辑）
        base_dir = getattr(self.plugin_instance, "base_dir", None)
        if not base_dir:
            return None

        cat_dir = Path(base_dir) / "categories" / category
        if not cat_dir.exists():
            logger.debug(f"情绪'{category}'对应的图片目录不存在")
            return None

        try:
            files = [p for p in cat_dir.iterdir() if p.is_file()]
            if not files:
                logger.debug(f"情绪'{category}'对应的图片目录为空")
                return None

            logger.debug(
                f"从'{category}'目录中找到 {len(files)} 张图片（{'智能选择失败，' if use_smart else ''}随机选择）"
            )

            # 改进的随机选择：避免重复
            recent_usage_key = f"recent_usage_{category}"
            recent_usage = getattr(self.plugin_instance, recent_usage_key, [])

            # 过滤最近使用的文件
            available_files = [f for f in files if f.as_posix() not in recent_usage]

            # 如果过滤后没有可用文件，清空历史
            if not available_files:
                available_files = files
                recent_usage.clear()

            # 随机选择
            picked_image = random.choice(available_files)

            # 更新最近使用历史
            picked_path = picked_image.as_posix()
            if picked_path in recent_usage:
                recent_usage.remove(picked_path)
            recent_usage.append(picked_path)

            # 保持历史队列大小
            max_recent = min(5, max(2, len(files) // 2))
            if len(recent_usage) > max_recent:
                recent_usage.pop(0)

            setattr(self.plugin_instance, recent_usage_key, recent_usage)

            return picked_path
        except Exception as e:
            logger.error(f"选择表情包失败: {e}")
            return None

    async def select_emoji_smart(self, category: str, context_text: str) -> str | None:
        """智能选择表情包（多样性+匹配度+文本相似度）。

        Args:
            category: 情绪分类
            context_text: 上下文文本

        Returns:
            str | None: 表情包路径
        """
        import math
        import os
        import random
        import time

        try:
            # 1. 加载索引，获取该分类下的所有表情包
            cache_service = getattr(self.plugin_instance, "cache_service", None)
            if not cache_service:
                return None

            idx = await cache_service.load_index()
            candidates = []
            current_time = time.time()

            # 获取最近使用历史（避免重复）
            recent_usage_key = f"recent_usage_{category}"
            recent_usage = getattr(self.plugin_instance, recent_usage_key, [])

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                # 匹配分类
                file_category = data.get("category", data.get("emotion", ""))
                if file_category != category:
                    continue

                # 检查文件是否存在
                if not os.path.exists(file_path):
                    continue

                candidates.append(
                    {
                        "path": file_path,
                        "data": data,
                        "last_used": data.get("last_used", 0),
                        "use_count": data.get("use_count", 0),
                        "desc": str(data.get("desc", "")).lower(),
                        "tags": [str(t).lower() for t in data.get("tags", [])],
                        "scenes": [str(s).lower() for s in data.get("scenes", [])],
                    }
                )

            if not candidates:
                logger.debug(f"分类 '{category}' 下没有可用的表情包")
                return None

            # 2. 强制避重：过滤最近使用的表情包
            available_candidates = [
                c for c in candidates if c["path"] not in recent_usage
            ]

            # 如果过滤后候选太少，只避免最近3个
            if len(available_candidates) < max(2, len(candidates) * 0.3):
                recent_3 = recent_usage[-3:] if len(recent_usage) >= 3 else recent_usage
                available_candidates = [
                    c for c in candidates if c["path"] not in recent_3
                ]

            # 如果还是没有候选，清空历史重新开始
            if not available_candidates:
                available_candidates = candidates
                recent_usage.clear()

            # 3. 计算多样性分数（权重70%）
            for candidate in available_candidates:
                diversity_score = 100  # 基础分数

                # 时间多样性（使用时间越久分数越高）
                time_since_last_use = current_time - candidate["last_used"]
                if time_since_last_use < 300:  # 5分钟内大幅减分
                    diversity_score -= 60
                elif time_since_last_use < 1800:  # 30分钟内
                    diversity_score -= 30
                elif time_since_last_use < 3600:  # 1小时内
                    diversity_score -= 10
                else:
                    # 超过1小时给予奖励
                    hours_passed = time_since_last_use / 3600
                    diversity_score += min(hours_passed * 5, 30)

                # 频率多样性（使用次数越少分数越高）
                use_count = candidate["use_count"]
                if use_count == 0:
                    diversity_score += 20  # 从未使用过的优先
                elif use_count < 3:
                    diversity_score += 10
                elif use_count < 10:
                    diversity_score += 0
                else:
                    diversity_score -= min(use_count * 2, 30)

                candidate["diversity_score"] = max(diversity_score, 10)

            # 4. 计算匹配分数（权重30%）
            has_context = context_text and len(context_text.strip()) > 5

            for candidate in available_candidates:
                match_score = 10  # 基础匹配分数

                if has_context:
                    context_lower = context_text.lower()

                    # 场景匹配
                    for scene in candidate["scenes"]:
                        if len(scene) > 2 and scene in context_lower:
                            match_score += 25

                    # 描述匹配
                    desc = candidate["desc"]
                    if desc and desc in context_lower:
                        match_score += 20
                    elif desc:
                        desc_words = [w for w in desc.split() if len(w) > 1]
                        matched_words = sum(
                            1 for word in desc_words if word in context_lower
                        )
                        match_score += matched_words * 5

                    # 标签匹配
                    for tag in candidate["tags"]:
                        if len(tag) > 1 and tag in context_lower:
                            match_score += 8

                    # 文本相似度加成
                    if desc and len(desc) > 3:
                        similarity = calculate_hybrid_similarity(context_text, desc)
                        similarity_bonus = similarity * 15
                        match_score += similarity_bonus

                        if similarity > 0.3:
                            logger.debug(
                                f"[相似度匹配] '{context_text[:20]}...' vs '{desc[:20]}...' = {similarity:.2f}"
                            )

                candidate["match_score"] = match_score

            # 5. 计算综合分数：多样性70% + 匹配度30%
            max_diversity = max(c["diversity_score"] for c in available_candidates)
            max_match = max(c["match_score"] for c in available_candidates)

            for candidate in available_candidates:
                # 标准化到0-100
                norm_diversity = (candidate["diversity_score"] / max_diversity) * 100
                norm_match = (candidate["match_score"] / max_match) * 100

                # 多样性权重更高
                candidate["final_score"] = norm_diversity * 0.7 + norm_match * 0.3

            # 6. 选择策略：从前40%候选中加权随机选择
            available_candidates.sort(key=lambda x: x["final_score"], reverse=True)
            top_40_percent = max(1, int(len(available_candidates) * 0.4))
            top_candidates = available_candidates[:top_40_percent]

            # 使用指数衰减权重，保持随机性
            weights = [math.exp(-i * 0.3) for i in range(len(top_candidates))]
            selected = random.choices(top_candidates, weights=weights, k=1)[0]

            # 7. 更新使用历史和统计
            selected_path = selected["path"]

            # 更新索引中的使用统计
            idx[selected_path]["last_used"] = int(current_time)
            idx[selected_path]["use_count"] = idx[selected_path].get("use_count", 0) + 1
            await cache_service.save_index(idx)

            # 更新最近使用历史
            if selected_path in recent_usage:
                recent_usage.remove(selected_path)
            recent_usage.append(selected_path)

            # 保持历史队列大小（最多记录8个）
            max_recent = min(8, max(3, len(candidates) // 2))
            if len(recent_usage) > max_recent:
                recent_usage.pop(0)

            setattr(self.plugin_instance, recent_usage_key, recent_usage)

            logger.info(
                f"选择表情包: 多样性={selected['diversity_score']:.1f}, "
                f"匹配度={selected['match_score']:.1f}, "
                f"综合={selected['final_score']:.1f}"
            )

            return selected_path

        except Exception as e:
            logger.error(f"智能选择表情包失败: {e}", exc_info=True)
            return None

    def check_send_probability(self) -> bool:
        """检查表情包发送概率。

        Returns:
            bool: 是否应该发送
        """
        import random

        try:
            emoji_chance = getattr(self.plugin_instance, "emoji_chance", 0.4)
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
    ):
        """发送表情包（异步场景下直接发送新消息）。

        Args:
            event: 消息事件
            emoji_path: 表情包路径
            cleaned_text: 清理后的文本
        """
        try:
            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Image as ImageComponent

            config_service = getattr(self.plugin_instance, "config_service", None)
            if config_service and not config_service.is_group_allowed(
                config_service.get_group_id(event)
            ):
                return

            image_processor = getattr(self.plugin_instance, "image_processor_service", None)
            if not image_processor:
                return

            b64 = await image_processor._file_to_gif_base64(emoji_path)
            await event.send(MessageChain([ImageComponent.fromBase64(b64)]))
            logger.debug(f"[Stealer] 已发送表情包: {emoji_path}")

        except Exception as e:
            logger.error(f"发送表情包失败: {e}", exc_info=True)

    async def send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ):
        """发送显式指定的表情包列表和文本。

        Args:
            event: 消息事件
            emoji_paths: 表情包路径列表
            cleaned_text: 清理后的文本
        """
        import os

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
            image_processor = getattr(self.plugin_instance, "image_processor_service", None)
            if not image_processor:
                return

            for path in emoji_paths:
                try:
                    if os.path.exists(path):
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
        """尝试发送表情包。

        Args:
            event: 消息事件
            emotions: 情绪列表
            cleaned_text: 清理后的文本

        Returns:
            bool: 是否成功发送
        """
        config_service = getattr(self.plugin_instance, "config_service", None)
        if config_service and not config_service.is_group_allowed(
            config_service.get_group_id(event)
        ):
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

    def cleanup(self):
        """清理资源。"""
        pass
