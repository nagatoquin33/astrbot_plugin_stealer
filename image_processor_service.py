import asyncio
import base64
import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class ImageProcessorService:
    """图片处理服务类，负责处理所有与图片相关的操作。"""

    def __init__(self, plugin_instance):
        """初始化图片处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        # 确保base_dir始终是字符串
        if hasattr(plugin_instance, 'base_dir') and plugin_instance.base_dir is not None:
            self.base_dir = str(plugin_instance.base_dir)
        else:
            # 如果没有base_dir，尝试从context获取数据目录
            self.base_dir = None
        self.emoji_mapping = {}
        
        # 尝试从插件实例获取提示词配置，如果不存在则使用默认值
        self.image_classification_prompt = getattr(plugin_instance, 'IMAGE_CLASSIFICATION_PROMPT', 
            """Please classify the image's emotion into a single English label from this exact list: happy, neutral, sad, angry, shy, surprised, smirk, cry, confused, embarrassed, sigh, speechless. Only return the single emotion word, no other text or JSON.""")
        
        self.content_filtration_prompt = getattr(plugin_instance, 'CONTENT_FILTRATION_PROMPT', 
            "请判断这张图片是否包含违反规定的内容，仅回复'是'或'否'。如果包含裸露、暴力、敏感或违法内容，回复'是'，否则回复'否'")
        # 配置参数
        self.categories = []
        self.content_filtration = False
        self.filtration_prompt = ""
        self.vision_provider_id = ""
        self.emoji_only = False

    def update_config(self, categories=None, content_filtration=None, filtration_prompt=None, vision_provider_id=None, emoji_only=None):
        """更新图片处理器配置。
        
        Args:
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            filtration_prompt: 内容过滤提示
            vision_provider_id: 视觉模型提供者ID
            emoji_only: 是否只处理表情
        """
        if categories is not None:
            self.categories = categories
        if content_filtration is not None:
            self.content_filtration = content_filtration
        if filtration_prompt is not None:
            self.filtration_prompt = filtration_prompt
        if vision_provider_id is not None:
            self.vision_provider_id = vision_provider_id
        if emoji_only is not None:
            self.emoji_only = emoji_only

    async def load_emoji_mapping(self):
        """加载表情关键字映射表。"""
        try:
            # 尝试获取配置目录
            config_dir = getattr(self.plugin, "config_dir", None)
            if config_dir is None and hasattr(self.plugin, "base_dir"):
                # 如果没有config_dir，使用base_dir作为替代
                config_dir = str(self.plugin.base_dir) if self.plugin.base_dir is not None else None
                
            if config_dir:
                map_path = os.path.join(config_dir, "emoji_mapping.json")
                if os.path.exists(map_path):
                    import json
                    with open(map_path, encoding="utf-8") as f:
                        self.emoji_mapping = json.load(f)
            else:
                # 默认的表情关键字映射表
                self.emoji_mapping = {
                    "开心": ["开心", "高兴", "快乐", "愉悦", "欢喜", "兴奋", "愉快", "愉悦", "欢快", "喜悦", "欢乐", "喜笑颜开", "眉开眼笑", "笑逐颜开", "哈哈大笑", "大笑", "傻笑", "痴笑", "笑哈哈", "笑", "乐呵呵"],
                    "难过": ["难过", "悲伤", "伤心", "哀伤", "悲痛", "沮丧", "忧郁", "忧伤", "哀愁", "悲凉", "悲切", "心如刀割", "肝肠寸断", "伤心欲绝", "悲痛欲绝", "哀痛", "哀戚", "愁眉苦脸", "闷闷不乐"],
                    "愤怒": ["愤怒", "生气", "恼火", "发火", "恼怒", "恼火", "愤恨", "愤慨", "愤然", "勃然大怒", "怒不可遏", "怒火中烧", "火冒三丈", "暴跳如雷", "气急败坏", "怒气冲天", "气冲冲", "气呼呼"],
                    "惊讶": ["惊讶", "吃惊", "震惊", "惊诧", "讶异", "意外", "意想不到", "大吃一惊", "目瞪口呆", "瞠目结舌", "震惊", "惊悉", "惊呆了", "惊了", "惊到", "惊", "讶"],
                    "恶心": ["恶心", "厌恶", "反感", "厌烦", "腻烦", "憎恶", "嫌恶", "讨厌", "反感", "恶感", "作呕", "反胃", "倒胃口", "讨厌", "嫌"]
                }
                # 保存默认映射表
                with open(map_path, "w", encoding="utf-8") as f:
                    json.dump(self.emoji_mapping, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"加载表情映射表失败: {e}")
            # 使用默认映射表
            self.emoji_mapping = {
                "开心": ["开心", "高兴", "快乐", "愉悦", "欢喜", "兴奋", "愉快", "愉悦", "欢快", "喜悦", "欢乐", "喜笑颜开", "眉开眼笑", "笑逐颜开", "哈哈大笑", "大笑", "傻笑", "痴笑", "笑哈哈", "笑", "乐呵呵"],
                "难过": ["难过", "悲伤", "伤心", "哀伤", "悲痛", "沮丧", "忧郁", "忧伤", "哀愁", "悲凉", "悲切", "心如刀割", "肝肠寸断", "伤心欲绝", "悲痛欲绝", "哀痛", "哀戚", "愁眉苦脸", "闷闷不乐"],
                "愤怒": ["愤怒", "生气", "恼火", "发火", "恼怒", "恼火", "愤恨", "愤慨", "愤然", "勃然大怒", "怒不可遏", "怒火中烧", "火冒三丈", "暴跳如雷", "气急败坏", "怒气冲天", "气冲冲", "气呼呼"],
                "惊讶": ["惊讶", "吃惊", "震惊", "惊诧", "讶异", "意外", "意想不到", "大吃一惊", "目瞪口呆", "瞠目结舌", "震惊", "惊悉", "惊呆了", "惊了", "惊到", "惊", "讶"],
                "恶心": ["恶心", "厌恶", "反感", "厌烦", "腻烦", "憎恶", "嫌恶", "讨厌", "反感", "恶感", "作呕", "反胃", "倒胃口", "讨厌", "嫌"]
            }

    async def process_image(
        self,
        event: Optional[AstrMessageEvent],
        file_path: str,
        is_temp: bool = False,
        idx: Optional[Dict[str, Any]] = None,
        categories: Optional[list[str]] = None,
        emoji_only: Optional[bool] = None,
        content_filtration: Optional[bool] = None,
        filtration_prompt: Optional[str] = None,
        backend_tag: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """统一处理图片：存储、分类、过滤。

        Args:
            event: 消息事件
            file_path: 图片路径
            is_temp: 是否为临时文件
            idx: 索引字典
            categories: 分类列表
            emoji_only: 是否只处理表情
            content_filtration: 是否进行内容过滤
            filtration_prompt: 内容过滤提示
            backend_tag: 后端标签

        Returns:
            tuple: (是否成功, 图片索引)
        """
        # 使用传入的索引或创建空索引
        if idx is None:
            idx = {}
            
        base_path = Path(file_path)
        if not base_path.exists():
            logger.error(f"图片文件不存在: {file_path}")
            return False, None

        # 计算图片哈希作为唯一标识符
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        hash_val = hasher.hexdigest()

        # 检查图片是否已存在
        for k, v in idx.items():
            if isinstance(v, dict) and v.get("hash") == hash_val:
                logger.debug(f"图片已存在: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await self._safe_remove_file(file_path)
                return False, None

        # 存储图片到raw目录
        if self.base_dir:
            raw_dir = os.path.join(self.base_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
            filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
            raw_path = os.path.join(raw_dir, filename)
            if is_temp:
                shutil.move(file_path, raw_path)
            else:
                shutil.copy2(file_path, raw_path)
        else:
            raw_path = file_path

        # 过滤图片
        try:
            # 使用传入的过滤参数
            filter_result = await self._filter_image(
                event, 
                raw_path, 
                filtration_prompt=filtration_prompt, 
                content_filtration=content_filtration
            )
            
            if filter_result:
                # 图片分类
                category = await self._classify_image(event, raw_path)
                # 处理分类失败的情况，使用默认分类
                if category is None:
                    category = "unknown"
                    logger.warning(f"图片分类失败，使用默认分类: {category}")
                else:
                    logger.debug(f"图片分类结果: {category}")

                # 复制图片到对应分类目录
                if self.base_dir:
                    cat_dir = os.path.join(self.base_dir, "categories", category)
                    os.makedirs(cat_dir, exist_ok=True)
                    cat_path = os.path.join(cat_dir, os.path.basename(raw_path))
                    shutil.copy2(raw_path, cat_path)

                # 更新索引
                idx[raw_path] = {
                    "hash": hash_val,
                    "category": category,
                    "created_at": int(time.time()),
                    "usage_count": 0,
                    "last_used": 0
                }

                return True, idx
            else:
                logger.debug("图片未通过内容过滤，已删除")
                await self._safe_remove_file(raw_path)
                return False, None
        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            await self._safe_remove_file(raw_path)
            return False, None

    async def classify_image(self, event: AstrMessageEvent, file_path: str, emoji_only=None, categories=None) -> tuple[str, list[str], str, str]:
        """使用视觉模型对图片进行分类。

        Args:
            event: 消息事件
            file_path: 图片路径
            emoji_only: 是否只处理表情
            categories: 分类列表

        Returns:
            tuple: (category, tags, desc, emotion)
        """
        try:
            # 调用原有的分类方法
            emotion = await self._classify_image(event, file_path)
            
            # 返回默认值以匹配main.py的预期
            category = emotion if emotion else "无语"
            tags = []
            desc = ""
            
            return category, tags, desc, emotion
        except Exception as e:
            logger.error(f"图片分类失败: {e}")
            fallback = "无语" if categories and "无语" in categories else "其它"
            return fallback, [], "", fallback
            
    async def _classify_image(self, event: AstrMessageEvent, img_path: str) -> str:
        """使用视觉模型对图片进行分类（内部方法）。

        Args:
            event: 消息事件
            img_path: 图片路径

        Returns:
            str: 分类结果
        """
        try:
            if not os.path.exists(img_path):
                logger.error(f"图片文件不存在: {img_path}")
                return "无语"

            # 选择视觉模型
            model = self.plugin.vision_model if hasattr(self.plugin, "vision_model") else "gpt-4o-mini"
            logger.debug(f"使用视觉模型 {model} 对图片进行分类")

            # 构建提示词
            prompt = self.image_classification_prompt

            # 调用LLM生成分类结果
            # 使用getattr获取配置，避免直接访问不存在的config属性
            max_retries = int(getattr(self.plugin, "vision_max_retries", 3))
            retry_delay = float(getattr(self.plugin, "vision_retry_delay", 1.0))

            for attempt in range(max_retries):
                try:
                    # 获取当前使用的聊天提供商ID
                    chat_provider_id = await self.plugin.context.provider_manager.get_using_provider_id()
                    # 使用正确的关键字参数调用llm_generate
                    result = await self.plugin.context.llm_generate(
                        chat_provider_id=chat_provider_id,
                        prompt=prompt,
                        image_urls=[img_path] if img_path else None,
                        model=model
                    )
                    if result:
                        # 获取实际的文本结果
                        llm_response_text = ""
                        if hasattr(result, 'result_chain') and result.result_chain:
                            llm_response_text = result.result_chain.get_plain_text()
                        elif hasattr(result, 'completion_text') and result.completion_text:
                            llm_response_text = result.completion_text
                        else:
                            llm_response_text = str(result)
                            
                        # 直接处理LLM响应，提示词已要求只返回特定格式结果
                        category = llm_response_text.strip().lower()
                        
                        # 检查分类结果是否在有效类别列表中
                        valid_categories = ["happy", "neutral", "sad", "angry", "shy", "surprised", "smirk", "cry", "confused", "embarrassed", "sigh", "speechless"]
                        if category and category in valid_categories:
                            logger.info(f"图片分类结果: {category}")
                            return category
                        else:
                            logger.warning(f"分类结果不在有效类别列表中: {category}")
                        logger.debug(f"无效的分类结果: {llm_response_text}")
                except Exception as e:
                    error_msg = str(e)
                    # 检查是否为限流错误
                    is_rate_limit = "429" in error_msg or "RateLimit" in error_msg or "exceeded your current request limit" in error_msg
                    if is_rate_limit:
                        logger.warning(f"图片分类请求被限流，正在重试 ({attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"图片分类失败 (尝试 {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                        else:
                            break
            logger.error("图片分类失败，达到最大重试次数")
            return None
        except Exception as e:
            logger.error(f"图片分类失败: {e}")
            return None

    async def _filter_image(self, event: AstrMessageEvent, img_path: str, filtration_prompt=None, content_filtration=None) -> bool:
        """使用LLM过滤图片内容。

        Args:
            event: 消息事件
            img_path: 图片路径
            filtration_prompt: 内容过滤提示（可选）
            content_filtration: 是否进行内容过滤（可选）

        Returns:
            bool: 是否通过过滤
        """
        # 使用传入的过滤提示或默认提示
        current_filtration_prompt = filtration_prompt if filtration_prompt else self.content_filtration_prompt
        
        # 确定是否进行内容过滤
        should_filter = content_filtration if content_filtration is not None else getattr(self.plugin, "content_filtration", True)
        
        if not should_filter:
            return True

        if not os.path.exists(img_path):
            logger.error(f"图片文件不存在: {img_path}")
            return True

        # 选择视觉模型
        model = self.plugin.vision_model if hasattr(self.plugin, "vision_model") else "gpt-4o-mini"
        logger.debug(f"使用视觉模型 {model} 对图片进行过滤")

        # 调用LLM进行内容过滤
        # 使用插件实例的属性或默认值
        max_retries = getattr(self.plugin, "vision_max_retries", 3)
        retry_delay = getattr(self.plugin, "vision_retry_delay", 1.0)

        for attempt in range(max_retries):
            try:
                # 获取当前使用的聊天提供商ID
                chat_provider_id = await self.plugin.context.provider_manager.get_using_provider_id()
                # 使用正确的关键字参数调用llm_generate
                result = await self.plugin.context.llm_generate(
                    chat_provider_id=chat_provider_id,
                    prompt=current_filtration_prompt,
                    image_urls=[img_path] if img_path else None,
                    model=model
                )
                
                # 获取实际的文本结果
                llm_response_text = ""
                if hasattr(result, 'result_chain') and result.result_chain:
                    llm_response_text = result.result_chain.get_plain_text()
                elif hasattr(result, 'completion_text') and result.completion_text:
                    llm_response_text = result.completion_text
                else:
                    llm_response_text = str(result)
                
                # 直接处理LLM响应，提示词已要求只返回特定格式结果
                if llm_response_text.strip() == "是":
                    logger.debug("图片未通过内容过滤")
                    return False
                return True
            except Exception as e:
                error_msg = str(e)
                # 检查是否为限流错误
                is_rate_limit = "429" in error_msg or "RateLimit" in error_msg or "exceeded your current request limit" in error_msg
                if is_rate_limit:
                    logger.warning(f"图片过滤请求被限流，正在重试 ({attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    logger.error(f"图片过滤失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        break
        logger.error("图片过滤失败，达到最大重试次数，默认允许通过")
        return True

    async def _extract_emotions_from_text(self, event: AstrMessageEvent, text: str) -> tuple[list, str]:
        """从文本中提取情绪关键词。

        Args:
            event: 消息事件
            text: 文本内容

        Returns:
            tuple: (情绪关键词列表, 清理后的文本)
        """
        try:
            import re

            # 清理文本
            cleaned_text = re.sub(r"\[图片.*?\]", "", text)
            cleaned_text = re.sub(r"\[表情.*?\]", "", cleaned_text)
            cleaned_text = cleaned_text.strip()
            if not cleaned_text:
                return [], cleaned_text

            # 使用情绪映射表提取关键词
            emotions = []
            for emotion, keywords in self.emoji_mapping.items():
                for keyword in keywords:
                    if keyword in cleaned_text:
                        emotions.append(emotion)
                        break

            return list(set(emotions)), cleaned_text
        except Exception as e:
            logger.error(f"提取情绪关键词失败: {e}")
            return [], text

    async def _file_to_base64(self, file_path: str) -> str:
        """将文件转换为base64编码。

        Args:
            file_path: 文件路径

        Returns:
            str: base64编码
        """
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"文件转换为base64失败: {e}")
            return ""

    async def _safe_remove_file(self, file_path: str) -> bool:
        """安全删除文件。

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否删除成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"已删除文件: {file_path}")
                return True
            logger.debug(f"文件不存在，无需删除: {file_path}")
            return True
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
            
    async def _store_image(self, src_path: str, category: str) -> str:
        """将图片存储到指定分类目录。

        Args:
            src_path: 源图片路径
            category: 分类名称

        Returns:
            str: 存储后的图片路径
        """
        try:
            if not self.base_dir:
                logger.error("base_dir未设置，无法存储图片")
                return src_path
                
            # 确保分类目录存在
            cat_dir = os.path.join(self.base_dir, "categories", category)
            os.makedirs(cat_dir, exist_ok=True)
            
            # 复制图片到分类目录
            filename = os.path.basename(src_path)
            dest_path = os.path.join(cat_dir, filename)
            
            # 如果文件已存在，生成新文件名
            if os.path.exists(dest_path):
                base_name, ext = os.path.splitext(filename)
                dest_path = os.path.join(cat_dir, f"{base_name}_{int(time.time())}{ext}")
                
            shutil.copy2(src_path, dest_path)
            logger.debug(f"图片已存储到分类目录: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"存储图片失败: {e}")
            return src_path

    async def _load_index(self) -> dict:
        """加载图片索引。

        Returns:
            dict: 图片索引
        """
        try:
            # 尝试获取配置目录
            config_dir = getattr(self.plugin, "config_dir", None)
            if config_dir is None and hasattr(self.plugin, "base_dir"):
                # 如果没有config_dir，使用base_dir作为替代
                config_dir = str(self.plugin.base_dir) if self.plugin.base_dir else None
            
            if not config_dir:
                return {}
                
            index_path = os.path.join(config_dir, "index.json")
            if os.path.exists(index_path):
                import json
                with open(index_path, encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return {}

    async def _save_index(self, idx: dict) -> bool:
        """保存图片索引。

        Args:
            idx: 图片索引

        Returns:
            bool: 是否保存成功
        """
        try:
            # 尝试获取配置目录
            config_dir = getattr(self.plugin, "config_dir", None)
            if config_dir is None and hasattr(self.plugin, "base_dir"):
                # 如果没有config_dir，使用base_dir作为替代
                config_dir = str(self.plugin.base_dir) if self.plugin.base_dir else None
            
            if not config_dir:
                return False
                
            index_path = os.path.join(config_dir, "index.json")
            # 确保目录存在
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            import json
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(idx, f, ensure_ascii=False, indent=4)
            logger.debug("索引文件已保存")
            return True
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False

    async def _compute_hash(self, file_path: str) -> str:
        """计算文件哈希值。

        Args:
            file_path: 文件路径

        Returns:
            str: 哈希值
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败: {e}")
            return ""
