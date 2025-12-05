import asyncio
import base64
import copy
import hashlib
import json
import os
import random
import re
import shutil
from functools import lru_cache, wraps
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

try:
    # 可选依赖，用于通过图片尺寸/比例进行快速过滤，未安装时自动降级
    from PIL import Image as PILImage  # type: ignore[import]
except Exception:  # pragma: no cover - 仅作为兼容分支
    PILImage = None


@register("astrbot_plugin_stealer", "nagatoquin33", "自动偷取并分类表情包，在合适时机发送", "1.0.0")
class StealerPlugin(Star):
    """表情包偷取与发送插件。

    功能：
    - 监听消息中的图片并自动保存到插件数据目录
    - 使用当前会话的多模态模型进行情绪分类与标签生成
    - 建立分类索引，支持自动与手动在合适时机发送表情包
    """
    
    # 常量定义
    BACKEND_TAG = "emoji_stealer"
    DEFAULT_FILTRATION_PROMPT = "符合公序良俗"
    
    # 提示词常量
    IMAGE_FILTER_PROMPT = "根据以下审核准则判断图片是否符合: {filtration_rule}。只返回是或否。"
    TEXT_EMOTION_PROMPT_TEMPLATE = """请基于这段文本的情绪选择一个最匹配的类别: {categories}。
请使用&&emotion&&格式返回，例如&&happy&&、&&sad&&。
只返回表情标签，不要添加任何其他内容。文本: {text}"""
    
    # 从外部文件加载的提示词
    EMOTION_DETECTION_PROMPT = ""
    
    # 缓存清理阈值
    _CACHE_MAX_SIZE = 1000  # 每个缓存的最大条目数
    
    async def _save_cache(self, cache_data: dict, cache_path: Path | None, max_size: int = None):
        """保存缓存数据到文件。"""
        if not cache_path or not cache_data:
            return
        
        try:
            def sync_save():
                nonlocal cache_data
                # 确保缓存大小在限制内
                current_max = max_size if max_size is not None else self._CACHE_MAX_SIZE
                if len(cache_data) > current_max:
                    keys_to_keep = list(cache_data.keys())[-current_max:]
                    cache_data = {k: cache_data[k] for k in keys_to_keep}
                cache_path.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
            
            await asyncio.to_thread(sync_save)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    # 情绪分类列表（英文标签）
    CATEGORIES = [
        "happy",
        "neutral",
        "sad",
        "angry",
        "shy",
        "surprised",
        "smirk",
        "cry",
        "confused",
        "embarrassed",
    ]
    
    # 预先声明类属性，避免实例化时出现AttributeError
    _EMOTION_MAPPING = {}
    
    # 情绪类别映射 - 实例属性，在 initialize 方法中从文件加载
    
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.enabled = True
        self.config = config  # 保存配置参数
        self.auto_send = True
        self.base_dir: Path | None = None
        # 情绪映射，在 initialize 方法中从文件加载
        self._EMOTION_MAPPING = {}
        # 默认情绪分类（英文标签，避免字符兼容性问题）
        # 语义对应关系：happy(开心)、neutral(无语/平静)、sad(伤心)、angry(愤怒)、
        # shy(害羞)、surprised(惊讶)、smirk(坏笑)、cry(哭)、confused(疑惑)、
        # embarrassed(尴尬)
        self.categories = self.CATEGORIES
        self.index_path: Path | None = None
        self.vision_provider_id: str | None = None
        self.text_provider_id: str | None = None
        self.alias_path: Path | None = None
        self.backend_tag: str = self.BACKEND_TAG
        self.emoji_chance: float = 0.4
        self.max_reg_num: int = 100
        self.do_replace: bool = True
        self.check_interval: int = 10
        self.steal_emoji: bool = True
        self.content_filtration: bool = False
        self.filtration_prompt: str = self.DEFAULT_FILTRATION_PROMPT
        self._scanner_task: asyncio.Task | None = None
        
        # 情绪类别映射（将在initialize方法中加载）
        self._EMOTION_MAPPING = {}
        self.desc_cache_path: Path | None = None
        self.emotion_cache_path: Path | None = None
        self._desc_cache: dict[str, str] = {}
        self._emotion_cache: dict[str, str] = {}
        # 图片分类结果缓存
        self.image_cache_path: Path | None = None
        self._image_cache: dict[str, tuple[str, list[str], str, str]] = {}
        # 文本分类结果缓存
        self.text_cache_path: Path | None = None
        self._text_cache: dict[str, str] = {}
        # LLM 调用频率限制
        self._last_llm_call_time: float = 0.0
        self._llm_call_cooldown: float = 2.0  # 2秒冷却时间
        self.emoji_only: bool = True  # 仅偷取表情包开关
        



    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        try:
            # 直接处理每个配置项，提高IDE类型推断和代码可读性
            if "enabled" in config_dict and isinstance(config_dict["enabled"], bool):
                self.enabled = config_dict["enabled"]
            if "auto_send" in config_dict and isinstance(config_dict["auto_send"], bool):
                self.auto_send = config_dict["auto_send"]
            if "emoji_chance" in config_dict and isinstance(config_dict["emoji_chance"], (int, float)):
                self.emoji_chance = float(config_dict["emoji_chance"])
            if "max_reg_num" in config_dict and isinstance(config_dict["max_reg_num"], int):
                self.max_reg_num = config_dict["max_reg_num"]
            if "do_replace" in config_dict and isinstance(config_dict["do_replace"], bool):
                self.do_replace = config_dict["do_replace"]
            if "check_interval" in config_dict and isinstance(config_dict["check_interval"], int):
                self.check_interval = config_dict["check_interval"]
            if "steal_emoji" in config_dict and isinstance(config_dict["steal_emoji"], bool):
                self.steal_emoji = config_dict["steal_emoji"]
            if "content_filtration" in config_dict and isinstance(config_dict["content_filtration"], bool):
                self.content_filtration = config_dict["content_filtration"]
            if "filtration_prompt" in config_dict and isinstance(config_dict["filtration_prompt"], str):
                self.filtration_prompt = config_dict["filtration_prompt"] if config_dict["filtration_prompt"] else self.DEFAULT_FILTRATION_PROMPT
            if "emoji_only" in config_dict and isinstance(config_dict["emoji_only"], bool):
                self.emoji_only = config_dict["emoji_only"]
            
            # 特殊处理categories（需要映射和去重）
            cats = config_dict.get("categories")
            if isinstance(cats, list) and cats:
                # 兼容旧版本配置，将中文/旧标签映射为英文情绪标签，并移除已废弃分类
                mapped: list[str] = []
                for c in cats:
                    norm = self._normalize_category(str(c))
                    if norm and norm in self.categories and norm not in mapped:
                        mapped.append(norm)
                if mapped:
                    self.categories = mapped
        except Exception as e:
            logger.error(f"更新配置失败: {e}")

    async def initialize(self):
        """初始化插件数据目录与配置。

        创建 raw、categories 目录并加载配置。
        """
        self.base_dir = StarTools.get_data_dir()
        (self.base_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "categories").mkdir(parents=True, exist_ok=True)
        for c in self.categories:
            (self.base_dir / "categories" / c).mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        self.alias_path = self.base_dir / "aliases.json"
        self.desc_cache_path = self.base_dir / "desc_cache.json"
        self.emotion_cache_path = self.base_dir / "emotion_cache.json"
        self.image_cache_path = self.base_dir / "image_cache.json"
        self.text_cache_path = self.base_dir / "text_cache.json"
        
        # 加载情绪映射文件
        mapping_file_path = Path(__file__).parent / "emotion_mapping.json"
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                self._EMOTION_MAPPING = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load emotion mapping: {e}")
            # 加载失败时使用空映射作为降级方案
            self._EMOTION_MAPPING = {}
        
        # 加载提示词文件
        prompts_file_path = Path(__file__).parent / "prompts.json"
        try:
            with open(prompts_file_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
            self.EMOTION_DETECTION_PROMPT = prompts.get("EMOTION_DETECTION_PROMPT", "")
            self.IMAGE_CLASSIFICATION_PROMPT = prompts.get("IMAGE_CLASSIFICATION_PROMPT", "")
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            # 加载失败时使用默认提示词
            self.EMOTION_DETECTION_PROMPT = "Based on the following description, choose ONE emotion word in English from this exact list: happy, neutral, sad, angry, shy, surprised, smirk, cry, confused, embarrassed. You must select the emotion that best matches the overall feeling described. If multiple emotions are mentioned, choose the most prominent one. Only return the single emotion word, with no other text, punctuation, or explanations. Examples: - Description: A cartoon cat with eyes curved into crescents and an upward smile, looking happy Response: happy - Description: An anime girl with red cheeks, looking down shyly Response: shy - Description: A character with tears streaming down their face, looking sad Response: cry Description: "
            self.IMAGE_CLASSIFICATION_PROMPT = "你是专业的表情包图片分析专家，请严格按照以下要求处理这张图片：\n1. 首先仔细观察图片内容，生成简洁准确的详细描述（10-30字）\n2. 基于描述选择最匹配的情绪类别（从以下列表选择：happy, neutral, sad, angry, shy, surprised, smirk, cry, confused, embarrassed）\n3. 提取2-5个能描述图片特征的关键词标签\n4. 必须返回严格的JSON格式，包含description（描述）、category（情绪类别）和tags（关键词数组）三个字段\n5. 不要添加任何JSON之外的内容，确保JSON可以被程序直接解析\n\n示例：\n{\"description\":\"一个卡通猫角色，眼睛弯成月牙，嘴角上扬，露出开心的笑容\",\"category\":\"happy\",\"tags\":[\"cute\",\"smile\",\"cartoon\"]}"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
        if self.alias_path and not self.alias_path.exists():
            self.alias_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
        if self.desc_cache_path and self.desc_cache_path.exists():
            try:
                self._desc_cache = json.loads(self.desc_cache_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"加载描述缓存失败: {e}")
                self._desc_cache = {}
        else:
            if self.desc_cache_path:
                self.desc_cache_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
        if self.text_cache_path and not self.text_cache_path.exists():
            if self.text_cache_path:
                self.text_cache_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
        if self.emotion_cache_path and self.emotion_cache_path.exists():
            try:
                self._emotion_cache = json.loads(self.emotion_cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._emotion_cache = {}
        if self.text_cache_path and self.text_cache_path.exists():
            try:
                self._text_cache = json.loads(self.text_cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._text_cache = {}
        else:
            if self.emotion_cache_path:
                self.emotion_cache_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")
        
        # 加载图片分类缓存
        if self.image_cache_path and self.image_cache_path.exists():
            try:
                cache_data = json.loads(self.image_cache_path.read_text(encoding="utf-8"))
                # 将缓存数据转换为正确的格式
                self._image_cache = {}
                for hash_key, data in cache_data.items():
                    if isinstance(data, list) and len(data) >= 4:
                        category = str(data[0])
                        tags = list(data[1])
                        desc = str(data[2])
                        emotion = str(data[3])
                        self._image_cache[hash_key] = (category, tags, desc, emotion)
            except Exception as e:
                logger.error(f"加载图片分类缓存失败: {e}")
                self._image_cache = {}
        else:
            if self.image_cache_path:
                self.image_cache_path.write_text(json.dumps({}, ensure_ascii=False), encoding="utf-8")

        # 移除了侵入式的人格修改功能，使用非侵入式的表情标签提取方式

        # 从插件配置读取模型选择
        try:
            if self.config:
                self._update_config_from_dict(self.config)
                # 读取模型ID配置（仅在config中可用）
                vpid = self.config.get("vision_provider_id")
                tpid = self.config.get("text_provider_id")
                self.vision_provider_id = str(vpid) if vpid else None
                self.text_provider_id = str(tpid) if tpid else None
        except Exception as e:
            logger.error(f"读取插件配置失败: {e}")

        if self._scanner_task is None:
            self._scanner_task = asyncio.create_task(self._scanner_loop())

    async def terminate(self):
        """插件销毁生命周期钩子。清理任务。"""

        # 取消后台扫描任务
        try:
            if self._scanner_task is not None:
                self._scanner_task.cancel()
        except Exception as e:
            logger.error(f"取消扫描任务失败: {e}")

        return

    def _persist_config(self):
        """持久化插件运行配置到AstrBotConfig。"""
        if not self.config:
            return
            
        payload = {
            "enabled": self.enabled,
            "auto_send": self.auto_send,
            "categories": self.categories,
            "backend_tag": self.backend_tag,
            "emoji_chance": self.emoji_chance,
            "max_reg_num": self.max_reg_num,
            "do_replace": self.do_replace,
            "check_interval": self.check_interval,
            "steal_emoji": self.steal_emoji,
            "content_filtration": self.content_filtration,
            "filtration_prompt": self.filtration_prompt,
            "emoji_only": self.emoji_only,
            "vision_provider_id": self.vision_provider_id,
            "text_provider_id": self.text_provider_id
        }
        
        try:
            # 使用AstrBotConfig的机制保存配置
            for key, value in payload.items():
                self.config[key] = value
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    # 旧的同步保存方法已被统一的_save_cache方法替代
    # def _sync_save_text_cache(self):
    #     """同步保存文本分类缓存到文件。"""
    #     try:
    #         if self.text_cache_path:
    #             with open(self.text_cache_path, "w", encoding="utf-8") as f:
    #                 json.dump(self._text_cache, f, ensure_ascii=False, indent=2)
    #     except Exception as e:
    #         logger.error(f"保存文本分类缓存失败: {e}")

    async def _load_index(self) -> dict:
        """加载分类索引文件。

        Returns:
            dict: 键为文件路径，值为包含 category 与 tags 的字典。
        """
        if not self.index_path or not self.index_path.exists():
            return {}
        
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return {}

    async def _save_index(self, idx: dict):
        """保存分类索引文件。"""
        if not self.index_path:
            return
        
        try:
            await self._save_cache(idx, self.index_path, max_size=None)  # 索引不受缓存大小限制
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}")

    async def _load_aliases(self) -> dict:
        if not self.alias_path:
            return {}
            
        try:
            def sync_load_aliases():
                return json.loads(self.alias_path.read_text(encoding="utf-8"))
            
            return await asyncio.to_thread(sync_load_aliases)
        except Exception as e:
            logger.error(f"加载别名失败: {e}")
            return {}

    async def _save_aliases(self, aliases: dict):
        if not self.alias_path:
            return
            
        try:
            def sync_save_aliases():
                self.alias_path.write_text(json.dumps(aliases, ensure_ascii=False), encoding="utf-8")
            
            await asyncio.to_thread(sync_save_aliases)
        except Exception as e:
            logger.error(f"保存别名文件失败: {e}")



    def _normalize_category(self, raw: str | None) -> str:
        """将模型返回的情绪类别规范化到内部英文标签。

        - 将中文情绪词映射为英文标签
        - 兼容旧配置中的“搞怪/其它”等泛用标签
        - 对常见同义词做映射
        """
        if not raw:
            return "neutral"
        text = str(raw).strip()

        if text in self.categories:
            return text

        # 旧分类兼容
        if text == "搞怪":
            # 搞怪类通常是调皮/夸张，可映射到 smirk / happy
            return "smirk" if "smirk" in self.categories else "happy"
        if text in {"其它", "其他", "其他表情", "其他情绪"}:
            return "neutral" if "neutral" in self.categories else self.categories[0]

        # 同义词 / 近义词映射（中文与英文别名）
        if text in self._EMOTION_MAPPING and self._EMOTION_MAPPING[text] in self.categories:
            return self._EMOTION_MAPPING[text]

        # 通过包含关系粗略匹配
        for key, val in self._EMOTION_MAPPING.items():
            if key in text and val in self.categories:
                return val

        for cat in self.categories:
            if cat in text:
                return cat

        # 默认回退
        return "neutral" if "neutral" in self.categories else self.categories[0]

    def _is_likely_emoji_by_metadata(self, file_path: str) -> bool:
        """基于文件大小与图像尺寸做一次启发式过滤，减少明显非表情图片。

        这里只做“明显不是表情包”的快速排除，避免误删正常表情：
        - 非常大的文件（>2MB）且分辨率较高时更像是照片/长图
        - 长宽比极端（>4:1）时更像长截图/漫画页
        - 过小的图片也直接排除
        """
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0

        # 特大图一般不是聊天用表情
        if size and size > 2 * 1024 * 1024:
            return False

        if PILImage is not None:
            try:
                with PILImage.open(file_path) as im:
                    width, height = im.size

                if width <= 0 or height <= 0:
                    return False

                long_side = max(width, height)
                short_side = min(width, height)

                # 过长的长图 / 截图
                aspect = long_side / short_side if short_side > 0 else 0
                if aspect > 4.0:
                    return False

                # 过小或超大分辨率都视为非典型表情
                if long_side < 40:
                    return False

                if long_side > 2048:
                    return False

            except Exception:
                # 无法读取时不做强制判定，交给后续多模态模型处理
                return True

        return True

    async def _classify_image(self, event: AstrMessageEvent | None, file_path: str) -> tuple[str, list[str], str, str]:
        """调用多模态模型对图片进行情绪分类与标签抽取。

        Args:
            event: 当前消息事件，用于获取 provider 配置。
            file_path: 本地图片路径。

        Returns:
            (category, tags, desc, emotion): 类别、标签、详细描述、情感标签。
        """
        try:
            h = await self._compute_hash(file_path)
            
            # 检查图片分类结果缓存
            if h and h in self._image_cache:
                cached_result = self._image_cache[h]
                if cached_result and len(cached_result) >= 4:
                    logger.debug(f"使用图片分类缓存结果: {h}")
                    return cached_result

            # 获取视觉模型
            prov_id = await self._pick_vision_provider(event)
            if not prov_id:
                return "其它", [], "", "其它"

            # 仅在启用表情包过滤时进行判断
            if self.emoji_only:
                # 先用元数据做一次快速过滤，明显不是表情图片的直接跳过
                if not self._is_likely_emoji_by_metadata(file_path):
                    return "非表情包", [], "", "非表情包"

                # 再使用多模态模型严格判断是否为表情包
                emoji_prompt = (
                    "你是聊天表情审核助手，请判断这张图片是否为聊天表情包"
                    "（emoji/meme/sticker），仅返回“是”或“否”，不要添加任何其他内容。"
                    "表情包通常具有以下特征："
                    "1）尺寸相对较小，主要用于聊天中快速表达情绪或态度；"
                    "2）画面主体清晰突出，通常集中在人物/卡通形象/动物或简洁抽象图案上；"
                    "3）可能包含少量文字、夸张表情或动作来强化情绪表达；"
                    "4）常以方图或接近方图的比例出现（宽高比通常在1:2到2:1之间）；"
                    "5）风格简洁明了，能在短时间内传达情绪。"
                    "以下情况一律回答“否”："
                    "- 风景照、生活照片、人像摄影等写实类图片"
                    "- 完整漫画页、长截图（高度远大于宽度）"
                    "- 聊天记录截图、社交媒体界面截图"
                    "- 宣传海报、商业广告、产品图片"
                    "- 电脑/手机壁纸（通常尺寸较大且内容复杂）"
                    "- 含大量说明文字的信息图、流程图、文档截图"
                    "- 视频帧截图、电影/动漫截图（非专门制作的表情）"
                    "- 像素极低或严重模糊无法识别内容的图片"
                )
                emoji_resp = await self.context.llm_generate(
                    chat_provider_id=prov_id,
                    prompt=emoji_prompt,
                    image_urls=[f"file:///{os.path.abspath(file_path)}"],
                )
                emoji_result = emoji_resp.completion_text.strip()
                is_emoji = ("是" in emoji_result) or ("yes" in emoji_result.lower())

                # 如果不是表情包，返回特定标识
                if not is_emoji:
                    return "非表情包", [], "", "非表情包"

            # 统一使用合并后的提示词进行图片分类
            resp = await self.context.llm_generate(
                chat_provider_id=prov_id,
                prompt=self.IMAGE_CLASSIFICATION_PROMPT,
                image_urls=[f"file:///{os.path.abspath(file_path)}"],
            )
            txt = resp.completion_text.strip()
            
            # 解析JSON结果
            cat = "无语"
            tags: list[str] = []
            desc = ""
            emotion = ""
            
            try:
                data = json.loads(txt)
                desc = str(data.get("description", "")).strip()
                c = str(data.get("category", "")).strip()
                if c:
                    cat = self._normalize_category(c)
                    emotion = cat  # 默认使用类别作为情绪
                t = data.get("tags", [])
                if isinstance(t, list):
                    tags = [str(x) for x in t][:8]
            except Exception:
                logger.warning(f"解析图片分类结果失败: {txt}")
            
            # 规范化类别和情绪
            cat = self._normalize_category(cat)
            emotion = self._normalize_category(emotion) if emotion else cat
            emo = emotion
            
            # 保存描述到缓存
            if desc and h:
                self._desc_cache[h] = desc
                
                # 清理描述缓存，保持在阈值以下
                if len(self._desc_cache) > self._CACHE_MAX_SIZE:
                    keys_to_keep = list(self._desc_cache.keys())[-self._CACHE_MAX_SIZE:]
                    self._desc_cache = {k: self._desc_cache[k] for k in keys_to_keep}
                
                if self.desc_cache_path:
                    try:
                        def sync_save_desc_cache():
                            self.desc_cache_path.write_text(json.dumps(self._desc_cache, ensure_ascii=False), encoding="utf-8")
                        await asyncio.to_thread(sync_save_desc_cache)
                    except Exception as e:
                        logger.error(f"保存描述缓存失败: {e}")
            
            # 保存分类结果到缓存
            if h:
                result = (cat, tags, desc or "", emo)
                self._image_cache[h] = result
                
                # 清理缓存，保持在阈值以下
                if len(self._image_cache) > self._CACHE_MAX_SIZE:
                    # 只保留最新的条目
                    keys_to_keep = list(self._image_cache.keys())[-self._CACHE_MAX_SIZE:]
                    self._image_cache = {k: self._image_cache[k] for k in keys_to_keep}
                
                if self.image_cache_path:
                    try:
                        def sync_save_image_cache():
                            self.image_cache_path.write_text(
                                json.dumps(self._image_cache, ensure_ascii=False), encoding="utf-8"
                            )
                        
                        await asyncio.to_thread(sync_save_image_cache)
                    except Exception as e:
                        logger.error(f"保存图片缓存失败: {e}")
            
            return cat, tags, desc or "", emo or cat
        except Exception as e:
            logger.error(f"视觉分类失败: {e}")
            fallback = "无语" if "无语" in self.categories else self.categories[0]
            return fallback, [], "", fallback

    async def _compute_hash(self, file_path: str) -> str:
        try:
            def sync_compute_hash():
                with open(file_path, "rb") as f:
                    data = f.read()
                return hashlib.sha256(data).hexdigest()
            return await asyncio.to_thread(sync_compute_hash)
        except Exception:
            return ""

    async def _file_to_base64(self, path: str) -> str:
        try:
            def sync_file_to_base64():
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            return await asyncio.to_thread(sync_file_to_base64)
        except Exception:
            return ""

    async def _filter_image(self, event: AstrMessageEvent | None, file_path: str) -> bool:
        try:
            if not self.content_filtration:
                return True
            prov_id = await self._pick_vision_provider(event)
            if not prov_id:
                return True
            prompt = self.IMAGE_FILTER_PROMPT.format(filtration_rule=self.filtration_prompt)
            resp = await self.context.llm_generate(
                chat_provider_id=prov_id,
                prompt=prompt,
                image_urls=[f"file:///{os.path.abspath(file_path)}"],
            )
            txt = resp.completion_text.strip()
            return ("是" in txt) or ("符合" in txt) or ("yes" in txt.lower())
        except Exception:
            return True

    async def _store_image(self, src_path: str, category: str) -> str:
        """将图片保存到 raw 与分类目录，并返回分类目录保存路径。"""
        if not self.base_dir:
            return src_path
        name = f"{int(asyncio.get_event_loop().time()*1000)}_{random.randint(1000,9999)}"
        ext = os.path.splitext(src_path)[1] or ".jpg"
        raw_dest = self.base_dir / "raw" / f"{name}{ext}"
        cat_dir = self.base_dir / "categories" / category
        cat_dest = cat_dir / f"{name}{ext}"
        
        try:
            def sync_store_image():
                # 同步部分
                cat_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_path, raw_dest)
                shutil.copyfile(src_path, cat_dest)
                return cat_dest.as_posix()
            
            return await asyncio.to_thread(sync_store_image)
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            return src_path
    
    async def _safe_remove_file(self, file_path: str) -> bool:
        """安全删除文件，处理可能的异常"""
        try:
            await asyncio.to_thread(os.remove, file_path)
            return True
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
    
    async def _process_image(self, event: AstrMessageEvent | None, file_path: str, is_temp: bool = False, idx: dict | None = None) -> tuple[bool, dict | None]:
        """统一处理图片的方法，包括过滤、分类、存储和索引更新
        
        Args:
            event: 消息事件对象，可为None
            file_path: 图片文件路径
            is_temp: 是否为临时文件，处理后需要删除
            idx: 可选的索引字典，如果提供则直接使用，否则加载新的
            
        Returns:
            (成功与否, 更新后的索引字典)
        """
        try:
            # 过滤图片
            ok = await self._filter_image(event, file_path)
            if not ok:
                if is_temp:
                    await self._safe_remove_file(file_path)
                return False, idx
            
            # 分类图片
            cat, tags, desc, emotion = await self._classify_image(event, file_path)
            
            # 如果分类为"非表情包"，跳过存储
            if cat == "非表情包" or emotion == "非表情包":
                if is_temp:
                    await self._safe_remove_file(file_path)
                return False, idx
            
            # 存储图片
            stored = await self._store_image(file_path, cat)
            
            # 更新索引
            if stored != file_path:  # 确保图片已成功保存
                # 如果没有提供索引，则加载新的
                if idx is None:
                    idx = await self._load_index()
                
                idx[stored] = {
                    "category": cat,
                    "tags": tags,
                    "backend_tag": self.backend_tag,
                    "created_at": int(asyncio.get_event_loop().time()),
                    "usage_count": 0,
                    "desc": desc,
                    "emotion": emotion,
                }
                
                # 删除源文件（如果是临时文件）
                if is_temp:
                    await self._safe_remove_file(file_path)
            
            return True, idx
        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            if is_temp:
                await self._safe_remove_file(file_path)
            return False, idx

    def _is_in_parentheses(self, text: str, index: int) -> bool:
        """判断字符串中指定索引位置是否在括号内。
        
        支持圆括号()和方括号[]。
        """
        parentheses_count = 0
        bracket_count = 0
        
        for i in range(index):
            if text[i] == '(':
                parentheses_count += 1
            elif text[i] == ')':
                parentheses_count -= 1
            elif text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
        
        return parentheses_count > 0 or bracket_count > 0

    async def _classify_text_category(self, event: AstrMessageEvent, text: str) -> str:
        """调用文本模型判断文本情绪并映射到插件分类。"""
        try:
            # 检查缓存
            import hashlib
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if text_hash in self._text_cache:
                return self._text_cache[text_hash]
            
            # 检查频率限制
            current_time = asyncio.get_event_loop().time()
            if current_time - self._last_llm_call_time < self._llm_call_cooldown:
                # 冷却时间未到，返回空
                return ""
            
            prov_id = await self._pick_text_provider(event)
            
            # 使用插件原有的分类体系构建提示词，要求输出&&emotion&&格式
            categories_str = ", ".join(self.categories)
            prompt = self.TEXT_EMOTION_PROMPT_TEMPLATE.format(
                categories=categories_str,
                text=text
            )
            
            if prov_id is None:
                return ""
                
            resp = await self.context.llm_generate(chat_provider_id=str(prov_id), prompt=prompt)
            txt = resp.completion_text.strip()
            
            # 提取&&emotion&&格式的内容
            match = re.search(r'&&([^&&]+)&&', txt)
            if match:
                emotion = match.group(1).strip()
            else:
                # 如果没有&&格式，直接使用返回值
                emotion = txt
            
            # 使用插件内置的_normalize_category方法进行类别映射
            normalized_category = self._normalize_category(emotion)
            result = normalized_category if normalized_category in self.categories else ""
            
            # 更新缓存
            self._text_cache[text_hash] = result
            # 限制缓存大小
            if len(self._text_cache) > self._CACHE_MAX_SIZE:
                keys_to_keep = list(self._text_cache.keys())[-self._CACHE_MAX_SIZE:]
                self._text_cache = {k: self._text_cache[k] for k in keys_to_keep}
            # 异步保存缓存
            await self._save_cache(self._text_cache, self.text_cache_path)
            
            # 更新最后调用时间
            self._last_llm_call_time = current_time
            
            return result
            
        except Exception as e:
            logger.error(f"文本情绪分类失败: {e}")
            return ""

    async def _extract_emotions_from_text(self, event: AstrMessageEvent | None, text: str) -> tuple[list[str], str]:
        """从文本中提取情绪关键词，本地提取不到时使用 LLM。

        支持的形式：
        - 形如 &&开心&& 的显式标记
        - 直接出现的类别关键词（如“开心”“害羞”“哭泣”等），按出现顺序去重
        - 本地提取不到时调用 LLM 进行情绪分类
        
        返回：
        - 提取到的情绪列表
        - 清理掉情绪标记和情绪词后的文本
        
        优化出处：参考 astrbot_plugin_meme_manager 插件的 resp 方法实现
        """
        if not text:
            return [], text

        res: list[str] = []
        seen: set[str] = set()
        cleaned_text = str(text)
        valid_categories = set(self.categories)
        
        # 1. 处理显式包裹标记：&&情绪&&
        hex_pattern = r"&&([^&&]+)&&"
        matches = list(re.finditer(hex_pattern, cleaned_text))

        # 收集所有匹配项，避免索引偏移问题
        temp_replacements = []
        for match in matches:
            original = match.group(0)
            emotion = match.group(1).strip()
            norm_cat = self._normalize_category(emotion)
            
            if norm_cat and norm_cat in valid_categories:
                temp_replacements.append((original, norm_cat))
            else:
                temp_replacements.append((original, ""))  # 非法或未知情绪静默移除

        # 保持原始顺序替换
        for original, emotion in temp_replacements:
            cleaned_text = cleaned_text.replace(original, "", 1)
            if emotion and emotion not in seen:
                seen.add(emotion)
                res.append(emotion)
        
        # 2. 处理直接出现的英文情绪词（直接匹配分类）
        for cat in self.categories:
            if cat in seen:
                continue
                
            # 使用边界匹配确保是完整单词
            pattern = rf'\b{re.escape(cat)}\b'
            matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
            
            # 检查是否有括号外的匹配
            has_external_match = False
            for match in matches:
                if not self._is_in_parentheses(cleaned_text, match.start()):
                    has_external_match = True
                    break
            
            if has_external_match:
                seen.add(cat)
                res.append(cat)
                
                # 注意：不再移除英文情绪词，保留原始文本的完整性
                # 只提取情绪，不修改文本内容
        
        # 3. 处理直接出现的中文情绪词（使用统一的EMOTION_MAPPING）
        # 按长度排序，优先处理长的情绪词
        sorted_cn_emotions = sorted(self._EMOTION_MAPPING.keys(), key=len, reverse=True)
        
        for cn_emotion in sorted_cn_emotions:
            en_emotion = self._EMOTION_MAPPING[cn_emotion]
            if en_emotion in valid_categories and en_emotion not in seen:
                positions = []
                start = 0
                
                # 收集所有匹配位置
                while True:
                    pos = cleaned_text.find(cn_emotion, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                # 检查是否有括号外的匹配
                external_positions = [pos for pos in positions if not self._is_in_parentheses(cleaned_text, pos)]
                
                if external_positions:
                    seen.add(en_emotion)
                    res.append(en_emotion)
                    
                    # 注意：不再移除中文情绪词，保留原始文本的完整性
                    # 只提取情绪，不修改文本内容
        
        # 清理多余的空格
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # 本地提取不到情绪时，调用 LLM 进行分类
        if not res and event:
            llm_emotion = await self._classify_text_category(event, cleaned_text)
            if llm_emotion and llm_emotion in valid_categories:
                res.append(llm_emotion)

        return res, cleaned_text

    async def _pick_vision_provider(self, event: AstrMessageEvent | None) -> str | None:
        if self.vision_provider_id:
            return self.vision_provider_id
        if event is None:
            return None
        return await self.context.get_current_chat_provider_id(event.unified_msg_origin)

    async def _pick_text_provider(self, event: AstrMessageEvent | None) -> str | None:
        if self.text_provider_id:
            return self.text_provider_id
        if event is None:
            return None
        return await self.context.get_current_chat_provider_id(event.unified_msg_origin)

    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs):
        """消息监听：偷取消息中的图片并分类存储。"""
        if not self.enabled:
            return
            
        # 收集所有图片组件
        imgs = [comp for comp in event.message_obj.message if isinstance(comp, Image)]
        
        for img in imgs:
            try:
                path = await img.convert_to_file_path()
                # 使用统一的图片处理方法
                success, idx = await self._process_image(event, path, is_temp=True)
                if success and idx:
                    await self._save_index(idx)
            except Exception as e:
                logger.error(f"处理图片失败: {e}")

    async def _scanner_loop(self):
        while True:
            try:
                await asyncio.sleep(max(1, int(self.check_interval)) * 60)
                if not self.steal_emoji:
                    continue
                await self._scan_register_emoji_folder()
            except Exception as e:
                logger.error(f"表情包扫描循环出错: {e}")
                continue

    async def _scan_register_emoji_folder(self):
        try:
            base = Path(get_astrbot_data_path()) / "emoji"
            base.mkdir(parents=True, exist_ok=True)
            
            # 获取所有支持格式的图片文件
            supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
            files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in supported_formats]
            
            if not files:
                return
            
            # 一次性加载索引，避免多次IO操作
            idx = await self._load_index()
            
            for f in files:
                try:
                    # 使用统一的图片处理方法
                    success, idx = await self._process_image(None, f.as_posix(), is_temp=True, idx=idx)
                    
                    # 如果处理失败，尝试删除源文件（可能是不合法内容）
                    if not success:
                        await self._safe_remove_file(f.as_posix())
                except Exception as e:
                    logger.error(f"处理文件失败: {f.as_posix()}, 错误: {e}")
                    await self._safe_remove_file(f.as_posix())
            
            # 在处理完所有文件后再检查容量和保存索引
            await self._enforce_capacity(idx)
            await self._save_index(idx)
        except Exception as e:
            logger.error(f"扫描注册表情文件夹失败: {e}")

    async def _enforce_capacity(self, idx: dict):
        try:
            if len(idx) <= int(self.max_reg_num):
                return
            if not self.do_replace:
                return
            items = []
            for k, v in idx.items():
                c = int(v.get("usage_count", 0)) if isinstance(v, dict) else 0
                t = int(v.get("created_at", 0)) if isinstance(v, dict) else 0
                items.append((k, c, t))
            items.sort(key=lambda x: (x[1], x[2]))
            remove_count = len(idx) - int(self.max_reg_num)
            for i in range(remove_count):
                rp = items[i][0]
                try:
                    if os.path.exists(rp):
                        await asyncio.to_thread(os.remove, rp)
                except Exception:
                    pass
                if rp in idx:
                    del idx[rp]
        except Exception:
            return

    



    @filter.on_decorating_result()
    async def before_send(self, event: AstrMessageEvent):
        if not self.auto_send or not self.base_dir:
            return
        result = event.get_result()
        # 只在有文本结果时尝试匹配表情包
        if result is None:
            return
        try:
            chance = float(self.emoji_chance)
            # 兜底保护，防止配置错误导致永远/从不触发
            if chance <= 0:
                logger.debug(f"表情包自动发送概率为0，未触发图片发送")
                return
            if chance > 1:
                chance = 1.0
            if random.random() >= chance:
                logger.debug(f"表情包自动发送概率检查未通过 ({chance}), 未触发图片发送")
                return
        except Exception:
            logger.error(f"解析表情包自动发送概率配置失败，未触发图片发送")
            pass
        
        logger.debug(f"表情包自动发送概率检查通过，开始处理图片发送")
        
        # 文本仅用于本地规则提取情绪关键字，不再请求额外的 LLM
        text = result.get_plain_text() or event.get_message_str()
        if not text or not text.strip():
            logger.debug(f"没有可处理的文本内容，未触发图片发送")
            return
            
        emotions, cleaned_text = await self._extract_emotions_from_text(event, text)
        if not emotions:
            logger.debug(f"未从文本中提取到情绪关键词，未触发图片发送")
            return
            
        logger.debug(f"提取到情绪关键词: {emotions}")
        
        # 目前只取第一个识别到的情绪类别
        category = emotions[0]
        cat_dir = self.base_dir / "categories" / category
        if not cat_dir.exists():
            logger.debug(f"情绪'{category}'对应的图片目录不存在，未触发图片发送")
            return
            
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            logger.debug(f"情绪'{category}'对应的图片目录为空，未触发图片发送")
            return
            
        logger.debug(f"从'{category}'目录中找到 {len(files)} 张图片")
        pick = random.choice(files)
        idx = await self._load_index()
        rec = idx.get(pick.as_posix())
        if isinstance(rec, dict):
            rec["usage_count"] = int(rec.get("usage_count", 0)) + 1
            rec["last_used"] = int(asyncio.get_event_loop().time())
            idx[pick.as_posix()] = rec
            await self._save_index(idx)
        # 创建新的结果对象并更新内容
        new_result = event.make_result().set_result_content_type(result.result_content_type)
        
        # 添加除了Plain文本外的其他组件
        for comp in result.chain:
            if not isinstance(comp, Plain):
                new_result.chain.append(comp)
        
        # 添加清除标签后的文本
        if cleaned_text.strip():
            new_result.message(cleaned_text.strip())
        
        # 添加图片
        b64 = await self._file_to_base64(pick.as_posix())
        new_result.base64_image(b64)
        
        # 设置新的结果对象
        event.set_result(new_result)

    @filter.command_group("meme")
    def meme(self):
        """meme 指令组。"""
        pass

    @meme.command("on")
    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        self.enabled = True
        self._persist_config()
        yield event.plain_result("已开启偷表情包")

    @meme.command("off")
    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        self.enabled = False
        self._persist_config()
        yield event.plain_result("已关闭偷表情包")

    @meme.command("auto_on")
    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        self.auto_send = True
        self._persist_config()
        yield event.plain_result("已开启自动发送")

    @meme.command("auto_off")
    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        self.auto_send = False
        self._persist_config()
        yield event.plain_result("已关闭自动发送")



    @meme.command("set_vision")
    async def set_vision(self, event: AstrMessageEvent, provider_id: str = ""):
        if not provider_id:
            yield event.plain_result("请提供视觉模型的 provider_id")
            return
        self.vision_provider_id = provider_id
        self._persist_config()
        yield event.plain_result(f"已设置视觉模型: {provider_id}")

    @meme.command("set_text")
    async def set_text(self, event: AstrMessageEvent, provider_id: str = ""):
        if not provider_id:
            yield event.plain_result("请提供主回复文本模型的 provider_id")
            return
        self.text_provider_id = provider_id
        self._persist_config()
        yield event.plain_result(f"已设置文本模型: {provider_id}")

    @meme.command("show_providers")
    async def show_providers(self, event: AstrMessageEvent):
        vp = self.vision_provider_id or "当前会话"
        tp = self.text_provider_id or "当前会话"
        yield event.plain_result(f"视觉模型: {vp}\n文本模型: {tp}")



    @meme.command("emoji_only")
    async def meme_emoji_only(self, event: AstrMessageEvent, enable: str = ""):
        """切换是否仅偷取表情包模式。"""
        if enable.lower() in ["on", "开启", "true"]:
            self.emoji_only = True
            self._persist_config()
            yield event.plain_result("已开启仅偷取表情包模式")
        elif enable.lower() in ["off", "关闭", "false"]:
            self.emoji_only = False
            self._persist_config()
            yield event.plain_result("已关闭仅偷取表情包模式")
        else:
            status = "开启" if self.emoji_only else "关闭"
            yield event.plain_result(f"当前仅偷取表情包模式: {status}")

    @meme.command("status")
    async def status(self, event: AstrMessageEvent):
        """显示当前偷取状态与后台标识。"""
        st_on = "开启" if self.enabled else "关闭"
        st_auto = "开启" if self.auto_send else "关闭"
        st_emoji_only = "开启" if self.emoji_only else "关闭"
        idx = await self._load_index()
        yield event.plain_result(
            f"偷取: {st_on}\n自动发送: {st_auto}\n仅偷取表情包: {st_emoji_only}\n已注册数量: {len(idx)}\n概率: {self.emoji_chance}\n上限: {self.max_reg_num}\n替换: {self.do_replace}\n周期: {self.check_interval}min\n自动偷取: {self.steal_emoji}\n审核: {self.content_filtration}"
        )

    async def get_count(self) -> int:
        idx = await self._load_index()
        return len(idx)

    async def get_info(self) -> dict:
        idx = await self._load_index()
        return {
            "current_count": len(idx),
            "max_count": self.max_reg_num,
            "available_emojis": len(idx),
        }

    async def get_emotions(self) -> list[str]:
        idx = await self._load_index()
        s = set()
        for v in idx.values():
            if isinstance(v, dict):
                emo = v.get("emotion")
                if isinstance(emo, str) and emo:
                    s.add(emo)
        return sorted(list(s))

    async def get_descriptions(self) -> list[str]:
        idx = await self._load_index()
        res = []
        for v in idx.values():
            if isinstance(v, dict):
                d = v.get("desc")
                if isinstance(d, str) and d:
                    res.append(d)
        return res

    async def _load_all_records(self) -> list[tuple[str, dict]]:
        idx = await self._load_index()
        return [(k, v) for k, v in idx.items() if isinstance(v, dict) and os.path.exists(k)]

    async def get_random_paths(self, count: int | None = 1) -> list[tuple[str, str, str]]:
        recs = await self._load_all_records()
        if not recs:
            return []
        n = max(1, int(count or 1))
        pick = random.sample(recs, min(n, len(recs)))
        res = []
        for p, v in pick:
            d = str(v.get("desc", ""))
            emo = str(v.get("emotion", v.get("category", self.categories[0] if self.categories else "开心")))
            res.append((p, d, emo))
        return res

    async def get_by_emotion_path(self, emotion: str) -> tuple[str, str, str] | None:
        recs = await self._load_all_records()
        cands = []
        for p, v in recs:
            emo = str(v.get("emotion", v.get("category", "")))
            tags = v.get("tags", [])
            if emotion and (emotion == emo or (isinstance(tags, list) and emotion in [str(t) for t in tags])):
                cands.append((p, v))
        if not cands:
            return None
        p, v = random.choice(cands)
        return (p, str(v.get("desc", "")), str(v.get("emotion", v.get("category", self.categories[0] if self.categories else "开心"))))

    async def get_by_description_path(self, description: str) -> tuple[str, str, str] | None:
        recs = await self._load_all_records()
        cands = []
        for p, v in recs:
            d = str(v.get("desc", ""))
            if description and description in d:
                cands.append((p, v))
        if not cands:
            for p, v in recs:
                tags = v.get("tags", [])
                if isinstance(tags, list):
                    if any(str(description) in str(t) for t in tags):
                        cands.append((p, v))
        if not cands:
            return None
        p, v = random.choice(cands)
        return (p, str(v.get("desc", "")), str(v.get("emotion", v.get("category", self.categories[0] if self.categories else "开心"))))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @meme.command("push")
    async def push(self, event: AstrMessageEvent, category: str = "", alias: str = ""):
        if not self.base_dir:
            return
        umo = event.unified_msg_origin
        if alias:
            aliases = await self._load_aliases()
            if alias in aliases:
                umo = aliases[alias]
            else:
                yield event.plain_result("别名不存在")
                return
        cat = category or (self.categories[0] if self.categories else "happy")
        cat_dir = self.base_dir / "categories" / cat
        if not cat_dir.exists():
            yield event.plain_result("分类不存在")
            return
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            yield event.plain_result("该分类暂无表情包")
            return
        pick = random.choice(files)
        b64 = await self._file_to_base64(pick.as_posix())
        chain = MessageChain().base64_image(b64)
        # 统一使用yield返回结果，保持交互体验一致
        yield event.result_with_message_chain(chain)




