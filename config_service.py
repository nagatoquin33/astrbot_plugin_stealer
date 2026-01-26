import json
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger


class ConfigService:
    """简化的配置服务，负责加载默认值、合并用户配置、提供属性访问。
    
    核心功能：
    1. 从 _conf_schema.json 读取默认值
    2. 合并用户配置（AstrBotConfig 优先级最高）
    3. 提供属性访问接口
    4. 向后兼容（覆盖升级时不报错）
    """

    def __init__(self, base_dir: Path, astrbot_config: AstrBotConfig | None = None):
        """初始化配置服务。

        Args:
            base_dir: 插件数据目录
            astrbot_config: AstrBot 全局配置（优先级最高）
        """
        self.base_dir = base_dir
        self.schema_path = Path(__file__).parent / "_conf_schema.json"
        self._astrbot_config = astrbot_config
        
        # 加载默认值
        self._defaults = self._load_defaults()
        
        # 当前配置（合并后的最终配置）
        self._config = {}
        
        # 别名配置
        self.alias_path = self.base_dir / "aliases.json"
        self._aliases = {}
        
        # 分类详细信息配置
        self.category_info_path = self.base_dir / "category_info.json"
        self.category_info = {
            "happy": {"name": "开心", "desc": "快乐、高兴、愉悦的情绪"},
            "sad": {"name": "难过", "desc": "悲伤、沮丧、失落的情绪"},
            "angry": {"name": "生气", "desc": "愤怒、恼火、不满的情绪"},
            "shy": {"name": "害羞", "desc": "羞涩、不好意思的情绪"},
            "surprised": {"name": "惊讶", "desc": "意外、震惊、惊奇的情绪"},
            "troll": {"name": "搞怪", "desc": "调皮、搞怪、发癫的情绪"},
            "cry": {"name": "大哭", "desc": "哭泣、流泪、伤心的情绪"},
            "confused": {"name": "困惑", "desc": "迷茫、不解、疑惑的情绪"},
            "embarrassed": {"name": "尴尬", "desc": "窘迫、尴尬、为难的情绪"},
            "love": {"name": "喜爱", "desc": "喜欢、爱慕、宠溺的情绪"},
            "disgust": {"name": "厌恶", "desc": "讨厌、反感、嫌弃的情绪"},
            "fear": {"name": "害怕", "desc": "恐惧、担心、害怕的情绪"},
            "excitement": {"name": "兴奋", "desc": "激动、亢奋、兴奋的情绪"},
            "tired": {"name": "疲惫", "desc": "劳累、疲倦、无力的情绪"},
            "sigh": {"name": "叹气", "desc": "无奈、叹气、失望的情绪"},
            "thank": {"name": "感谢", "desc": "感谢、道谢、感恩的情绪"},
            "dumb": {"name": "无语", "desc": "呆滞、无语、傻眼的状态"},
            "troll": {"name": "搞怪", "desc": "发癫、搞怪、调皮的状态"},
        }

    def _load_defaults(self) -> dict[str, Any]:
        """从 _conf_schema.json 加载默认值。"""
        try:
            with open(self.schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            
            defaults = {}
            for key, props in schema.items():
                # 跳过分隔符（invisible 字段）
                if isinstance(props, dict) and props.get("invisible"):
                    continue
                # 提取默认值
                if isinstance(props, dict) and "default" in props:
                    defaults[key] = props["default"]
            
            return defaults
        except Exception as e:
            logger.error(f"加载配置模式失败: {e}")
            return {}

    def initialize(self):
        """初始化配置：合并默认值、用户配置、AstrBotConfig。"""
        # 1. 从默认值开始
        config = self._defaults.copy()
        
        # 2. 从 AstrBotConfig 合并（如果存在）
        if self._astrbot_config:
            try:
                config.update(self._astrbot_config)
            except Exception as e:
                logger.error(f"从 AstrBotConfig 加载配置失败: {e}")
        
        # 3. 应用配置到实例属性
        self._config = config
        self._apply_config()
        
        # 4. 加载别名
        self._load_aliases()

        # 5. 加载分类详细信息
        self._load_category_info()

    def _load_category_info(self):
        """加载分类详细信息。"""
        if not self.category_info_path.exists():
            return
        
        try:
            with open(self.category_info_path, encoding="utf-8") as f:
                saved_info = json.load(f)
                self.category_info.update(saved_info)
        except Exception as e:
            logger.error(f"加载分类详细信息失败: {e}")

    def save_category_info(self):
        """保存分类详细信息。"""
        try:
            with open(self.category_info_path, "w", encoding="utf-8") as f:
                json.dump(self.category_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存分类详细信息失败: {e}")

    def _apply_config(self):
        """将配置字典应用到实例属性。"""
        # 基础功能
        self.steal_emoji = self._config.get("steal_emoji", True)
        self.auto_send = self._config.get("auto_send", True)
        self.emoji_chance = self._config.get("emoji_chance", 0.4)
        self.smart_emoji_selection = self._config.get("smart_emoji_selection", True)
        
        # 模型配置
        self.vision_provider_id = self._config.get("vision_provider_id")
        self.content_filtration = self._config.get("content_filtration", False)
        self.enable_natural_emotion_analysis = self._config.get("enable_natural_emotion_analysis", True)
        self.emotion_analysis_provider_id = self._config.get("emotion_analysis_provider_id")
        
        # 存储管理
        self.max_reg_num = self._config.get("max_reg_num", 100)
        self.do_replace = self._config.get("do_replace", True)
        
        # 节流控制
        self.image_processing_mode = self._config.get("image_processing_mode", "probability")
        self.image_processing_probability = self._config.get("image_processing_probability", 0.3)
        self.image_processing_interval = self._config.get("image_processing_interval", 60)
        self.image_processing_cooldown = self._config.get("image_processing_cooldown", 30)
        
        # WebUI 配置
        self.webui_enabled = self._config.get("webui_enabled", True)
        self.webui_host = self._config.get("webui_host", "0.0.0.0")
        self.webui_port = self._config.get("webui_port", 8899)
        
        # 高级选项
        self.enable_raw_cleanup = self._config.get("enable_raw_cleanup", True)
        self.raw_cleanup_interval = self._config.get("raw_cleanup_interval", 30)
        self.enable_capacity_control = self._config.get("enable_capacity_control", True)
        self.capacity_control_interval = self._config.get("capacity_control_interval", 60)
        self.raw_retention_minutes = self._config.get("raw_retention_minutes", 30)
        
        # 分类列表
        self.categories = self._config.get("categories", [
            "happy", "sad", "angry", "shy", "surprised", "troll",
            "cry", "confused", "embarrassed", "love", "disgust",
            "fear", "excitement", "tired", "sigh", "thank", "dumb"
        ])

    def _load_aliases(self):
        """加载别名文件。"""
        if not self.alias_path.exists():
            return
        
        try:
            with open(self.alias_path, encoding="utf-8") as f:
                self._aliases = json.load(f)
        except Exception as e:
            logger.error(f"加载别名文件失败: {e}")

    def save_aliases(self):
        """保存别名文件。"""
        try:
            with open(self.alias_path, "w", encoding="utf-8") as f:
                json.dump(self._aliases, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存别名文件失败: {e}")

    def get_aliases(self) -> dict:
        """获取别名字典。"""
        return self._aliases.copy()

    def update_aliases(self, aliases: dict):
        """更新别名字典并保存。"""
        self._aliases = aliases
        self.save_aliases()

    def update_config(self, updates: dict[str, Any]) -> bool:
        """更新配置（仅更新内存，不持久化）。
        
        注意：配置由 AstrBot 管理，插件不应持久化配置。
        """
        if not updates:
            return False
        
        try:
            self._config.update(updates)
            self._apply_config()
            return True
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    def update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return
        
        try:
            # 特殊处理 categories（去重）
            if "categories" in config_dict and isinstance(config_dict["categories"], list):
                categories = config_dict["categories"]
                seen = set()
                unique_categories = [
                    cat for cat in categories if not (cat in seen or seen.add(cat))
                ]
                config_dict["categories"] = unique_categories
            
            # 更新配置
            self.update_config(config_dict)
        except Exception as e:
            logger.error(f"从字典更新配置失败: {e}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """设置配置值（仅更新内存）。"""
        try:
            self._config[key] = value
            if hasattr(self, key):
                setattr(self, key, value)
        except Exception as e:
            logger.error(f"设置配置失败: {e}")

    def _normalize_category(self, raw: str) -> str:
        """将模型返回的情绪类别规范化到内部英文标签。"""
        if not raw:
            return self.categories[0]
        
        text = str(raw).strip()
        
        if text in self.categories:
            return text
        
        # 旧分类兼容
        if text == "搞怪":
            return "troll" if "troll" in self.categories else "happy"
        if text in {"其它", "其他", "其他表情", "其他情绪"}:
            return self.categories[0]
        
        return self.categories[0]

    def get_category_info(self) -> list[dict]:
        """获取所有分类的详细信息。"""
        result = []
        for cat_key in self.categories:
            info = self.category_info.get(cat_key, {"name": cat_key, "desc": ""})
            result.append({
                "key": cat_key,
                "name": info.get("name", cat_key),
                "desc": info.get("desc", "")
            })
        return result

    def cleanup(self):
        """清理资源。"""
        pass
