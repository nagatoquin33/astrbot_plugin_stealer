import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context, StarTools


class WebuiConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8899
    auth_enabled: bool = True
    password: str = ""
    session_timeout: int = 3600


class PluginConfig(BaseModel):
    # === 基础功能 ===
    steal_emoji: bool = True
    steal_mode: str = "probability"  # "probability" 或 "cooldown"
    steal_chance: float = 0.6  # 概率模式下的偷图概率
    auto_send: bool = True
    emoji_chance: float = 0.4
    send_emoji_as_gif: bool = True

    # === 群聊过滤 ===
    group_whitelist: list[str] = []
    group_blacklist: list[str] = []

    # === 模型配置 ===
    vision_provider_id: str = ""

    # === WebUI 管理界面 ===
    webui: WebuiConfig = Field(default_factory=WebuiConfig)

    # === 内部常量/高级配置 ===
    max_reg_num: int = 100
    content_filtration: bool = False  # 内容审核开关
    image_processing_cooldown: int = 10
    enable_natural_emotion_analysis: bool = True  # 情绪识别模式
    emotion_analysis_provider_id: str = ""  # 情绪分析专用模型
    smart_emoji_selection: bool = True  # 智能表情包选择

    # === 内化常量（不再暴露给用户） ===
    DO_REPLACE: ClassVar[bool] = True  # 达到上限始终替换旧表情
    ENABLE_RAW_CLEANUP: ClassVar[bool] = True  # raw 始终自动清理
    RAW_CLEANUP_INTERVAL: ClassVar[int] = 30  # 清理周期(分钟)，固定
    ENABLE_CAPACITY_CONTROL: ClassVar[bool] = True  # 始终启用容量控制
    CAPACITY_CONTROL_INTERVAL: ClassVar[int] = 60  # 容量检查周期(分钟)，固定
    RAW_RETENTION_MINUTES: ClassVar[int] = 60  # 原始图片保留时间(分钟)，固定

    # === 分类信息 ===
    categories: list[str] = []
    category_info: dict[str, dict[str, str]] = {}

    # === 内部状态 (不作为 Pydantic 字段) ===
    # 使用 PrivateAttr 或在 __init__ 中设置且不包含在 __annotations__ 中
    # 但 Pydantic v1/v2 处理方式不同。这里使用 __private_attributes__ 机制或直接忽略

    # 忽略额外字段
    class Config:
        extra = "ignore"
        arbitrary_types_allowed = True

    # === 常量 ===
    # 使用 ClassVar 标注，避免被 Pydantic 识别为字段
    DEFAULT_CATEGORIES: ClassVar[list[str]] = [
        "happy",
        "sad",
        "angry",
        "shy",
        "surprised",
        "troll",
        "cry",
        "confused",
        "embarrassed",
        "love",
        "disgust",
        "fear",
        "excitement",
        "tired",
        "sigh",
        "thank",
        "dumb",
    ]

    DEFAULT_CATEGORY_INFO: ClassVar[dict[str, dict[str, str]]] = {
        "happy": {"name": "开心", "desc": "快乐、愉悦、满足、好心情"},
        "sad": {"name": "难过", "desc": "悲伤、沮丧、失落、emo"},
        "angry": {"name": "生气", "desc": "愤怒、恼火、不满、暴躁"},
        "shy": {"name": "害羞", "desc": "羞涩、不好意思、腼腆"},
        "surprised": {"name": "惊讶", "desc": "意外、震惊、惊奇、啊？"},
        "troll": {"name": "整活", "desc": "调皮、搞怪、发癫、抽象"},
        "cry": {"name": "哭哭", "desc": "哭泣、流泪、委屈、破防"},
        "confused": {"name": "困惑", "desc": "迷茫、不解、疑惑、问号脸"},
        "embarrassed": {"name": "尴尬", "desc": "社死、窘迫、为难、脚趾抠地"},
        "love": {"name": "喜欢", "desc": "喜爱、爱慕、宠溺、心动"},
        "disgust": {"name": "嫌弃", "desc": "厌恶、反感、讨厌、yue"},
        "fear": {"name": "害怕", "desc": "恐惧、担心、紧张、怂"},
        "excitement": {"name": "兴奋", "desc": "激动、亢奋、嗨、上头"},
        "tired": {"name": "困倦", "desc": "疲惫、困、无力、想躺"},
        "sigh": {"name": "无奈", "desc": "叹气、摆烂、算了、心累"},
        "thank": {"name": "感谢", "desc": "道谢、感恩、收到、爱了"},
        "dumb": {"name": "无语", "desc": "呆住、傻眼、离谱、沉默"},
    }

    DEFAULT_CATEGORY_ALIASES: ClassVar[dict[str, str]] = {
        "开心": "happy",
        "高兴": "happy",
        "快乐": "happy",
        "哈哈": "happy",
        "笑": "happy",
        "难过": "sad",
        "伤心": "sad",
        "emo": "sad",
        "沮丧": "sad",
        "失落": "sad",
        "生气": "angry",
        "愤怒": "angry",
        "恼火": "angry",
        "暴躁": "angry",
        "害羞": "shy",
        "不好意思": "shy",
        "腼腆": "shy",
        "惊讶": "surprised",
        "震惊": "surprised",
        "意外": "surprised",
        "搞怪": "troll",
        "整活": "troll",
        "发癫": "troll",
        "抽象": "troll",
        "哭": "cry",
        "大哭": "cry",
        "哭哭": "cry",
        "委屈": "cry",
        "破防": "cry",
        "困惑": "confused",
        "疑惑": "confused",
        "迷茫": "confused",
        "问号": "confused",
        "尴尬": "embarrassed",
        "社死": "embarrassed",
        "为难": "embarrassed",
        "喜欢": "love",
        "喜爱": "love",
        "爱": "love",
        "心动": "love",
        "嫌弃": "disgust",
        "厌恶": "disgust",
        "反感": "disgust",
        "yue": "disgust",
        "害怕": "fear",
        "恐惧": "fear",
        "紧张": "fear",
        "怂": "fear",
        "兴奋": "excitement",
        "激动": "excitement",
        "嗨": "excitement",
        "上头": "excitement",
        "疲惫": "tired",
        "困": "tired",
        "困倦": "tired",
        "想睡": "tired",
        "无奈": "sigh",
        "叹气": "sigh",
        "摆烂": "sigh",
        "算了": "sigh",
        "感谢": "thank",
        "谢谢": "thank",
        "多谢": "thank",
        "感恩": "thank",
        "无语": "dumb",
        "傻眼": "dumb",
        "离谱": "dumb",
        "沉默": "dumb",
        "其它": "",
        "其他": "",
        "其他表情": "",
        "其他情绪": "",
    }

    def __init__(self, config: AstrBotConfig | None, context: Context | None = None):
        # 1. 初始化 Pydantic 模型
        # config 可能是 AstrBotConfig (dict-like) 或 None
        initial_data = config if config else {}
        super().__init__(**initial_data)

        # 2. 保存 AstrBotConfig 引用以便回写
        # 使用 object.__setattr__ 绕过 Pydantic 的 setattr 检查
        object.__setattr__(self, "_data", config)
        object.__setattr__(self, "_plugin_name", "astrbot_plugin_stealer")

        # 3. 初始化路径和目录
        data_dir = StarTools.get_data_dir(self._plugin_name)
        object.__setattr__(self, "data_dir", data_dir)
        object.__setattr__(self, "categories_path", data_dir / "categories.json")
        object.__setattr__(self, "raw_dir", data_dir / "raw")
        object.__setattr__(self, "categories_dir", data_dir / "categories")
        object.__setattr__(self, "cache_dir", data_dir / "cache")
        object.__setattr__(self, "category_info_path", data_dir / "category_info.json")

        # 确保目录存在
        self.ensure_base_dirs()

        self._load_category_state()
        self._migrate_category_config()

    def _read_json_file(self, path: Path):
        try:
            if not path.exists():
                return None
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_json_file(self, path: Path, data: Any) -> bool:
        """写入 JSON 文件。

        Args:
            path: 文件路径
            data: 要写入的数据

        Returns:
            bool: 是否写入成功
        """
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except PermissionError as e:
            logger.error(f"[Config] 权限不足，无法写入文件 {path}: {e}")
            return False
        except OSError as e:
            logger.error(f"[Config] 写入文件失败 {path}: {e}")
            return False
        except Exception as e:
            logger.error(f"[Config] 写入 JSON 文件时发生未知错误 {path}: {e}")
            return False

    def _load_category_state(self) -> None:
        stored_categories = self._read_json_file(self.categories_path)
        stored_info = self._read_json_file(self.category_info_path)

        config_categories = None
        config_info = None
        if isinstance(self._data, dict):
            if "categories" in self._data:
                config_categories = self._data.get("categories")
            if "category_info" in self._data:
                config_info = self._data.get("category_info")

        categories = (
            stored_categories
            if isinstance(stored_categories, list) and stored_categories
            else config_categories
            if isinstance(config_categories, list) and config_categories
            else list(self.DEFAULT_CATEGORIES)
        )
        info = (
            stored_info
            if isinstance(stored_info, dict)
            else config_info
            if isinstance(config_info, dict)
            else {}
        )

        # 使用 BaseModel.__setattr__ 绕过自定义 __setattr__ 中的写文件逻辑，
        # 避免初始化期间重复写文件（最后统一写一次即可）
        BaseModel.__setattr__(self, "categories", list(categories))
        merged_info = dict(self.DEFAULT_CATEGORY_INFO)
        merged_info.update(info)
        BaseModel.__setattr__(self, "category_info", merged_info)
        self.save_categories()
        self.save_category_info()

    def _migrate_category_config(self) -> None:
        if not isinstance(self._data, dict):
            return
        removed = False
        if "categories" in self._data:
            try:
                del self._data["categories"]
                removed = True
            except Exception:
                pass
        if "category_info" in self._data:
            try:
                del self._data["category_info"]
                removed = True
            except Exception:
                pass
        if removed and hasattr(self._data, "save_config"):
            self._data.save_config()

    def save_webui_config(self) -> None:
        """保存 WebUI 配置。"""
        if hasattr(self, "_data") and hasattr(self._data, "save_config"):
            self._data.save_config({"webui": self.webui.model_dump()})

    def __setattr__(self, key: str, value: Any):
        # 更新 Pydantic 模型
        super().__setattr__(key, value)

        # 如果是私有属性或路径属性，跳过回写
        if key.startswith("_") or key in (
            "data_dir",
            "categories_path",
            "raw_dir",
            "categories_dir",
            "cache_dir",
            "category_info_path",
        ):
            return
        if key in ("categories", "category_info"):
            if key == "categories":
                self.save_categories()
            else:
                self.save_category_info()
            return

        # 回写到 AstrBotConfig
        if hasattr(self, "_data") and self._data is not None:
            if hasattr(self._data, "save_config"):
                try:
                    # 对于 webui 这种嵌套模型，如果是直接替换整个 webui 对象，这里可以处理
                    # 但如果是修改 webui.port，不会触发这里的 __setattr__
                    # 需要手动调用 save_webui_config
                    if key == "webui" and isinstance(value, WebuiConfig):
                        self._data.save_config({key: value.model_dump()})
                    else:
                        self._data.save_config({key: value})
                except Exception:
                    pass
            elif isinstance(self._data, dict):
                self._data[key] = value

    def update_config(self, updates: dict) -> bool:
        """批量更新配置项。

        Args:
            updates: 配置更新字典

        Returns:
            bool: 是否更新成功
        """
        try:
            for key, value in updates.items():
                setattr(self, key, value)

            # 回写到 AstrBotConfig
            if hasattr(self, "_data") and self._data is not None:
                if hasattr(self._data, "save_config"):
                    self._data.save_config(updates)
                elif isinstance(self._data, dict):
                    self._data.update(updates)
            return True
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    def save_categories(self) -> None:
        self._write_json_file(self.categories_path, self.categories)

    def save_category_info(self) -> None:
        self._write_json_file(self.category_info_path, self.category_info)

    def ensure_category_dir(self, category: str) -> Path:
        category_dir = self.categories_dir / str(category)
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir

    def ensure_category_dirs(self, categories: list[str] | None) -> None:
        if not categories:
            return
        for category in categories:
            self.ensure_category_dir(category)

    def ensure_raw_dir(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        return self.raw_dir

    def ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def ensure_base_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.categories_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def normalize_category_strict(self, category: str) -> str | None:
        """严格归一化情绪分类。"""
        if not category:
            return None

        category = category.lower().strip()

        # 1. 直接匹配当前配置的分类列表（包括用户自定义分类）
        if category in self.categories:
            return category

        # 2. 匹配默认分类（兜底）
        if category in self.DEFAULT_CATEGORIES:
            return category

        # 3. 别名查找
        return self.DEFAULT_CATEGORY_ALIASES.get(category)

    def get_keyword_map(self) -> dict[str, str]:
        """获取关键词映射表。"""
        return self.DEFAULT_CATEGORY_ALIASES

    def get_category_info(self) -> list[dict[str, str]]:
        categories = self.categories or list(self.DEFAULT_CATEGORIES)
        info_map = self.category_info or {}

        result: list[dict[str, str]] = []
        for key in categories:
            info = info_map.get(key, {}) if isinstance(info_map, dict) else {}
            name = str(info.get("name", "") or key)
            desc = str(info.get("desc", "") or "")
            result.append({"key": str(key), "name": name, "desc": desc})
        return result

    def get_group_id(self, event: AstrMessageEvent) -> str:
        """获取群号。"""
        try:
            return str(event.message_obj.group_id)
        except Exception:
            return ""

    def is_group_allowed(self, group_id: str) -> bool:
        """检查群组是否允许。"""
        if not group_id:
            return True

        if self.group_whitelist and group_id not in self.group_whitelist:
            return False

        if self.group_blacklist and group_id in self.group_blacklist:
            return False

        return True
