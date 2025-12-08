import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from astrbot.api import AstrBotConfig, logger


class PluginConfig(BaseModel):
    """插件配置的Pydantic模型，用于简化配置验证逻辑。"""
    auto_send: bool = Field(default=True, description="是否自动随聊追加表情包")
    vision_provider_id: str | None = Field(default=None, description="视觉模型提供商ID")
    emoji_chance: float = Field(default=0.4, description="触发表情动作的基础概率")
    max_reg_num: int = Field(default=100, description="允许注册的最大表情数量")
    do_replace: bool = Field(default=True, description="达到上限时是否替换旧表情")
    maintenance_interval: int = Field(default=10, description="后台维护任务的执行周期")
    steal_emoji: bool = Field(default=True, description="是否开启表情包偷取和清理功能")
    content_filtration: bool = Field(default=False, description="是否开启内容审核")
    filtration_prompt: str = Field(default="符合公序良俗", description="内容审核提示词")
    raw_retention_hours: int = Field(default=24, description="raw目录图片保留期限")
    raw_clean_interval: int = Field(default=3600, description="raw目录清理时间间隔")
    categories: list[str] = Field(default_factory=lambda: ["happy", "sad", "angry", "shy", "surprised", "smirk", "cry", "confused", "embarrassed", "love", "disgust", "fear", "excitement", "tired"], description="分类列表")
    config_version: str = Field(default="1.0.0", description="配置版本")

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> Any:
        """在验证前对字段进行预处理。"""
        if isinstance(data, dict):
            # 确保emoji_chance在0-1之间
            if "emoji_chance" in data:
                data["emoji_chance"] = max(0.0, min(1.0, data["emoji_chance"]))

            # 确保maintenance_interval大于0
            if "maintenance_interval" in data and data["maintenance_interval"] <= 0:
                data["maintenance_interval"] = 10

            # 确保max_reg_num大于0
            if "max_reg_num" in data and data["max_reg_num"] <= 0:
                data["max_reg_num"] = 100

            # 确保filtration_prompt不为空
            if "filtration_prompt" in data and not data["filtration_prompt"]:
                data["filtration_prompt"] = "符合公序良俗"
        return data

class ConfigManager:
    """统一的配置管理器，负责配置的加载、更新、验证和持久化。"""

    def __init__(self, config_path: Path, schema_path: Path, astrbot_config: AstrBotConfig | None = None):
        """初始化配置管理器。"""
        self.config_path = config_path
        self.schema_path = schema_path
        # 只在初始化时使用astrbot_config，不再持有对它的引用
        self._initial_astrbot_config = astrbot_config

        # 加载配置模式
        self.schema = self._load_schema()

        # 从模式中提取默认值和版本信息
        self.defaults = self._extract_defaults()

        # 从schema中获取版本信息，若未提供则使用默认版本
        self.CONFIG_VERSION = self.schema.get("config_version", {}).get("default", "1.0.0")

        # 当前配置
        self.config = self.defaults.copy()

        # 加载现有配置
        self.load_config()

    def _load_schema(self) -> dict[str, Any]:
        """加载配置模式文件。"""
        try:
            with open(self.schema_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置模式失败: {e}")
            return {}

    def _extract_defaults(self) -> dict[str, Any]:
        """从配置模式中提取默认值。"""
        defaults = {}
        for key, props in self.schema.items():
            if "default" in props:
                defaults[key] = props["default"]
        return defaults

    def load_config(self) -> None:
        """加载配置，优先使用初始的AstrBotConfig，其次是本地文件，最后使用默认值。"""
        # 加载默认值作为基础
        config = self.defaults.copy()

        # 从本地文件加载配置
        try:
            if self.config_path.exists():
                with open(self.config_path, encoding="utf-8") as f:
                    file_config = json.load(f)
                    config.update(file_config)
        except Exception as e:
            logger.error(f"从文件加载配置失败: {e}")

        # 从初始AstrBotConfig加载配置（优先级最高）
        if self._initial_astrbot_config:
            try:
                config.update(self._initial_astrbot_config)
            except Exception as e:
                logger.error(f"从AstrBotConfig加载配置失败: {e}")

        # 验证并应用配置
        self.config = self._validate_config(config)

        # 迁移旧版本配置
        self._migrate_config()

    def _validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """使用Pydantic模型验证配置的类型和范围。"""
        try:
            # 使用PluginConfig模型进行验证
            validated_config = PluginConfig.model_validate(config).model_dump()
            return validated_config
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            logger.debug(f"原始配置: {config}")
            # 如果验证失败，返回默认配置
            return PluginConfig().model_dump()

    def _migrate_config(self) -> None:
        """迁移旧版本配置到新版本格式。"""
        # 获取当前配置版本
        current_version = self.config.get("config_version", "0.0.0")

        # 如果是旧版本，执行迁移
        if current_version < self.CONFIG_VERSION:
            logger.info(f"迁移配置从版本 {current_version} 到 {self.CONFIG_VERSION}")

            # 版本迁移逻辑 - 实现实际的迁移规则
            # 1.0.0版本迁移：确保所有配置项都有正确的类型和默认值
            if current_version < "1.0.0":
                # 检查并迁移所有配置项
                for key, props in self.schema.items():
                    if key in self.config:
                        # 重新验证每个配置项
                        self.config[key] = self._validate_config({key: self.config[key]}).get(key)
                    else:
                        # 添加缺失的配置项
                        self.config[key] = self.defaults[key]

            # 设置当前配置版本
            self.config["config_version"] = self.CONFIG_VERSION

            # 保存迁移后的配置
            self.save_config()

    def update_config(self, updates: dict[str, Any]) -> bool:
        """更新配置并验证。"""
        try:
            # 创建临时配置进行验证
            temp_config = self.config.copy()
            temp_config.update(updates)

            # 验证新配置
            validated_config = self._validate_config(temp_config)

            # 更新配置
            self.config = validated_config

            # 保存配置
            self.save_config()

            return True
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """设置单个配置值。"""
        return self.update_config({key: value})

    def save_config(self) -> None:
        """持久化配置到本地文件。"""
        # 只保存到本地文件，不再修改AstrBotConfig
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存配置到文件失败: {e}")

    def get_config_dict(self) -> dict[str, Any]:
        """获取完整的配置字典。"""
        return self.config.copy()

class ConfigService:
    """配置管理服务，负责插件配置的加载、更新和持久化。"""

    def __init__(self, base_dir: Path, astrbot_config: AstrBotConfig | None = None):
        """初始化配置服务。

        Args:
            base_dir: 插件数据目录
            astrbot_config: AstrBot 全局配置
        """
        self.base_dir = base_dir
        self.config_path = base_dir / "config.json"
        self.astrbot_config = astrbot_config
        self.config_manager = None

        # 配置属性
        self.auto_send = True
        self.emoji_chance = 0.3
        self.max_reg_num = 500
        self.do_replace = True
        self.maintenance_interval = 30
        self.steal_emoji = True  # 控制偷取和扫描功能的开关
        self.content_filtration = True
        self.filtration_prompt = "符合公序良俗"
        self.vision_provider_id = None
        self.raw_retention_hours = 1  # raw目录图片保留期限
        self.raw_clean_interval = 1800  # raw目录清理时间间隔（秒）
        self.categories = [
            "happy",
            "sad",
            "angry",
            "shy",
            "surprised",
            "smirk",
            "cry",
            "confused",
            "embarrassed",
            "love",
            "disgust",
            "fear",
            "excitement",
            "tired",
        ]

        # 别名配置
        self.alias_path = self.base_dir / "aliases.json"
        self._aliases = {}

    def initialize(self):
        """初始化配置管理器并加载配置。"""
        self._initialize_config_manager()
        self._load_initial_config()
        self._load_aliases()

    def _initialize_config_manager(self):
        """初始化配置管理器。"""
        self.config_manager = ConfigManager(
            config_path=self.config_path,
            schema_path=Path(__file__).parent / "_conf_schema.json",
            astrbot_config=self.astrbot_config
        )

    def _load_initial_config(self):
        """从配置管理器加载初始配置。"""
        if not self.config_manager:
            return

        self.auto_send = self.config_manager.get("auto_send")
        self.emoji_chance = self.config_manager.get("emoji_chance")
        self.max_reg_num = self.config_manager.get("max_reg_num")
        self.do_replace = self.config_manager.get("do_replace")
        self.maintenance_interval = self.config_manager.get("maintenance_interval")
        self.steal_emoji = self.config_manager.get("steal_emoji")  # 控制偷取和扫描功能的开关
        self.content_filtration = self.config_manager.get("content_filtration")
        self.filtration_prompt = self.config_manager.get("filtration_prompt")
        self.vision_provider_id = self.config_manager.get("vision_provider_id")
        self.raw_retention_hours = self.config_manager.get("raw_retention_hours")
        self.raw_clean_interval = self.config_manager.get("raw_clean_interval")

        # 处理分类配置
        categories_config = self.config_manager.get("categories")
        if categories_config and isinstance(categories_config, list):
            self.categories = categories_config

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
        """获取别名字典。

        Returns:
            dict: 别名字典
        """
        return self._aliases.copy()

    def update_aliases(self, aliases: dict):
        """更新别名字典并保存。

        Args:
            aliases: 新的别名字典
        """
        self._aliases = aliases
        self.save_aliases()

    def update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict or not self.config_manager:
            return

        try:
            # 特殊处理categories（需要映射和去重）
            if "categories" in config_dict and isinstance(config_dict["categories"], list):
                categories = config_dict["categories"]
                # 去重并保持原始顺序
                seen = set()
                unique_categories = [cat for cat in categories if not (cat in seen or seen.add(cat))]
                config_dict["categories"] = unique_categories

            # 更新配置管理器
            for key, value in config_dict.items():
                self.config_manager.set(key, value)

            # 保存到文件
            self.config_manager.save_config()

            # 更新内存中的配置
            self._load_initial_config()
        except Exception as e:
            logger.error(f"从字典更新配置失败: {e}")

    def persist_config(self):
        """持久化插件运行配置。"""
        if not self.config_manager:
            return

        try:
            # 更新ConfigManager中的配置值
            config_updates = {
                "auto_send": self.auto_send,
            "categories": self.categories,
            "emoji_chance": self.emoji_chance,
            "max_reg_num": self.max_reg_num,
            "do_replace": self.do_replace,
            "maintenance_interval": self.maintenance_interval,
            "content_filtration": self.content_filtration,
            "filtration_prompt": self.filtration_prompt,
            "vision_provider_id": self.vision_provider_id
            }

            # 更新并保存配置
            self.config_manager.update_config(config_updates)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    def _normalize_category(self, raw: str) -> str:
        """将模型返回的情绪类别规范化到内部英文标签。

        Args:
            raw: 原始类别字符串

        Returns:
            规范化后的英文标签
        """
        if not raw:
            return self.categories[0]  # 默认使用第一个分类

        text = str(raw).strip()

        if text in self.categories:
            return text

        # 旧分类兼容
        if text == "搞怪":
            # 搞怪类通常是调皮/夸张，可映射到 smirk / happy
            return "smirk" if "smirk" in self.categories else "happy"
        if text in {"其它", "其他", "其他表情", "其他情绪"}:
            return self.categories[0]  # 移除"neutral"后，默认使用第一个分类

        # 不再需要同义词/近义词映射，直接返回第一个分类
        return self.categories[0]

    def get_config(self, key: str, default: any = None) -> any:
        """获取配置值。

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值
        """
        if not self.config_manager:
            return default
        return self.config_manager.get(key, default)

    def set_config(self, key: str, value: any) -> None:
        """设置配置值。

        Args:
            key: 配置键
            value: 配置值
        """
        if not self.config_manager:
            return

        try:
            self.config_manager.update_config({key: value})
            # 更新实例属性
            if hasattr(self, key):
                setattr(self, key, value)
        except Exception as e:
                logger.error(f"设置配置失败: {e}")

    def cleanup(self):
        """清理资源。"""
        pass







