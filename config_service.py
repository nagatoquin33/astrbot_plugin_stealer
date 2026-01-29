import json
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger


class ConfigService:
    """配置与插件数据服务。

    - 插件配置：由 AstrBot 通过 _conf_schema.json 管理，并在启动时注入 AstrBotConfig
    - 插件数据：别名、分类信息等存放在 data/plugin_data/{plugin_name}/ 下
    """

    def __init__(self, base_dir: Path, astrbot_config: AstrBotConfig | None = None):
        """初始化服务。

        Args:
            base_dir: 插件数据目录
            astrbot_config: AstrBot 注入的插件配置
        """
        self.base_dir = base_dir
        self._astrbot_config = astrbot_config
        self._config: dict[str, Any] | AstrBotConfig = astrbot_config or {}

        # 别名配置
        self.alias_path = self.base_dir / "aliases.json"
        self._aliases = {}

        self.categories_path = self.base_dir / "categories.json"

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
        }

    def initialize(self):
        """初始化服务。"""
        self._apply_config()

        # 加载别名
        self._load_aliases()

        # 加载分类列表
        self._load_categories()

        # 加载分类详细信息
        self._load_category_info()

    def _load_categories(self):
        if self.categories_path.exists():
            try:
                with open(self.categories_path, encoding="utf-8-sig") as f:
                    saved = json.load(f)
                if isinstance(saved, list):
                    normalized = []
                    seen = set()
                    for item in saved:
                        if not isinstance(item, str):
                            continue
                        key = item.strip()
                        if not key or key in seen:
                            continue
                        normalized.append(key)
                        seen.add(key)
                    if normalized:
                        self.categories = normalized
                return
            except Exception as e:
                logger.error(f"加载分类列表失败: {e}")

        try:
            legacy_path = self.base_dir / "cache" / "categories_cache.json"
            if not legacy_path.exists():
                return
            with open(legacy_path, encoding="utf-8-sig") as f:
                legacy = json.load(f)
            if not isinstance(legacy, dict):
                return
            legacy_categories = legacy.get("categories")
            if not isinstance(legacy_categories, list):
                return

            normalized = []
            seen = set()
            for item in legacy_categories:
                if not isinstance(item, str):
                    continue
                key = item.strip()
                if not key or key in seen:
                    continue
                normalized.append(key)
                seen.add(key)

            if not normalized:
                return

            self.categories = normalized
            self.save_categories()
        except Exception:
            return

    def save_categories(self):
        try:
            with open(self.categories_path, "w", encoding="utf-8") as f:
                json.dump(self.categories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存分类列表失败: {e}")

    def _load_category_info(self):
        """加载分类详细信息。"""
        if not self.category_info_path.exists():
            try:
                for cat_key in getattr(self, "categories", []) or []:
                    if cat_key not in self.category_info:
                        self.category_info[cat_key] = {"name": cat_key, "desc": ""}
                self.save_category_info()
            except Exception:
                pass
            return

        try:
            with open(self.category_info_path, encoding="utf-8-sig") as f:
                saved_info = json.load(f)

            normalized: dict[str, dict[str, str]] = {}
            if isinstance(saved_info, dict):
                for k, v in saved_info.items():
                    if not isinstance(k, str):
                        continue
                    key = k.strip()
                    if not key:
                        continue
                    if isinstance(v, dict):
                        name = str(v.get("name", key)).strip() or key
                        desc = str(v.get("desc", "")).strip()
                        normalized[key] = {"name": name, "desc": desc}
                    elif isinstance(v, str):
                        normalized[key] = {"name": v.strip() or key, "desc": ""}
            elif isinstance(saved_info, list):
                for item in saved_info:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("key", "")).strip()
                    if not key:
                        continue
                    name = str(item.get("name", key)).strip() or key
                    desc = str(item.get("desc", "")).strip()
                    normalized[key] = {"name": name, "desc": desc}

            if normalized:
                self.category_info.update(normalized)

            changed = False
            for cat_key in getattr(self, "categories", []) or []:
                if cat_key not in self.category_info:
                    self.category_info[cat_key] = {"name": cat_key, "desc": ""}
                    changed = True

            to_add_categories = []
            current_categories = list(getattr(self, "categories", []) or [])
            current_set = set(current_categories)
            for cat_key in normalized.keys():
                if cat_key not in current_set:
                    to_add_categories.append(cat_key)
                    current_set.add(cat_key)

            if to_add_categories:
                self.categories = current_categories + to_add_categories
                self.save_categories()
                changed = True

            if isinstance(saved_info, list) or changed:
                self.save_category_info()
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

        self.categories = [
            "happy", "sad", "angry", "shy", "surprised", "troll",
            "cry", "confused", "embarrassed", "love", "disgust",
            "fear", "excitement", "tired", "sigh", "thank", "dumb",
        ]

    def _load_aliases(self):
        """加载别名文件。"""
        if not self.alias_path.exists():
            return

        try:
            with open(self.alias_path, encoding="utf-8-sig") as f:
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
        """更新配置。

        如果存在 AstrBot 注入的配置对象，则会写回配置文件并同步内存视图。
        """
        if not updates:
            return False

        try:
            if self._astrbot_config:
                self._astrbot_config.save_config(updates)
                self._config = self._astrbot_config
            else:
                self._config.update(updates)
            self._apply_config()
            self._load_categories()
            return True
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    def update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return

        try:
            categories = None
            if "categories" in config_dict and isinstance(config_dict["categories"], list):
                seen = set()
                unique_categories = []
                for item in config_dict["categories"]:
                    if not isinstance(item, str):
                        continue
                    key = item.strip()
                    if not key or key in seen:
                        continue
                    unique_categories.append(key)
                    seen.add(key)
                categories = unique_categories
                config_dict = dict(config_dict)
                config_dict.pop("categories", None)

            if config_dict:
                self.update_config(config_dict)

            if categories is not None:
                self.categories = categories
                self.save_categories()
        except Exception as e:
            logger.error(f"从字典更新配置失败: {e}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        try:
            self.update_config({key: value})
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
