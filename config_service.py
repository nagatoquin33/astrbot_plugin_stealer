import json
import re
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

        self.category_aliases: dict[str, str] = {
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

        def normalize_id_list(value: object) -> list[str]:
            if not isinstance(value, list):
                return []
            normalized: list[str] = []
            seen: set[str] = set()
            for item in value:
                if item is None:
                    continue
                s = str(item).strip()
                if not s or s in seen:
                    continue
                normalized.append(s)
                seen.add(s)
            return normalized

        # 基础功能
        self.steal_emoji = self._config.get("steal_emoji", True)
        self.auto_send = self._config.get("auto_send", True)
        self.emoji_chance = self._config.get("emoji_chance", 0.4)
        self.smart_emoji_selection = self._config.get("smart_emoji_selection", True)
        self.send_emoji_as_gif = self._config.get("send_emoji_as_gif", True)

        self.group_blacklist = normalize_id_list(
            self._config.get("group_blacklist", [])
        )
        self.group_whitelist = normalize_id_list(
            self._config.get("group_whitelist", [])
        )

        # 模型配置
        self.vision_provider_id = self._config.get("vision_provider_id")
        self.content_filtration = self._config.get("content_filtration", False)
        self.enable_natural_emotion_analysis = self._config.get(
            "enable_natural_emotion_analysis", True
        )
        self.emotion_analysis_provider_id = self._config.get(
            "emotion_analysis_provider_id"
        )

        # 存储管理
        self.max_reg_num = self._config.get("max_reg_num", 100)
        self.do_replace = self._config.get("do_replace", True)

        # 节流控制
        self.image_processing_mode = self._config.get(
            "image_processing_mode", "probability"
        )
        self.image_processing_probability = self._config.get(
            "image_processing_probability", 0.3
        )
        self.image_processing_interval = self._config.get(
            "image_processing_interval", 60
        )
        self.image_processing_cooldown = self._config.get(
            "image_processing_cooldown", 30
        )

        # WebUI 配置
        self.webui_enabled = self._config.get("webui_enabled", True)
        self.webui_host = self._config.get("webui_host", "0.0.0.0")
        self.webui_port = self._config.get("webui_port", 8899)
        self.webui_auth_enabled = self._config.get("webui_auth_enabled", True)
        self.webui_password = str(self._config.get("webui_password", "") or "").strip()
        raw_timeout = self._config.get("webui_session_timeout", 3600)
        try:
            self.webui_session_timeout = int(raw_timeout) if raw_timeout else 3600
        except Exception:
            self.webui_session_timeout = 3600

        # 高级选项
        self.enable_raw_cleanup = self._config.get("enable_raw_cleanup", True)
        self.raw_cleanup_interval = self._config.get("raw_cleanup_interval", 30)
        self.enable_capacity_control = self._config.get("enable_capacity_control", True)
        self.capacity_control_interval = self._config.get(
            "capacity_control_interval", 60
        )
        self.raw_retention_minutes = self._config.get("raw_retention_minutes", 30)

        self.categories = [
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

    def is_group_allowed(self, group_id: str | None) -> bool:
        if not group_id:
            return True
        gid = str(group_id).strip()
        if not gid:
            return True
        whitelist = getattr(self, "group_whitelist", []) or []
        blacklist = getattr(self, "group_blacklist", []) or []
        if whitelist:
            return gid in whitelist
        if blacklist:
            return gid not in blacklist
        return True

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
            if "categories" in config_dict and isinstance(
                config_dict["categories"], list
            ):
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

    def normalize_category_strict(self, raw: str) -> str:
        if not raw:
            return ""

        text = str(raw).strip()
        if not text:
            return ""

        if text in self.categories:
            return text

        lowered = text.lower()
        for cat in self.categories:
            if cat.lower() == lowered:
                return cat

        alias_hit = self.category_aliases.get(text)
        if alias_hit and alias_hit in self.categories:
            return alias_hit

        alias_hit_lower = self.category_aliases.get(lowered)
        if alias_hit_lower and alias_hit_lower in self.categories:
            return alias_hit_lower

        for k, v in sorted(
            self.category_aliases.items(),
            key=lambda kv: len(str(kv[0] or "")),
            reverse=True,
        ):
            if not v or v not in self.categories:
                continue

            key = str(k or "").strip()
            if len(key) <= 1:
                continue

            key_lower = key.lower()
            if key_lower.isascii():
                if re.search(rf"\b{re.escape(key_lower)}\b", lowered):
                    return v
            else:
                if key_lower in lowered:
                    return v

        return ""

    def _normalize_category(self, raw: str) -> str:
        """将模型返回的情绪类别规范化到内部英文标签。"""
        if not raw:
            return self.categories[0]

        text = str(raw).strip()
        strict = self.normalize_category_strict(text)
        if strict:
            return strict

        if text in self.categories:
            return text

        # 旧分类兼容
        if text in {"其它", "其他", "其他表情", "其他情绪"}:
            return self.categories[0]

        return self.categories[0]

    def get_category_info(self) -> list[dict]:
        """获取所有分类的详细信息。"""
        result = []
        for cat_key in self.categories:
            info = self.category_info.get(cat_key, {"name": cat_key, "desc": ""})
            result.append(
                {
                    "key": cat_key,
                    "name": info.get("name", cat_key),
                    "desc": info.get("desc", ""),
                }
            )
        return result

    def get_group_id(self, event) -> str | None:
        """从事件中提取群组ID。

        Args:
            event: 消息事件对象

        Returns:
            str | None: 群组ID，如果不存在则返回None
        """
        try:
            if hasattr(event, "get_group_id"):
                gid = event.get_group_id()
                if gid:
                    return str(gid)
        except Exception:
            pass

        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            try:
                gid = getattr(message_obj, "group_id", "") or ""
                gid = str(gid).strip()
                return gid or None
            except Exception:
                return None
        return None

    def cleanup(self):
        """清理资源。"""
        pass
