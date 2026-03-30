# 🌟 表情包小偷

<div align="center">

<img src="https://count.getloli.com/@nagatoquin33?name=nagatoquin33&theme=rule34&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto" alt="Moe Counter">

**让 Bot 自动偷走群友的表情包，分类入库，聊天时看心情自动甩出来。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%E2%89%A54.10.4-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
[![Last Commit](https://img.shields.io/github/last-commit/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/commits/master)

</div>

---

## 📢 简介

灵感来自 maibot 的表情包偷取和 meme_manager 的标签注入思路。全人机代码，本人全程监工调试。

表情包小偷是一款基于多模态 AI 的 [AstrBot](https://github.com/Soulter/AstrBot) 娱乐插件——自动偷取聊天中的表情包，利用视觉模型进行情绪分类，对话时按概率发送一张情绪匹配的表情包，让 Bot 的回复更有人味。偷够了可以随时金盆洗手，偷取和自动发送可独立开关。

本插件完全开源免费，欢迎 Issue 和 PR。

## ✨ 核心功能

| 功能 | 说明 |
|:---|:---|
| **自动偷图** | 监听群聊图片，按概率或冷却模式自动收集，支持内容审核过滤 |
| **智能分类** | 利用 VLM 识别图片内容与情绪（happy、sad、angry 等 17+ 种预设分类） |
| **情感共鸣** | 分析 Bot 回复的情绪，自动追加一张匹配的表情包 |
| **LLM 主动选图** | 对话中 LLM 可通过工具调用搜索并发送最合适的表情包 |
| **双模式情绪分析** | LLM 模式（后置轻量模型分析，不改回复）/ 被动标签模式（LLM 直接标注情绪） |
| **WebUI 管理** | 可视化查看、搜索、删除、拉黑、设置 `public/local` 作用域，支持来源群展示与批量管理 |
| **群聊过滤** | 黑白名单控制哪些群启用偷取/发送 |

## 🚀 快速开始

### 1. 安装

在 AstrBot 插件管理中搜索并安装 `astrbot_plugin_stealer`。

### 2. 前置条件

**必须配置视觉模型**——插件依赖 VLM 对图片进行分类。可以在 AstrBot 中配置全局图片描述模型，也可以在插件配置中指定 `vision_provider_id`。

### 3. 开始使用

```
/meme on        # 开启偷图
/meme auto_on   # 开启自动发送
```

偷够了？

```
/meme off       # 关闭偷图（已收集的表情包仍可使用）
```

### 4. WebUI 管理

启动后访问 `http://<你的IP>:9191/`，默认启用登录验证，密码在插件配置中查看（首次自动生成 6 位随机密码）。

- `blacklist` 和 `scope` 为即开即用功能，无需额外新增配置项；未设置作用域的图片默认按 `public` 处理
- WebUI 已支持表情包作用域管理：`public` 为公共图库，`local` 仅允许在来源群发送
- 可在列表和详情页查看 `origin_target`，并批量调整作用域

## 💡 推荐用法

### 全自动模式（适合 Token 充足）

**开启自动偷图 + 自动发送**

1. 启用偷图：`/meme on`
2. 开启自动发送：`/meme auto_on`
3. Bot 会自动偷取群聊中的表情包并分类
4. Bot 回复时会根据情绪自动追加匹配的表情包

> LLM 模式：Bot 回复后由后置轻量模型分析情绪，不影响回复内容
> 被动标签模式：LLM 直接标注情绪，更快但会在回复中插入标签

**LLM 主动选图**：对话中 Bot 可通过工具调用搜索最合适的表情包

### 半自动模式（适合 Token 吃紧）

**手动管理表情包 **

1. 手动放入表情包到 `data/emoji_store/` 分类目录
2. 或使用 WebUI 批量上传并指定分类
3. Bot 回复时仍会自动发送匹配的表情包（仅使用已有分类，不调用 VLM 分析）

### 精准入库模式（适合不想乱偷）

**精准收录 + WebUI 管理**

1. 使用指令精准入库：`/meme` 进入强制收录模式，期间发送的图片直接入库
2. 或 WebUI 批量导入 + 自动分析：勾选"自动识别"由 VLM 分析每张图并分类
3. 通过 WebUI 查看、编辑、删除、设置作用域

## ⚙️ 配置说明

所有配置项均可在 AstrBot 管理面板中修改，无需手动编辑文件。

### 偷图设置

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `steal_emoji` | `false` | 总开关 |
| `steal_mode` | `probability` | `probability` 概率模式 / `cooldown` 冷却模式 |
| `steal_chance` | `0.3` | 概率模式下每次偷取的概率 |
| `image_processing_cooldown` | `30` | 冷却模式下两次偷取的最小间隔（秒） |
| `content_filtration` | `false` | 内容审核，开启后过滤不当图片 |

### 发送设置

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `auto_send` | `false` | 自动随聊发送表情包 |
| `emoji_chance` | `0.2` | 自动发送的概率（0.0 ~ 1.0） |
| `send_emoji_as_gif` | `false` | 以 GIF 格式发送（更像真表情包，但高频场景内存占用略高） |
| `emoji_send_delay` | `5.0` | 发送延迟（秒），避免与分段插件冲突，设为 0 立即发送 |
| `emoji_send_delay_random` | `false` | 开启后在 [延迟] ~ [最大延迟] 之间随机等待，更自然 |
| `emoji_send_delay_max` | `8.0` | 随机延迟最大值（秒），仅随机模式生效 |
| `smart_emoji_selection` | `true` | 智能评分选择；关闭则随机选取 |

### 情绪识别

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `enable_natural_emotion_analysis` | `true` | `true` = LLM 模式（推荐） / `false` = 被动标签模式 |
| `emotion_analysis_provider_id` | `""` | LLM 模式使用的轻量模型，留空用默认模型 |

### 模型配置

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `vision_provider_id` | `""` | 视觉模型，留空自动使用 AstrBot 全局图片描述模型 |

### 群聊过滤

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `send_target_whitelist` | `[]` | 发表情白名单。使用 `group:群号` 或 `user:QQ号` |
| `send_target_blacklist` | `[]` | 发表情黑名单。白名单为空时生效 |
| `steal_target_whitelist` | `[]` | 偷表情白名单。使用 `group:群号` 或 `user:QQ号` |
| `steal_target_blacklist` | `[]` | 偷表情黑名单。白名单为空时生效 |

### 存储管理

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `max_reg_num` | `100` | 表情包存储上限，超出自动清理最旧的 |

### WebUI

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| `webui.enabled` | `false` | 是否启用 WebUI |
| `webui.host` | `0.0.0.0` | 监听地址 |
| `webui.port` | `9191` | 监听端口 |
| `webui.auth_enabled` | `true` | 启用登录验证 |
| `webui.password` | `""` | 访问密码，留空自动生成 |
| `webui.session_timeout` | `3600` | 登录会话超时（秒） |

### WebUI 批量上传

批量上传时，**分类选择**和**自动分析**互斥：
- **选择分类**：图片保存到指定分类，不调用VLM
- **自动分析**：不选择分类，启用VLM自动识别每张图片的分类（会高并发调用API）

> ⚠️ **高并发警告**：启用自动分析时，会同时并发处理多张图片，可能造成API限流。请根据API服务限流情况分批次分析。

## 🔄 情绪分析模式详解

| | LLM 模式（默认） | 被动标签模式 |
|:---|:---|:---|
| **原理** | Bot 回复后，后台用轻量模型分析语义情绪 | 注入提示词让 LLM 在回复开头插入 `&&emotion&&` 标签 |
| **对回复的影响** | ✅ 不修改 LLM 回复 | ❌ 会在回复中插入标签（插件自动清理） |
| **适用场景** | 日常使用，保持对话自然 | 需要精确控制情绪分类 |

> ⚠️ **切换模式后务必执行 `/reset` 清除对话上下文**，否则 LLM 可能延续旧模式的输出习惯。

## 🎮 指令列表

所有指令以 `/meme` 为前缀。

### 基础指令（所有人可用）

| 指令 | 说明 |
|:---|:---|
| `on` / `off` | 开启 / 关闭表情包收集 |
| `auto_on` / `auto_off` | 开启 / 关闭自动发送 |
| `status` | 查看运行状态和表情包统计 |
| `list [分类] [每页数量] [页码]` | 列出已收集的表情包（默认每页 10，页码从 1 开始） |
| `emotion_stats` | 查看情绪分析统计和当前模式 |
| `clean [force]` | 清理未分类的原始缓存 |

### 管理指令（仅管理员）

| 指令 | 说明 |
|:---|:---|
| `偷` | 进入 30 秒强制收录模式，期间发送的图片直接入库 |
| `group show` | 查看当前偷表情/发表情名单 |
| `group <send\|steal> <wl\|bl> <add\|del\|clear> [group:群号\|user:QQ号]` | 管理目标黑白名单 |
| `delete <序号\|文件名>` | 删除指定表情包 |
| `blacklist <序号\|文件名>` | 删除指定表情包并加入黑名单 |
| `scope <序号\|文件名> <public\|local>` | 设置表情包作用域，`local` 仅来源群可发送 |
| `capacity` | 立即执行容量控制 |
| `rebuild_index` | 重建索引（版本迁移或索引异常时使用） |
| `natural_analysis <on\|off>` | 切换情绪识别模式 |
| `clear_emotion_cache` | 清空情绪分析缓存 |

### LLM 工具调用（对话中自动触发）

| 工具 | 说明 |
|:---|:---|
| `search_emoji` | LLM 搜索匹配的表情包候选列表（包含标签与场景） |
| `send_emoji_by_id` | LLM 从候选列表中选择并发送表情包 |

## ⚠️ 注意事项

- **WebUI 删除分类**会同时删除该分类下所有图片，且无法恢复，操作前请确认。
- 开启 `send_emoji_as_gif` 时，大分辨率图片或高帧动图的 GIF 转换会产生瞬时内存峰值，低内存环境建议关闭。
- 插件依赖视觉模型（VLM）进行分类，未配置视觉模型时偷图功能无法正常工作。

### 📝 提示词格式更新（v2.4.5+）

从 v2.4.5 开始，VLM 分类提示词改为 **JSON 格式输出**，结构更稳定，解析更准确。

**如果你之前自定义过提示词**，建议：
1. 清空 `custom_emoji_classification_prompt` 和 `custom_emoji_classification_with_filter_prompt` 配置
2. 重启插件，自动加载新版 JSON 格式提示词

旧格式（管道符分隔）仍兼容，但 JSON 格式更可靠。

## 📄 许可证

本项目基于 [MIT](LICENSE) 许可证开源。

---

<div align="center">

好用的话给个 ⭐ Star 吧，谢谢大伙！

有问题欢迎提 [Issue](https://github.com/nagatoquin33/astrbot_plugin_stealer/issues) 或群里找我，一般很快会看到。

</div>
