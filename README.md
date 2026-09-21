# 🌟 表情包小偷

<div align="center">

<img src="https://count.getloli.com/@nagatoquin33?name=nagatoquin33&theme=rule34&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto" alt="Moe Counter">

**让 Bot 自动偷走群友的表情包，分类入库，聊天时看心情自动发出来。**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%E2%89%A54.24.1-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
[![CI](https://github.com/nagatoquin33/astrbot_plugin_stealer/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/nagatoquin33/astrbot_plugin_stealer/actions/workflows/ci.yml)
[![Last Commit](https://img.shields.io/github/last-commit/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/commits/master)

**Language / 语言**

[![中文](https://img.shields.io/badge/中文-当前-blue)](README.md)
[![English](https://img.shields.io/badge/English-README-lightgrey)](README_EN.md)

</div>

---

## 📢 简介

灵感来自 maibot 的表情包偷取思路，也曾参考 meme_manager 的标签注入机制（该机制在当前版本已弃用）；当前版本提供可供 LLM 调用的表情包工具。

表情包小偷是一款基于多模态 AI 的 [AstrBot](https://github.com/AstrBotDevs/AstrBot) 娱乐插件：自动收集聊天中的图片，使用视觉模型进行语义与情绪分类，在对话中按概率、冷却和目标过滤规则发送匹配的表情包。收集和自动发送可以分别开关。

本插件完全开源免费，欢迎提交 [Issue](https://github.com/nagatoquin33/astrbot_plugin_stealer/issues) 和 PR。

## ✨ 核心功能

| 功能 | 说明 |
|:---|:---|
| **自动偷图** | 监听群聊图片，按概率或冷却模式收集，并支持待审核池容量控制 |
| **待审核池** | 自动收集的图片可先进入审核区，人工通过后再进入图库 |
| **智能分类** | VLM 识别图片内容、文字、场景和情绪；GIF 从完整帧序列等距抽取九帧，生成 3×3 时间分镜 |
| **语义检索** | 将图上文字、描述和适用场景用于检索；可选远程 Embedding，未启用或不可用时使用 BM25，不运行本地 CLIP |
| **情绪匹配** | 分析 Bot 回复的情绪，在原回复之后追加匹配的表情包 |
| **LLM 主动选图** | LLM 可通过 `search_meme`、`send_meme`、`steal_meme` 工具搜索、发送和收录表情包 |
| **表情选择后端** | 保留轻量模型提取检索词与情绪先验；可选 JEV 根据最近对话从 Top 10 候选选图，也支持直接用回复原文检索 |
| **VLM A/B 复核** | WebUI 详情页可再次分析图片，并在确认前对照当前标注与新结果 |
| **角色库** | WebUI 手工为整套图片指定已有或新建角色，角色与情绪分类相互独立 |
| **外部表情包源** | 导入 Meme Manager / AstrBot Meme Pack、GitHub 资源包或分页 HTTPS JSON API，支持预检、映射、去重和来源溯源 |
| **WebUI 双区管理** | 审核区处理待入库图片；表情包库支持分类浏览、排序和批量操作 |
| **群聊过滤** | 为偷取和发送分别配置白名单、黑名单及冲突优先级 |

## 🚀 快速开始

### 1. 安装与更新

- 在 AstrBot 插件管理中搜索并安装 `astrbot_plugin_stealer`。
- 手动安装或更新时，下载 GitHub Release 中的 `astrbot_plugin_stealer-vX.Y.Z.zip`，解压后将顶层 `astrbot_plugin_stealer/` 目录放入 AstrBot 插件目录，再重启 AstrBot。更新插件代码不会删除 `plugin_data/astrbot_plugin_stealer/` 中的图库和数据库；避免多套嵌套目录导致插件无法加载。

### 2. 前置条件

**必须配置视觉模型**。插件依赖 VLM 对图片进行分类，可以使用 AstrBot 全局图片描述模型，也可以在插件配置中指定 `vision_provider_id`。Embedding 检索为可选项，启用后需配置可用的 Embedding Provider。

### 3. 开始使用

```
/meme on        # 开启偷图
/meme auto_on   # 开启自动发送
```

想暂停收集时：

```
/meme off       # 关闭偷图，已收集的表情包仍可使用
```

### 4. WebUI 管理

在 AstrBot 插件面板中进入插件详情页，点击「表情管理」即可打开管理页面，无需额外端口或密码。

- **表情包浏览**：按分类筛选、搜索、排序查看已收集的表情包。
- **作用域管理**：`public` 为公共图库，`local` 仅允许在来源群发送。
- **审核区**：批量通过、删除待审核图片，并查看失败原因。
- **单张上传**：上传图片后可使用 AI 自动识别分类、描述、标签和场景。
- **VLM A/B 复核**：详情页点击重新分析，确认后才应用新分类、文字、标签、场景和情绪；角色、作用域、收藏及使用记录会保留。
- **批量导入**：分类选择与自动分析互斥。选择分类时不调用 VLM；启用自动分析时会并发调用 VLM，请按 API 限流情况分批处理。
- **存储维护**：扫描并清理失效索引、孤儿文件、缩略图缓存和临时文件。
- **分类管理**：新增、编辑和删除表情包分类。
- **主题**：支持跟随宿主、暗色、亮色、我的世界和辐射 4；页面选择会持续保存，配置中的 `webui_theme` 可作为默认值。

## 🔌 v3 外部表情包源

在 WebUI 点击「外部源」，可直接完成以下操作：

- 自动发现同一 AstrBot 实例中的 Meme Manager v4 资源包。
- 上传 AstrBot Meme Pack 的 ZIP / `.meme-pack` 导出并预检；也接受带 `memes/` 图片目录的通用 ZIP。
- 填写 GitHub 仓库（`owner/repo` 或 HTTPS 地址），按分支和子目录读取资源包；例如 [DDZS987/astrbot-meme-pack-semantic-01](https://github.com/DDZS987/astrbot-meme-pack-semantic-01)。GitHub 源只读取公开归档，不执行仓库代码。
- 连接可分页的 HTTPS JSON 表情目录，后续一键同步。
- 将来源分类映射到本插件分类，选择直接入库或进入待审核池。
- 导入整套角色表情时，可选择已有角色或创建新角色，整套图片会写入同一角色标记。
- 查看导入、重复、失败和 stale 条目；同步缺失项只更新来源状态，图库副本会继续保留。

推荐资源包结构：

```
pack/
├── manifest.json                 # 资源包信息
├── memes/<分类>/<图片>            # 必选图片目录
├── meme_pack_export.json         # 可选导出元数据
└── semantic_metadata.json        # 可选逐图描述、标签、OCR 和远端哈希
```

纯图片 ZIP 也能导入；分类会从 `memes/` 下一级目录或图片父目录推断，`previews/` 和缩略图目录会跳过。导入过程会逐张验证格式、大小与像素数，再复制到本插件自己的目录。源插件和原始资源包不会被修改；相同图片按 SHA-256 去重，授权、署名和来源 URL 会写入溯源记录。外部导入项使用独立保留等级，不占用聊天偷图的自动淘汰额度。

HTTP 源默认只接受 HTTPS，并拒绝本机、私网、链路本地地址与危险重定向。启用内容审核时，外部图片会强制进入待审核池。协议字段、分页方式和压缩包限制见 [外部表情包源协议](docs/external-sources.md)。

## 💡 推荐用法

### 全自动模式（适合 Token 充足）

1. 启用偷图：`/meme on`
2. 开启自动发送：`/meme auto_on`
3. Bot 自动收集并分类群聊图片
4. Bot 回复完成后按意图、概率或冷却规则追加匹配表情包

LLM 情绪模式会用轻量模型提取检索词和情绪先验；被动检索模式直接使用回复原文。是否发送仍由原有门控决定，回复内容保持不变。LLM 也可以在对话中调用工具主动搜索和发送。

### 半自动模式（适合 Token 吃紧）

1. 手动将图片放入 `plugin_data/astrbot_plugin_stealer/categories/<分类>/`
2. 或使用 WebUI 批量上传并指定分类
3. 自动发送只使用已有分类，不额外调用 VLM

### 精准入库模式（适合不想乱偷）

1. 使用 `/meme 偷` 进入 30 秒强制收录模式，期间发送的图片直接入库
2. 或在 WebUI 批量导入时启用自动分析
3. 通过审核区和图库管理页面检查、编辑、删除及设置作用域

## ⚙️ 配置说明

所有公开配置项均可在 AstrBot 管理面板中修改。默认值与 `_conf_schema.json`、运行时配置保持一致。

### 偷图设置

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **开启表情包偷取功能** | `false` | 总开关 |
| **偷图模式** | `probability` | `probability` 按概率尝试；`cooldown` 两次收集至少间隔 30 秒 |
| **偷图概率** | `0.3` | 概率模式下每次收到图片的收集概率 |
| **内容审核** | `false` | 开启后使用 VLM 过滤不当图片，会增加处理时间 |
| **待审核池容量上限** | `200` | 达到上限后暂停自动偷取，审核处理后自动恢复 |
| **自动偷取需人工审核** | `true` | 开启后先进入审核区；关闭后通过基础校验即可直接入库 |
| **表情收录模式（QQ_Official）** | `cdn_only` | `all_images` 收录所有图片；`cdn_only` 仅收录表情 CDN 特征；`gif_only` 仅收录 GIF |

### 发送设置

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **自动随聊发送表情包** | `true` | 是否在 Bot 回复后自动发送 |
| **自动发送意图门控** | `true` | 跳过命令、错误/严肃回复、过短回复和疑问过多的内容 |
| **新消息取消待发送表情** | `true` | 同一会话有新消息时取消上一轮延迟发送 |
| **表情包发送概率** | `0.2` | 自动发送概率（0.0 ~ 1.0） |
| **真表情包样式（GIF 发送）** | `false` | 强制以 GIF 输出，内存占用会增加 |
| **以 QQ 表情包形式发送** | `true` | 仅 aiocqhttp/NapCat 生效；关闭后按普通图片发送 |
| **表情包发送延迟/秒** | `5.0` | 延迟发送以避开分段插件冲突，设为 0 立即发送 |
| **随机延迟** | `false` | 在固定延迟和最大延迟之间随机等待 |
| **最大随机延迟/秒** | `8.0` | 随机延迟上限 |
| **智能表情包选择** | `true` | 使用综合评分选择表情包；关闭后随机选择但仍避免短期重复 |

### 情绪识别

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **智能提取检索词** | `true` | 用轻量模型提取检索词和情绪先验；关闭后直接使用回复原文 |
| **情绪分析专用模型** | `""` | 留空使用当前会话模型 |
| **启用 JEV 选图** | `false` | 与“小模型分析”同时开启后由官方 JEV 从 Top 10 中选图；未勾选时保留传统 LLM 链路 |
| **TypeSafe API Key** | `""` | 在 console.typesafe.ai 获取；缺少 key 或 JEV 调用失败时回退小模型 |
| **TypeSafe API Base URL** | `https://api.typesafe.ai` | 默认使用官方 API；支持可信的兼容代理地址，API Key 会发送到该地址 |
| **情绪分析提示词** | `""` | 留空使用内置模板，支持 `{emotion_list}`、`{llm_reply}` 和 `{user_message}` |

### 模型配置

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **视觉模型** | `""` | 留空自动使用 AstrBot 全局图片描述模型 |
| **启用嵌入向量检索** | `false` | 开启后使用 FaissVecDB 做语义召回，不可用时自动降级 BM25 |
| **嵌入模型 ID** | `""` | 留空自动使用 AstrBot 首个 Embedding Provider；填写 Embedding 模型 ID，不填聊天模型或视觉模型 ID |

### 群聊过滤

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **发表情白名单** | `[]` | 使用 `group:群号` 或 `user:QQ号` |
| **发表情黑名单** | `[]` | 可与白名单同时生效 |
| **发表情名单优先级** | `whitelist_first` | 可选 `whitelist_first` 或 `blacklist_first` |
| **偷表情白名单** | `[]` | 使用 `group:群号` 或 `user:QQ号` |
| **偷表情黑名单** | `[]` | 可与白名单同时生效 |
| **偷表情名单优先级** | `whitelist_first` | 可选 `whitelist_first` 或 `blacklist_first` |

### 存储、提示词与智能选择

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **通用表情库自动淘汰上限** | `100` | 仅计算未收藏、无角色的普通表情；收藏、角色库和受保护导入不占额度 |
| **低使用次数权重** | `0.7` | `eviction_usage_weight`，入库时间权重为 `1 - 此值` |
| **VLM 分类提示词** | `""` | 自定义 VLM 分类提示词，留空使用内置 `prompts.json` |
| **VLM 分类提示词（带审核）** | `""` | 内容审核开启时使用，留空使用内置模板 |
| **文字距离融合预设** | `balanced` | `balanced`、`keyword`、`semantic`、`strict`；控制智能选择的匹配权重 |

### 外部表情包源

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **启用外部表情包源** | `true` | 控制资源包和 JSON API 导入功能 |
| **允许明文 HTTP 外部源** | `false` | 建议保持关闭；关闭时只接受 HTTPS |
| **外部导入默认进入待审核池** | `false` | 导入界面可逐次覆盖；内容审核开启时会强制进入审核池 |
| **单个外部源最大图片数** | `2000` | 限制预检与单次同步规模 |
| **单图 / 压缩包 / 解压后 / 像素上限** | `32 MiB / 1 GiB / 4 GiB / 4000 万` | 防止超大响应、压缩炸弹和超大图片 |

### WebUI

| 配置项 | 默认值 | 说明 |
|:---|:---|:---|
| **WebUI 默认主题** | `auto` | 可选 `auto`、`dark`、`light`、`minecraft`、`fallout`；页面内选择会持久化 |

## 🔄 情绪分析模式详解

| | LLM 模式（默认推荐） | 被动检索模式 |
|:---|:---|:---|
| **原理** | 轻量模型从 Bot 回复中提取检索词和 1~3 个情绪先验；发送仍由概率、冷却和意图门控决定 | 直接用 Bot 回复原文检索，不注入标签 |
| **对回复的影响** | ✅ 不修改 LLM 原生回复内容 | ✅ 不修改 LLM 原生回复内容 |
| **适用场景** | 希望增强匹配、愿意增加一次轻量模型调用 | Token 紧张或希望减少一次模型调用 |

角色归档在 WebUI 中手工完成，与情绪分类独立：VLM 负责语义，角色由你指定。

同时勾选“小模型分析”（`enable_natural_emotion_analysis`）和“JEV 选图”（`enable_jev`），填写 `typesafe_api_key` 后，
自动表情采用“本轮用户消息与助手回复 → 相关度粗筛 Top 10 → JEV 选择候选编号 → 发送”。
API Base URL 默认是 `https://api.typesafe.ai`，插件自动补全 `/v1/systemone`；也兼容以 `/v1`
结尾或直接填写完整 endpoint。模型固定为 `jev-latest`。修改 Base URL 时，API Key 会发送到该地址，
因此应只配置可信服务。
JEV 粗筛始终使用相关度排序，不受 `smart_meme_selection` 的随机模式影响；传统 LLM 链路继续沿用该开关。
粗筛先排除当前会话不可见或文件缺失的条目，不随机抽取候选；候选不足 10 张时按实际数量提供。
JEV 收到的是候选的图上文字、描述、分类、情绪、标签、场景和角色，不包含图片文件及本地路径。
同时会把最近 6 条用户/助手文本（约 3 轮，每条最多 400 字符）和本轮消息、回复（各最多 2000 字符）
发送到配置的 TypeSafe API Base URL。系统提示、工具结果和历史图片不发送；历史读取失败时使用本轮对话。

JEV 可选择 `none`，此时本轮不发送，也不回退随机选图；无候选时同样跳过。
超时（10 秒）、HTTP 错误或非法候选编号会回退原有小模型分析与选图链路。
发送概率、冷却、意图门控仍然有效，发送前再次核验候选文件与可见范围。
关闭小模型分析时恢复原文检索，JEV 开关也不生效；仅开启小模型分析时使用传统 LLM，无需 TypeSafe key。

## 🎮 指令列表

所有指令以 `/meme` 为前缀。

### 展示类指令（所有人可用）

| 指令 | 说明 |
|:---|:---|
| `status` | 查看运行状态和表情包统计 |
| `list [分类] [每页数量] [页码]` | 列出已收集的表情包（默认每页 10，页码从 1 开始） |
| `emotion_stats` | 查看情绪分析统计和当前模式 |

### 管理类指令（仅管理员）

| 指令 | 说明 |
|:---|:---|
| `on` / `off` | 开启 / 关闭表情包收集 |
| `auto_on` / `auto_off` | 开启 / 关闭自动发送 |
| `clean [force]` | 清理未分类的原始暂存图文件 |
| `偷` | 进入 30 秒强制收录模式，期间发送的图片直接入库 |
| `group show` | 查看当前偷表情/发表情名单配置 |
| `group <send\|steal> priority <wl\|bl>` | 设置白黑名单冲突时的优先级 |
| `group <send\|steal> <wl\|bl> <add\|del\|clear> [group:群号\|user:QQ号]` | 管理目标黑白名单策略 |
| `delete <序号\|文件名>` | 删除指定表情包 |
| `blacklist <序号\|文件名>` | 删除指定表情包并加入黑名单，禁止再次收录 |
| `scope <序号\|文件名> <public\|local>` | 设置表情包作用域 |
| `capacity` | 立即执行容量控制判断 |
| `rebuild_index` | 重建索引（版本迁移或索引异常时使用） |
| `natural_analysis <on\|off>` | 切换两套情绪识别模式 |
| `clear_emotion_cache` | 清空情绪分析结果缓存 |
| `tag_stats [N]` | 查看标签/场景统计，N 默认为 15 |

### LLM 工具调用（对话中自动触发）

| 工具 | 说明 |
|:---|:---|
| `search_meme` | 搜索候选表情包，返回分类、场景、作用域和使用次数等参考 |
| `send_meme` | 从候选列表选择并发送表情包，失败时返回明确 reason |
| `steal_meme` | 收录用户明确要求保存的图片；`image_ref` 可留空以使用当前消息第一张图片，VLM 自动完成分类、标签、描述和场景分析 |

## ⚠️ 注意事项

- WebUI 删除分类会同时删除该分类下的图片文件，请谨慎操作。
- 开启 `send_meme_as_gif` 时，超大图片转换为 GIF 会造成瞬时内存占用，低内存环境建议关闭。
- 插件底层需要可用的视觉模型；未配置时图片分类、自动收录和 VLM 复核无法完成。

### 📝 提示词与 GIF

- VLM 分类提示词使用严格 JSON 输出，字段包括分类、情绪、描述、标签、场景和图上文字。
- 自定义 VLM 提示词与情绪分析提示词均支持配置；留空时使用内置模板。
- GIF 分类从完整帧序列中等距抽取九帧，按时间顺序生成 3×3 分镜，并清理临时采样文件。
- 已有的管道符分隔响应仍保持兼容，JSON 格式更稳定。

### 表情库与自动淘汰

管理页提供通用表情库（原未分配）、收藏、角色表情库三个入口，支持按使用最少或入库最旧排序后手动批量删除。已收藏的角色表情显示在收藏库，角色标记保留；取消收藏后回到对应角色库。收藏和角色库均不参与自动淘汰，也不占 `max_reg_num` 额度。外部导入和 pinned 项保持已有保护。

超限时，只在可自动淘汰的通用表情中计算：`淘汰分 = w × 低使用分 + (1-w) × 最旧分`。两项分别按候选集的使用次数 `use_count` 和入库时间 `created_at` 做最小最大归一化（越少、越旧越接近 1；所有值相同时该项为 0）。默认 `w=0.7`，分数越高越先删除；同分依次按使用次数、入库时间、路径排序。`w=0` 优先最旧，`w=1` 优先使用最少。只删除超出额度的数量。

取消收藏或移除角色标记后，普通表情会重新计入通用库额度，下次容量控制可能淘汰它。定时容量检查、`/meme capacity` 和重建后的容量检查使用相同规则。入库时间和使用历史不因切换库而重置。

## 🚢 维护者发布流程

1. 在 `metadata.yaml` 更新版本，并在 `CHANGELOG.md` 增加同版本、带日期的章节。
2. 本地运行 pytest、Ruff、Python 编译、前端语法和发布脚本校验。
3. 提交并推送包含 `metadata.yaml` 变化的 commit 到 `master` 或 `main`，例如 `git commit -m "release: vX.Y.Z"` 后执行 `git push origin master`。
4. Release Action 只响应包含 `metadata.yaml` 的推送；它会检查版本严格递增、运行完整测试、构建 ZIP 与 `.sha256`，创建 `vX.Y.Z` 标签和 GitHub Release，并回下载校验哈希。
5. 在 Actions 页面手动运行 Release 时，`publish=false` 会按 `ref` 做校验、打包并上传 7 天临时产物，不创建标签或正式 Release；确认无误后再使用 `publish=true`。

普通代码提交和 PR 仍由 CI 检查，单独修改其他文件不会触发正式发布流程。

## 📄 许可证

本项目基于 [GNU AGPL v3.0](LICENSE) 许可证开源。

---

<div align="center">

好用的话给个 ⭐ Star 吧，谢谢大伙！

有问题欢迎提 [Issue](https://github.com/nagatoquin33/astrbot_plugin_stealer/issues) 或群里找我。

</div>
