# 🌟 表情包小偷

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.10.14%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen)](CONTRIBUTING.md)
[![Last Commit](https://img.shields.io/github/last-commit/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/commits/main)

</div>

## 📑 目录

- [🌟 表情包小偷](#-表情包小偷)
  - [📑 目录](#-目录)
  - [📢 简介](#-简介)
  - [🚀 功能特点](#-功能特点)
  - [📦 安装方法](#-安装方法)
  - [🛠️ 快速上手](#️-快速上手)
  - [⚙️ 配置说明](#️-配置说明)
  - [🎮 使用指令](#-使用指令)
  - [🗑️ 文件清理机制](#️-文件清理机制)
  - [🎯 图片处理节流](#-图片处理节流)
  - [⚠️ 注意事项](#️-注意事项)
  - [📄 许可证](#-许可证)

## 📢 简介

我想仿照麦麦的表情包偷取做个娱乐性的插件的，于是就有了这个表情包小偷插件

全人机代码，不过本人一直在监工调试，但不保证什么问题都没有

我尽量做到在上架前让这个插件能正常工作

表情包小偷是一款基于多模态AI的娱乐性插件，能够自动偷取聊天中的图片，进行视觉理解与情绪分类，并在发送回复前按概率追加一张匹配情绪的表情，显著提升聊天互动体验。

如果偷够了可以金盆洗手，支持偷取功能单独开启或关闭

本插件设计灵活，支持自动使用当前会话的视觉模型，无需额外配置即可开始使用。

## 📝 更新历史
version: 1.0.2：修改正则，现在应该不会吞掉换行和[]()这一类符号了，同时加强对&&的匹配
version: 1.0.3：修复指令状态，去除无用指令，改进状态显示


## 🚀 功能特点

### 核心功能
- **自动图片偷取**：实时监听聊天中的图片消息并自动保存
- **AI多模态理解**：使用视觉模型生成图片描述、标签与情绪分类
- **情绪智能匹配**：支持开心、悲伤、愤怒、惊讶等多种情绪分类（基于提示词，不保证真的100%准确）
- **自动表情追加**：发送回复前按概率追加匹配情绪的表情，提升互动性
- **内容安全审核**：可选开启内容审核功能，过滤不符合要求的图片
- **智能存储管理**：自动清理过期文件，优化存储空间使用

### 🆕 新功能

#### 🎯 智能节流系统
- **4种节流模式**：probability（概率）、interval（间隔）、cooldown（冷却）、always（总是）
- **大幅减少API消耗**：最高可减少90%的VLM调用
- **灵活配置**：可根据API额度和需求自由调整
- **实时生效**：无需重启，配置即时生效

#### 🔧 独立后台任务
- **Raw清理任务**：独立的原始图片清理任务，可单独配置周期和启用状态
- **容量控制任务**：独立的表情包数量控制任务，可单独配置周期和启用状态
- **灵活控制**：两个任务互不影响，可按需启用
- **精细调整**：分别设置不同的执行周期

### 未来功能计划

我们计划在后续版本中添加以下功能：

- **自定义表情库**：支持用户上传和管理个人表情库，增加更多情感表达
- **手动调整表情的标签**：用户可以手动调整已偷取的图片的标签，修正分类错误

### 技术特性
- **自动模型选择**：未指定视觉模型时，自动使用当前会话的视觉模型
- **灵活配置选项**：支持通过指令或配置文件调整各项功能参数
- **后台维护机制**：定期执行容量控制和文件清理，确保系统稳定
- **错误容错处理**：完善的异常处理机制，保障插件稳定运行（大概？）
- **代码质量保证**：通过ruff格式化和检查，符合PEP8规范

## 📦 安装方法

1. 确保已安装 AstrBot
2. 将插件复制到 AstrBot 的插件目录：`AstrBot/data/plugins`
3. 或在 Dashboard 插件中心直接安装
4. 重启 AstrBot 或使用热加载命令

## 🛠️ 快速上手

### 1. 模型配置

设置视觉模型（用于图片分类）：

```bash
/meme set_vision <provider_id>
```

> **智能模型选择**：若未设置视觉模型，插件会自动使用当前会话的视觉模型，无需额外配置即可开始使用
> 
> **支持模型**：需要支持图片输入的视觉模型（如 Gemini, 豆包, qwen vl 等）

### 2. 功能开启

```bash
# 开启/关闭插件
/meme on
/meme off

# 开启/关闭自动随聊表情
/meme auto_on
/meme auto_off
```

### 3. 节流配置（推荐）✨

为了避免过度消耗API，建议配置节流：

```bash
# 推荐配置：概率模式，30%处理概率
/meme throttle_mode probability
/meme throttle_probability 0.3

# 查看节流状态
/meme throttle_status
```

### 4. 基本使用

```bash
# 查看插件状态
/meme status

# 查看后台任务状态
/meme task_status
```

## ⚙️ 配置说明

### 基础配置

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `auto_send` | bool | true | 是否自动随聊追加表情包 |
| `vision_provider_id` | string | null | 视觉模型提供商ID，未设置时自动使用当前会话模型 |
| `emoji_chance` | float | 0.4 | 触发表情动作的基础概率 |
| `max_reg_num` | int | 100 | 允许注册的最大表情数量 |
| `do_replace` | bool | true | 达到上限时是否替换旧表情 |
| `steal_emoji` | bool | true | 是否开启表情包偷取功能 |
| `content_filtration` | bool | false | 是否开启内容审核 |
| `raw_retention_minutes` | int | 60 | raw目录中图片的保留期限（分钟） |

### 节流配置 🆕

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `image_processing_mode` | string | probability | 图片处理模式：always/probability/interval/cooldown |
| `image_processing_probability` | float | 0.3 | 概率模式下的处理概率（0.0-1.0） |
| `image_processing_interval` | int | 60 | 间隔模式下的处理间隔（秒） |
| `image_processing_cooldown` | int | 30 | 冷却模式下的冷却时间（秒） |

### 后台任务配置 🆕

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `raw_cleanup_interval` | int | 30 | raw目录清理任务的执行周期（分钟） |
| `capacity_control_interval` | int | 60 | 容量控制任务的执行周期（分钟） |
| `enable_raw_cleanup` | bool | true | 是否启用raw目录自动清理 |
| `enable_capacity_control` | bool | true | 是否启用容量自动控制 |

## 🎮 使用指令

### 基础指令

| 指令 | 描述 |
|------|------|
| `/meme on` | 开启偷表情包功能 |
| `/meme off` | 关闭偷表情包功能 |
| `/meme auto_on` | 开启自动随聊表情 |
| `/meme auto_off` | 关闭自动随聊表情 |
| `/meme status` | 查看插件状态和表情包统计 |
| `/meme set_vision <provider_id>` | 设置视觉模型 |
| `/meme clean` | 手动触发清理 |

### 高级配置指令

| 指令 | 描述 |
|------|------|
| `/meme throttle [action] [value]` | 配置图片处理节流 |
| `/meme task [type] [action] [value]` | 配置后台任务 |

#### 节流配置示例
```bash
/meme throttle                    # 查看节流状态
/meme throttle mode probability   # 设置概率模式
/meme throttle probability 0.3    # 设置30%处理概率
/meme throttle interval 60        # 设置60秒间隔
/meme throttle cooldown 30        # 设置30秒冷却
```

#### 任务配置示例
```bash
/meme task cleanup on             # 启用清理任务
/meme task cleanup interval 30    # 设置清理周期30分钟
/meme task capacity off           # 禁用容量控制
/meme task capacity interval 60   # 设置容量控制周期60分钟
```

### 管理员指令

| 指令 | 描述 |
|------|------|
| `/meme push <category> [alias]` | 推送指定分类的表情包 |
| `/meme debug_image` | 调试图片处理 |

## 🗑️ 文件清理机制

插件实现了灵活的文件清理机制，确保不会占用过多存储空间。

### 🆕 独立任务系统

清理机制已拆分为两个独立任务：

#### 1. Raw目录清理任务 📁
- **功能**：定期清理过期的原始图片
- **默认周期**：30分钟
- **保留期限**：60分钟
- **控制**：可单独启用/禁用

#### 2. 容量控制任务 📊
- **功能**：定期检查并控制表情包数量
- **默认周期**：60分钟
- **数量上限**：100张
- **控制**：可单独启用/禁用

### 配置示例

```bash
# 只启用raw清理，不控制容量
/meme raw_cleanup on
/meme capacity_control off

# 设置不同的执行周期
/meme raw_cleanup_interval 15    # 15分钟清理一次
/meme capacity_interval 120      # 2小时控制一次
```

### 工作流程

1. **Raw清理任务**：
   - 每30分钟执行一次（可配置）
   - 删除超过60分钟的原始图片
   - 只在 `steal_emoji` 和 `enable_raw_cleanup` 都启用时执行

2. **容量控制任务**：
   - 每60分钟执行一次（可配置）
   - 检查表情包数量是否超过上限
   - 超过时删除最不常用的表情
   - 只在 `steal_emoji` 和 `enable_capacity_control` 都启用时执行

## 🎯 图片处理节流

为了避免过度调用VLM API，插件提供了智能节流功能。

### 四种节流模式

| 模式 | 说明 | API消耗 | 适用场景 |
|-----|------|---------|---------|
| **probability** | 按概率随机处理 | ⭐⭐ 可控 | **日常使用（推荐）** |
| **interval** | 每N秒处理一次 | ⭐ 最低 | API额度紧张 |
| **cooldown** | 两次间隔至少N秒 | ⭐⭐ 较低 | 避免刷屏 |
| **always** | 每张都处理 | ⭐⭐⭐⭐⭐ 最高 | API充足 |

### 推荐配置 ✨

```bash
# 平衡模式（推荐）- 减少70%的API调用
/meme throttle_mode probability
/meme throttle_probability 0.3

# 节省模式 - 减少90%的API调用
/meme throttle_mode probability
/meme throttle_probability 0.1

# 严格限制 - 每5分钟最多处理一次
/meme throttle_mode interval
/meme throttle_interval 300
```

### 效果对比

假设每天收到100张图片：

| 配置 | 处理数量 | API节省 |
|-----|---------|---------|
| always | 100张 | 0% |
| probability 30% | ~30张 | 70% ⬇️ |
| probability 10% | ~10张 | 90% ⬇️ |
| interval 300秒 | 取决于分布 | 大幅减少 |

## ⚠️ 注意事项

- 插件支持自动使用当前会话的视觉模型，无需额外配置即可开始体验
- 视觉模型调用会消耗相应的API token，**强烈建议配置节流功能**以减少API消耗
- 实验性插件，不同的视觉模型对表情包分类的效果可能不同，建议根据实际情况选择合适的模型
- 如果有对我目前插件中提示词有优化想法的可以提个issue我测试测试
- 推荐使用 `probability` 模式 + 30%概率，可以在保持功能的同时大幅减少API消耗

## 📚 详细文档

- [图片处理节流功能使用指南](image_processing_throttle_guide.md)
- [独立后台任务功能使用指南](independent_tasks_guide.md)
- [节流功能快速总结](throttle_feature_summary.md)

## 📄 许可证

本项目基于 MIT 许可证开源。

[![GitHub license](https://img.shields.io/github/license/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/blob/main/LICENSE)

---

<div align="center">

**🎭 Happy meme collecting! 🎭**

如果觉得这个插件有用，欢迎给个 ⭐ Star！

</div>
