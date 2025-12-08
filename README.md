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
  - [📝 使用指令](#-使用指令)
  - [⚠️ 注意事项](#️-注意事项)
  - [📄 许可证](#-许可证)

## 📢 简介
我想仿照麦麦的表情包偷取做个娱乐性的插件的，于是就有了这个表情包小偷插件
全人机代码，不过本人一直在监工调试，但不保证什么问题都没有
我尽量做到在上架前让这个插件能正常工作
表情包小偷是一款基于多模态AI的娱乐性插件，能够自动偷取聊天中的图片，进行视觉理解与情绪分类，并在发送回复前按概率追加一张匹配情绪的表情，显著提升聊天互动体验。
如果偷够了可以金盆洗手，支持偷取功能单独开启或关闭
本插件设计灵活，支持自动使用当前会话的视觉模型，无需额外配置即可开始使用。


## 🚀 功能特点

### 核心功能
- **自动图片偷取**：实时监听聊天中的图片消息并自动保存
- **AI多模态理解**：使用视觉模型生成图片描述、标签与情绪分类
- **情绪智能匹配**：支持开心、悲伤、愤怒、惊讶等多种情绪分类（基于提示词，不保证真的100%准确）
- **自动表情追加**：发送回复前按概率追加匹配情绪的表情，提升互动性
- **内容安全审核**：可选开启内容审核功能，过滤不符合要求的图片
- **智能存储管理**：自动清理过期文件，优化存储空间使用

### 未来功能计划

我们计划在后续版本中添加以下功能：

- **自定义表情库**：支持用户上传和管理个人表情库，增加更多情感表达
- **手动调整表情的标签**：用户可以手动调整已偷取的图片的标签，修正分类错误


### 技术特性
- **自动模型选择**：未指定视觉模型时，自动使用当前会话的视觉模型
- **灵活配置选项**：支持通过指令或配置文件调整各项功能参数
- **后台维护机制**：定期执行容量控制和文件清理，确保系统稳定
- **错误容错处理**：完善的异常处理机制，保障插件稳定运行（大概？）

## 📦 安装方法

1. 确保已安装 AstrBot
2. 将插件复制到 AstrBot 的插件目录：`AstrBot/data/plugins`
3. 或在 Dashboard 插件中心直接安装
4. 重启 AstrBot 或使用热加载命令

## 🛠️ 快速上手

### 1. 模型配置

设置视觉模型（用于图片分类）：

> **智能模型选择**：若未设置视觉模型，插件会自动使用当前会话的视觉模型，无需额外配置即可开始使用
> 
> **支持模型**：需要支持图片输入的视觉模型（如 Gemini, 豆包, qwen vl 等）

### 2. 功能开启

开启插件：
```
meme on
```
关闭插件：
```
meme off
```
开启自动随聊表情：
```
meme auto_on
```
关闭自动随聊表情：
```
meme auto_off
```

### 3. 基本使用

查看当前插件状态：
```
meme status
```

## ⚙️ 配置说明

插件提供以下配置选项，可在后台或配置文件中设置：

| 配置项 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `auto_send` | bool | true | 是否自动随聊追加表情包 |
| `vision_provider_id` | string | null | 视觉模型提供商ID（用于图片分类），未设置时自动使用当前会话模型 |
| `emoji_chance` | float | 0.4 | 触发表情动作的基础概率 |
| `max_reg_num` | int | 100 | 允许注册的最大表情数量 |
| `do_replace` | bool | true | 达到上限时是否替换旧表情 |
| `maintenance_interval` | int | 10 | 后台维护任务的执行周期（分钟），包括容量控制和文件清理 |
| `steal_emoji` | bool | true | 是否开启表情包偷取和清理功能（关闭后将停止偷取新图片并暂停所有清理操作） |
| `content_filtration` | bool | false | 是否开启内容审核 |
| `raw_retention_hours` | int | 1 | raw目录中图片的保留期限（小时） |
| `raw_clean_interval` | int | 30 | （已弃用）raw目录的清理时间间隔（分钟） |

## 🗑️ 文件清理机制

插件实现了自动的文件清理机制，确保不会占用过多存储空间。清理机制与偷图功能绑定，由以下配置项协同工作：

### 配置项协同原理

1. **核心开关 (`steal_emoji`)**：
   - 控制偷图和清理功能的总开关
   - 关闭后将停止偷取新图片并暂停所有清理操作

2. **保留期限 (`raw_retention_hours`)**：
   - 定义文件在系统中可以保留的最长时间
   - 默认1小时，超过此时间的文件会被视为"过期文件"

3. **清理时间间隔 (`raw_clean_interval`)**：
   - （已弃用）原用于定义清理操作的执行频率
   - 现在清理操作会在每次维护任务执行时直接进行

### 工作流程

1. 当`steal_emoji`开启时，系统每经过`maintenance_interval`分钟执行一次维护任务
2. 每次维护任务执行时，会自动检查raw目录中所有文件的修改时间
3. 删除所有修改时间超过`raw_retention_hours`小时的文件

### 清理范围

- **raw目录**：存放原始图片文件
- **已分类的图片**：通过容量控制机制(`max_reg_num`和`do_replace`)进行管理

### 示例（默认配置）

- 系统每10分钟执行一次维护任务（包括文件清理）
- 每次维护时，删除所有超过1小时的文件
- 这样确保了文件不会无限期保留，同时避免了过于频繁的清理操作影响性能

### 自定义配置示例

如果需要更频繁的清理：
1. 设置`maintenance_interval`为5（每5分钟执行一次维护任务）
2. 设置`raw_retention_hours`为0.5（删除超过30分钟的文件）
3. 重启插件或等待配置自动生效


## 📝 使用指令

| 指令 | 描述 |
|------|------|
| `meme on` | 开启偷表情包功能 |
| `meme off` | 关闭偷表情包功能 |
| `meme set_vision <provider_id>` | 设置视觉模型 |
| `meme show_providers` | 查看当前模型配置 |
| `meme auto_on` | 开启自动随聊表情 |
| `meme auto_off` | 关闭自动随聊表情 |
| `meme status` | 查看当前插件状态 |
| `meme push <category> [alias]` | 推送指定分类的表情包（管理员指令） |
| `meme debug_image` | 调试图片处理（管理员指令） |

## ⚠️ 注意事项

- 插件支持自动使用当前会话的视觉模型，无需额外配置即可开始体验
- 视觉模型调用会消耗相应的API token，建议根据实际使用情况调整配置
- 实验性插件，不同的视觉模型对表情包分类的效果可能不同，建议根据实际情况选择合适的模型
- 如果有对我目前插件中提示词有优化想法的可以提个issue我测试测试

## 📄 许可证

本项目基于 MIT 许可证开源。

[![GitHub license](https://img.shields.io/github/license/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/blob/main/LICENSE)


