# 🌟 表情包小偷

<div align="center">

<img src="https://count.getloli.com/@nagatoquin33?name=nagatoquin33&theme=rule34&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto" alt="Moe Counter">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.10.14%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
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
  - [📄 许可证](#-许可证)

## 📢 简介

我想仿照麦麦的表情包偷取做个娱乐性的插件的，于是就有了这个表情包小偷插件

全人机代码，不过本人一直在监工调试，但不保证什么问题都没有

我尽量做到在上架前让这个插件能正常工作

表情包小偷是一款基于多模态AI的娱乐性插件，能够自动偷取聊天中的图片，进行视觉理解与情绪分类，并在发送回复前按概率追加一张匹配情绪的表情，显著提升聊天互动体验。

如果偷够了可以金盆洗手，支持偷取功能单独开启或关闭

本插件设计灵活，支持自动使用当前会话的视觉模型，无需额外配置即可开始使用。




## 🌟 核心功能

*   **自动收集**: 默默保存群聊中的表情包（过滤色图/无关图片）。
*   **智能分类**: 利用 LLM/VLM 识别不仅是内容，更是**情绪** (happy, sad, angry...)。
*   **情感共鸣**: 结合 Bot 当前回复的情绪，自动发送匹配的表情包。
*   **不那么聪明的后置分析**：使用小参数量的llm进行语义分析，后置发送表情包。
*   **库腾堡贼王——亨利的战利品 (WebUI)**: 
    *   可视化查看、搜索、删除表情包。
    *   支持“永不再偷”功能，永久拉黑不喜欢的表情包。
    *   访问地址: `http://<ip>:8899/`

## 🚀 快速开始

1.  **安装**: 在 AstrBot 插件管理中安装 `astrbot_plugin_stealer`。
2.  **配置视觉模型** (必须):
    ```bash
    /meme set_vision <provider_id>  # 例如: doubao, gemini, qwenvl...
    ```
3.  **开启**:
    ```bash
    /meme on      # 开启偷图
    /meme auto_on # 开启自动发送
    ```

## ⚙️ 配置说明

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `max_reg_num` | 100 | 最大存储表情数量 (超过自动清理旧图) |
| `emoji_chance` | 0.2 | 自动回复时附带表情的概率 (0-1) |
| `raw_cleanup_interval` | 30 | 原始缓存清理间隔 (分钟) |
| `webui_port` | 8899 | WebUI 端口 |

## 🔄 模式切换说明

### LLM模式 vs 被动标签模式

| 模式 | 特点 | 适用场景 |
|:---|:---|:---|
| **LLM模式** (默认) | ✅ 不修改LLM回复<br>✅ 后台轻量模型分析语义<br>✅ 保持对话自然流畅 | 希望保持AI回复的原始性和完整性 |
| **被动标签模式** | ❌ LLM会插入情绪标签<br>✅ 直接获取LLM判断的情绪<br>❌ 需要清理消息链 | 需要精确控制情绪分类的场景 |

### ⚠️ 重要提醒

**切换模式后请使用 `/reset` 命令清除AI的对话上下文**

原因：
- 被动标签模式下，LLM学会了在回复开头插入 `&&emotion&&` 标签
- 切换到LLM模式后，LLM可能仍保留这种习惯性行为
- 使用 `/reset` 可以清除AI的对话记忆，避免继续输出残留标签

**操作步骤：**
```bash
# 1. 修改配置切换模式
# 2. 重启插件或重载配置
# 3. 清除AI上下文
/reset
```

这样能确保模式切换立即生效，避免新旧模式混淆。

## 🎮 指令列表 (前缀: `/meme`)

| 指令 | 说明 |
| :--- | :--- |
| `on` / `off` | 开启/关闭 表情包收集 |
| `auto_on` / `auto_off` | 开启/关闭 自动发送表情 |
| `set_vision <id>` | 设置视觉模型提供商 (Provider ID) |
| `status` | 查看当前统计与状态 |
| `push [分类] [别名]` | 手动发送一张表情 (不填则随机) |
| `偷` | 开启30秒强制收录窗口（无视概率，发送1张图将自动分类并入库） |
| `clean` | 清理未分类的原始缓存 |
| `rebuild_index` | 重建索引并恢复旧数据 |
| `debug_image` | [回复图片] 调试图片识别结果 |

### ⚠️ 分类删除提醒

WebUI「分类管理」中删除分类会 **同时删除该分类下所有图片**，并且无法恢复；操作前请确认无误。



好用就给个star吧谢谢大伙
有问题了群里找我或者提issue，我一般很快会看到

## 📄 许可证

本项目基于 MIT 许可证开源。

[![GitHub license](https://img.shields.io/github/license/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/blob/main/LICENSE)





