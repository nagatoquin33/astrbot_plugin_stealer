# 🌟 Emoji Stealer

<div align="center">

<img src="https://count.getloli.com/@nagatoquin33?name=nagatoquin33&theme=rule34&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto" alt="Moe Counter">

**Inspired by maibot's emoji-stealing and meme-manager's tag-injection systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%E2%89%A54.24.1-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
[![Last Commit](https://img.shields.io/github/last-commit/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/commits/master)

**Language / 语言**

[![中文](https://img.shields.io/badge/中文-README-lightgrey)](README.md)
[![English](https://img.shields.io/badge/English-current-blue)](README_EN.md)

</div>

---

## 📢 Introduction

An [AstrBot](https://github.com/AstrBotDevs/AstrBot) entertainment plugin powered by multimodal AI. Automatically steals emojis from chat, classifies them with a vision model, and sends a mood-matching emoji during conversations to make Bot replies feel more human. Stealing and auto-sending can be toggled independently.

This plugin is fully open-source and free. Issues and PRs are welcome.

## ✨ Core Features

| Feature | Description |
|:---|:---|
| **Auto Steal** | Monitor group chat images, auto-collect by probability or cooldown, with pending pool capacity control |
| **Pending Review Pool** | Auto-stolen images enter a review queue first; manual approval prevents low-quality images from polluting the library |
| **Smart Classification** | Use VLM to identify image content and classify by emotion preset categories |
| **Semantic Search** | FaissVecDB vector embedding matches emojis by meaning, not just keywords; auto-degrades to BM25 when unavailable |
| **Emotion Matching** | Analyze the emotion of Bot replies and append a matching emoji |
| **LLM Proactive Selection** | LLM can search and send the best emoji via tool calls during conversation |
| **Dual-Mode Emotion Analysis** | Smart keyword extraction (a lightweight model extracts keywords and emotion priors from the reply; it does not decide whether to send) / Raw-text retrieval (directly uses the reply text, no tags injected) |
| **WebUI Dual-Section** | Review Queue: manually approve/reject pending emojis / Library: browse, sort, batch manage |
| **Sort & Filter** | Most used / Recently used / Newest / Oldest — all via SQL ORDER BY |
| **Group Filtering** | Whitelist/blacklist control over which groups allow stealing/sending |

## 🚀 Quick Start

### 1. Installation

Search and install `astrbot_plugin_stealer` in the AstrBot plugin manager.

### 2. Prerequisites

**A vision model is required.** The plugin relies on VLM for image classification. You can configure the global image caption model in AstrBot, or specify `vision_provider_id` in the plugin config.

### 3. Getting Started

```
/meme on        # Enable emoji stealing
/meme auto_on   # Enable auto-sending
```

Done stealing?

```
/meme off       # Disable stealing (collected emojis remain available)
```

### 4. WebUI Management

Open the plugin detail panel in the AstrBot Dashboard and click "Emoji Manager" to access the management page.

- **Browse**: Filter by category, search, and sort collected emojis.
- **Scope Management**: `public` for shared library, `local` for origin-group-only sending.
- **Batch Operations**: Move categories, delete, set scopes, and repair origin targets in batch.
- **Single Upload**: Upload an image and use AI to auto-detect category and scene tags.
- **Batch Import**: Upload multiple emojis at once. **Category selection** and **Auto-analysis** are mutually exclusive:
  - **Select Category**: Saves images to the specified category without calling VLM.
  - **Auto-analysis**: VLM automatically identifies each image's category (high concurrent API calls).
- **Storage Maintenance**: Scan and clean stale index rows, orphan files, thumbnail cache, and temporary files.
- **Category Management**: Add, edit, and delete emoji categories.

> ⚠️ **High Concurrency Warning**: Auto-analysis processes multiple images concurrently and may trigger API rate limits. Batch your imports accordingly.

## 💡 Recommended Usage

### Fully Automatic (for token-rich setups)

**Enable auto-steal + auto-send**

1. Enable stealing: `/meme on`
2. Enable auto-send: `/meme auto_on`
3. The bot auto-collects and classifies group emojis
4. The bot appends mood-matched emojis to replies

> LLM mode: A lightweight model analyzes emotion after the reply, without modifying the reply content.
> Passive tag mode: The LLM directly tags emotion — faster but modifies the reply.

**LLM Proactive Selection**: The bot can search for the best emoji via tool calls during conversation.

### Semi-Automatic (for token-constrained setups)

**Manual emoji management**

1. Place emojis in `plugin_data/astrbot_plugin_stealer/categories/<category>/` manually
2. Or use WebUI batch upload with a specified category
3. The bot still auto-sends matching emojis (using existing categories only, no VLM calls)

### Precision Collection (for controlled setups)

**Targeted collection + WebUI management**

1. Use command: `/meme 偷` to enter 30-second forced collection mode
2. Or use WebUI batch import + auto-analysis
3. Manage via WebUI: view, edit, delete, set scopes

## ⚙️ Configuration

All settings can be modified in the AstrBot admin panel.

### Stealing Settings

| Setting | Default | Description |
|:---|:---|:---|
| **Enable emoji stealing** | `false` | Master toggle |
| **Steal mode** | `probability` | `probability` = roll chance per image / `cooldown` = enforce interval |
| **Steal probability** | `0.3` | Chance of stealing each received image (probability mode) |
| **Steal cooldown (seconds)** | `30` | Minimum interval between two steals (cooldown mode) |
| **Content filtration** | `false` | Filter inappropriate images; may increase processing time |
| **Allow import on audit service errors** | `false` | Policy hint for audit-service failures; explicitly rejected unsafe images still won't be imported |

### Sending Settings

| Setting | Default | Description |
|:---|:---|:---|
| **Auto-send emojis** | `false` | Auto-send emojis during conversation |
| **Auto-send intent gate** | `true` | Skip commands, serious/error replies, very short replies, and question-heavy messages |
| **Cancel pending auto-send on new message** | `true` | Cancel the previous delayed auto-send when a new message arrives in the same session |
| **Emoji send probability** | `0.2` | Probability of auto-sending (0.0 ~ 1.0) |
| **Send as GIF** | `false` | Send as GIF (closer to native emoji style, slightly higher memory) |
| **Send delay (seconds)** | `5.0` | Delay before sending to avoid conflicts with message segmentation plugins. Set to 0 for immediate. |
| **Random delay** | `false` | Random delay between [delay] ~ [max delay] for more natural timing |
| **Max random delay (seconds)** | `8.0` | Upper bound for random delay |
| **Smart emoji selection** | `true` | Composite scoring; disable for completely random picks |

### Emotion Recognition

| Setting | Default | Description |
|:---|:---|:---|
| **Emotion recognition mode** | `true` | `true` = LLM mode (recommended, doesn't pollute chat) / `false` = passive tag mode |
| **Emotion analysis model** | `""` | Lightweight model for LLM mode; leave blank to use the current session default |
| **Emotion analysis prompt** | `""` | Custom prompt for the lightweight emotion-analysis model; leave blank to use the built-in template |

### Model Configuration

| Setting | Default | Description |
|:---|:---|:---|
| **Vision model** | `""` | For image classification; leave blank to auto-use the global image caption model |

### Group Filtering

| Setting | Default | Description |
|:---|:---|:---|
| **Send whitelist** | `[]` | Format: `group:<id>` or `user:<id>`. Non-empty = only whitelisted targets. |
| **Send blacklist** | `[]` | Can coexist with whitelist; priority controls conflict resolution. |
| **Send filter priority** | `whitelist_first` | `whitelist_first` or `blacklist_first` |
| **Steal whitelist** | `[]` | Format: `group:<id>` or `user:<id>`. |
| **Steal blacklist** | `[]` | Can coexist with whitelist. |
| **Steal filter priority** | `whitelist_first` | `whitelist_first` or `blacklist_first` |

### Pending Pool & Embedding Search

| Setting | Default | Description |
|:---|:---|:---|
| **Pending pool capacity** | `200` | Auto-stolen images enter the pool first; stealing pauses when full. Resumes after manual review. Adjustable in real-time |
| **Enable embedding search** | `true` | FaissVecDB semantic similarity matching; auto-degrades to BM25 when disabled or unavailable |
| **Embedding model ID** | `""` | Leave blank to auto-use AstrBot's first Embedding Provider. Check AstrBot Admin → LLM Config for available embedding model IDs (e.g., `openai_embedding`) |

### Storage & Advanced

| Setting | Default | Description |
|:---|:---|:---|
| **Max emoji count** | `100` | Storage limit; oldest emojis are cleaned when exceeded |
| **VLM classification prompt** | `""` | Custom prompt for VLM emotion classification; leave blank to use bundled `prompts.json` |
| **VLM classification prompt (with filtration)** | `""` | Prompt used when content filtration is enabled; leave blank to use bundled `prompts.json` |

### WebUI

The management page is served through the AstrBot Dashboard plugin page system. Click "Emoji Manager" in the plugin detail panel to access it. No additional port or password configuration is needed.

Since v2.8.8, theme choices made in the WebUI are persisted. When `webui_theme` is changed in the plugin configuration, the previous page-level choice is invalidated automatically and the new configured default takes effect without clearing browser storage.

## 🔄 Emotion Analysis Modes

| | LLM mode (recommended) | Passive retrieval |
|:---|:---|:---|
| **How it works** | A lightweight model extracts search keywords and emotion priors from the reply; sending is still gated by probability/cooldown/intent rules | No tags are injected; the reply text is used as the search query |
| **Effect on replies** | ✅ Does not modify the LLM reply | ✅ Does not modify the LLM reply |
| **Best for** | Want stronger emoji matching and can afford one extra lightweight model call | Tight token budget, skip the extra model call |

Character archives are assigned by hand in the WebUI and are independent of emotion categories. VLM only writes semantics; you pick the character (for example neurosama).

## 🎮 Command Reference

All commands use the `/meme` prefix.

### Display Commands (everyone)

| Command | Description |
|:---|:---|
| `status` | View running status and emoji stats |
| `list [category] [page_size] [page]` | List collected emojis (default: 10 per page, page 1) |
| `emotion_stats` | View emotion analysis stats and current mode |

### Admin Commands (admins only)

| Command | Description |
|:---|:---|
| `on` / `off` | Enable / disable emoji collection |
| `auto_on` / `auto_off` | Enable / disable auto-sending |
| `clean [force]` | Clean unclassified raw staging images |
| `偷` | Enter 30-second forced collection mode |
| `group show` | View current send/steal filter config |
| `group <send\|steal> priority <wl\|bl>` | Set whitelist/blacklist conflict priority |
| `group <send\|steal> <wl\|bl> <add\|del\|clear> [group:<id>\|user:<id>]` | Manage group/user filter lists |
| `delete <index\|filename>` | Delete a specific emoji |
| `blacklist <index\|filename>` | Delete and add to blacklist, preventing re-collection |
| `scope <index\|filename> <public\|local>` | Set emoji scope; `local` is restricted to the origin group |
| `capacity` | Trigger capacity control immediately |
| `rebuild_index` | Rebuild the index (for version migration or corruption) |
| `natural_analysis <on\|off>` | Toggle between the two emotion recognition systems |
| `clear_emotion_cache` | Clear emotion analysis cache |

### LLM Tool Calls (triggered automatically in conversation)

| Tool | Description |
|:---|:---|
| `search_meme` | LLM searches for matching meme candidates with category, scene, scope, and usage hints |
| `send_meme` | LLM selects and sends a meme from the candidate list; failures include explicit reason codes |
| `steal_meme` | LLM imports an image when the user asks to collect it or the current message contains a suitable meme; `image_ref` may be omitted to use the first image in the current message, and VLM handles category, tags, description, and scene analysis |

## ⚠️ Notes

- **Deleting a category via WebUI** also deletes all associated image files. Use with caution.
- With `send_meme_as_gif` enabled, very large images converted to GIF may cause memory spikes. Turn it off on low-memory environments.
- A **vision model (VLM)** is required. Without one, the plugin's core functions won't work.

### 📝 Prompt Format Update

Since `v2.4.5+`, the core classification prompt uses a **strict JSON output format** for more stable and accurate parsing.

Starting with v2.9.0, the VLM classification prompts and the lightweight LLM emotion-analysis prompt are configurable again: a non-empty custom value overrides the bundled prompt, while leaving it blank uses the built-in template.

The old pipe-delimited format is still compatible, but the JSON format is more reliable.

## 📄 License

This project is open-source under the [MIT](LICENSE) license.

---

<div align="center">

If you find this useful, please give it a ⭐ Star — thank you!

Report issues at [GitHub Issues](https://github.com/nagatoquin33/astrbot_plugin_stealer/issues) or find me in the community group.

</div>
