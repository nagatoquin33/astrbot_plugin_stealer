# 🌟 Emoji Stealer

<div align="center">

<img src="https://count.getloli.com/@nagatoquin33?name=nagatoquin33&theme=rule34&padding=7&offset=0&align=top&scale=1&pixelated=1&darkmode=auto" alt="Moe Counter">

**Let your Bot collect memes from chat, understand their mood, and send the right one at the right moment.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)
![AstrBot](https://img.shields.io/badge/AstrBot-%E2%89%A54.24.1-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)
[![CI](https://github.com/nagatoquin33/astrbot_plugin_stealer/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/nagatoquin33/astrbot_plugin_stealer/actions/workflows/ci.yml)
[![Last Commit](https://img.shields.io/github/last-commit/nagatoquin33/astrbot_plugin_stealer)](https://github.com/nagatoquin33/astrbot_plugin_stealer/commits/master)

**Language / 语言**

[![中文](https://img.shields.io/badge/中文-README-lightgrey)](README.md)
[![English](https://img.shields.io/badge/English-current-blue)](README_EN.md)

</div>

---

## 📢 Introduction

Inspired by maibot and meme-manager, this [AstrBot](https://github.com/AstrBotDevs/AstrBot) entertainment plugin provides LLM-callable meme tools.

Emoji Stealer collects images from chat, uses a vision model for semantic and emotion labels, and sends a matching meme according to probability, cooldown, intent, and target-filter rules. Collection and auto-send can be toggled independently.

The plugin is open source and free to use. Issues and pull requests are welcome at [GitHub](https://github.com/nagatoquin33/astrbot_plugin_stealer/issues).

## ✨ Core Features

| Feature | Description |
|:---|:---|
| **Auto Steal** | Monitor group-chat images and collect them by probability or cooldown, with a pending-pool capacity limit |
| **Pending Review Pool** | Route automatically collected images through a review queue before adding them to the library |
| **Smart Classification** | VLM extracts image text, description, scenes, and emotion; GIFs use nine evenly sampled frames in a 3×3 timeline storyboard |
| **Semantic Search** | Search over visible text, descriptions, and reply scenes; optional remote Embedding improves recall, with BM25 fallback |
| **Emotion Matching** | Analyze the Bot reply and append a mood-matched meme after it |
| **LLM Proactive Selection** | `search_meme`, `send_meme`, and `steal_meme` let the LLM search, send, and collect memes |
| **Dual-Mode Emotion Analysis** | Extract keywords and emotion priors with a lightweight model, or search directly with the reply text; neither mode rewrites the reply |
| **VLM A/B Review** | Re-analyze an image in the WebUI and compare the current labels with the new result before applying it |
| **Character Library** | Assign a whole series to an existing or newly created character in the WebUI; characters are independent of emotion categories |
| **External Meme Sources** | Import Meme Manager / AstrBot Meme Packs, GitHub packs, or paginated HTTPS JSON catalogs with preflight, mapping, deduplication, and provenance |
| **Two-Part WebUI** | Review Queue handles pending images; Library supports browsing, sorting, and batch operations |
| **Group Filtering** | Configure separate send/steal whitelists, blacklists, and conflict priorities |

## 🚀 Quick Start

### 1. Install or update

- Search for `astrbot_plugin_stealer` in the AstrBot plugin manager and install it.
- For a manual install or update, download `astrbot_plugin_stealer-vX.Y.Z.zip` from a GitHub Release. Extract the top-level `astrbot_plugin_stealer/` directory into the AstrBot plugin directory, then restart AstrBot. Replacing the plugin code keeps the library and database under `plugin_data/astrbot_plugin_stealer/`; avoid an extra nested directory.

### 2. Prerequisites

**A vision model is required.** Use AstrBot's global image-caption provider or set `vision_provider_id` in the plugin configuration. Embedding search is optional and requires an available Embedding Provider when enabled.

### 3. Getting started

```text
/meme on        # Enable emoji stealing
/meme auto_on   # Enable auto-sending
```

Pause collection at any time:

```text
/meme off       # Disable stealing; collected memes remain available
```

### 4. WebUI management


Open the plugin detail panel in the AstrBot Dashboard and click **Emoji Manager**. No extra port or password is needed.

- **Browse**: Filter by category, search, and sort collected memes.
- **Scope**: `public` is shared; `local` restricts sending to the source group.
- **Review Queue**: Approve or delete pending images in batches and inspect failure reasons.
- **Single upload**: Upload an image and let AI detect category, description, tags, and scenes.
- **VLM A/B review**: Click re-analyze in the detail view. Apply the new category, visible text, tags, scenes, or emotions only after confirmation; character, scope, favorites, and usage history are preserved.
- **Batch re-analysis**: Click **Batch**, select images, then **Batch re-analysis**. Background analysis only prepares results; it never saves automatically. Review each image and its before/after description, tags, scenes, emotions and image text, then choose **Save this image** or **Keep original**. Categories, characters, favorites, provenance and scope are preserved. Saving checks for edits or deletions since analysis began. Progress, stopping and failure retries are supported; retries retain other results awaiting review. Closing the page keeps analysis active and reloading restores review. Restarting the plugin ends the task and clears unsaved results. Each batch supports up to 500 images, with one active batch at a time. Resolve pending results before starting a new batch.
- **Batch import**: Category selection and auto-analysis are mutually exclusive. Selecting a category skips VLM; auto-analysis processes images sequentially using the configured model request interval.
- **Maintenance**: Scan and clean stale index rows, orphan files, thumbnail cache, and temporary files.
- **Categories**: Add, edit, and delete categories.
- **Themes**: Choose host default, dark, light, Minecraft, or Fallout. The page choice persists, while `webui_theme` supplies the default.

## 🔌 v3 External Meme Sources

Open **External Sources** in the WebUI to:

- Discover Meme Manager v4 packs in the same AstrBot instance.
- Upload AstrBot Meme Pack ZIP / `.meme-pack` exports, including generic ZIPs with a `memes/` directory.
- Enter a GitHub repository (`owner/repo` or an HTTPS URL) and read a branch or subdirectory, such as [DDZS987/astrbot-meme-pack-semantic-01](https://github.com/DDZS987/astrbot-meme-pack-semantic-01). GitHub sources use public archives and never execute repository code.
- Register a paginated HTTPS JSON catalog for later synchronization.
- Map source categories to local categories and choose direct import or Pending Review.
- Assign an entire character series to an existing character or create a new one.
- Inspect imported, duplicate, failed, and stale entries. Missing remote items are marked stale while the local copy remains available.

Recommended pack layout:

```text
pack/
├── manifest.json                 # pack metadata
├── memes/<category>/<image>      # required image directory
├── meme_pack_export.json         # optional export metadata
└── semantic_metadata.json        # optional descriptions, tags, OCR, remote hashes
```

A plain ZIP of supported images is accepted; categories are inferred from the first directory below `memes/` or from the image parent directory, and `previews/` or thumbnail directories are skipped. Each image is validated for format, size, and pixel count before being copied into this plugin's storage. Source files stay untouched, SHA-256 prevents duplicate copies, and license, attribution, and source URLs are retained as provenance. External imports have a protected retention class and do not evict chat-collected memes.

HTTPS is required by default. Loopback, private, link-local, and reserved hosts, unsafe redirects, oversized responses, archive traversal, symlinks, decompression bombs, and excessive pixel counts are rejected. See the [External Source Protocol](docs/external-sources.md) for the JSON contract, pagination, and limits.

## 💡 Recommended Usage

### Fully automatic (for token-rich setups)

1. Enable stealing: `/meme on`
2. Enable auto-send: `/meme auto_on`
3. The Bot collects and classifies group-chat images.
4. After each reply, intent, probability, and cooldown gates decide whether to append a matching meme.

LLM emotion mode uses a lightweight model to extract search terms and emotion priors. Passive retrieval searches the reply text directly. Automatic sending is controlled by probability, cooldown, and intent gates. The LLM can also call the meme tools proactively.

### Semi-automatic (for token-constrained setups)

1. Place images under `plugin_data/astrbot_plugin_stealer/categories/<category>/`.
2. Or use WebUI batch upload with a chosen category.
3. Auto-send uses existing categories without an additional VLM call.

### Precision collection (for controlled setups)

1. Use `/meme 偷` to enter 30-second forced collection mode; images received during that window go straight to the library.
2. Or use WebUI batch import with auto-analysis.
3. Review, edit, delete, and scope images from the Review Queue and Library pages.

## ⚙️ Configuration

All public settings can be changed in the AstrBot admin panel. Defaults below match `_conf_schema.json` and the runtime configuration.

<details open>
<summary><strong>Stealing Settings</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **Enable emoji stealing** | `false` | Master toggle |
| **Steal mode** | `probability` | `probability` rolls per image; `cooldown` keeps at least 30 seconds between collections |
| **Steal probability** | `0.3` | Collection probability in probability mode |
| **Content filtration** | `false` | Use VLM to filter inappropriate images; adds processing time |
| **Pending pool capacity** | `200` | Stealing pauses at this many pending images and resumes after review |
| **Require manual review for auto-steal** | `true` | Save originals before vision calls. Approval immediately queues background analysis; successful results enter the library. With review off, automatic analysis continues |
| **QQ_Official collection mode** | `cdn_only` | `all_images` collects every image; `cdn_only` uses emoji CDN markers; `gif_only` keeps GIFs only |

</details>

<details open>
<summary><strong>Sending Settings</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **Auto-send emojis** | `true` | Send a meme after eligible Bot replies |
| **Auto-send intent gate** | `true` | Skip commands, error or serious replies, very short replies, and question-heavy content |
| **Cancel pending auto-send on new message** | `true` | Cancel the previous delayed send when a new message arrives in the same session |
| **Emoji send probability** | `0.2` | Auto-send probability (0.0 ~ 1.0) |
| **Send as GIF** | `false` | Force GIF output; large images use more transient memory |
| **Send as QQ sticker** | `true` | On aiocqhttp/NapCat, show a custom QQ sticker; otherwise send a normal image |
| **Send delay (seconds)** | `5.0` | Delay to avoid message-segmentation conflicts; 0 sends immediately |
| **Random delay** | `false` | Randomize between the fixed delay and the maximum delay |
| **Maximum random delay (seconds)** | `8.0` | Upper bound for random delay |
| **Smart emoji selection** | `true` | Use composite matching scores; disabled mode picks randomly while avoiding short-term repeats |

</details>

<details>
<summary><strong>Emotion Recognition</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **Understand conversation context with a model** | `true` | Extract sticker usage scenes from your reply and the user message; if no model is set, the current session model is used. Adds model calls; disabled mode uses local matching |
| **Context understanding model (optional)** | `""` | Leave blank to use the current session model; a fast text model is enough, and JEV in advanced settings can replace this step |
| **Enable JEV selection** | `false` | Turned on together with "Understand conversation context with a model", JEV picks from the top 10 pre-filtered candidates; otherwise the traditional LLM path is kept |
| **TypeSafe API Key** | `""` | Create one at console.typesafe.ai; a missing key or a failed JEV call falls back to the traditional LLM |
| **TypeSafe API Base URL** | `https://api.typesafe.ai` | The official API by default; trusted compatible proxies are accepted, and the API key is sent to this address |
| **JEV model ID** (`typesafe_model_id`) | `""` | Leave blank or whitespace-only for `jev-latest`; surrounding whitespace is trimmed. Compatible proxies can use their required model ID, such as `typesafe/jev` for Command Code |
| **Emotion analysis prompt** | `""` | Leave blank for the bundled context prompt; supports `{emotion_list}`, `{llm_reply}`, and `{user_message}` |

JEV requests use the HTTP/HTTPS proxy settings AstrBot writes to the environment and respect the `no_proxy` bypass list. No separate network proxy setting is needed in the plugin.

</details>

<details>
<summary><strong>Model Configuration</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **Vision model** | `""` | Leave blank to use AstrBot's global image-caption provider |
| **Enable embedding search** | `false` | Use FaissVecDB semantic retrieval; falls back to BM25 when unavailable |
| **Minimum model request interval (seconds)** | `5.0` | Vision and embedding requests run serially per provider. Increase for low free-tier quotas; configurable in advanced settings |
| **Embedding model ID** | `""` | Leave blank for AstrBot's first Embedding Provider; enter an Embedding model ID, not a chat or vision model ID |

Embedding dimensions are detected from actual vectors. Model or dimension changes rebuild vectors in the background while local matching remains available. If the API rejects the `dimensions` request parameter, disable that parameter or enter a supported value in the AstrBot provider settings.

Approval returns before vision analysis, allowing continuous review. The review area shows queued, analyzing and failed images. Failed originals are kept; approve again to retry. Jobs recover after restart and remain within the pending capacity limit. Legacy pending images and external imports with metadata keep their existing promotion flow.

</details>

<details>
<summary><strong>Group Filtering</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **Send whitelist** | `[]` | Use `group:<id>` or `user:<id>` |
| **Send blacklist** | `[]` | Can coexist with the whitelist |
| **Send filter priority** | `whitelist_first` | `whitelist_first` or `blacklist_first` |
| **Steal whitelist** | `[]` | Use `group:<id>` or `user:<id>` |
| **Steal blacklist** | `[]` | Can coexist with the whitelist |
| **Steal filter priority** | `whitelist_first` | `whitelist_first` or `blacklist_first` |

</details>

<details>
<summary><strong>Storage, Prompts, and Smart Selection</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **General library eviction limit** | `100` | Counts ordinary general stickers only; favorites, character stickers and protected imports are excluded |
| **Low-usage weight** | `0.7` | `eviction_usage_weight`; creation-age weight is `1 - weight` |
| **VLM classification prompt** | `""` | Custom VLM prompt; blank uses bundled `prompts.json` |
| **VLM classification prompt (with filtration)** | `""` | Used when content filtration is enabled; blank uses the bundled template |
| **Text-distance weight preset** | `balanced` | `balanced`, `keyword`, `semantic`, or `strict` for smart-selection weights |

</details>

<details>
<summary><strong>External Meme Sources</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **Enable external meme sources** | `true` | Enable pack and JSON-catalog imports |
| **Allow plain HTTP sources** | `false` | Keep disabled to require HTTPS |
| **Default external import to Review Queue** | `false` | The import dialog can override this; content filtration still forces review |
| **Maximum images per source** | `2000` | Limit preflight and one synchronization run |
| **Per-image / archive / expanded / pixel limit** | `32 MiB / 1 GiB / 4 GiB / 40 million` | Protect against oversized responses, decompression bombs, and huge images |

</details>

<details>
<summary><strong>WebUI</strong></summary>

| Setting | Default | Description |
|:---|:---|:---|
| **WebUI default theme** | `auto` | `auto`, `dark`, `light`, `minecraft`, or `fallout`; page-level choices persist |

</details>

## 🔄 Emotion Analysis Modes

| | LLM mode (recommended) | Passive retrieval |
|:---|:---|:---|
| **How it works** | A lightweight model extracts search terms and 1–3 emotion priors; probability, cooldown, and intent gates decide sending | The reply text is used directly as the search query |
| **Effect on replies** | ✅ Does not modify the LLM reply | ✅ Does not modify the LLM reply |
| **Best for** | Stronger matching with one extra lightweight model call | Tight token budgets or lower latency |

Character assignment is done manually in the WebUI and remains independent of emotion categories. VLM writes semantic labels; you choose the character.

## 🎮 Command Reference

Commands below use the default `/` wake prefix as an example; use the prefix configured for your bot.

### Viewing and Test Sends (everyone)

| Command | Description |
|:---|:---|
| `menu` / `help` | Open the role-aware function panel |
| `status` | View running status and meme statistics |
| `emotion_stats` | Inspect retrieval-analysis diagnostics |
| `random` | Ask the bot to send one random meme allowed in the current session |

Use `/meme menu` or `/meme help` to open the panel. Its compact body shows current state; functions are grouped as buttons. Members can view status and request random test sends. AstrBot administrators also see Settings and Maintenance. Official QQ bots, including Webhook adapters, use native callback buttons. Clicks execute directly without sending control commands into chat, invoking the LLM or depending on a wake prefix. Other platforms and failed keyboard requests fall back to text commands generated with the current session's wake prefix. `/` in this documentation is the default-prefix example.

Settings covers collection, automatic delivery, retrieval mode, forced collection and target filters. Maintenance covers collected-image inspection, tag statistics, emotion cache, index rebuilds and cleanup; see the command table below for delete, blacklist and scope usage. Management callbacks reuse existing command services and recheck the current administrator configuration, actor, bot instance and session. Typed commands retain framework permission checks. The `list` command requires administrator access.

Random test sends respect session send filters and meme scope, and avoid images already sent through the same active panel. `/meme random` also works directly. All panel buttons belong to their creator and original session and expire after 5 minutes. Refresh updates displayed state while retaining active-session deduplication. Closing invalidates every button in that panel and keeps the message in chat.

When enabling callbacks for the first time, restart WebSocket adapters so interaction intents are subscribed during connection. Webhook deployments must receive `INTERACTION_CREATE` from QQ. Unloading removes this plugin's handler and tasks while preserving other plugins' callbacks.

If the creator cannot operate a freshly opened panel, inspect `[MemeMenu]` logs. Panel delivery records identify the raw QQ identity source; callback receipt confirms the click reached the plugin. Identity rejection records distinguish `sender`, `group`, `client` and other mismatches. User, group and panel identifiers are hashed. When no receipt is logged, check QQ-side button permissions, the client and event subscriptions before attributing the rejection to the backend.

### Admin Commands (admins only)

| Command | Description |
|:---|:---|
| `list [category] [page_size] [page]` | Inspect memes collected by the bot (default: 10 per page) |
| `on` / `off` | Enable / disable meme collection |
| `auto_on` / `auto_off` | Enable / disable auto-send |
| `clean [force]` | Clean unclassified raw staging images |
| `偷` | Enter 30-second forced collection mode |
| `group show` | View send/steal filter configuration |
| `group <send\|steal> priority <wl\|bl>` | Set whitelist/blacklist conflict priority |
| `group <send\|steal> <wl\|bl> <add\|del\|clear> [group:<id>\|user:<id>]` | Manage group and user filter lists |
| `delete <index\|filename>` | Delete a meme |
| `blacklist <index\|filename>` | Delete a meme and block it from future collection |
| `scope <index\|filename> <public\|local>` | Set meme scope |
| `capacity` | Run capacity control immediately |
| `rebuild_index` | Rebuild the index after migration or corruption |
| `natural_analysis <on\|off>` | Switch between the two emotion-analysis modes |
| `clear_emotion_cache` | Clear cached emotion-analysis results |
| `tag_stats [N]` | Inspect tag/scene statistics; N defaults to 15 |

### LLM Tool Calls (automatic during conversation)

| Tool | Description |
|:---|:---|
| `search_meme` | Search candidate memes with category, scene, scope, and usage hints |
| `send_meme` | Select and send a meme from the candidate list; failures include a reason code |
| `steal_meme` | Save an image when the user asks for it; omit `image_ref` to use the first image in the current message, while VLM supplies category, tags, description, and scenes |

## ⚠️ Notes

- Deleting a category in the WebUI also deletes its image files.
- With `send_meme_as_gif` enabled, converting a very large image can cause a transient memory spike; disable it on low-memory systems.
- A working vision model is required for image classification, auto-collection, and VLM re-analysis.

### 📝 Prompts and GIFs

- The VLM classification prompt uses strict JSON with category, emotions, description, tags, scenes, and visible text.
- Custom VLM and emotion-analysis prompts are configurable; blank values use the bundled templates.
- GIF analysis samples nine evenly spaced frames in chronological order, builds a 3×3 storyboard, and removes temporary sample files.
- Legacy pipe-delimited responses remain supported, while JSON is more robust.

### Libraries and automatic eviction

The dashboard separates General (formerly Unassigned), Favorites, and Character libraries. Favorited character stickers appear under Favorites while retaining their character assignment. These protected libraries and external/pinned imports neither count toward `max_reg_num` nor participate in automatic eviction. Use Least used or Oldest sorting and batch deletion to manage them manually.

Eligible general stickers receive `score = w * low_usage + (1-w) * old_age`, using min-max normalization of `use_count` and `created_at` within the eligible set. Lower counts and older creation times score higher; a constant component contributes zero. Default `w=0.7`; only the overflow is removed, highest score first. Ties use lower usage, older creation, then path. Weight 0 prioritizes age; weight 1 prioritizes low usage.

Removing favorite/character protection returns ordinary stickers to the quota, preserving their usage and creation history. Scheduled cleanup, `/meme capacity`, and post-rebuild capacity enforcement share these rules.

## 🚢 Maintainer Release Flow

1. Update the version in `metadata.yaml` and add a dated section for that version to `CHANGELOG.md`.
2. Run only relevant tests, static checks, and `python scripts/release.py validate` locally. Package builds and packaging tests run in GitHub Actions; do not generate local ZIPs, checksum files or release previews, or download release assets for duplicate verification.
3. Commit and push the change to `master` or `main`, including the `metadata.yaml` change (for example, `git commit -m "release: 发布 vX.Y.Z"` followed by `git push origin master`).
4. The Release Action responds only when the pushed commit changes `metadata.yaml`. It requires a strictly higher version, runs the full checks, builds the ZIP and `.sha256`, creates the `vX.Y.Z` tag and GitHub Release, and downloads the asset again to verify its hash.
5. A manual Release run with `publish=false` validates and packages the selected `ref`, uploading a seven-day dry-run artifact without creating a tag or release. Use `publish=true` only after that check succeeds.
6. Verify the Actions result, release tag and listed ZIP/checksum assets. The Release Action downloads the published package and checks its SHA-256; local copies are unnecessary for release verification.

Pull requests run CI automatically; ordinary code commits can run CI manually from Actions. Packaging checks run on Python 3.12, while the full checks cover Python 3.12 and 3.13. Changing other files alone does not start the publishing workflow.

## 📄 License

This project is open source under the [GNU AGPL v3.0](LICENSE) license.

---

<div align="center">

If you find this useful, please give it a ⭐ Star — thank you!

Report issues at [GitHub Issues](https://github.com/nagatoquin33/astrbot_plugin_stealer/issues).

</div>
