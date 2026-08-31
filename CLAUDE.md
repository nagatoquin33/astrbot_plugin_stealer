# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`astrbot_plugin_stealer` (表情包小偷) is an AstrBot plugin that automatically collects images from group chats, classifies them via a vision-language model (VLM), and sends contextually matching emojis during conversations based on emotional tone. It provides a WebUI for management and supports LLM tool calls for emoji selection.

## Architecture

### Plugin Lifecycle

AstrBot loads `main.py` as the entry point. The `Main` class (inheriting from `astrbot.api.star.Star`) initializes all services in `__init__` and wires them together:

```
Main (main.py)
├── PluginConfig (core/config/config.py)      -- Pydantic-backed config, wraps AstrBotConfig
├── DatabaseService (core/db/database_service.py) -- SQLite with WAL, stores emoji index
├── CacheService (cache_service.py)             -- In-memory caches (index, image, cooldown)
├── CommandHandler (core/commands/command_handler.py) -- /meme commands
├── EventHandler (core/events/event_handler.py) -- Message listeners, image download, force-capture window
├── ImageProcessorService (core/processing/image_processor_service.py) -- VLM classification, dedup, tagging
├── MemeSelector (core/search/meme_selector.py) -- Search, selection strategy, BM25 + fuzzy matching
├── MemeSenderEngine (core/events/meme_sender_engine.py) -- Auto-send decision and cooldowns
├── EventContext / EmojiDelivery (core/events/) -- Shared event identity and platform delivery helpers
├── SmartEmotionMatcher (core/processing/natural_emotion_analyzer.py) -- LLM-based emotion analysis
├── PluginAPI (plugin_api.py)                   -- Web API routes for the dashboard page
├── Shared utilities (core/util/)               -- Path/metadata normalization, blacklist writes, safe IO
└── TaskScheduler (task_scheduler.py)         -- Periodic tasks (cleanup, capacity control)
```

### Data Flow

1. **Collection**: `EventHandler` listens to messages → downloads images → `ImageProcessorService` computes perceptual hash and calls VLM for category/tags → `DatabaseService` stores metadata in SQLite.
2. **Selection**: `MemeSelector` uses `MemeSearchEngine` (BM25 pre-filter + fuzzy re-ranking) and `MemeSelectionStrategy` (recent-usage penalty + randomness) to pick a matching emoji.
3. **Sending**: `MemeSenderEngine` intercepts LLM responses via `on_decorating_result`, decides whether to send an emoji based on cooldowns and probability, and dispatches the selected image asynchronously.
4. **WebUI**: `PluginAPI` registers routes under `/astrbot_plugin_stealer/*` via `context.register_web_api()`. The frontend is in `pages/表情管理/`.

### Key Design Patterns

- **Plugin instance as dependency container**: `Main` passes `self` to most services. Services access each other through `self.plugin.{service}`.
- **Async locks**: `DatabaseService` uses `asyncio.Lock` for writes; `ImageProcessorService` uses an `asyncio.Lock` plus a `_processing_hashes` set to prevent duplicate concurrent processing of the same image.
- **Stubs for testing**: `tests/conftest.py` injects fake `astrbot.*` modules so tests can run without the full AstrBot framework. Do not change stub names without updating all tests.

## Development Commands

### Running Tests

The project uses `pytest`. Run from the repository root:

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_database_service.py

# Run a specific test
pytest tests/test_database_service.py::test_some_function -v
```

### Linting / Formatting

There is no configured linter or formatter in this repository. Follow the existing style (PEP 8, 4-space indentation, type hints where appropriate).

### Running the Plugin

This is an AstrBot plugin, not a standalone application. To develop and test:

1. Install AstrBot (see its documentation).
2. Clone or symlink this repository into AstrBot's `data/plugins/` directory.
3. Restart AstrBot to load the plugin.
4. Configure a VLM provider in AstrBot; the plugin requires a vision model for classification.

### Dependencies

Only extra dependency declared in `requirements.txt`:

```
Pillow>=10.0.0
```

AstrBot itself provides `aiohttp`, `pydantic`, and other core libraries. Do not add heavy dependencies without consideration.

## Important Files and Conventions

- **`_conf_schema.json`**: AstrBot configuration schema. When adding new user-facing settings, update both `_conf_schema.json` and `core/config/config.py`.
- **`prompts.json`**: VLM prompts for classification. Fallback prompts exist in `ImageProcessorService` if the file is missing.
- **Database migrations**: `DatabaseService` uses `SCHEMA_VERSION` and `_init_schema()` for table creation. For schema changes, increment `SCHEMA_VERSION` and add migration logic.
- **`pages/表情管理/`**: WebUI frontend (HTML/JS). Backend API routes are in `plugin_api.py` and must match the frontend's expected endpoints.
- **i18n**: Translations are in `.astrbot-plugin/i18n/`. Keys must stay in sync with `core/commands/command_handler.py` and other user-facing strings.

## Testing Notes

- `tests/conftest.py` patches `sys.modules` with fake `astrbot.*` modules before any plugin code is imported. If you add a new import from `astrbot.*` in plugin code, add the corresponding stub in `conftest.py`.
- Tests should not require a real AstrBot instance or external network. Mock external HTTP calls and VLM responses.
