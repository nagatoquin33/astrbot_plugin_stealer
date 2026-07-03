# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.7.3] - 2026-07-03

### fixed
- 修复 `send_meme_as_gif` 开启时 QQ (aiocqhttp) sticker 路径未执行 GIF 转换的问题，`_send_qq_image_as_sticker` 新增 GIF 联动逻辑 (#84) — 感谢 @Foolllll-J

### added
- 新增 `meme_send_char_delay` 按字延迟配置（0.3s/字），发送延迟与 LLM 回复长度成正比，设为 0 回退原有固定/随机延迟 (#84) — 感谢 @Foolllll-J
- 新增空回复跳过保护：LLM 主回复被置空时自动跳过表情发送，避免"纯表情无文本"的异常场景 (#84) — 感谢 @Foolllll-J

### changed
- `meme_send_delay` 描述从"表情包发送延迟/秒"改为"表情包发送固定延迟/秒"，与新增的按字延迟做区分 (#84) — 感谢 @Foolllll-J

## [2.7.2] - 2026-07-02

### added
- QQ (aiocqhttp) 平台发送表情包时外显文案从 `[图片]` 改为 `[动画表情]`，模拟真实 QQ 表情包效果 (#81) — 感谢 @Foolllll-J

### changed
- WebUI 仪表盘前端重构：模板与逻辑分离（`template.js` + `app.js`），页面目录重命名为 `pages/dashboard/`
- WebUI 主题模式改为跟随 AstrBot 框架上下文自动切换亮色/暗色，移除页面内独立主题切换按钮
- WebUI 新增中英双语 i18n（`zh-CN` / `en-US`），页面标题、按钮、弹窗、提示、批量导入与审核区文案均接入多语言 (#82) — 感谢 @Zhalslar


## [2.7.1] - 2026-06-28

### added
- WebUI 搜索扩展覆盖 hash/分类/来源/群号/文件名/路径，搜索框支持按群号或文件名快速定位表情包 (#76)
- 批量导入支持选择文件夹并递归检索子文件夹内所有图片 (#75)

## [2.7.0] - 2026-06-28

### added
- 待审核池完整闭环 (#78)：自动偷取先进 pending，人工审核通过后入库
  - `/pending` 分页列表、`/pending/approve` 批量通过、`/pending/reject` 批量拒绝+拉黑、`/pending/stats` 容量统计
  - 哈希去重：插入前三重检查（正式库/pending/黑名单），避免重复图片进入审核区
  - 原子写入 + 审核回滚：文件移动失败不写 DB，DB 写入失败文件移回 pending
  - 容量护栏：`steal_pool_capacity` 满则暂停偷取，审核通过后自动恢复
  - WebUI 双区管理：📋 审核区（审核卡片、批量操作、容量进度条）/ 🗄️ 表情包库
  - 缩略图 pending 回退：正式库未命中时自动回退查待审核池
- 嵌入向量检索：FaissVecDB 语义搜索优先，不可用自动降级 BM25
  - 启动时自动回填旧数据向量，provider 维度不匹配自动修复
  - auto-emoji 选图加入嵌入语义增强评分 (bonus × 0.20)
  - LLM 工具 `search_meme` / `send_meme` 搜索路径嵌入优先
  - `faiss-cpu>=1.8.0` 列为正式依赖
- 启动孤儿扫描：清理无文件索引/无索引文件（categories + pending 双目录）
- WebUI 排序新增"最常发送"/"最近发送"选项

### changed
- 统一命名 emoji/sticker → meme：LLM 工具 3 个（`search_meme`/`send_meme`/`steal_meme`）、配置键 12 个、i18n 三语言同步
- `_sync_index` 不再在审核通过后调用，改为仅 invalidate BM25 索引（修复了新入库 emoji 被 sync diff 误删的 bug）
- WebUI 审核卡片加大、描述两行显示

### deprecated
- 旧配置键（`steal_emoji`/`auto_send`/`emoji_chance` 等）已全部重命名，升级后需在配置面板重新设置

## [2.6.14] - 2026-06-24

### fixed
- 修复后台批量导入表情后 BM25 索引表不更新的问题。所有写入路径（批量导入、单张上传、删除、更新、移动、偷图、容量控制）现在都会调用 `_invalidate_bm25_index()` 使 BM25 失效，下一次搜索时自动重建
- 修复搜索热路径的安全网：即使某条写入路径遗漏了 BM25 失效调用，搜索时也会对比语料签名，发现不一致则强制重建
- 修复重启插件后把用户已删除的预定义类别重新加回的问题。`_auto_merge_existing_categories` 现在以用户的 `categories.json` 为合并基线，仅自动发现自定义类别，已删除的预定义类别即使磁盘有残留文件也不会复活
- 修复储存清理中路径大小写/分隔符不匹配导致全部有效表情被误判为孤儿文件而删除的问题（Windows NTFS）。新增 `_norm_path_key()` 归一化比较，`stale_index` 检测改用 `Path().resolve().is_file()` 规范判断
- 修复储存清理中 `stale_index` 只删索引条目不删物理文件的边界问题——回滚了误加的 `_safe_remove_file` 循环，`stale_index` 设计语义就是"文件已丢失、只清索引"，绝不删文件

### changed
- `send_emoji_as_gif` 发送路径不再对图片做二次压缩：取消 PIL GIF 重编码中的 `optimize=True` 与 2048 尺寸缩放限制，保留原始帧数据与尺寸
- blacklist 从 `cache/blacklist_cache.json` 迁移到数据库 `blacklist` 表（SCHEMA_VERSION 2→3）。新增 `blacklisted_hashes` / `add_blacklist` / `remove_blacklist` / `add_blacklist_batch` CRUD 方法。插件启动时自动从旧 JSON 缓存迁移数据，迁移后黑名单读写以 DB 为准
- 初始化时自动清理遗留的迁移残渣文件（`.backup`/`.migrated`/`categories/*/index.json`/`cache/index_cache.json`/`index.json`/`image_index.json`），仅当 DB 已有数据时执行，保留活跃缓存与配置文件

## [2.6.13] - 2026-06-18

### fixed
- 修复 NapCat/QQ 平台表情包"偷取"失败：`download_original_image()` 只取 `img.url` 发起 HTTP 下载，但 NapCat 平台 `url` 字段为本地文件路径，导致 `aiohttp` 请求失败（日志: `下载图片失败: /home/.../temp/media_image_xxx.jpg`）
  - 新增本地路径优先检测：按 `path` → `file` → `url` 顺序检查，若为本地已存在文件则直接返回路径和 GIF 检测结果，跳过 HTTP 下载
  - 仅当所有字段均非本地路径时才回退到远程下载

## [2.6.12] - 2026-06-18

### fixed
- 修复 v4.26.0-beta.4 中 `@filter.llm_tool` 工具接收 `ContextWrapper` 导致 `send_emoji_by_id` / `search_emoji` / `steal_sticker` 调用失败 (#76)
  - 新增 `_unwrap_event` 工具函数，检测 `ContextWrapper` 自动解包为 `AstrMessageEvent`
  - 三个工具入口处统一解包，消除 `AttributeError: 'ContextWrapper' object has no attribute 'send'` / `'set_extra'` 等错误

### removed
- 移除 WebUI 独立密码/认证残留代码（`WebuiConfig` 类、`auth_enabled`、`password`、`session_timeout` 等字段）
  - 插件已迁移至 AstrBot Pages 系统，认证由面板统一管理，这些配置项不再生效

## [2.6.11] - 2026-06-15

### fix
- 修复 GIF 动图拼接后超出 VLM 输入限制(2048x2048)导致分类失败 (#72)：拼接图超限时自动 LANCZOS 等比缩放至限制内
- 修复 `zh-CN.json` 键名 `auto_send` 使用中文引号导致 JSON 解析失败 (#74)
- 修复 `README_EN.md` 两处中文书名号 `「」` 改为英文引号

### improved
- 智能选图预筛选优化：模糊候选初筛改用更轻量的 n-gram Jaccard（`calculate_simple_similarity`），减少昂贵策略融合计算
- 智能选图打分循环添加提前终止：≥5 个最终评分 f≥0.7 的高分候选时跳过剩余条目，避免全量遍历

## [2.6.10] - 2026-06-12

### improved
- 优化 `steal_sticker` 主动偷取工具描述，明确 `image_ref` 必填、来源限制与 VLM 自动打标行为，降低 LLM 误调用/偷错图概率
- 移除 `_inject_emotion_instruction` 中注入式偷取指引，避免每次 LLM 请求都追加额外提示词，提升提示词缓存命中率
- 移除 `steal_by_llm` 独立配置与相关 i18n 残留，LLM 主动偷取统一由 `steal_emoji` 总开关控制

## [2.6.9] - 2026-06-09

### added
- WebUI 新增存储扫描与清理入口，可清理失效索引、孤儿文件、缩略图缓存和临时文件
- WebUI 新增批量作用域来源修复工具，便于为缺失 `origin_target` 的表情补齐来源并设为 `local`
- 新增自动表情意图门控配置，跳过命令、严肃错误回复、过短回复和连续提问等不适合自动发图的场景
- 新增同会话新消息取消待发送自动表情配置，避免延迟发送打断新的对话节奏

### improved
- 基于 PR #69（清理自动表情标签后再过发送门控）和 PR #70（WebUI 单张上传 base64 JSON 兼容）同步后的代码继续做稳定性增强
- 单张 WebUI 上传改为原子化写入图片和元数据，分类、标签、描述、场景与作用域不再依赖二次更新
- LLM 工具搜索结果补充作用域、使用次数等选择参考，发送失败时返回明确 reason 便于重新选择
- LLM 主动偷图增加轻量图片预检，提前拦截无效扩展名、空文件、超大文件和损坏图片
- 批量上传任务增加 TTL 与更新时间，自动分析临时文件改为 `finally` 清理

### docs
- README/README_EN 增加双语言切换 badge，并同步说明新增 WebUI、配置和 LLM 工具行为
- metadata 描述与版本更新至 `2.6.9`

## [2.6.8] - 2026-06-07

### fix
- 修复数据库并发锁超时：新增 `PRAGMA busy_timeout=30000` 防止写操作被饿死
- 修复删除/拉黑时索引清理逻辑：支持按哈希匹配，避免路径不一致导致索引残留
- 修复容量控制 overflow 计算错误：`remove_count` 逻辑改为 `max(0, overflow)` 防止负数
- 修复容量控制删除确认：新增路径规范化，先尝试映射 raw 路径到分类目录，文件确认删除后才删索引
- 修复 WebUI 删除/移动操作原子性：新增事务回滚机制，文件移动失败时自动恢复

### improved
- 数据库新增 `get_emoji_by_hash`、`delete_paths`、`update_path`、`move_path` 方法，支持事务和回滚
- PluginAPI 重构：优先使用数据库操作，新增 `_build_full_index_snapshot` 统一索引快照
- WebUI 批量操作原子性增强：批量删除只移除实际删除成功的路径
- WebUI 移动端响应式优化：新增 768px/420px 断点适配

## [2.6.7] - 2026-06-07

### added
- 新增表情包收藏功能：WebUI 支持单张/批量收藏，侧边栏支持只看收藏
- 收藏状态接入自动发送策略：随机模式提高收藏项权重，智能模式为收藏项加分
- 收藏表情包加入容量清理保护，不参与自动淘汰

### fix
- 完成收藏功能上线前接口修复：`/images` 返回 `favorite_count`，数据库分页返回 `is_favorite` / `last_used_at`
- 修正 WebUI 删除图片/分类的请求方法与后端路由一致
- 修复分页越界重试被请求锁拦截的问题

### improved
- WebUI 分类数据兼容数组和对象两种返回格式
- 优化 WebUI 样式性能并清理少量冗余 CSS

## [2.6.6] - 2026-06-04

### fix
- 修复 `_enforce_capacity` 只删索引不删文件：返回值 `files_to_delete` 被所有调用方忽略，导致容量控制循环和 `/meme capacity` 命令均沦为"索引清理表演"，磁盘文件持续堆积
- 修复重建索引无视容量限制：`rebuild_index` 扫描全部文件后直接入库，完全不检查 `max_reg_num`，重建后索引瞬间突破限制
- 修复 `_enforce_capacity` 原子性缺陷：原代码先 `del` 索引再删文件，文件删除失败后产生新的僵尸文件且下次循环不再清理；改为先删文件、成功后才删索引
- 修复 `_restore_metadata` 滥发 `True`：只要旧数据是 dict 就返回成功，空 `{}` 也算恢复；改为仅当真正写入了 `desc`/`tags`/`source`/`scenes` 等字段时才返回 `True`
- 修复 `recovered_count` 统计时机错误：在容量控制前统计，包含了后来被物理删除的文件；改为容量控制后重新统计最终保留的文件
- 修复重建索引统计文案 misleading：`"当前索引数量"` 实为重建前旧数据汇总，改为 `"重建前旧数据"` 和 `"重建后索引/文件"`

### refactor
- `core/events/event_handler.py`：统一使用 `pathlib.Path` 替代 `os.path`（`basename`/`join`/`exists`）

## [2.6.5] - 2026-06-01

### fix
- 修复 LLM 工具发送表情包被概率拦截（issue #66）：`send_emoji_by_id` 不再检查 `resolve_auto_emoji_turn_permission` 和 `claim_auto_emoji_turn`
- 修复 `_update_result_with_cleaned_text_safe` 完全无效：`MessageEventResult` 没有 `cleaned_text` 和 `result` 属性，改为遍历 `chain` 修改 `Plain` 组件
- 修复 `_compute_hash` 全文件读取导致大图片 OOM：改为 64KB 分块读取
- 修复无效分类后临时文件泄漏：`_handle_classification_result` 中无效分类时立即清理文件

### changed
- pHash 计算失败日志级别 debug→warning，生产环境可见去重失败

## [2.6.4] - 2026-05-28

### fix
- 修复动图转 GIF 发送时出现拖影：GIF 保存缺少 `disposal=2`，导致帧叠加产生残影

## [2.6.3] - 2026-05-11

### added
- LLM 自主偷取功能：新增 `steal_sticker` LLM tool，bot 可在私聊/群聊中自主偷取图片入库
- 偷取统一走 VLM 自动分析路线，入库后回传分类/标签/描述/场景结果给 LLM
- 新增 `steal_by_llm` 配置项，控制 LLM 自主偷取开关
- `on_llm_request` 注入偷取指引，包含分类库存提示和使用时机

### fix
- 补回 `resolve_auto_emoji_turn_permission` 丢失的 debug 日志（重构遗漏）
- 修复 `steal_image_direct` 空索引覆盖全库的 bug

## [2.6.2] - 2026-05-11

### fix
- 修复 `claim_auto_emoji_turn` 提前设置 `_active_sent`，导致 `EmojiSmartSelectService.try_send_emoji` 误判已发送而跳过自动发送
- 修复 `EmojiSmartSelectService.try_send_emoji` 调用 `self.select_emoji()` 不存在，改为 `self.plugin.emoji_selector.select_emoji()`
- 修复 `EmojiSmartSelectService` 缺失 `SMART_FAST_PREFILTER_*` 和 `SMART_BM25_*` 类常量
- 修复 `EmojiSmartSelectService.__getattr__` 白名单过窄，缺失 `_recent_usage` 等多个委托属性；改为通用委托
- 修复 `async_analyze_and_send_emoji` 未接入 `SmartEmotionMatcher.analyze_and_match_emotion`，自然情绪分析功能无效
- 修复 `_prepare_emoji_response` 未传入 `user_query`，自然情绪分析器缺少 QA 上下文

## [2.6.1] - 2026-05-11

### fix
- 修复 `_enforce_capacity` 方法名不匹配：重构时误改名为 `_enforce_capacity_sync`，导致容量控制循环崩溃
- 修复 `_clean_raw_directory` 方法缺失：Phase 4 拆分时被误删，导致 raw 目录定时清理失败
- 修复 `send_emoji_by_id` 工具消息乱码：编码损坏导致用户可见文本变乱码
- 修复 `calculate_hybrid_similarity` 导入缺失：`emoji_smart_select_service.py` 未导入该函数
- 修复 `PHashDedupService` 类名引用错误：未导入直接使用类名
- 修复 `LANCZOS` 常量引用错误：Pillow 新版移除该常量，改为 `PILImage.LANCZOS`
- 清理所有未使用的 import 和冗余代码

### changed
- `image_mgmt_command.py`：list 命令改用数据库分页查询，limit 上限提升至 100
- `plugin_api.py`：补充 list 接口的数据库分页路径与分类统计口径对齐

## [2.6.0] - 2026-05-09

### changed
- WebUI 迁移至 AstrBot Pages 系统，删除独立 WebServer
- PluginAPI 重写，handler 直接实现消除委托链
- `core/` 拆分 6 子目录，按职责分离 15 个子服务

### fix
- 修复 iframe 沙箱拦截 `confirm`/`alert` 导致操作不可用
- 补回重构丢失的 `initialize`、hook、辅助方法等
- 修复 VLM 调用链、子服务委托缺失、导入缺失等重构问题

### removed
- 删除 `api/` 旧 handler 委托层，逻辑合并到 `plugin_api.py`

## [2.5.8] - 2026-05-02
### improved
- 哈希查重改用 SQL 直查，不再每次加载全量索引，大幅降低图片处理内存占用
- WebUI 页面请求优先使用内存缓存，减少数据库全量重建
- 容量控制循环改用内存缓存代替 DB 重建
- `update_index` 减少一次冗余 deepcopy
- VLM 调用修复：删除多余的 `file://` URI 构造和错误回退逻辑
- `__setattr__` 不再每次属性赋值写盘，改为显式批量保存

### removed
- 删除 ~900 行无用代码（未调用方法、死参数、死缓存逻辑）
- 删除重复的 db/cache 索引获取模式，提取为 `_get_index()` 助手方法
- 删除 `task_scheduler` 中 6 个从未调用的方法
- 删除 `database_service` 中 11 个从未调用的方法和死缓存逻辑
- 删除 `command_handler` 中 9 个从未调用的 CRUD 方法

### fixed
- 修复关键路径裸 `except Exception` 静默吞异常问题

## [2.5.7] - 2026-04-25
### add
- 新增黑白名单优先级机制：白名单和黑名单可同时启用，通过 `/meme group <send|steal> priority <wl|bl>` 或 WebUI 配置切换冲突时的优先级
- 表情包发送路径优化：Telegram 贴纸 → file_image → base64 三级自动选择，非 QQ 平台优先使用 file_image 降低发送开销

### improved
- 图片下载和 WebUI 上传流程去重重构，减少重复代码
- 目标名单检查增强：同时提取群组和用户目标，支持黑白名单同时生效并处理冲突

## [2.5.6] - 2026-04-19
### fix
- 修复 `rebuild_index` 元数据恢复稳定性：按路径/哈希/文件名多级匹配并补充大小写无关 stem 回退，减少标签与描述丢失
- 修复 WebUI 编辑后预览卡片与统计信息可能未及时刷新的问题，编辑保存后优先使用服务端最新数据回填
- 修复分页排序在时间戳相同场景下的不稳定问题：新增路径/哈希作为次级排序键，避免列表顺序抖动

### improved
- 数据库批量关联查询复用 `_load_related_map`，大集合读取改为分块 `IN` 查询，降低长 SQL 参数列表风险
- WebServer 运行时索引读取优先数据库快照并统一写入路径，减少 cache/DB 双源数据不一致
- 统计接口优先使用数据库计数（含今日新增），回退内存遍历作为兜底，提升大数据量下性能与一致性

### refactor
- 抽取搜索语料签名构建逻辑，统一 BM25 签名与数据库语料签名计算口径
- 分类与批量更新路径统一通过 `_update_runtime_index` 收敛同步流程，减少重复同步代码

## [2.5.5] - 2026-04-14
### fix
- 修复 WebUI 分类统计口径："全部"分类计数改为使用全量统计值，避免与当前筛选结果混淆
- 修复分页边界问题：当前页超出最后一页时自动回退，避免删除或筛选后出现空白页
- 修复批量导入任务状态字段不一致问题：前后端统一为 `success_count` / `failed_count`，并返回失败原因
- 修复分类管理写入异常：新增分类列表去重与空值过滤，禁止写入空分类

### improved
- 批量上传默认分类体验优化：未手动选择时自动回退到当前选中分类或配置首项，减少误操作
- 自动分析分类结果增加白名单校验，仅接受已配置分类，异常时回退兜底分类
- 分类更新流程增强：同步维护 `category_info` 键集并确保分类目录存在
- 缓存裁剪策略优化：允许 `index_cache` 作为无上限缓存，避免索引数据被 LRU 误裁剪

## [2.5.4] - 2026-04-14
### improved
- `CacheService.update_index` 支持异步 updater，新增 `IndexCache` / `IndexUpdater` 类型别名
- `WebServer._sync_index_to_db` 新增 `raise_on_error` 参数，写操作同步失败时向上抛出异常而非静默忽略
- `_sync_index_to_db` 通过 `_get_db_service()` 安全获取数据库服务，兼容 `sync_index` 与 `save_index` 两种接口
- `_collect_removed_paths_by_hashes` 新增 `raise_on_sync_error` 参数，调用方可控同步失败行为
- 异常日志增加 `exc_info=True`，记录完整堆栈便于排查

## [2.5.3] - 2026-04-13
### fix
- 修复 `/meme rebuild_index` 后标签、描述、场景丢失的问题，支持从旧 JSON 与备份恢复索引元数据
- 补强缓存加载异常处理、WebUI 分页查询兼容性以及 `sync_index` 异常回滚逻辑

## [2.5.01] - 2026-04-11
### fix
- 修复 LLM 调用工具时多段回复导致表情包重复发送的问题

### refactor
- 新增 `_EmojiTurnState` 类封装事件状态管理，替代散落的 `event.get_extra/set_extra` 调用
- 优化索引加载策略：优先从数据库加载，旧数据自动迁移

## [2.4.15] - 2026-04-03
### fix
- 修复 `natural_emotion_analyzer.py` 缺失 `calculate_hybrid_similarity` 导入导致的 NameError
- 修复否定词误判问题："不开心"不再错误匹配到 "happy" 分类

### improved
- 相似度算法优化：调整权重（提高编辑距离权重、降低子串权重）
- 新增否定词检测：支持"不/没/无/非"等否定词语义反转识别
- 代码复用：统一使用 `_has_negation_prefix` 函数，消除重复代码

## [2.4.14] - 2026-04-02
### perf
- 优化了readme，调整了部分默认设置与工具描述

## [2.4.13] - 2026-04-02
### perf
- BM25 索引接入缓存持久化：新增 `bm25_cache`，支持按语料签名恢复，减少冷启动重建开销
- 智能选图热路径优化：先按分类/作用域过滤，再做轻量词法预筛，降低全量扫描与重排成本
- 预筛召回增强：加入 `SMART_FAST_PREFILTER_FUZZY_RESERVE`，在保性能的同时保留部分模糊候选，减少误杀
### refactor
- `EmojiSelector` 的 BM25 构建流程改为异步调用，统一搜索与智能选图路径的索引加载行为

## [2.4.12] - 2026-03-31
### add
- WebUI 字体优化：更换为思源黑体 (Noto Sans SC)，提升可读性
- WebUI 登录页面风格统一：角落装饰、边框阴影与主页面保持一致

### improved
- WebUI 表情包网格排版优化：增加间距、悬停效果增强、分类标签更清晰
- WebUI 工具栏布局优化：按钮按功能分组，视觉层次更清晰
- WebUI 侧边栏样式优化：增加宽度、悬停交互、选中效果增强
- WebUI 详情弹窗布局优化：图片预览区加大、信息面板更宽敞、标签样式优化

## [2.4.11] - 2026-03-30
### add
- WebUI 新增批量导入功能，支持多图同时上传
- WebUI 登录页面拆分为独立页面，便于预览
- WebUI 批量导入支持自动分析模式，由 VLM 自动识别每张图片分类
- 搜索算法升级，新增 BM25 混合搜索，提升搜索准确性

### fix
- WebUI 批量导入分类选择与自动分析互斥，避免状态混乱

## [2.4.10] - 2026-03-27
### perf
- VLM 分类调用并行化，锁范围优化，处理速度提升
- 图片下载与处理并行化
- aiohttp session 复用，减少连接开销

### fix
- 修复 `_auto_emoji_cooldowns` 内存泄漏，添加上限1000条
- 修复 `_force_capture_windows` 竞态条件，添加 RLock 保护
- 修复 `terminate()` 异常吞没，独立 try-except 确保各服务清理
- 修复 `update_index` 异常后数据不一致，添加快照恢复
- 修复 `threading` 模块未导入问题

## [2.4.9] - 2026-03-24
### add
-新增支持电报telegram

## [2.4.8] - 2026-03-24
### add
- 新增表情包发送延迟配置，避免与分段插件冲突
  - `emoji_send_delay`: 固定延迟时间（秒），默认 5.0
  - `emoji_send_delay_random`: 是否开启随机延迟，默认关闭
  - `emoji_send_delay_max`: 随机延迟最大值（秒），默认 8.0，支持滑块配置
- 随机延迟模式下，实际延迟在 [延迟] ~ [最大延迟] 之间随机，更加自然

## [2.4.7] - 2026-03-22
### add
- 新增 `/meme blacklist <序号|文件名>`，支持按 `/meme list` 显示的全局序号删除表情并加入黑名单
- 新增表情包作用域 `public/local`，收集时记录 `origin_target`，可将指定表情限制为仅来源群发送
- WebUI 新增作用域展示与管理，支持查看来源群、单图切换 `public/local`、批量设置作用域

### fix
- 搜索、随机选图、LLM 工具发送统一遵守表情包作用域限制，避免限定表情串群发送
- 随机选图在索引缓存为空时改为安全失败，避免在缺少元数据时误发 `local` 表情
- `rebuild_index` 现在会保留 `origin_target`、`scope_mode` 等扩展元数据，减少重建索引后的信息丢失

## [2.4.6] - 2026-03-18
### fix
- 修复 VLM 分类提示词模板渲染，避免 JSON 花括号触发格式化异常
- 修复 VLM 分类响应解析，兼容 JSON 前后缀文本、代码块和脏字段
- 移除干扰 JSON 输出的旧版管道格式提示词尾巴
- 分类列表为空时直接报出明确错误，便于定位配置问题

## [2.4.5] - 2026-03-17
### perf
- 提示词输出改为 JSON 结构，确保分类可被准确解析
- 新增 JSON 响应解析器，支持从 markdown 代码块提取 JSON
- 保留旧格式兼容层（管道符分隔），实现平滑过渡

## [2.4.4] - 2026-03-16
- 修复 VLM 分类解析：支持 "审核通过：分类名" 格式的前缀处理
- 修复情绪分析解析：支持从带解释的文字中提取分类名
- 文档同步：README 配置默认值与 schema 保持一致

## [2.4.3] - 2026-03-15
- 新增 VLM 提示词配置：支持在插件配置界面直接编辑提示词
- 分类失败处理：未识别到分类时跳过图片，不再回退到默认分类
- 优化分类提示词：减少 troll/dumb 分类过度倾向，添加分类判断指导
- 提示词恢复机制：清空配置后重启插件自动恢复默认提示词

## [2.4.2] - 2026-03-15
- 修复 FC 工具 search_emoji 未展示场景的问题（兼容场景字段格式）
- 文档更新：补充 /meme list 翻页参数与工具输出说明

## [2.4.1] - 2026-03-10
- 修复 GIF 识别 bug：Pillow 版本兼容性处理 (LANCZOS 常量)
- 修复 numpy 导入问题：移至模块级别，避免 GIF 帧提取时的重复导入
- 并发安全优化：CacheService.get() 和自动表情冷却机制添加异步锁保护
- 修复日志中文乱码问题
- 代码质量：类型注解修复、异常日志改进

## [2.4.0] - 2026-03-08
- 群聊过滤升级为目标过滤：偷表情与发表情分离配置，支持 `group:群号` 与 `user:QQ号`
- 命令入口整理为 `command_group` 结构，保留 `/meme 偷`，优化名单帮助与状态输出
- 自动发表情体验优化：新增轻量门控、会话冷却、最近使用强惩罚，减少连续重复发图
- 被动标签模式发送顺序优化：增加轻微延迟，避免表情包先于文字发出
- 文本相似度与选图热路径优化：增加缓存、修正短语包含评分、减少 tags/scenes 重复拆词
- 清理旧版群聊黑白名单兼容层，统一到新目标名单配置
- 新增 GIF 动图多帧分析：VLM 分析时自动提取关键帧横向拼接，完整理解动图内容
- GIF 原始文件下载：检测动图 URL 并下载原始 GIF，避免框架转换丢失动效
- 大图发送优化：GIF 转换时自动缩放大图防止内存溢出，保持 VLM 分析可用性
- Bug 修复：cleanup 后空引用风险、重复清理、并发字典修改、WebUI 重启失败无恢复等问题

## [2.3.9] - 2026-03-03
- WebUI 新增白天/黑夜主题切换
- 登录页重新设计，添加动画效果
- 修复登录页渲染 bug

## [2.3.8] - 2026-03-02
- 修复bug
- 新增 QA 上下文情绪分析：将用户消息与 LLM 回复组成 Q/A 上下文，提升情绪识别准确度


## [2.3.7] - 2026-02-26
- 增强 FC 工具模糊匹配：搜索无结果时返回推荐分类
- 优化智能选择性能：避免二次遍历索引，一次遍历同时收集高低分候选
- 优化搜索过滤：分词匹配替代字符级匹配，更适合中文场景
- 新增分词评分：解决"猫娘"搜"小猫"等中文词汇匹配问题
- 优化情绪分析器：本地关键词预匹配跳过不必要的 LLM 调用，LLM 失败时降级用分词匹配
- 智能选择增强：描述匹配增加分词补充，提升中文上下文匹配效果
- 重构代码结构：`find_similar_categories` 和 `_find_best_category_match` 集中到 EmojiSelector

## [2.3.6] - 2026-02-25
- WebUI 代码重构瘦身：拆分公共逻辑，保持接口与行为不变
- 配置更新流程去重：保留双模式与旧版本迁移链路
- 代码质量检查：执行 `ruff check . --no-cache` 并修复 import 排序

## [2.3.5] - 2026-02-25
- 智能选择算法：多维度评分 + 加权随机选择
- 内存优化：GIF缓存限制50条/10MB
- 增强错误日志，便于诊断模型配置问题
- WebUI 移动端适配

## [2.3.4] - 2026-02-25
- 精简代码，修复默认视觉模型bug

## [2.3.1] - 2026-02-22
- 代码规范修复：路径安全、异常处理、类型标注、常量提取

## [2.3.0] - 2026-02-21
- 性能优化：asyncio.to_thread、SHA-256、缓存驱逐
- 新增：steal_mode/steal_chance 配置项
- 代码重构：I/O 迁移至 initialize()
- Bug 修复：aiofiles 回退、资源清理、竞态条件

## [2.2.3]
- 修复 FC 工具 search_emoji 缺少 tags 字段

## [2.2.2]
- 增加 webui 登录功能

## [2.2.1]
- 代码拆分优化

## [2.2.0]
- 群聊黑白名单、"真"表情包模式

## [2.1.9]
- `/meme 偷` 主动收录窗口

## [2.1.8]
- 表情包黑名单功能

## [2.1.7]
- WebUI 界面优化

## [2.1.6]
- 函数工具优化

## [2.1.5]
- 后置轻量 LLM 语义分析

## [2.1.4]
- 优化表情识别和 VLM 提示词

## [2.1.1]
- 智能表情选择、使用统计、NapCat 兼容

## [2.1.0]
- 表情描述、WebUI 管理、FC 工具

## [2.0.9]
- 优化提示词和残缺标签处理

## [2.0.8]
- 代码规范性优化

## [2.0.7]
- 代码重构优化

## [2.0.0]
- 增强存储系统

## [1.0.4]
- 修复 VLM 模型调用

## [1.0.3]
- 修复指令状态

## [1.0.2]
- 修复正则匹配
