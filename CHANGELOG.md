# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.2] - 2026-09-22

### added
- 新增可选 JEV 表情决策链路：本地相关度粗筛最多 10 张候选，结合最近对话、本轮消息和回复，由 TypeSafe Choice 选择候选或 `none`
- 新增 JEV 开关、TypeSafe API Key 和 API Base URL 配置；默认使用官方地址，也兼容可信代理的 `/v1` 或完整 endpoint

### changed
- 保留传统轻量 LLM 查询改写链路；仅在同时开启小模型分析和 JEV 时启用候选决策，JEV 请求失败后回退传统 LLM
- JEV 选择规则优先保持助手说话立场、指代和互动意图，并将原始响应 JSON 与实际发送路径分开记录为 debug 日志
- JEV 粗筛按确定性相关度排序，过滤文件缺失、重复及当前会话不可见的候选，发送前再次校验文件和作用域

### fixed
- 修复智能候选评分中局部变量覆盖 `entry_category()` 函数导致智能选图持续失败并退回随机选择的问题

## [3.1.1] - 2026-09-20

### added
- 新增旧版中文分类 key 的自动迁移，兼容已存在的分类目录、SQLite 元数据和待审核记录

### changed
- 兼容 3.1.0 之前使用中文或其他不安全分类 key 的旧数据：已知中文情绪复用安全英文 key，自定义 key 使用稳定 legacy slug，并保留中文显示名
- 图库、审核卡片、详情与 VLM 对比共用多情绪标签组件；待审核和正式库共用详情字段映射，旧数据以主分类作为情绪回退
- 合并待审核/图库详情与 A/B 对比模板，移除多余情绪分析包装层；入库、审核、编辑、移动和删除共用检索同步入口
- 移除通用 CacheService 和空 API 包；BM25 引擎直接原子持久化文档缓存，旧图库与黑名单迁移统一交给 IndexManager，黑名单运行期只读写 SQLite
- 命令入口直接调用对应处理器，图片列表渲染和发送编码直接使用 ImageRenderService，删除中间转发方法
- 测试统一使用 conftest 的框架替身和隔离数据目录，移除多份会互相覆盖的初始化代码
- 删除 Main 的 lambda 方法别名和搜索服务的动态属性转发；搜索、使用记录和作用域检查通过明确的依赖连接，共享文本转换改为独立函数
- 合并 VLM Provider 解析、提示词默认值和分类解析职责；单图重分析优先按 hash 查询，避免读取整个图库

### fixed
- 修复旧中文分类导致插件升级后在 `ensure_category_dir()` 阶段无法加载的问题；启动时同步迁移分类目录、SQLite 元数据和待审核记录
- 详情图片按预览区域约束尺寸并裁切缩放溢出，避免大图遮挡 VLM 重分析区域；适配桌面、窄屏及各套主题
- 待审核接口补齐 emotions；增量同步正确保存多情绪，并在提交成功后仅刷新语义变化的条目
- 命令删除、拉黑和自动淘汰显式删除数据库记录及向量，不再依赖增量字典同步表达删除，保留并发入库保护
- 修改视觉模型后实际请求同步使用新 Provider；框架默认视觉模型变更后不再沿用旧缓存
- 无效 VLM 文本或缺少必要字段的响应进入有限重试，失败后不再补成成功分类
- 应用 VLM 标注时保留原作用域；描述、标签、适用对话和情绪修改后刷新检索向量
- 分析服务不可用或请求解析失败时，清理逻辑不再覆盖原始错误

## [3.1.0] - 2026-09-18

### added
- 通用表情库新增按使用次数和入库时间加权的自动淘汰策略；默认低使用次数权重为 0.7，可在配置中调整
- WebUI 将图库拆分为通用表情库、收藏和角色表情库，新增使用最少排序、库容量统计、卡片右键菜单和可折叠侧栏
- 新增分类 key、图库分组、淘汰排序、重建元数据和 `/meme` 指令注册契约的回归测试

### changed
- `max_reg_num` 仅统计可自动淘汰的通用表情；收藏、角色表情和受保护外部导入独立管理，不占自动淘汰额度
- 自动淘汰使用候选集内归一化分数：低使用次数占 70%，入库时间越旧占 30%，同分时按使用次数、入库时间和路径稳定排序
- WebUI 重排为“表情库 → 情绪/角色筛选”层级，卡片固定展示归属、使用次数、作用域、保护状态和入库时间
- WebUI 侧栏折叠、主题和网格/列表视图均可持久化；快速切库只应用最后一次请求结果
- README 许可证说明与仓库根目录的 AGPL-3.0 许可证保持一致

### fixed
- 重建索引时恢复收藏、角色、保留等级、使用次数和最近使用时间，避免受保护表情重建后误入自动淘汰
- 收藏或角色状态变更后按当前库重新刷新列表与计数；文件删除失败时继续保留索引记录

### security
- 分类 key 统一限制为可移植的小写 slug，阻止目录穿越、路径分隔符、系统/插件保留名、大小写重名和重复显示名

## [3.0.3] - 2026-09-14

### added
- WebUI 审核区支持放大预览：在弹窗中直接通过、删除、拉黑或编辑，动作后自动跳到下一张待审核表情包
- 页头新增「表情包库 / 审核区」分区导航（审核区带待审数徽标），桌面端侧栏只保留分类；Fallout 主题继续使用 Pip-Boy 顶部标签
- 审核区键盘快捷键新增 Enter 放大预览

### changed
- WebUI 排版体系化：全站字号收敛为 7 级、行高 3 档的设计令牌；展示衬线字体只用于标题与统计大数，按钮、输入框等控件改用无衬线 UI 字体；计数与时间戳使用等宽数字
- 页头改为三列网格（品牌+导航 | 居中统计 | 状态/主题），统计块相对窗口真正居中；工具栏重排为「搜索+排序/视图切换」与「左浏览操作、右增改操作」两行，窄窗口不再全部堆在左侧
- 图库与审核网格居中排布，稀疏分类不再左偏留白；卡片图区统一 1:1，卡片按序错峰入场
- 可访问性：卡片、分类、主题等可点击元素改为原生 button 并补键盘支持，12 个弹窗补齐 role/aria 标注，增加焦点可见样式，Esc 按层级从最上层弹窗开始关闭

### fixed
- WebUI 删除表情包只检查 HTTP 状态导致假成功：现在读取响应体的 success 字段，失败时提示错误并保留弹窗


## [3.0.2] - 2026-09-11

### added
- 新增基于 `metadata.yaml` 版本提交的自动发布校验与可复现安装包流程，并提供手动 dry-run 产物

### changed
- 发布流程改为 PR CI 门禁与 `metadata.yaml` 版本提交驱动；版本递增后自动检查、打包、创建标签与 Release，并附带 SHA-256
- README 中英文同步整理当前功能、安装更新方式、外部资源包布局、配置默认值与维护者发布流程；补充 VLM A/B 复核和 GIF 处理说明，并标注旧的 meme_manager 标签注入机制已弃用

### fixed
- 统一 Linux 与 Windows 的路径规范化行为，修复跨平台检索去重测试失败

## [3.0.1] - 2026-09-11

### added
- WebUI 详情预览支持重新调用 VLM，并提供当前标注与新结果的 A/B 对比；确认后才应用新分类、描述、标签、场景、图上文字和情绪

### fixed
- 已是 GIF 的表情包发送时直接保留原始帧与时序；非 GIF 动图转换时保留总时长、循环设置和稀疏变化帧，修复动图加速或表现为静态图
- 本地事件图片和后台暂存按文件魔数校正 GIF/WebP/PNG 等后缀，避免动图因错误扩展名在预览或发送端表现异常
- 合并 PR #109 的事件文件保护与 PR #112 的 QQ 自定义表情发送支持

## [3.0.0] - 2026-09-04

### added
- 新增统一外部表情包源：支持 Meme Manager / AstrBot Meme Pack 目录、ZIP 与 `.meme-pack` 导出、GitHub 资源包仓库，以及可分页的 HTTPS JSON 目录 API
- WebUI 新增「外部源」控制台，提供上传、预检、分类映射、公开/本地作用域、待审核导入、后台进度、取消、同步与来源注销
- 数据库新增独立来源注册表与条目溯源，记录授权、署名、远端哈希、最后出现时间和失效状态；旧版 schema 6 数据库可原位增量升级
- 外部导入采用 SHA-256 去重，并把描述、图上文字、标签、场景、情绪、原文件名和来源 URL 映射到现有检索元数据
- 兼容 `semantic_metadata.json` 逐图语义字段；导入整套角色表情时可批量分配已有角色或创建新角色

### changed
- 外部图片会验证后复制到本插件存储，源插件和资源包文件保持原样；同步时缺失条目标记为 stale，图库副本继续保留
- 外部导入项使用独立保留等级，不参与原生偷图容量淘汰，避免大型资源包挤掉聊天收集的表情
- 开启内容审核时外部导入强制进入待审核池；已注册源同步会复用保存的分类映射、作用域和认证头

### security
- ZIP 预检拒绝目录穿越、绝对路径、符号链接、超量成员、解压炸弹、超大文件和空资源包
- HTTP 源默认仅允许 HTTPS，并在每次请求和重定向前拦截本机、私网、链路本地及保留地址；响应、图片像素和并发任务均有上限
- 外部目录元数据仅保留公开字段，API 凭据不会出现在来源列表或条目溯源中

### fixed
- 外部源认证头在跨域图片地址或重定向时自动剥离，避免凭据发送到未授权主机
- 自动表情改从 `on_llm_response` 捕获流式回复，补齐流式结果跳过装饰阶段时的发送路径
- 会话状态键优先采用 AstrBot 统一消息来源，群聊/私聊及跨平台的相同本地 ID 不再互相覆盖
- 导入范围和待审核容量在角色创建前完成校验；失败任务不会留下新角色或来源登记
- 浏览器上传的资源包加入过期暂存清理；注销来源后立即释放无引用的上传归档
- 外部源协议文档纳入发布文件，管理页改用随插件发布的 Vue 构建并固定字体 CDN 版本

## [2.8.10] - 2026-09-03

### changed
- 嵌入向量检索默认改为关闭；新安装默认使用 BM25，已有显式配置保持不变
- GIF 的 VLM 预处理改为在完整帧序列中等距抽取九帧并生成 3×3 分镜，只保留采样帧，不再缓存全帧或依赖 NumPy 相似帧过滤
- 精简并统一 VLM 默认提示词，强化 GIF 时间顺序、OCR 去重、动作变化、分类键与适用对话字段约束

### fixed
- 修复 WebUI 亮色主题下角色筛选栏沿用暗色背景、导致标签与计数文字对比度不足的问题（#108）

## [2.8.9] - 2026-08-31

### changed
- 移除旧的情绪标签注入、标签清理和相关兼容钩子；被动发送直接使用 LLM 回复内容进行检索
- 提取事件解包、路径/标签规范化、黑名单和媒体发送等跨模块重复逻辑，统一插件内部实现
- LLM 表情工具统一使用 `search_meme` → `send_meme`；搜索自然短句时自动展开情绪别名参与分类召回

### fixed
- 被动表情的 info 日志移动到概率、冷却和意图门控通过之后，避免造成“检测到就会偷取”的误解
- LLM 通过通用图片发送工具绕过插件工具时，同轮不会再次触发被动表情发送
- WebUI 主题跟随宿主参数不再覆盖页面已保存的主题选择，刷新后可稳定恢复主题状态

## [2.8.8] - 2026-08-30

### added
- PR #103：支持在 QQ Official（QQ 官方机器人平台）按 CDN 特征、全部图片或 GIF 模式收录表情，并补充平台声明、配置项与回归测试

### fixed
- WebUI 成功读取服务端偏好后不再被浏览器旧 `localStorage` 覆盖；页面主题偏好会记录配置默认值快照，配置项 `webui_theme` 再次变更后，下次打开或刷新管理页时自动清理旧覆盖并采用新默认主题

## [2.8.7] - 2026-08-28

### fixed
- 情绪分析提示词渲染不再使用 `str.format` 把输出示例里的 `{"query": ...}` 误当成占位符，修复 `提示词模板缺少占位符 '"query"'` / `LLM调用失败` 报错

## [2.8.6] - 2026-08-27

### added
- VLM 分类提示词与 LLM 小模型情绪分析提示词恢复为配置项：非空覆盖内置，留空回退内置模板
- 文字距离融合权重改为预设选择：均衡 / 关键词优先 / 语义优先 / 严格匹配，移除 5 个手动权重滑块
- 智能选择新增角色召回、适用对话（scenes）召回；BM25 对图上文字 / 角色 / scenes 加权
- 动图 VLM 分析提示词强化；拼接图增加帧分隔线与帧序号
- WebUI 编辑 / 上传 / 待审核表单支持 `overlay_text`（图上文字）

### changed
- 移除 `other` 分类：无法判断时归入最接近的已有情绪类
- VLM 提示词中立化：每个分类机会均等；tags 0~3、scenes 1~2 句；即使图上无文字也要求写适用对话
- LLM 小模型不再决定是否发送表情，只负责从 AI 回复提取检索词与情绪先验
- 被动模式不再向 LLM 注入 `&&emotion&&` 标签，改为直接用回复原文做语义匹配
- 自动发送匹配改为：图上文字 / 角色 / 适用对话 → 嵌入 → BM25 → 分类兜底；分类只作弱先验
- 配置面板收起普通用户难懂的参数：意图门控、字间延迟、随机延迟、偷图冷却、审核异常策略、存储清理策略等
- README / i18n 同步更新

### fixed
- 辐射 4 避难所小子图片在 AstrBot 实际页面因资产 token 缺失无法加载：由 JS 模板中的 `<img>` 改为 CSS `background-image`，走框架资源重写

## [2.8.5] - 2026-08-27

### added
- WebUI 主题选择会保存为默认：写入插件 KV（`/prefs`），配置项 `webui_theme` 作为初次默认
- 辐射 4 主题使用 FO4 支线任务图标作为避难所小子素材（`pages/dashboard/vaultboy.png`）

### changed
- WebUI 辐射 4 主题改成 Pip-Boy 腕机外壳 + RobCo 终端内屏：STAT/INV/DATA 页签、底部 HP/AP/LVL、终端菜单；分类栏改进行内网格，避免被外壳/顶栏挡住
- 详情弹窗才加载原图（LRU 4 张）；网格 / 审核区始终用 JPEG 缩略图，hover 不再预取原图（issue #101）
- 辐射 4 主题去掉全屏 `mix-blend-mode` 扫描层与图片 CSS filter，避免大 GIF 解码时整页卡死

### fixed
- 缩略图生成失败时不再回退发送原图（大 GIF 会把列表卡死）；改为返回错误，列表保留占位色块

## [2.8.4] - 2026-08-24

### added
- WebUI 游戏背包主题：我的世界（箱子物品栏）与辐射 4（Pip-Boy 仓库）；主题选择器保留跟随宿主 / 暗色鎏金 / 亮色
- WebUI 图片展示框按窗口宽度自适应（`--slot-size`），库网格、游戏槽位、详情预览与审核卡片随视口缩放
- WebUI 网格 / 列表视图切换、预览缩放拖拽、分类 accent 色点
- tests/web：Dashboard 预览服务器与冒烟测试（mock bridge、静态资源、审核 / 上传 / 删除流）

### changed
- WebUI 改用 Vue 生产构建，并补 favicon（`pages/dashboard/logo.png`）

### fixed
- PR #100：预览弹窗未导出 `formatBytes` / `formatAddMethod`，新入库图片点开详情崩溃

## [2.8.3] - 2026-08-20

### added
- PR #98：自动偷图改为有界后台管线（32 容量 + 2 worker）；本地媒体先快照再释放事件，远程下载后台执行
- 后台暂存/下载任务限流（capture_limit），队列满时远程 URL 在下载前直接丢弃，避免任务无界增长
- 冷却窗口在 worker 真正开始处理时刷新，慢 VLM 下冷却间隔不再失效
- 测试补强：后台队列限流、满队列跳过下载、shutdown 等待拷贝线程、force capture 同步回归

### fixed
- PR #96：prompts.json 的 UTF-8 BOM 导致初始化 JSONDecodeError；统一用 utf-8-sig 读取并在 json.loads 前剥离残留 BOM
- shutdown 清理 staging 时与 in-flight 拷贝线程的竞态：改用专用 executor shutdown(wait=True)
- `/meme 偷` 入队后丢失同步成功/失败反馈的问题；恢复同步处理与 convert_to_file_path 兜底
- WebUI 拉黑确认框把转义换行显示成字面量 `\n`（issue #97）；i18n 改为真实换行，确认框/toast 支持 pre-line 渲染

## [2.8.2] - 2026-08-14

### added
- DB schema v5：`emoji` / `emoji_pending` 新增图片元数据列（`source_url` / `original_name` / `width` / `height` / `format` / `bytes` / `add_method`，`emoji` 另有 `reviewed_at`），入库与审核流程自动填充；建表语句直接含全列（新旧库均幂等迁移）
- 待审核池标签/场景存储对齐正式库：新增 `emoji_pending_tag` / `emoji_pending_scene` 关联表，取代 `tags_text` / `scenes_text` 逗号拼接列（v4→v5 自动迁移；重复标签按正式表语义去重）
- VLM 打标数量约束：默认提示词明确 tags 输出 2~4 个、scenes 1~2 个；解析器统一规范化（保序去重 + 截断上限），旧管道分隔格式同样生效
- 分类缓存绑定 VLM 模型签名（`model_sig`）：切换视觉模型后旧分类结果自动失效，与嵌入向量表的 model_sig 机制对齐
- 文字距离融合权重配置化：`sim_weight_ngram/cosine/substring/char/edit` + `sim_negation_penalty` 六个权重可调（新增「智能选择：文字距离融合权重」分组），`configure_similarity()` 覆盖默认值并清缓存，启动与配置更新时自动同步
- 新命令 `/meme tag_stats [N]`：高频标签 Top、低频单次标签（噪声提示）、零标签条目数、高频场景 Top 5
- WebUI 预览面板新增图片信息（尺寸 · 格式 · 大小）、入库方式、审核时间、来源链接展示

### changed
- 配置面板按用户路径重排：模型配置组（视觉模型 / 情绪分析模式 / 情绪分析模型 / 嵌入检索开关 / 嵌入模型 ID）上移到「偷图设置」之后，`emotion_analysis_prompt` 归入高级配置，删除独立的情绪识别分组
- `DatabaseService` 统一 `_INSERT_EMOJI_SQL` / `_EMOJI_SCALAR_COLUMNS` / `_META_COLUMNS` 常量，元数据列在全部 CRUD/同步/迁移路径透传；`get_stats` 补充 `pending_count`
- i18n 三语言同步：zh-CN / en-US / ru-RU 补齐新增配置键与前端字段键（含 ru-RU 此前缺失的 `fields` 段）
- 死代码清理：`meme_selector` 遗留的 6 个未使用常量、`DatabaseService._join_multi`、未使用 import
- 测试补强：v5 迁移与关联表回归、fresh 库建表回归、标签规范化、权重配置化（tests/test_schema_v5.py、tests/test_text_similarity.py）

### fixed
- pending 标签含逗号时审核通过会错误拆分的隐患（逗号拼接列 → 关联表后消除）
- WebUI 预览面板元数据显示为空壳：后端 `_build_image_item` / `_build_pending_item` 丢弃元数据列，现已透传
- `/meme tag_stats` 同步 DB 查询阻塞事件循环：改为 `asyncio.to_thread` 丢线程池

## [2.8.1] - 2026-08-01

### added
- metadata.yaml 补充市场展示字段：`tags`（表情包/图片识别/情绪匹配/LLM 工具调用/WebUI/群聊）、`category: 娱乐`、`author_url`、`help`（/meme 指令帮助）、`pages`（表情包管理页）
- tests: 新增 TestDiscoveryFields 覆盖 tags/category/author_url/help/pages 结构断言（tests/test_metadata.py）

## [2.8.0] - 2026-08-01

### added
- PR #91: 情绪分析提示词可配置化（`emotion_analysis_prompt`），支持模型输出 `none` 跳过发送（abstain）
- PR #90: 待审核区分页返回 `category_total`，按分类筛选时保留其他分类计数
- metadata.yaml 补充 `short_desc`（市场卡片短描述，v4.24+ 规范）
- 测试纳入版本控制并补齐：情绪分析器、待审核池、metadata 结构、PluginAPI 分类列表（tests/）

### fixed
- PR #90: WebUI 响应式布局与图片加载修复（dashboard CSS/JS）
- PR #90: 缩略图回退支持从 pending 池按 hash 取图（`get_pending_by_hash`）

## [2.7.7] - 2026-07-23

### changed
- 优化代码规范
- 修正部分编码错误

## [2.7.6] - 2026-07-23

### changed
- emoji→meme 命名统一：搜索/事件模块文件重命名，代码内全部引用同步更新
- Main 新增 `__getattr__` 代理，自动将 `self.steal_meme` 等直接属性访问路由到 `plugin_config` (Pydantic)
- 新增 `core/maintenance/` 维护服务模块，从 `main.py` 和 `event_handler.py` 重构抽出
- 新增 `core/util/safe_io.py` 安全文件操作工具模块

### fixed
- 修复 `sync_index` 并发竞态：删除逻辑移除避免误删其他消息刚入库的条目，删除移交 `MaintenanceService` 孤儿扫描
- 修复 VLM 分类失败后 raw 目录孤儿文件泄漏
- 修复 `classification_parser` Unicode 转义比较死代码
- 修复 `meme_send_delay_random`(bool) 被当 float 用、`meme_send_delay_max` 从未被引用的语义错误
- 修复 `plugin_api` 调用不存在的 `cache.mark_dirty()` 静默空操作
- 修复审核通过时间戳使用 VLM 识别时间而非审核入库时间
- 添加 `super().initialize()` / `super().terminate()` 确保 SDK 生命周期兼容

### added
- 审核池批量模式「全选 / 取消全选」按钮
- i18n 新增 `deselect_all` (zh-CN / en-US)

## [2.7.5] - 2026-07-21

### fixed
- 修复 LLM `steal_meme` 工具传入相对路径（如 `./image.png`）时返回"图片文件不存在"的问题 (#88)

### added
- WebUI 审核区信息编辑功能 (#87)：在每张待审核卡片上新增 ✏️ 编辑按钮，弹窗可修改分类 / 作用域 / 描述 / 标签 / 场景
  - 弹窗底部提供「仅保存」与「保存并通过」两个动作
  - 新增后端 API：`POST /astrbot_plugin_stealer/pending/update`（`handle_pending_update`）
  - 新增 DB 层方法：`DatabaseService.update_pending()`，使用字段白名单防止改写 `path` / `hash` / `source` / `origin_target`
  - 同时修复 `update_pending()` 对值未变的 no-op UPDATE 误判"行不存在"的问题：SQLite 的 no-op UPDATE 返回 rowcount=0，统一改为回查 SELECT 判定存在性
  - 中英俄 i18n 同步新增：`actions.edit_approve` / `actions.save_only` / `actions.save_and_approve` / `modal.edit_pending` / `messages.no_preview` / `labels.hash` / `alerts.pending_saved`

### tests
- 新增 4 个 steal_meme 路径解析回归测试（相对路径 / 绝对路径 / `convert_to_file_path` / 无图回退）
- 新增 5 个 `update_pending` 单元测试（白名单、字段过滤、未知 id、空字段、no-op 仍返回行）

## [2.7.4] - 2026-07-04

### fixed
- 修复 `enable_embedding_search` 开关未被所有写入路径尊重的问题：入库、审核通过、初始化回填均改为仅在开关开启时操作，避免关闭后仍产生无效向量与重复日志 (#86)
- 修复 `main.py` 中嵌入服务初始化块的 Python 缩进错误 (#86)
- `EmbeddingService` 新增 provider 负缓存：同配置下未找到 provider 时不再重复探测与打印日志；配置变化（开关/provider ID）时自动重置状态 (#86)
- 移除 `napcat_token` 配置及其作为 `Authorization` header 附加到任意图片下载 URL 的逻辑，避免 token 泄露与部分服务器/安全 agent 拒绝请求 (#86)

### changed
- 清理 `_conf_schema.json`、i18n（zh-CN/en-US/ru-RU）与 README 中已废弃的 `napcat_token` 相关文案 (#86)

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
