# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
