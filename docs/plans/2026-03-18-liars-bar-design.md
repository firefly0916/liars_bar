# Liar's Bar Project Design (2026-03-18)

## 1. 目标与范围
本项目构建一个用于研究 LLM 欺骗博弈行为的《骗子酒馆》本地实验环境。系统采用纯 Python 异步架构，核心是“中心化环境 + 隔离化智能体 + 可复现实验日志”。

首版范围聚焦“可运行实验骨架 + 可扩展规则引擎”：
- 异步事件循环可完整跑一局。
- 基础规则可完成声明/质疑/惩罚结算。
- 俄罗斯轮盘惩罚可配置且可复现。
- 多玩家与多模型通过同一配置文件扩展。
- Prompt 模板可配置、可版本化。
- 输出格式对 Agent 严格声明，但运行时解析具备宽松容错。

非首版范围：完整道具系统、复杂特殊牌效果、分布式部署。

## 2. 架构总览
- `engine/`: 客观规则层，不依赖任何 LLM 实现。
- `agents/`: 玩家决策层，包含默认 `MockAgent` 和 `LangChainAgent` 骨架。
- `experiment/`: 实验编排、回合调度、日志写入。
- `config/`: 单一实验配置文件（非敏感配置）和加载器。
- `prompts/`: Prompt profile 模板（可多版本实验对比）。
- `logs/`: 每日变更摘要与详细修改记录。

核心循环（Orchestrator）流程：
1. 获取当前玩家。
2. 生成该玩家局部观察。
3. 调用 Agent 输出 `thought + action`。
4. Environment 结算 action 并推进 phase。
5. Logger 写入结构化日志（含错误链路）。

## 3. 配置策略
采用“`.env + 单一实验配置文件`”双层：
- `.env`：仅存 OpenRouter API Key / Base URL / 默认超时等敏感项。
- `config/experiment.yaml`：玩家配置、模型参数、规则参数、日志参数、Prompt profile 绑定等全部实验变量。

多玩家多模型能力由 `players[]` 配置驱动：每个玩家可指向不同模型（API 相同）与不同 prompt profile。

## 4. 状态机与规则模块
Environment 使用 phase pipeline：
`TURN_START -> DECLARE -> RESPONSE_WINDOW -> RESOLUTION -> PENALTY -> TURN_END`

规则模块通过统一接口参与校验与结算：
- 声明出牌规则
- 质疑规则
- 轮盘惩罚规则

这样新增“特殊牌/道具/阶段插队”等能力时，只需新增规则模块并注册，不破坏主循环。

## 5. 错误可观测与输出容错
Agent 输出采用“规范 + 宽松解析”：
- Prompt 强制声明目标结构。
- 解析器先严格 JSON 校验，再尝试代码块提取、键名归一化、轻量类型修复。
- 失败时把具体错误反馈给 Agent 重试（最多 3 次）。
- 若最终失败，执行可配置降级动作，并记录完整错误链路。

标准错误码：
- `E_AGENT_FORMAT_INVALID`
- `E_ACTION_SCHEMA_MISSING`
- `E_ACTION_RULE_VIOLATION`
- `E_ENV_PHASE_MISMATCH`
- `E_AGENT_FALLBACK_APPLIED`

## 6. 测试与验证
严格 TDD：
1. 配置加载测试（`.env` + `experiment.yaml` 覆盖）。
2. Agent 输出宽松解析与错误码测试。
3. Environment 阶段流转与质疑结算测试。
4. Orchestrator + Logger 的端到端回合记录测试。

完成标准：测试通过、默认 mock 模式可运行、日志输出完整、文档与记忆文件齐全。

