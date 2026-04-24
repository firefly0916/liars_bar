# 骗子酒馆虚拟环境引擎 (Liar's Game Engine) - Project Memory

## 1. 项目概述 (Project Overview)
本项目使用纯 Python 异步事件循环构建《骗子酒馆》本地实验环境，用于多智能体博弈、对手建模与 LLM 欺骗行为研究。系统默认可离线运行（MockAgent），并提供 OpenRouter + LangChain 接入骨架用于真实模型实验。

## 2. 核心架构设计 (Architecture Blueprint)
- **Environment (`liars_game_engine/engine`)**: 维护客观状态机（phase pipeline），不包含 AI 推理。
- **Agent (`liars_game_engine/agents`)**: 仅消费 `observation` 输出结构化动作，支持 `MockAgent` 与 `LangChainAgent`。
- **Orchestrator (`liars_game_engine/experiment/orchestrator.py`)**: 运行事件循环，调度 Agent 与 Environment。
- **Logger (`liars_game_engine/experiment/logger.py`)**: 写入 JSONL 回合日志，记录 observation/thought/action/result/error。

## 3. 技术栈 (Tech Stack)
- **核心语言:** Python 3.10+
- **并发模型:** asyncio
- **目标模型约束:** Pydantic v2（作为项目依赖声明，当前离线环境无法安装时采用 dataclass 兜底结构）
- **配置与数据:** YAML + `.env` + JSONL
- **Agent框架目标:** LangChain（以可选依赖方式接入）

## 4. 配置中心规范 (Single Config Entry)
- **敏感项统一在 `.env`**：`OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_TIMEOUT_SECONDS`。
- **实验参数统一在 `config/experiment.yaml`**：玩家、模型、规则、日志、解析器、运行参数。
- **多玩家/多模型**：通过 `players[]` 配置扩展；API 可以相同，`model` 可不同。
- **Prompt实验**：`prompts/profiles/*.yaml`，每个玩家通过 `prompt_profile` 绑定模板。

## 5. Agent 设计约束 (Agent Constraints)
- Agent 不得直接读取 Environment 内部隐藏状态，只能使用 `observation`。
- 目标输出必须包含 `thought` 与 `action`。
- 运行时解析采用“严格约束 + 宽松容错”：支持 Markdown JSON 块提取、别名键映射、重试反馈。
- 解析失败最多重试 `max_retries` 次；失败后执行可配置降级动作，并记录错误码。

## 6. 错误排查与可观测性 (Debuggability)
关键错误码：
- `E_AGENT_FORMAT_INVALID`
- `E_ACTION_SCHEMA_MISSING`
- `E_ACTION_RULE_VIOLATION`
- `E_ENV_PHASE_MISMATCH`
- `E_AGENT_PROVIDER_UNAVAILABLE`
- `E_AGENT_FALLBACK_APPLIED`

每个回合日志必须包含：`trace_id`, `turn`, `player_id`, `observation`, `thought`, `action`, `raw_output`, `parser_error`, `fallback_used`, `step_result`。

## 7. 强制维护文件规范 (Mandatory Memory & Changelog Maintenance)
以下文件在后续每次开发必须同步维护：
1. `PROJECT_MEMORY.md`
   - 维护架构约束、阶段进展、不可违反规则。
2. `logs/CHANGELOG_DAILY.md`
   - 记录“每日/每次修改”的概括摘要（目标、影响、结果）。
3. `logs/CHANGELOG_DETAILED.md`
   - 记录本次改动的详细位置与变化类型。
   - 需包含：文件路径、起始行号、修改类型（`+`/`-`/`~`）、修改说明。
   - 类似 diff 视角，便于精确回溯。

## 8. 当前阶段 (Current Phase)
- [x] Phase 0: 确定纯本地异步架构与目录结构。
- [x] Phase 1: 完成核心数据模型（Action/State/StepResult）。
- [x] Phase 2: 完成基础 Environment 状态机与规则模块（declare/challenge/roulette）。
- [x] Phase 3: 完成 Agent 基类、MockAgent、LangChainAgent 骨架与解析器。
- [x] Phase 4: 完成 Orchestrator + JSONL logger + 基础测试闭环。
- [~] Phase 5: 规则深化与分析工具。
  - [x] Phase 5A: 基础规则对齐（2~4人、每轮翻桌牌、Innocent/Liar 判定、1~3出牌、轮结束重发牌、起手位继承规则）。
  - [ ] Phase 5B: 道具/特殊牌/阶段插入。
  - [x] Phase 5C: 回放分析工具（Shapley 反事实采样 + `credit_report.csv` 聚合导出）。
  - [~] Phase 5D: Proxy-based Attribution（代理模型快速归因 + 对齐校验）。
    - [x] 已实现 `ProxyValuePredictor`、`attribute_step_proxy`、`proxy_alignment_report.json` 与 `credit_report_proxy.csv` 导出。
    - [~] 已从 6 维升级到 8 维（新增 `hand_truth_ratio` 与 `action_consistency_score`），并复用同一编码函数贯通训练/推理。
    - [ ] 8 维代理仍未通过对齐门槛：当前 `val_mse=0.2047`、`pearson=0.3339`，未达到 `MSE < 0.15` 与 `Pearson > 0.6` 的目标，故仍不得替代物理 rollout 归因。
    - [~] 已完成法医诊断：`proxy_forensics_report.json` 复原了 Task I 的 20 个对齐点；前三个离群点全部位于 `response_window`，且都需要“当前声明 + 手牌构成”才能解释其真实价值。

## 9. 后续开发注意事项
- 新增规则优先放 `engine/rules/`，避免在 `environment.py` 内硬编码堆叠。
- 若引入真实 LangChain 运行，请在联网环境安装依赖并补充集成测试。
- 若调整 Agent 输出结构，必须同步更新解析器、Prompt profile、测试用例与日志字段。
- 当前基础规则采用标准牌义：`table_type` + `JOKER` 视为 Innocent，其余为 Liar。
- 每轮结算后必须由 Environment 负责重置回合状态并重发手牌，避免由 Agent 推断轮边界。
- Task C 归因流水线固定使用“非原动作合法集合”作为反事实基线，不允许引入 No-op。
- 回放采样必须启用 100 步终止保护；无法在上限内结束的路径按平局 `0.5` 计分。
- Null Player 实验通过 `runtime.enable_null_player_probe` 注入 `Null_Probe_Skill`；日志中以 `skill_category=Probe` 标识。
- 当前日志版本升级为 `log_version=v2_probe`，用于区分含 Probe 字段的新轨迹。
- Task D 可通过 `liars_game_engine.analysis.task_d_axiomatic_runner` 执行 100 局 Probe 采样并导出 `axiomatic_brief.json`。
- Task I 可通过 `liars_game_engine.analysis.task_i_proxy_runner` 执行最新一批 100 局 Probe 日志的代理归因与双线对齐校验。
- Proxy 模式必须复用 `train_value_proxy.py` 的特征编码逻辑；当前 8D 编码为 `phase/table_type/must_call_liar/alive_player_count/hand_count/death_probability/hand_truth_ratio/action_consistency_score`。
- 若对齐报告 `pearson_correlation <= 0.75`，只能输出参考报表，不得宣称可替代 MC / rollout。
- 当前已验证 proxy 输入顺序没有漂移：`ProxyValuePredictor.encode_state_features()` 直接调用 `train_value_proxy._build_feature_vector()`，因此负相关不是由数组索引错位导致。
- `logs/task_d_probe/value_proxy_saliency.json` 的初步梯度显著性显示：当前模型最敏感的是 `alive_player_count`，其次是 `death_probability`、`must_call_liar`；`hand_count` 不是主导维度，因此问题更像“缺少 claim/inventory 语义”而不是“单纯过拟合手牌数”。
- `logs/task_d_probe/proxy_forensics_report.json` 的前三个离群点中，没有一个样本出现正的 `inventory_logic_conflict`；但它们都依赖当前声明数量、手中真牌比例或牌面库存语义，说明新增 `hand_truth_ratio` 与 `inventory_logic_conflict` 有机会改善挑战/续报类决策。
- 当前已落地的第 7/8 维是 `hand_truth_ratio` 与 `action_consistency_score`；尽管 8D 版本已使 Pearson 从负值转正到 `0.3339`，但因训练目标仍是 observational winner label，而对齐指标比较的是 causal `phi`，改进幅度有限。
- 若特征从 6 维扩展到 8 维，只要新增特征可由现有日志字段（`observation.private_hand`、`pending_claim`、`action`、`table_type`）推导，就**不需要**重新生成原始对局或重新做物理采样；需要的是基于现有 JSONL 重新提特征、重建训练张量并重新训练代理模型。
- EXP1 / 论文数据沉淀阶段默认继续使用物理 rollout 归因；Proxy 仅保留为辅助观测、诊断与候选特征筛查工具。
- `RuntimeSettings` 当前额外支持 `null_probe_action_probability`，用于控制 `MockAgent` 触发 `Null_Probe_Skill` 的概率；Task L 会把该值提升到高概率以收集“反面教材”轨迹。
- Task K 现已具备独立入口 `liars_game_engine.analysis.task_k_gold_runner`：默认配置为 4 Mock、物理 rollout、`rollout_samples=200`、输出 `logs/task_k_gold/credit_report_final.csv`，但本地分支只做代码接线，不在非 4090 环境执行大规模金标准采样。
- Task L 现已具备独立入口 `liars_game_engine.analysis.task_l_proxy_refine_runner`：会先生成高 Probe 负采样日志，再分别训练 elite/mixed 两个 proxy，并在最新一批 100 局 Probe 日志上做同一组 20 点 alignment 对比。
- 2026-04-23 本地 Task L 实跑结果已落盘到 `logs/task_l_proxy_refine/proxy_refine_report.json`：新增 10,438 条负采样记录（300 局），混合模型 `value_proxy_mlp_v2.pt` 将 Pearson 从 `0.3339` 提升到 `0.4249`，但 `val_mse` 从 `0.2047` 变为 `0.2088`，仍未达到 `Pearson > 0.6`，所以 Proxy 依然不能替代物理 rollout。
