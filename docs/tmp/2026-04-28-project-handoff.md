下面这份可以直接给新主控台，用来理解当前整个项目，而不是只看当前分支。

**总览**
项目名是 `liars_game_engine`，目标是做《骗子酒馆》多智能体博弈实验，并在日志上做 SAVI / Shapley 归因。代码结构稳定分成 5 层：

- 配置层：加载 `YAML + .env + 环境变量`
- 环境层：实现《骗子酒馆》规则、状态、合法动作、观测视图
- Agent 层：Mock / LangChain / LLM 三类智能体
- 编排层：跑完整局、落 JSONL 日志、保存 checkpoint
- 分析层：Task C / D / I / K / L / M / N 这几条实验流水线

当前本地主要分支有 4 条：

- `main`
  基础主线。包含游戏引擎、Mock/Skill 归因、Task C/D/I、服务器化 Task K、部署脚本。
- `dev-proxy-refine`
  Proxy / phi 蒸馏线。包含 Task K 离线回填、Task L proxy refine、Task N LLM 行为审计。
- `dev-llm-agent`
  LLM 演习线。包含 Task M：`LlmAgent`、本地 HF 后端、llm drill、三段式 prompt、容错 parser。
- `task-k-label-main`
  辅助分支。是主线某个阶段用于持久化 Task K attributed logs 的过渡分支，不是当前主要开发线。

**各分支职责**
`main` 的最近主线职责是把 Task K 服务器化并补日志可观测性。提交主线是：
`1961656` serverize Task K gold runner
`486f896` server deployment and vllm scripts
`0655e34` editable install on server
`9d3f063` Task K progress reporting
`06528e8` isolate Task K server outputs per run
`5ea11d1` Task K diagnostics logging

`dev-proxy-refine` 相对 `main` 增加的是 Proxy / Distill 路线：
[task_k_backfill_labels.py](git show dev-proxy-refine:liars_game_engine/analysis/task_k_backfill_labels.py)
[task_k_phi_distill_runner.py](git show dev-proxy-refine:liars_game_engine/analysis/task_k_phi_distill_runner.py)
[task_l_proxy_refine_runner.py](git show dev-proxy-refine:liars_game_engine/analysis/task_l_proxy_refine_runner.py)
[audit_llm_behavior.py](git show dev-proxy-refine:liars_game_engine/analysis/audit_llm_behavior.py)
以及更强的 [train_value_proxy.py](git show dev-proxy-refine:liars_game_engine/analysis/train_value_proxy.py)

`dev-llm-agent` 相对 `main` 增加的是 Task M：
[llm_agent.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/llm_agent.py)
[local_backend.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/local_backend.py)
[prompts.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/prompts.py)
[llm_drill.py](/home/user/workplace/repos/liars_bar/liars_game_engine/experiment/llm_drill.py)
[run_llm_drill.py](/home/user/workplace/repos/liars_bar/scripts/run_llm_drill.py)

**配置层**
[config/schema.py](/home/user/workplace/repos/liars_bar/liars_game_engine/config/schema.py)
这里定义整个系统的强类型配置：

- `ApiSettings`
  输入：原始 `dict`
  输出：标准化后的 API 配置
  关键字段：`openrouter_api_key`, `openrouter_base_url`, `timeout_seconds`
- `RuntimeSettings`
  关键字段：`max_turns`, `random_seed`, `fallback_action`, `enable_null_player_probe`, `null_probe_action_probability`
- `ParserSettings`
  关键字段：`max_retries`, `allow_markdown_json`, `allow_key_alias`
- `LoggingSettings`
  关键字段：`run_log_dir`, `level`
- `PlayerConfig`
  关键字段：`player_id`, `name`, `agent_type`, `model`, `prompt_profile`, `temperature`
- `RulesSettings`
  关键字段：`deck_ranks`, `cards_per_player`, `roulette_slots`, `enable_items`
- `AppSettings.from_dict(raw)`
  输入：YAML / env 聚合出来的原始字典
  输出：完整 `AppSettings`

[config/loader.py](/home/user/workplace/repos/liars_bar/liars_game_engine/config/loader.py)
核心函数是 `load_settings(config_file="config/experiment.yaml", env_file=".env")`
输入：
- YAML 文件
- `.env`
- 运行时环境变量

输出：
- `AppSettings`

覆盖优先级：
- 先读 YAML
- 再读 `.env`
- 再被进程环境变量覆盖
- `OPENAI_* / OPENROUTER_* / VLLM_*` 三套别名都支持

当前默认配置文件是 [experiment.yaml](/home/user/workplace/repos/liars_bar/config/experiment.yaml)。
在 `dev-llm-agent` 上，它默认已经是 Task M drill 配置：
- `api.base_url = local://hf`
- `api.api_key = LOCAL`
- `p1 = llm`
- `p2/p3/p4 = mock`
- 默认模型 `Qwen/Qwen2.5-0.5B-Instruct`

**入口层**
[main.py](/home/user/workplace/repos/liars_bar/liars_game_engine/main.py)
默认入口只有两步：

- `run()`
  输入：无
  内部：`load_settings -> GameEnvironment -> build_agents -> ExperimentLogger -> GameOrchestrator`
  输出：一局游戏的 summary 字典
- `main()`
  输入：CLI 直接执行
  输出：打印 summary

这条路径本质上是“跑一局默认配置的游戏”。

**状态与环境层**
[engine/game_state.py](/home/user/workplace/repos/liars_bar/liars_game_engine/engine/game_state.py)
定义核心数据结构：

- `ActionModel`
  输入/输出动作统一结构：`type`, `claim_rank`, `cards`
- `ParseError`, `ParseResult`
  给 Agent parser 用
- `ClaimState`
  一次声明的运行态
- `PlayerRuntimeState`
  关键运行字段：`hand`, `eliminated`, `revolver_deck`
  关键属性：
  `is_safe`
  `death_probability`
- `RuntimeGameState`
  全局运行态：玩家表、当前玩家、phase、pending_claim、pile_history、table_type
- `StepResult`
  环境执行一步后的统一结果
- `RouletteOutcome`
  轮盘惩罚结果

[engine/environment.py](/home/user/workplace/repos/liars_bar/liars_game_engine/engine/environment.py)
这是游戏环境主类 `GameEnvironment`。

关键函数：

- `__init__(settings)`
  输入：`AppSettings`
  输出：初始化后的环境对象
  作用：随机 turn order、给每个玩家初始化 revolver、抽本轮 table_type、发牌
- `_build_liar_deck()`
  输出：标准牌堆，`A/K/Q` 各 6，加 2 张 `JOKER`
- `_deal_round_cards()`
  输出：给存活玩家重发手牌
- `get_current_player()`
  输出：当前该谁走
- `save_checkpoint()`
  输出：`{"state": RuntimeGameState, "rng_state": ...}`
- `load_checkpoint(checkpoint)`
  输入：先前保存的 checkpoint
  作用：恢复环境
- `serialize_checkpoint(checkpoint)`
  输出：可写进 JSONL 的 base64 文本
- `deserialize_checkpoint(encoded_checkpoint)`
  输入：base64 字符串
  输出：checkpoint 字典
- `get_legal_actions(player_id)`
  输入：玩家 ID
  输出：当前合法动作模板列表
- `get_observation_for(player_id)`
  输入：玩家 ID
  输出：agent 可见的 observation 字典
  关键字段：
  `player_id`, `phase`, `alive_players`, `private_hand`, `pending_claim`, `table_type`, `must_call_liar`, `player_states`, `pile_history`, `legal_actions`

也就是说，所有 agent 的输入统一都是 `observation: dict[str, object]`。

**规则层**
[rules/declare_rule.py](/home/user/workplace/repos/liars_bar/liars_game_engine/engine/rules/declare_rule.py)
`DeclareRule.validate(state, player_id, action)`
输入：游戏状态、玩家、动作
输出：`RuleResult`
作用：检查 `play_claim` 是否合法，检查牌数、claim rank、玩家是否真的持有这些牌。

[rules/challenge_rule.py](/home/user/workplace/repos/liars_bar/liars_game_engine/engine/rules/challenge_rule.py)
- `validate(...)`
  检查 challenge 是否能在当前 phase 执行
- `has_liar_cards(state)`
  判断被挑战牌组是否有假牌
- `evaluate_truth(state)`
  输出声明是否真实

[rules/roulette_rule.py](/home/user/workplace/repos/liars_bar/liars_game_engine/engine/rules/roulette_rule.py)
- `build_revolver_deck()`
  输出一份随机左轮序列
- `apply_penalty(player)`
  输入：玩家运行态
  输出：`RouletteOutcome`
  作用：触发俄罗斯轮盘，决定是否淘汰

**Agent 层**
[agents/base_agent.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/base_agent.py)
统一接口：

- `AgentDecision`
  输出结构：
  `thought`, `action`, `raw_output`, `parse_error`, `selected_skill`, `skill_parameters`, `decision_bias`
- `BaseAgent.act(observation)`
  所有 agent 都实现这个接口

[agents/factory.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/factory.py)
`build_agents(settings)`
输入：`AppSettings`
输出：`dict[player_id, BaseAgent]`
根据 `player.agent_type` 创建：
- `mock`
- `langchain`
- `llm`

[agents/liar_planner.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/liar_planner.py)
这是 SAVI skill 抽象的核心。

这里定义 5 个主 skill：
- `Truthful_Action`
- `Calculated_Bluff`
- `Aggressive_Deception`
- `Logical_Skepticism`
- `Strategic_Drain`

可选第 6 个：
- `Null_Probe_Skill`

关键对象和函数：

- `SKILL_DEFINITIONS`
  五 skill 的语义定义
- `ObservationParser.parse(observation)`
  输入：结构化 observation
  输出：给 LLM 看的人类可读局势文本
- `ParameterResolver.resolve_strategic_drain(hand, table_rank, skill_parameters)`
  输入：手牌、桌面牌型、`bluff_ratio/intended_total_cards`
  输出：具体真假牌组合与解析后的参数
- `SkillExecutioner.execute(selected_skill, skill_parameters, observation)`
  输入：skill 名、参数、observation
  输出：`(ActionModel, resolved_parameters)`
  作用：把 skill 变成真实动作
- `LiarPlanner.build_skill_prompt_block()`
  输出：一段 prompt，要求模型必须从 skill 列表中选一个
- `LiarPlanner.resolve_outcome(thought, selected_skill, skill_parameters, observation)`
  输入：thought + skill + params + observation
  输出：`PlannerOutcome`
  作用：合法性纠偏，然后产出最终动作

这就是 Mock 与 skill-first LLM 的公共执行骨架。

[agents/mock_agent.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/mock_agent.py)
`MockAgent.act(observation)`
输入：observation
输出：`AgentDecision`
机制：
- 不调用外部模型
- 直接用启发式规则先选 `selected_skill`
- 再交给 `LiarPlanner.resolve_outcome()`
- 日志里会稳定产出 `selected_skill` 和 `skill_parameters`

[agents/langchain_agent.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/langchain_agent.py)
这是 skill-first 的 LLM agent。

关键函数：

- `_build_planner_prompt(observation)`
  输入：observation
  输出：完整 planner prompt
  内容包括：
  `system`
  `SKILL_SYSTEM`
  `OBSERVATION_TEXT`
  `OBSERVATION_STRUCT`
- `act(observation)`
  输入：observation
  输出：`AgentDecision`
  机制：
  1. 调用模型
  2. 让模型输出 `{"thought","selected_skill","skill_parameters"}`
  3. `parse_planner_output()`
  4. `LiarPlanner.resolve_outcome()`

这个 agent 和 SAVI 的五 skill 抽象是完全对齐的。

[agents/llm_agent.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/llm_agent.py)
这是 Task M 新加的 action-first LLM agent。

关键函数：

- `act(observation)`
  输入：observation
  输出：`AgentDecision`
  机制：
  1. `build_openai_messages()`
  2. 调用后端
  3. 期待模型直接返回 `{"Reasoning","Action"}`
  4. `parse_agent_output()`
  5. 失败就 fallback 成 `challenge`
- `_request_raw_output(messages)`
  输入：chat messages
  输出：模型原始文本或 provider-unavailable fallback
  支持两条后端：
  `local://hf`
  普通 OpenAI-compatible HTTP
- `_provider_unavailable_decision(message)`
  输出：降级用 `AgentDecision`

注意：`LlmAgent` 不选择五个 skill，它直接选择最终动作。这是和 `MockAgent`、`LangChainAgent` 最大的结构差异。

[agents/prompts.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/prompts.py)
Task M 的 prompt 工程在这里。

关键函数：

- `load_prompt_profile(profile_name)`
  输入：profile 名
  输出：profile dict
- `format_observation_for_llm(observation)`
  输入：observation
  输出：三段式文本：
  `Status Report`
  `Qualitative Context`
  `Decision Request`
- `build_openai_messages(profile, observation)`
  输入：profile + observation
  输出：OpenAI-compatible `[{role, content}, ...]`

这里还会从 observation 派生：
- `诚实度参考`
- `人设稳定性`
- `轮盘死亡概率`
- `legal actions`

[agents/parsers.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/parsers.py)
这里有两类 parser：

- `parse_agent_output(raw_output)`
  解析 Task M 的 `Reasoning + Action`
- `parse_planner_output(raw_output)`
  解析 skill-first 的 `thought + selected_skill + skill_parameters`

容错能力包括：
- 支持 code fence JSON
- 支持正文里的花括号 JSON 抽取
- 支持字段别名归一

[agents/local_backend.py](/home/user/workplace/repos/liars_bar/liars_game_engine/agents/local_backend.py)
Task M 的本地 HF 后端。

关键函数：

- `_load_model_bundle(model_name, device, device_map)`
  输出：`tokenizer + model`
- `_build_prompt(messages, tokenizer)`
  把 chat messages 拼成模型可吃的 prompt
- `_generate_local_chat_completion_sync(...)`
  真正本地生成
- `generate_local_chat_completion(...)`
  异步包装，供 `LlmAgent` 调用

**实验与日志层**
[experiment/logger.py](/home/user/workplace/repos/liars_bar/liars_game_engine/experiment/logger.py)
`ExperimentLogger.record_turn(payload)`
输入：单回合 payload
输出：追加到 `*.jsonl`

[experiment/orchestrator.py](/home/user/workplace/repos/liars_bar/liars_game_engine/experiment/orchestrator.py)
这是每局游戏的主循环。

关键函数：

- `_build_state_features(observation, player_id)`
  输入：observation
  输出：六个核心状态特征：
  `phase`, `table_type`, `must_call_liar`, `alive_player_count`, `hand_count`, `death_probability`
- `run_game_loop()`
  输入：无
  输出：一局 summary
  每回合写入 JSONL 的字段包括：
  `turn`, `player_id`, `observation`, `thought`, `action`, `raw_output`, `skill_name`, `skill_parameters`, `state_features`, `checkpoint`, `parser_error`, `fallback_used`, `step_result`

这就是后续所有 Task C/D/I/K/L/N 能复用的统一日志格式。

[experiment/llm_drill.py](/home/user/workplace/repos/liars_bar/liars_game_engine/experiment/llm_drill.py)
这是 Task M 的 drill 运行器。

关键函数：

- `_extract_llm_turns(log_file, llm_player_id)`
  输入：一局 JSONL 日志
  输出：仅 LLM 玩家回合列表
- `select_representative_decisions(decisions, limit=3)`
  输出：代表性决策片段
- `select_high_risk_reasoning_snippets(decisions, limit=3, risk_threshold=1/3)`
  输出：高风险 reasoning 片段
- `run_llm_drill(settings, games=5, log_dir="logs/llm_drill")`
  输入：配置、局数、输出目录
  输出：summary dict
  落盘：
  `logs/llm_drill/games/*.jsonl`
  `logs/llm_drill/progress.log`
  `logs/llm_drill/high_risk_reasoning.json`
  `logs/llm_drill/summary.json`

**分析层核心**
[analysis/shapley_analyzer.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/shapley_analyzer.py)
这是整套 SAVI 的核心。

关键数据类：

- `TurnTrajectory`
  一条已记录决策轨迹
- `GameTrajectory`
  一局轨迹
- `ShapleyAttribution`
  单决策点归因结果：
  `value_action`, `value_counterfactual`, `phi`
- `CreditLedger`
  聚合归因台账

关键类：

- `LogIterator(log_paths)`
  输入：JSONL 文件列表
  输出：`list[GameTrajectory]`
- `ProxyValuePredictor(model_path)`
  输入：训练好的 `.pt`
  输出：可对 state/action 特征做快速打分的预测器
- `ShapleyAnalyzer(settings, rollout_samples, rollout_policy, max_workers, baseline_mode)`
  核心分析器

关键函数 / 方法：

- `_rollout_once(...)`
  输入：
  `settings_raw`, `encoded_checkpoint`, `initial_action_payload`, `target_player_id`, `sample_seed`, `rollout_policy`, `counterfactual`, `baseline_mode`
  输出：一次 rollout 的终局价值分数
- `attribute_step_rollout(trajectory, winner)`
  输入：一条日志轨迹
  输出：物理 rollout 的 `ShapleyAttribution`
- `attribute_step_proxy(trajectory, winner, predictor)`
  输入：轨迹 + proxy model
  输出：proxy 近似归因
- `analyze_logs(log_paths)`
  输入：JSONL 列表
  输出：`(attributions, ledger)`
  用物理 rollout 算 phi
- `analyze_logs_proxy(log_paths, predictor)`
  输入：JSONL 列表 + proxy
  输出：proxy 版 `(attributions, ledger)`
- `run_proxy_alignment(log_paths, predictor, sample_size, sample_seed)`
  输出：proxy 与 physical rollout 的 Pearson / MAE / speedup
- `export_credit_report(attributions, output_path)`
  输出：`credit_report.csv`
  聚合维度：
  `skill_name x death_prob_bucket`
- `summarize_probe_skill(attributions)`
  专门汇总 `Null_Probe_Skill`

**训练层**
[analysis/train_value_proxy.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/train_value_proxy.py)
当前 `dev-llm-agent` / `main` 上，这个文件的基础版本是“从日志提取 8 维特征，训练 MLP 预测 winner-style value”。

关键对象和函数：

- `ValueSample`
  单训练样本：`game_id`, `features`, `target`
- `ValueProxyMLP`
  输入维度 `8`
  输出 `1`
- `build_value_proxy_feature_context(...)`
  输入：
  `state_features`, `observation`, `player_id`, `action`
  输出：统一特征上下文字典
- `encode_value_proxy_features(state_features)`
  输出：8 维数值特征
- `load_value_samples(log_root)`
  输入：日志目录
  输出：训练样本列表
  在当前分支里，默认 target 还是 `winner==player ? 1.0 : 0.0`
- `split_train_val(samples, val_ratio, seed)`
  输出：train/val 样本
- `train_value_proxy(output_dir, log_root/log_roots, ...)`
  输出：metrics dict
  落盘：
  `value_proxy_mlp.pt`
  `value_proxy_metrics.json`

`dev-proxy-refine` 这条分支在这个基础上继续扩展成了 phi supervision / distill 路线。

**任务流水线**
[analysis/task_c_runner.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/task_c_runner.py)
Task C：物理 Shapley 反事实采样

关键函数：
- `_ensure_four_mock_players(settings)`
- `generate_baseline_logs(settings, game_count, log_dir)`
- `run_task_c_pipeline(...)`

输入：
- 配置
- 对局局数
- rollout_samples

输出：
- `logs/task_c/baseline_logs/*.jsonl`
- `logs/task_c/credit_report.csv`

[analysis/task_d_axiomatic_runner.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/task_d_axiomatic_runner.py)
Task D：Probe 公理验证

关键函数：
- `_build_probe_experiment_settings(settings)`
- `count_probe_actions(log_paths)`
- `compute_efficiency_error(...)`
- `compute_symmetry_deviation(...)`
- `compute_force_original_alignment(...)`
- `run_task_d_probe_pipeline(...)`

输出目录：
- `logs/task_d_probe/probe_logs/*.jsonl`
- `logs/task_d_probe/axiomatic_brief.json`
- `logs/task_d_probe/axiomatic_details.json`

[analysis/task_i_proxy_runner.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/task_i_proxy_runner.py)
Task I：proxy-based attribution

关键函数：
- `_select_latest_probe_logs(log_root, game_count)`
- `run_task_i_proxy_pipeline(...)`

流程：
1. 选最新一批 Task D logs
2. `train_value_proxy`
3. `run_proxy_alignment`
4. `analyze_logs_proxy`
5. 导出 alignment report 和 proxy credit report

输出：
- `logs/task_d_probe/proxy_alignment_report.json`
- `logs/task_d_probe/credit_report_proxy.csv`
- 模型与 metrics 默认也会落在 `logs/task_d_probe/`

[analysis/task_k_gold_runner.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/task_k_gold_runner.py)
Task K：金标准物理 rollout 大规模归因

关键函数：
- `_append_progress_log(...)`
- `_export_attributed_logs(...)`
- `run_task_k_gold_pipeline(...)`

流程：
1. 生成或复用 baseline logs
2. 对每批日志跑 `ShapleyAnalyzer.analyze_logs`
3. 把 phi 回填进原始日志
4. 输出最终 credit report

输出目录：
- `logs/task_k_gold/baseline_logs/*.jsonl`
- `logs/task_k_gold/attributed_logs/*.jsonl`
- `logs/task_k_gold/progress.log`
- `logs/task_k_gold/credit_report_final.csv`

`task-k-label-main` 上的 `task_k_gold_runner.py` 是这条线的过渡版本，当前 `main` 已继续前进，优先看 `main` 本身。

[analysis/task_l_proxy_refine_runner.py](/home/user/workplace/repos/liars_bar/liars_game_engine/analysis/task_l_proxy_refine_runner.py)
当前分支也有这个文件，但 `dev-proxy-refine` 上是增强版。它负责 Task L：

- 生成高 probe 概率的 negative logs
- 分别训练 elite model / mixed model
- 在同一批 alignment logs 上做 proxy 对齐比较
- 输出 refine 报告

输出目录在 `dev-proxy-refine` 上主要是：
- `logs/task_l_proxy_refine/negative_logs/*.jsonl`
- `logs/task_l_proxy_refine/elite_model/*.pt|*.json`
- `logs/task_l_proxy_refine/mixed_model/*.pt|*.json`
- `logs/task_l_proxy_refine/proxy_refine_report.json`

[dev-proxy-refine:audit_llm_behavior.py](git show dev-proxy-refine:liars_game_engine/analysis/audit_llm_behavior.py)
这条线相当于 Task N / LLM 行为审计。

关键函数：
- `classify_reasoning_confidence(reasoning)`
- `run_llm_behavior_audit(log_root, model_path, output_dir, phi_threshold, llm_player_id, summary_path)`

作用：
- 读取 Task M 的 LLM drill logs
- 用 proxy 预测 `phi_pred`
- 按 reasoning 的“强硬/保守”语气做冲突审计
- 找出 `phi_pred < threshold` 但 reasoning 很强硬的样本

输出：
- `logs/task_n_llm_behavior_audit/conflict_cases.jsonl`
- `logs/task_n_llm_behavior_audit/summary.json`

[dev-proxy-refine:task_k_backfill_labels.py](git show dev-proxy-refine:liars_game_engine/analysis/task_k_backfill_labels.py)
离线给已有 Task K baseline logs 回填 `shapley_value/phi`。

输出：
- `logs/task_k_gold/attributed_logs/*.jsonl`
- `logs/task_k_gold/progress.log`

[dev-proxy-refine:task_k_phi_distill_runner.py](git show dev-proxy-refine:liars_game_engine/analysis/task_k_phi_distill_runner.py)
直接从 Task K attributed logs 提取 `(State, Action, Phi)` 数据集并训练 phi proxy。

输出通常是：
- `task_k_phi_dataset.jsonl`
- `value_proxy_mlp_distill.pt`
- `value_proxy_metrics_distill.json`
- `diagnostics.jsonl`
- `task_k_phi_audit_report.json`

**脚本层**
[scripts/setup_server_env.sh](/home/user/workplace/repos/liars_bar/scripts/setup_server_env.sh)
作用：服务器初始化
输入：`CONDA_ENV_NAME`, `PYTHON_VERSION`, `INSTALL_VLLM`
输出：创建好 `liar_bar` 环境并安装依赖

[scripts/start_vllm.sh](/home/user/workplace/repos/liars_bar/scripts/start_vllm.sh)
作用：启动 vLLM 服务
输入：`VLLM_MODEL`, `VLLM_HOST`, `VLLM_PORT`, `VLLM_API_KEY`
输出：
- 后台 tmux 服务
- `logs/vllm/vllm-<port>.log`

[scripts/remote_run.sh](/home/user/workplace/repos/liars_bar/scripts/remote_run.sh)
作用：服务器执行入口
`RUN_MODE=task-k`
输出：
- 后台 `task_k_gold_runner`
- `logs/task_k_gold/server-run-*.log`
- `logs/task_k_gold/progress.log`

`RUN_MODE=llm`
输出：
- 后台 `python -m liars_game_engine.main`
- `logs/server-llm-*.log`

[scripts/run_llm_drill.py](/home/user/workplace/repos/liars_bar/scripts/run_llm_drill.py)
作用：Task M CLI 入口
输入：
- `--config`
- `--games`
- `--log-dir`
输出：
- 调用 `run_llm_drill`
- 打印 summary JSON

**日志与数据落盘**
当前仓库常见日志目录：

- `logs/runs/`
  默认 `main.py` 的单局运行日志
- `logs/task_c/baseline_logs/`
  Task C baseline JSONL
- `logs/task_d_probe/probe_logs/`
  Task D probe JSONL
- `logs/task_d_probe/`
  Task D briefs 与 Task I proxy 报表
- `logs/task_k_gold/`
  Task K baseline / attributed / progress / final report
- `logs/task_l_proxy_refine/`
  Task L negative logs / model / refine report
- `logs/llm_drill/games/`
  Task M drill per-game JSONL
- `logs/llm_drill/`
  Task M progress, summary, high-risk reasoning
- `logs/task_n_llm_behavior_audit/`
  只在 `dev-proxy-refine` 那条线使用，用于 LLM 行为审计
- `logs/task_h_reward_amplifier*`
  当前仓库里主要是历史产物目录，不对应现行生产 runner

**必须让主控台知道的关键事实**
- 项目的归因核心单位是 `skill_name`，这一点在 `MockAgent` 和 `LangChainAgent` 上是天然成立的，因为它们是 skill-first。
- 当前 `dev-llm-agent` 的 `LlmAgent` 是 action-first，不输出 `selected_skill`。这意味着 Task M 目前更适合“行为演习”，而不是直接进入按五 skill 聚合的 SAVI 主分析口径。
- `main` 是稳定基础线，偏 Task K 物理 rollout 与服务器执行。
- `dev-proxy-refine` 是 proxy / phi / distill / audit 线。
- `dev-llm-agent` 是 LLM 接入与 drill 线。
- 如果新主控台只看当前分支，会漏掉 `dev-proxy-refine` 的 Task K/L/N 代码；如果只看 `main`，又会漏掉 Task M 的 LLM drill 代码。

如果你要，我下一步可以继续做两件事之一：
1. 把这份说明压缩成“主控台决策摘要版”，只保留任务、分支、风险和建议。
2. 再进一步把“每个分支现在应该承担什么任务”明确写成一个调度建议。

## 2026-05-04 English-Only Hardening Note

- Task M prompt output is now expected to be English-only.
- SAVI dataset builders now reject CJK content in `thought`, prompt text, and chat message payloads.
- `feat-savi-grpo/scripts/train_savi_alignment.py` now rejects non-English dataset rows at load time.
- Existing Chinese Task M / Task 1.1 / Task 2.x artifacts should be treated as legacy references only.
- Before Task O.3 checkpoint work and any post-train comparison, rerun the English path in this order:
  1. Task M drill
  2. Task 1.1 EV-gap audit
  3. Task 2.1 / Task 2.3 dataset rebuild
