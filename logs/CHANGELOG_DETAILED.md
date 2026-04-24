# CHANGELOG DETAILED

## 2026-04-23 / Session 1

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `docs/plans/2026-04-23-proxy-based-attribution.md` | 1 | `+` | 新增 Task I 实施计划，明确 proxy 归因、双线对齐校验与验证步骤。 |
| `liars_game_engine/analysis/shapley_analyzer.py` | 108 | `~` | 新增 `ProxyValuePredictor`、状态特征复用编码、单步 successor-state proxy 归因、对齐指标计算与报告导出。 |
| `liars_game_engine/analysis/task_i_proxy_runner.py` | 1 | `+` | 新增 Task I 运行入口，默认选择最新一批 100 局 Probe 日志生成 proxy 对齐报告与信用报表。 |
| `tests/test_shapley_analyzer.py` | 437 | `~` | 新增 proxy 特征缩放一致性、合法动作均值归因和 alignment 报告字段测试。 |
| `README.md` | 69 | `~` | 新增 Task I 命令、输出路径与“相关性未过线仅供参考”的说明。 |
| `PROJECT_MEMORY.md` | 53 | `~` | 新增 Phase 5D 记录，并写明 proxy 对齐门槛未达标前不可替代物理 rollout。 |
| `logs/CHANGELOG_DAILY.md` | 26 | `~` | 记录 Task I 实现、真实结果与测试回归情况。 |
| `logs/task_d_probe/proxy_alignment_report.json` | 1 | `+` | 新增 20 点双线对齐报告（Pearson、MAE、Speedup）。 |
| `logs/task_d_probe/credit_report_proxy.csv` | 1 | `+` | 新增 100 局 proxy 快速归因汇总报表。 |

## 2026-04-23 / Session 2

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `docs/plans/2026-04-23-proxy-quality-diagnosis.md` | 1 | `+` | 新增 Task J 法医诊断计划，约束离群点复原、特征一致性核验与 8 维扩展评估。 |
| `logs/task_d_probe/proxy_forensics_report.json` | 1 | `+` | 新增 Task I 同批 20 点的逐点诊断结果，含 top-3 绝对误差/符号反转样本。 |
| `logs/task_d_probe/value_proxy_saliency.json` | 1 | `+` | 新增当前 6 维代理模型的梯度显著性与首层权重统计。 |
| `PROJECT_MEMORY.md` | 63 | `~` | 补充 Task J 结论：根因更像缺失 claim/inventory 特征，8 维扩展可复用现有日志重训，EXP1 默认继续走物理 rollout。 |
| `logs/CHANGELOG_DAILY.md` | 35 | `~` | 记录法医诊断、saliency 结果和 EXP1 决策。 |

## 2026-04-23 / Session 3

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `liars_game_engine/analysis/train_value_proxy.py` | 29 | `~` | 将 value proxy 编码从 6 维扩展到 8 维，新增 `hand_truth_ratio` 与 `action_consistency_score`，并把训练默认 epoch 调整为 40。 |
| `liars_game_engine/analysis/shapley_analyzer.py` | 14 | `~` | `ProxyValuePredictor` 改为直接复用训练模块的 8D 编码与输入维度常量；proxy successor-state 特征上下文携带 action/hand 信息。 |
| `liars_game_engine/analysis/task_i_proxy_runner.py` | 11 | `~` | Task I 流水线改为先基于现有日志重训 8D 代理，再执行对齐与 proxy 报表导出，并回传训练指标。 |
| `tests/test_shapley_analyzer.py` | 443 | `~` | 新增 8D 编码内容、训练编码复用和 8D proxy attribution 的回归测试。 |
| `PROJECT_MEMORY.md` | 63 | `~` | 记录 8D 升级后的真实指标与“仍不能替代物理 rollout”的结论。 |
| `logs/CHANGELOG_DAILY.md` | 41 | `~` | 记录 Task J 特征升级、重训结果与全量测试验证。 |
| `logs/task_d_probe/value_proxy_metrics.json` | 1 | `~` | 训练指标更新为 8D/40epoch 结果（11,127 样本，val MSE 0.2047）。 |
| `logs/task_d_probe/proxy_alignment_report.json` | 1 | `~` | 对齐结果更新为 8D 模型输出（Pearson 0.3339，MAE 0.0948，Speedup 451.45x）。 |
| `logs/task_d_probe/credit_report_proxy.csv` | 1 | `~` | 100 局 proxy 信用报表更新为 8D 模型生成结果。 |

## 2026-04-23 / Session 4

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `liars_game_engine/config/schema.py` | 15 | `~` | `RuntimeSettings` 增加 `null_probe_action_probability: float = 0.12`，用于控制 MockAgent 的 Probe 触发概率。 |
| `config/experiment.yaml` | 3 | `~` | 新增 `runtime.null_probe_action_probability` 默认配置。 |
| `liars_game_engine/agents/mock_agent.py` | 11 | `~` | `MockAgent` 支持注入 Probe 概率，并用该值替代硬编码的 `0.12`。 |
| `liars_game_engine/agents/factory.py` | 33 | `~` | 构建 MockAgent 时透传 `settings.runtime.null_probe_action_probability`。 |
| `liars_game_engine/analysis/train_value_proxy.py` | 275 | `~` | 新增多日志根目录加载能力，并允许自定义模型/指标文件名，供 Task L 训练 elite/mixed 两套模型。 |
| `liars_game_engine/analysis/task_k_gold_runner.py` | 1 | `+` | 新增 Task K 物理金标准采样入口，导出 `logs/task_k_gold/credit_report_final.csv` 并统计平均归因耗时。 |
| `liars_game_engine/analysis/task_l_proxy_refine_runner.py` | 1 | `+` | 新增 Task L 负采样、混合重训、alignment 对比与 `proxy_refine_report.json` 导出入口。 |
| `tests/test_mock_agent_pipeline.py` | 106 | `~` | 新增 Probe 概率可配置断言，确保 `1.0` 时稳定触发 `Null_Probe_Skill`。 |
| `tests/test_task_k_task_l_runners.py` | 1 | `+` | 新增 Task K/Task L runner 单测，锁定返回摘要字段与负采样配置行为。 |
| `PROJECT_MEMORY.md` | 101 | `~` | 记录 Task K/L 入口约束、Task L 本地实跑结果与“仍不可替代物理 rollout”的结论。 |
| `logs/CHANGELOG_DAILY.md` | 47 | `~` | 记录 Task K/L 接线、Task L 实跑结果与全量测试回归。 |
| `logs/task_l_proxy_refine/proxy_refine_report.json` | 1 | `+` | 新增 elite vs mixed proxy 的对齐对比报告。 |
| `logs/task_l_proxy_refine/elite_model/value_proxy_metrics.json` | 1 | `+` | 新增 Task L elite-only 重训指标。 |
| `logs/task_l_proxy_refine/elite_model/value_proxy_mlp.pt` | 1 | `+` | 新增 Task L elite-only 模型权重。 |
| `logs/task_l_proxy_refine/mixed_model/value_proxy_metrics_v2.json` | 1 | `+` | 新增负采样混合训练后的 v2 模型指标。 |
| `logs/task_l_proxy_refine/mixed_model/value_proxy_mlp_v2.pt` | 1 | `+` | 新增负采样混合训练后的 v2 模型权重。 |
| `logs/task_l_proxy_refine/negative_logs/*.jsonl` | 1 | `+` | 新增 300 局高 Probe 负采样轨迹，共 10,438 条记录。 |

## 2026-04-24 / Session 1

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `liars_game_engine/analysis/train_value_proxy.py` | 230 | `~` | 在 value proxy 训练样本加载阶段新增“少于 3 回合整局跳过”过滤，作为 8D 路线的止损数据清洗。 |
| `tests/test_shapley_analyzer.py` | 139 | `~` | 新增 `load_value_samples()` 会跳过少于 3 回合日志的回归测试。 |
| `PROJECT_MEMORY.md` | 65 | `~` | 记录 8D 止损重训结果，并说明现有 elite/negative 日志的短局数均为 0，过滤未改变训练集。 |
| `logs/CHANGELOG_DAILY.md` | 54 | `~` | 记录 8D 止损回滚、短局过滤和重训结果。 |
| `logs/task_l_proxy_refine/elite_model/value_proxy_metrics.json` | 1 | `~` | 重新写入 8D elite-only 重训指标。 |
| `logs/task_l_proxy_refine/elite_model/value_proxy_mlp.pt` | 1 | `~` | 重新写入 8D elite-only 模型权重。 |
| `logs/task_l_proxy_refine/mixed_model/value_proxy_metrics_v2.json` | 1 | `~` | 重新写入 8D mixed 重训指标。 |
| `logs/task_l_proxy_refine/mixed_model/value_proxy_mlp_v2.pt` | 1 | `~` | 重新写入 8D mixed 模型权重。 |
| `logs/task_l_proxy_refine/proxy_refine_report.json` | 1 | `~` | 更新为 8D 止损重训后的 alignment 对比报告，并补充 elite/negative 短局计数为 0。 |

## 2026-04-17 / Session 3

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `liars_game_engine/analysis/task_d_axiomatic_runner.py` | 1 | `+` | 新增 Task D Probe 采样运行器（100 局对战 + 公理验证简报计算）。 |
| `tests/test_task_d_axiomatic_runner.py` | 1 | `+` | 新增有效性误差与对称性偏离计算的单元测试。 |
| `README.md` | 56 | `~` | 新增 Task D 命令与输出路径说明。 |
| `PROJECT_MEMORY.md` | 72 | `~` | 补充 Task D 运行器入口约束。 |
| `logs/CHANGELOG_DAILY.md` | 19 | `~` | 新增 Session 3 概述与真实实验结果。 |
| `logs/task_d_probe/axiomatic_brief.json` | 1 | `+` | 新增公理验证简报（三项核心指标）。 |
| `logs/task_d_probe/axiomatic_details.json` | 1 | `+` | 新增 Probe/效率/对称性详细统计结果。 |
| `logs/task_d_probe/probe_logs/*.jsonl` | 1 | `+` | 新增 100 局 Probe 模式对战轨迹日志。 |

## 2026-04-17 / Session 2

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `docs/plans/2026-04-17-null-probe-injection.md` | 1 | `+` | 新增 Null Probe 配置注入与日志/分析对齐实现计划。 |
| `liars_game_engine/config/schema.py` | 13 | `~` | `RuntimeSettings` 增加 `enable_null_player_probe: bool = False`。 |
| `config/experiment.yaml` | 7 | `~` | 增加 `runtime.enable_null_player_probe` 默认值。 |
| `liars_game_engine/agents/factory.py` | 20 | `~` | 将 Null Probe 开关注入 Mock/LangChain Agent 初始化。 |
| `liars_game_engine/agents/mock_agent.py` | 10 | `~` | 支持 Probe 开关，启用时可选择 `Null_Probe_Skill`。 |
| `liars_game_engine/agents/langchain_agent.py` | 10 | `~` | Planner 初始化支持 Probe 开关；重试提示动态使用可用技能集合。 |
| `liars_game_engine/agents/parsers.py` | 12 | `~` | `VALID_SKILLS` 增加 `Null_Probe_Skill`。 |
| `liars_game_engine/agents/liar_planner.py` | 8 | `~` | 新增 Null Probe 技能注册、最小代价动作执行逻辑、可用技能校验与 Probe 参数标记。 |
| `liars_game_engine/experiment/orchestrator.py` | 11 | `~` | 日志新增 `log_version=v2_probe` 与 `skill_category`（Probe/Standard）。 |
| `liars_game_engine/analysis/shapley_analyzer.py` | 18 | `~` | 新增 `summarize_probe_skill` 专项统计接口。 |
| `tests/test_config_loader.py` | 28 | `~` | 增加 `enable_null_player_probe` 配置加载断言。 |
| `tests/test_mock_agent_pipeline.py` | 7 | `~` | 新增 Agent 工厂挂载 Null Probe 技能断言。 |
| `tests/test_liar_planner.py` | 122 | `~` | 新增 `Null_Probe_Skill` Probe 标记与合法动作断言。 |
| `tests/test_orchestrator_logging.py` | 93 | `~` | 新增 `log_version` 与 `skill_category` 日志字段断言。 |
| `tests/test_shapley_analyzer.py` | 226 | `~` | 新增 `summarize_probe_skill` 专项统计测试。 |
| `README.md` | 36 | `~` | 配置说明新增 `runtime.enable_null_player_probe`。 |
| `PROJECT_MEMORY.md` | 70 | `~` | 补充 Null Probe 注入规则与 `v2_probe` 日志版本约束。 |
| `logs/CHANGELOG_DAILY.md` | 12 | `~` | 新增 Session 2 概述。 |

## 2026-04-17 / Session 1

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `docs/plans/2026-04-17-shapley-sampling-validation.md` | 1 | `+` | 新增 Task C 实施计划，明确 TDD、50 局基线日志与 N=50 反事实采样步骤。 |
| `tests/test_orchestrator_logging.py` | 93 | `~` | 新增 `state_features` 字段断言，要求日志包含 6 个核心状态特征。 |
| `tests/test_shapley_analyzer.py` | 1 | `~` | 新增反事实动作排除原动作、100 步平局计分、`credit_report.csv` 导出字段测试。 |
| `liars_game_engine/experiment/orchestrator.py` | 39 | `~` | 新增 `state_features` 组装逻辑并写入回合日志。 |
| `liars_game_engine/analysis/shapley_analyzer.py` | 1 | `~` | 重构反事实采样核心：非原合法动作 baseline、100 步终止保护、平局 0.5 计分、CSV 聚合导出。 |
| `liars_game_engine/analysis/task_c_runner.py` | 1 | `+` | 新增任务 C 一键流水线入口（4 Mock Agent × 50 局 + Shapley 分析 + 报表输出）。 |
| `PROJECT_MEMORY.md` | 93 | `~` | 阶段更新为 Phase 5C 完成，并补充 Task C 反事实基线与回放上限约束。 |
| `logs/CHANGELOG_DAILY.md` | 3 | `~` | 新增 2026-04-17 会话级摘要。 |

## 2026-03-19 / Session 1

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `docs/plans/2026-03-19-basic-rules-alignment-implementation.md` | 1 | `+` | 新增本轮规则对齐实现计划（TDD任务拆解）。 |
| `tests/test_environment_pipeline.py` | 1 | `~` | 重写环境测试，覆盖响应窗口继续出牌、1~3张限制、Innocent/Liar判定、轮结束重发牌、强制LIAR。 |
| `liars_bar/engine/game_state.py` | 1 | `~` | 扩展运行态字段：`round_index`、`table_type`、`first_turn_of_round`、`revolver_deck`。 |
| `liars_bar/engine/environment.py` | 1 | `~` | 重构核心状态机：每轮桌牌、发牌、LIAR结算、下一轮起手位继承与跳过淘汰玩家。 |
| `liars_bar/engine/rules/declare_rule.py` | 1 | `~` | 允许响应窗口继续出牌；增加每次出牌 1~3 张约束与重复牌计数校验。 |
| `liars_bar/engine/rules/challenge_rule.py` | 1 | `~` | 真伪判定改为 `table_type + JOKER` 为 Innocent，其余为 Liar。 |
| `liars_bar/engine/rules/roulette_rule.py` | 1 | `~` | 轮盘惩罚改为“翻左轮牌堆顶牌”模型（`LETHAL/BLANK`）。 |
| `liars_bar/agents/mock_agent.py` | 1 | `~` | 更新 mock 决策：支持 1~3 张出牌，识别 `must_call_liar` 场景并优先 challenge。 |
| `liars_bar/config/schema.py` | 1 | `~` | 默认 `deck_ranks` 调整为 `[A, K, Q, JOKER]`。 |
| `config/experiment.yaml` | 27 | `~` | 默认规则配置同步使用 `JOKER` 标记。 |
| `PROJECT_MEMORY.md` | 53 | `~` | 更新阶段进度：Phase 5A 基础规则对齐已完成。 |
| `logs/CHANGELOG_DAILY.md` | 3 | `~` | 新增 2026-03-19 会话概述。 |

## 2026-03-19 / Session 2

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `liars_bar/` -> `social_deduction_lab/` | 1 | `~` | 包目录重命名，统一为可覆盖多社交推理游戏的命名。 |
| `social_deduction_lab/**/*.py` | 1 | `~` | 全量更新内部 import 前缀到 `social_deduction_lab`。 |
| `tests/*.py` | 1 | `~` | 更新测试导入路径以适配新包名。 |
| `pyproject.toml` | 1 | `~` | 项目名由 `liars-bar` 调整为 `social-deduction-lab`，并更新描述。 |
| `README.md` | 1 | `~` | 标题与项目介绍改为通用社交推理实验框架。 |
| `PROJECT_MEMORY.md` | 1 | `~` | 项目定位从单一《骗子酒馆》扩展为多社交推理游戏框架。 |
| `docs/plans/2026-03-18-liars-bar-bootstrap.md` | 1 | `~` | 计划文档中的包路径引用同步为 `social_deduction_lab`。 |
| `docs/plans/2026-03-19-basic-rules-alignment-implementation.md` | 1 | `~` | 计划文档中的包路径引用同步为 `social_deduction_lab`。 |
| `logs/CHANGELOG_DAILY.md` | 3 | `~` | 增加 Session 2 的重命名变更摘要。 |

## 2026-03-19 / Session 3

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `social_deduction_lab/` -> `liars_game_engine/` | 1 | `~` | 包目录改为与当前单游戏逻辑一致的命名。 |
| `liars_game_engine/**/*.py` | 1 | `~` | 更新内部 import 前缀到 `liars_game_engine`。 |
| `tests/*.py` | 1 | `~` | 测试导入路径同步为 `liars_game_engine`。 |
| `pyproject.toml` | 1 | `~` | 项目名改为 `liars-game-engine`，描述回归《骗子酒馆》引擎定位。 |
| `README.md` | 1 | `~` | 标题改为 Liar's Game Engine，并保持包入口命令一致。 |
| `PROJECT_MEMORY.md` | 1 | `~` | 项目定位改回《骗子酒馆》专用引擎描述。 |
| `liars_game_engine/__init__.py` | 1 | `~` | 顶层包说明文案与当前定位保持一致。 |
| `logs/CHANGELOG_DAILY.md` | 3 | `~` | 新增 Session 3 变更摘要。 |

## 2026-03-18 / Session 1

| File | Start Line | Change | Detail |
|---|---:|:---:|---|
| `docs/plans/2026-03-18-liars-bar-design.md` | 1 | `+` | 新增设计文档，定义架构、配置、错误与测试策略。 |
| `docs/plans/2026-03-18-liars-bar-bootstrap.md` | 1 | `+` | 新增分任务实现计划（TDD顺序）。 |
| `pyproject.toml` | 1 | `+` | 新增项目元信息与依赖声明。 |
| `.env.example` | 1 | `+` | 新增 OpenRouter 环境变量模板。 |
| `config/experiment.yaml` | 1 | `+` | 新增单一实验配置文件（玩家/规则/日志/解析器）。 |
| `liars_bar/config/schema.py` | 1 | `+` | 新增配置数据模型。 |
| `liars_bar/config/loader.py` | 1 | `+` | 新增 `.env + YAML` 合并加载逻辑。 |
| `liars_bar/engine/game_state.py` | 1 | `+` | 新增动作、解析结果、游戏阶段、运行态模型。 |
| `liars_bar/engine/environment.py` | 1 | `+` | 新增 phase pipeline 状态机与 step 结算主逻辑。 |
| `liars_bar/engine/rules/base.py` | 1 | `+` | 新增规则模块基类与规则返回结构。 |
| `liars_bar/engine/rules/declare_rule.py` | 1 | `+` | 新增声明动作校验规则。 |
| `liars_bar/engine/rules/challenge_rule.py` | 1 | `+` | 新增质疑动作校验与真伪判定。 |
| `liars_bar/engine/rules/roulette_rule.py` | 1 | `+` | 新增轮盘惩罚逻辑。 |
| `liars_bar/agents/parsers.py` | 1 | `+` | 新增宽松解析器（代码块提取、别名映射、错误码）。 |
| `liars_bar/agents/base_agent.py` | 1 | `+` | 新增 Agent 抽象基类与决策结构。 |
| `liars_bar/agents/mock_agent.py` | 1 | `+` | 新增默认离线智能体实现。 |
| `liars_bar/agents/langchain_agent.py` | 1 | `+` | 新增 LangChain/OpenRouter 接入骨架与降级策略。 |
| `liars_bar/agents/prompts.py` | 1 | `+` | 新增 Prompt profile 加载与拼装逻辑。 |
| `liars_bar/agents/factory.py` | 1 | `+` | 新增多玩家 agent 构建工厂。 |
| `liars_bar/experiment/logger.py` | 1 | `+` | 新增 JSONL 实验日志记录器。 |
| `liars_bar/experiment/orchestrator.py` | 1 | `+` | 新增异步主循环调度器与 fallback 流程。 |
| `liars_bar/main.py` | 1 | `+` | 新增程序启动入口。 |
| `prompts/profiles/baseline.yaml` | 1 | `+` | 新增基线 Prompt profile。 |
| `tests/test_config_loader.py` | 1 | `+` | 新增配置加载测试。 |
| `tests/test_action_parser.py` | 1 | `+` | 新增解析容错与错误码测试。 |
| `tests/test_environment_pipeline.py` | 1 | `+` | 新增阶段流转与惩罚结算测试。 |
| `tests/test_orchestrator_logging.py` | 1 | `+` | 新增编排器日志落盘测试。 |
| `tests/test_project_policy_files.py` | 1 | `+` | 新增策略文件存在性测试。 |
| `PROJECT_MEMORY.md` | 1 | `+` | 新增项目记忆与强制维护规范。 |
| `logs/CHANGELOG_DAILY.md` | 1 | `+` | 新增每日概述日志。 |
| `logs/CHANGELOG_DETAILED.md` | 1 | `+` | 新增详细改动日志。 |

### Line-level delta summary (`+` means added lines)

| File | Delta |
|---|---:|
| `docs/plans/2026-03-18-liars-bar-design.md` | `+71` |
| `docs/plans/2026-03-18-liars-bar-bootstrap.md` | `+210` |
| `pyproject.toml` | `+21` |
| `.env.example` | `+4` |
| `.env` | `+3` |
| `config/experiment.yaml` | `+31` |
| `liars_bar/config/schema.py` | `+71` |
| `liars_bar/config/loader.py` | `+42` |
| `liars_bar/engine/game_state.py` | `+81` |
| `liars_bar/engine/environment.py` | `+170` |
| `liars_bar/engine/rules/base.py` | `+22` |
| `liars_bar/engine/rules/declare_rule.py` | `+45` |
| `liars_bar/engine/rules/challenge_rule.py` | `+36` |
| `liars_bar/engine/rules/roulette_rule.py` | `+37` |
| `liars_bar/agents/parsers.py` | `+97` |
| `liars_bar/agents/base_agent.py` | `+26` |
| `liars_bar/agents/mock_agent.py` | `+45` |
| `liars_bar/agents/langchain_agent.py` | `+83` |
| `liars_bar/agents/prompts.py` | `+45` |
| `liars_bar/agents/factory.py` | `+34` |
| `liars_bar/experiment/logger.py` | `+19` |
| `liars_bar/experiment/orchestrator.py` | `+93` |
| `liars_bar/main.py` | `+37` |
| `prompts/profiles/baseline.yaml` | `+12` |
| `tests/test_config_loader.py` | `+66` |
| `tests/test_action_parser.py` | `+37` |
| `tests/test_environment_pipeline.py` | `+65` |
| `tests/test_orchestrator_logging.py` | `+77` |
| `tests/test_project_policy_files.py` | `+20` |
| `PROJECT_MEMORY.md` | `+64` |
| `logs/CHANGELOG_DAILY.md` | `+13` |
| `logs/CHANGELOG_DETAILED.md` | `+37` (initial section) |
| `README.md` | `+47` |
| `.gitignore` | 1 | `+` | 新增忽略规则（`__pycache__`、`.env`、运行日志）。 |

> 附加增量：`.gitignore` `+4`。
