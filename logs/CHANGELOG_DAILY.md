# CHANGELOG DAILY

## 2026-04-25

### Session 1 - Task K label persistence repair
- 复核确认 `task_k_gold_runner` 仅导出聚合 `credit_report_final.csv`，没有把逐条 `phi/shapley_value` 回写进 JSONL，因此服务器同步回来的 `baseline_logs` 无法直接用于 `task_k_phi_distill_runner`。
- 新增 `attributed_logs/*.jsonl` 导出：按 `(game_id, turn, player_id)` 对齐归因结果，并把 `phi` 同步落盘到 `shapley_value` 与 `phi` 字段，同时补充 `value_action/value_counterfactual/winner/rollout_samples` 等审计字段。
- 为 `task_k_gold_runner` 新增 CLI 参数 `--game-count`、`--rollout-samples`、`--output-dir`、`--max-workers`，支持服务器先跑 `10x2` smoke test，再跑 `2000x200` 正式采样。
- 验证通过：`tests.test_task_k_task_l_runners`、`tests.test_task_k_phi_distill_runner` 全绿；本地真实 `1x1` smoke run 已落盘 `baseline_logs`、`attributed_logs` 与 `credit_report_final.csv`。

## 2026-04-17

### Session 1 - Task C Shapley sampling & algorithm validation pipeline
- 为 Orchestrator 回合日志新增 `state_features`（6 个核心状态特征），并保持 `checkpoint/skill_name/skill_parameters` 全量记录。
- 重构 `ShapleyAnalyzer`：反事实基线切换为“非原合法动作集合”，并加入 100 步回放终止与平局 `0.5` 计分。
- 新增 `credit_report.csv` 聚合导出能力，按 `skill_name + death_prob_bucket` 汇总平均 Shapley 值与采样数。
- 新增 `liars_game_engine.analysis.task_c_runner`，支持一键运行 4 Mock Agent × 50 局基线日志 + N=50 反事实采样。
- 新增/更新测试后，全量 `unittest` 通过（32/32）；完成真实流水线运行并生成 `logs/task_c/credit_report.csv`。

### Session 2 - Null Probe skill config injection (Option 2)
- 在 `runtime` 新增 `enable_null_player_probe` 配置开关，并贯通到 `build_agents -> Agent -> LiarPlanner`。
- 新增 `Null_Probe_Skill`（最小代价合法动作策略），并在输出参数中强制附带 `probe_type=Probe`。
- Orchestrator 日志新增 `log_version=v2_probe` 与 `skill_category`，使 Probe 轨迹可与旧日志版本隔离。
- ShapleyAnalyzer 新增 `summarize_probe_skill`，可单独汇总 `Null_Probe_Skill` 的 `phi` 统计。
- 新增相关测试后，全量 `unittest` 通过（35/35）。

### Session 3 - Task D probe experiment (100-game real run)
- 新增 `liars_game_engine.analysis.task_d_axiomatic_runner`，用于 100 局 Probe 模式采样与《公理验证简报》生成。
- 新增指标计算：`phi_probe_avg`、`efficiency_total_bias`、`symmetry_deviation`，并分别导出 brief/details JSON。
- 执行真实流水线（4 Mock Agent + enable_null_player_probe + rollout_samples=100），生成 100 局日志与验证简报。
- 本次真实结果：`phi_probe_avg=0.0913`，`efficiency_total_bias=0.4331`，`symmetry_deviation=0.1351`。
- 全量 `unittest` 回归通过（38/38）。

## 2026-04-23

### Session 1 - Task I proxy-based attribution
- 在 `ShapleyAnalyzer` 中新增 `ProxyValuePredictor`、单步 successor-state 代理归因、20 点双线对齐报告导出，以及 `analyze_logs_proxy` 快速分析路径。
- 新增 `liars_game_engine.analysis.task_i_proxy_runner`，默认选择 `logs/task_d_probe/probe_logs/` 中最新一批 100 局日志生成 `proxy_alignment_report.json` 和 `credit_report_proxy.csv`。
- 新增 proxy 归因测试，覆盖特征缩放一致性、合法动作均值归因与对齐报告字段；定向测试通过。
- 执行真实 Task I 流水线：速度提升约 `226.25x`，但 Pearson 相关系数仅 `-0.1974`，未通过 `>0.75` 对齐门槛。
- 全量 `unittest` 回归通过（45/45）；当前 proxy 报表仅保留为参考结果，不替代物理 rollout。

### Session 2 - Task J proxy forensics & feature expansion assessment
- 复原 Task I 的 20 个对齐点，新增 `logs/task_d_probe/proxy_forensics_report.json`，可直接查看逐点 `rollout_phi/proxy_phi/state_features` 与前 3 个离群样本。
- 新增 `logs/task_d_probe/value_proxy_saliency.json`，对当前 6 维输入做初步梯度显著性分析；结果显示 `alive_player_count` 与 `death_probability` 比 `hand_count` 更主导。
- 法医结论：负相关更像“缺少 claim/inventory 语义特征”，不是 proxy 输入顺序错位，也不是单纯过拟合 `hand_count`。
- 记录决策：EXP1/论文数据默认继续使用物理 rollout；若扩展到 8 维（如 `hand_truth_ratio`、`inventory_logic_conflict`），可复用现有 JSONL 日志重新提特征并重训，无需重跑原始对局采样。

### Session 3 - Task J feature engineering upgrade
- 将 value proxy 特征从 6 维升级到 8 维，新增 `hand_truth_ratio` 与 `action_consistency_score`，并把训练/推理统一到同一编码入口。
- `task_i_proxy_runner` 现在会先基于现有 11,127 条日志样本重训 40 epoch，再执行 20 点对齐与 100 局 proxy 报表生成。
- 真实结果：`val_mse=0.2047`、`best_val_mse=0.2047`、`pearson=0.3339`、`mae=0.0948`、`speedup≈451.45x`；Pearson 已从负值转正，但仍未达到主控目标。
- 全量 `unittest` 回归通过（46/46）；当前结论维持不变：EXP1/论文数据继续以物理 rollout 为准，8D proxy 仅作辅助观测。

### Session 4 - Task K/L branch wiring and negative-sampling run
- 新增 `runtime.null_probe_action_probability` 并贯通到 `MockAgent`，使 Task L 可显式提高 `Null_Probe_Skill` 触发概率来收集“作死”轨迹。
- 新增 `liars_game_engine.analysis.task_k_gold_runner`，为 4090 `main` 分支准备 4 Mock + `rollout_samples=200` + `credit_report_final.csv` 的金标准物理归因入口；本次未在本地执行大规模采样。
- 新增 `liars_game_engine.analysis.task_l_proxy_refine_runner`，支持生成负采样日志、混合重训 `value_proxy_mlp_v2.pt`、并对比 elite vs mixed 的 proxy alignment。
- 执行真实 Task L 流水线：生成 `10,438` 条负采样记录（300 局），混合模型将 Pearson 从 `0.3339` 提升到 `0.4249`，但 `val_mse` 由 `0.2047` 恶化到 `0.2088`，仍未达到 `Pearson > 0.6`。
- 回归验证：全量 `unittest discover tests -v` 通过（50/50）；产物已落盘到 `logs/task_l_proxy_refine/`。

## 2026-04-24

### Session 1 - 8D stop-loss retrain after abandoning 9D
- 按主控要求将 `dev-proxy-refine` 回滚到 `d2a3bc7`，放弃 9D 特征方案，仅保留 8D（`hand_truth_ratio` + `action_consistency_score`）。
- 在 `train_value_proxy.py` 中新增训练样本过滤：整局跳过少于 `3` 回合的对局日志。
- 新增定向单测验证短局日志会被过滤，并完成相关回归测试。
- 基于现有 `logs/task_d_probe/probe_logs` 与 `logs/task_l_proxy_refine/negative_logs` 重新训练 elite/mixed 两个 8D proxy，并重跑 alignment。
- 实际结果：elite/negative 两批日志中的短局数都为 `0`，因此过滤未改变训练集；重训后 elite `val_mse=0.2047`、Pearson=`0.3339`，mixed `val_mse=0.2088`、Pearson=`0.4249`，与回滚基线基本一致。

## 2026-03-19

### Session 2 - Project renaming to generic social-deduction scope
- 将 Python 包目录由 `liars_bar/` 重命名为 `social_deduction_lab/`，消除仓库名与包名重复带来的歧义。
- 全量更新源码与测试 import 路径到 `social_deduction_lab.*`。
- 更新项目元信息：`pyproject.toml` 项目名改为 `social-deduction-lab`。
- 更新 `README.md` 与 `PROJECT_MEMORY.md`，将项目定位扩展为“狼人杀/骗子酒馆/阿瓦隆”等社交推理实验框架。
- 运行全量 `unittest` 验证改名后行为一致。

### Session 3 - Package rename aligned with current scope
- 将 Python 包目录由 `social_deduction_lab/` 调整为 `liars_game_engine/`，与当前“仅实现骗子酒馆规则”的逻辑范围保持一致。
- 全量更新源码、测试、计划文档中的 import 与路径引用到 `liars_game_engine.*`。
- 调整 `pyproject.toml` 项目名为 `liars-game-engine`，并同步回收描述为《骗子酒馆》引擎定位。
- 更新 `README.md`、`PROJECT_MEMORY.md` 与包内说明文案，避免“多游戏平台”语义超前。
- 运行全量 `unittest`，验证重命名后功能不变。

### Session 1 - Basic rules alignment
- 基于用户提供的官方基础规则，重构 Environment 回合流程与判定逻辑。
- 对齐每轮机制：翻桌牌（A/K/Q）、Liar 牌堆重洗重发、`table_type + JOKER` Innocent 判定。
- 对齐行动规则：`play_claim` 支持响应窗口继续出牌、限制每次 1~3 张、仅剩单人持牌时强制 call LIAR。
- 对齐 LIAR 结算：按“抓到骗子/错报 LIAR”决定受罚者，并按规则决定下一轮起手位。
- 将轮盘惩罚改为“左轮牌堆翻顶牌（LETHAL/BLANK）”模型。
- 更新 `MockAgent`，支持 1~3 张出牌与强制 LIAR 场景。
- 新增/更新环境测试，覆盖规则偏差点；全量 `unittest` 通过（11/11）。

## 2026-03-18

### Session 1 - Project bootstrap
- 新建 `liars_bar` 项目并落地异步事件循环架构。
- 完成单一实验配置中心：`.env`（敏感项）+ `config/experiment.yaml`（所有实验参数）。
- 完成阶段化状态机与规则模块（声明/质疑/轮盘惩罚）。
- 完成 `MockAgent` 默认可跑模式与 `LangChainAgent` OpenRouter 接入骨架。
- 完成宽松输出解析与结构化错误码体系。
- 完成 Orchestrator + JSONL 实验日志。
- 新增项目记忆文件与两级变更日志维护机制。
- 完成内置 `unittest` 测试闭环（配置、解析、环境、编排、策略文件存在性）。
- 新增 `.gitignore`，忽略缓存文件、`.env` 与运行日志输出。
