# Liar's Game Engine (`liars_game_engine`)

一个基于异步事件循环的《骗子酒馆》LLM 多智能体实验项目。

## 目录
- `liars_game_engine/engine/`: 游戏环境与规则模块
- `liars_game_engine/agents/`: 智能体、Prompt、输出解析
- `liars_game_engine/experiment/`: 回合编排与日志记录
- `config/experiment.yaml`: 单一实验配置文件（可统一修改所有实验参数）
- `.env` / `.env.example`: API 敏感配置
- `prompts/profiles/`: Prompt 实验模板
- `PROJECT_MEMORY.md`: 项目记忆与长期约束
- `logs/CHANGELOG_DAILY.md`: 每次修改概览
- `logs/CHANGELOG_DETAILED.md`: 详细修改记录

## 快速开始
1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```
2. 填写 `.env` 中的 OpenRouter 配置。
3. 根据实验需要修改 `config/experiment.yaml`。
4. 运行：
   ```bash
   python -m liars_game_engine.main
   ```

## 配置说明
- `.env`（敏感配置）
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_BASE_URL`
  - `OPENROUTER_TIMEOUT_SECONDS`
- `config/experiment.yaml`（非敏感参数）
  - 运行参数、规则参数、日志参数
  - 多玩家配置（每个玩家可不同模型）
  - Prompt profile 绑定
  - `runtime.enable_null_player_probe`：是否启用 `Null_Probe_Skill`（默认 `false`）

## 测试
当前仓库默认使用 `unittest`（离线环境无需额外安装）：

```bash
python -m unittest discover tests -v
```

## Task C（Shapley 反事实采样）
一键执行 4 名 Mock Agent 的 50 局基线对战、N=50 反事实回放采样与信用报表导出：

```bash
python -m liars_game_engine.analysis.task_c_runner
```

输出目录：
- `logs/task_c/baseline_logs/*.jsonl`
- `logs/task_c/credit_report.csv`

## Task D（Probe 公理验证采样）
一键执行 100 局 Probe 模式（4 Mock Agent + `Null_Probe_Skill`）并生成公理验证简报：

```bash
python -m liars_game_engine.analysis.task_d_axiomatic_runner
```

输出目录：
- `logs/task_d_probe/probe_logs/*.jsonl`
- `logs/task_d_probe/axiomatic_brief.json`
- `logs/task_d_probe/axiomatic_details.json`

## Task I（Proxy-based Attribution）
基于已训练的 `value_proxy_mlp.pt` 执行代理归因，并输出 20 点对齐报告与 100 局 proxy 报表：

```bash
python -m liars_game_engine.analysis.task_i_proxy_runner
```

输出目录：
- `logs/task_d_probe/proxy_alignment_report.json`
- `logs/task_d_probe/credit_report_proxy.csv`

说明：
- 默认使用 `logs/task_d_probe/probe_logs/` 中最新一批 100 局日志。
- 对齐校验固定比较 `Physical_Rollout_Phi (N=200)` 与 `Proxy_Phi` 的 Pearson 相关系数。
- 若相关性未超过 `0.75`，应将 proxy 报表视为参考结果，而不是替代物理回放的正式归因。

## 说明
- 默认 `mock` 智能体可直接离线运行。
- `langchain` 智能体是可选扩展，若依赖缺失会产生可观测错误并回退策略。
