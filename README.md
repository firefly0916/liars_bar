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
2. 填写 `.env` 中的 OpenAI-compatible API 配置。
3. 根据实验需要修改 `config/experiment.yaml`。
4. 运行：
   ```bash
   python -m liars_game_engine.main
   ```

## 配置说明
- `.env`（敏感配置）
  - 首选：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_TIMEOUT_SECONDS`
  - 兼容旧命名：`OPENROUTER_API_KEY`、`OPENROUTER_BASE_URL`、`OPENROUTER_TIMEOUT_SECONDS`
- `config/experiment.yaml`（非敏感参数）
  - 运行参数、规则参数、日志参数
  - 多玩家配置（每个玩家可不同模型）
  - Prompt profile 绑定
  - `runtime.enable_null_player_probe`：是否启用 `Null_Probe_Skill`（默认 `false`）
  - `api.base_url` / `api.api_key`：可选的通用 API 配置字段；若同时设置了环境变量，则环境变量优先

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
- `LangChainAgent` 通过 OpenAI-compatible `base_url` 调用模型，因此既可接 OpenRouter，也可接本地 `vLLM`。

## Server Deployment
下面这套流程的目标是：代码只在本地改，服务器只执行脚本，不手改 Python 文件。

### 1. 第一次部署
在服务器项目根目录执行：

```bash
bash scripts/setup_server_env.sh
```

这会：
- 创建 `liar_bar` conda 环境（若不存在）
- 安装当前项目
- 安装 `langchain-openai`
- 安装 `vllm`

说明：
- `Task K` 的物理采样默认使用 `mock` 玩家，不依赖 `vLLM`。如果你只跑 `Task K`，可以使用 `INSTALL_VLLM=0 bash scripts/setup_server_env.sh` 跳过 `vLLM`。
- 当前脚本默认使用 `python=3.10`。如需修改，可传 `PYTHON_VERSION=3.11` 之类的环境变量。

### 2. 启动 vLLM（仅在运行真实 LLM agent 时需要）
如果你要让 `langchain` 玩家通过本地模型服务推理，在服务器执行：

```bash
bash scripts/start_vllm.sh
```

默认行为：
- 模型：`Qwen/Qwen2.5-7B-Instruct`
- 地址：`http://127.0.0.1:8000/v1`
- `HF_HOME`：项目内 `models/huggingface/`
- 后台方式：`tmux` 会话 `vllm_service`

常见自定义例子：

```bash
VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct VLLM_PORT=8001 bash scripts/start_vllm.sh
```

启动后查看日志：

```bash
tail -f logs/vllm/vllm-8000.log
```

### 3. 运行服务器任务
如果你只是执行主控要求的 `Task K` 金标准采样：

```bash
bash scripts/remote_run.sh
```

这会：
- `git pull --ff-only` 当前分支
- 在后台启动 `conda run -n liar_bar python -m liars_game_engine.analysis.task_k_gold_runner`
- 为这次运行单独创建 `logs/task_k_gold/<timestamp>/`
- 将 stdout/stderr 写到该目录下的 `run.log`
- 将 `baseline_logs/`、`attributed_logs/`、`progress.log`、`credit_report_final.csv` 全部收口到同一个 run 目录

如果你要运行接入本地 `vLLM` 的 LLM 对局：

```bash
RUN_MODE=llm OPENAI_API_KEY=EMPTY OPENAI_BASE_URL=http://127.0.0.1:8000/v1 bash scripts/remote_run.sh
```

如果不显式传 `OPENAI_*`，`remote_run.sh` 会自动回退到：
- `OPENAI_API_KEY=${VLLM_API_KEY:-EMPTY}`
- `OPENAI_BASE_URL=http://127.0.0.1:${VLLM_PORT:-8000}/v1`

### 4. 后续更新
本地 `git push` 之后，服务器通常只需要：

```bash
bash scripts/remote_run.sh
```

如果你要在前台运行并把所有产物也收进独立目录，可以执行：

```bash
bash scripts/run_task_k_gold_server.sh
```

如果是 LLM 任务，先确认 `vLLM` 服务还在，再执行：

```bash
RUN_MODE=llm bash scripts/remote_run.sh
```

### 5. vLLM 安装注意事项
vLLM 官方当前文档推荐使用 `pip install vllm` 并通过 `vllm serve` 启动 OpenAI-compatible server。若服务器 CUDA/驱动版本与默认轮子不兼容，需要按官方对应版本说明改成匹配的 wheel 安装方式，而不是修改本仓库代码。
