# Experiment Recovery Report (2026-07-10)

## Current Code State

本分支 `codex/recovery-current-20260710` 整理出的代码以 `recovery/hlw3090/code/liars_bar` 为主，因为它包含 2026-06-17 到 2026-06-20 的 action-scope、action-json、eval-schema repair 相关代码。同步时保留并修回了本地 `main` 上较新的 Task K server script 独立运行目录逻辑。

已进入代码分支的主要新增能力：

- `liars_game_engine/agents/llm_agent.py`、`local_backend.py`、`prompts.py`：本地 LoRA adapter、action-only/action-json 输出约束与 eval repair 路径。
- `liars_game_engine/analysis/candidate_expansion.py`、`token_alignment.py`、`hicra_preprocessor.py`、`eval_scorecard.py`、`formal_eval_driver.py`：CRC-GRA/HICRA 训练与评估工具链。
- `scripts/train_crc_gra_ablation.py`、`run_crc_gra_ablation_suite.sh`、`run_action_json_eval_repair.sh`、`run_selected_formal_eval.py`：主要训练/评估入口。
- `openspec/changes/refine-crc-gra-post-proxy-workflow` 与 `openspec/changes/add-online-grpo-baseline`：实验设计和约束记录。

`recovery/` 被加入 `.gitignore`，大模型、adapter、proxy、tar 包和恢复原始资料不进入 Git。

## Recovered Sources

| source | role | main contribution |
| --- | --- | --- |
| hlw4090 | best-result source | 20260617 action-only frozen 300-game eval, `win_rate=0.610000`, clean access/parse/illegal; restored 20260523 action-only adapter. |
| hlw3090 | latest-code source | 20260617-20260620 action-scope/action-json/eval-schema repair code and experiments. |
| autodl4090 | provenance source | Missing `20260504-task2_1-english/savi_alignment_train.jsonl`, 20260510 best-main adapter, root-level git bundles/patches. |

## Scorecard Summary

Combined parsed scorecards: 91 rows. Clean means `access_ok=True`, `parse_error_rate=0`, and `illegal_chosen_turn_count=0`.

### Top Clean Rows

| win_rate | access | parse | illegal | server | experiment | variant |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1.000000 | True | 0.000000 | 0 | hlw4090 | `20260617-hlw4090-action-only-frozen-smoke-v2` | `action_only_proxy_frozen_20260523` |
| 0.800000 | True | 0.000000 | 0 | hlw4090 | `experiments` | `20260523-online-grpo-pilot-3u` |
| 0.800000 | True | 0.000000 | 0 | autodl4090 | `20260523-online-grpo-pilot-3u` | `20260523-online-grpo-pilot-3u` |
| 0.610000 | True | 0.000000 | 0 | hlw4090 | `20260617-action-only-frozen-20260523-300g` | `action_only_proxy_frozen_20260523` |
| 0.345000 | True | 0.000000 | 0 | hlw4090 | `experiments` | `20260525-hicra-adv-rescue-200g` |
| 0.345000 | True | 0.000000 | 0 | autodl4090 | `20260525-hicra-adv-rescue-200g` | `scoped_hicra_adv_reshape` |
| 0.295000 | True | 0.000000 | 0 | hlw3090 | `20260620-action-json-eval-schema-repair` | `conservative_expanded_action_json_eval_repair` |
| 0.290000 | True | 0.000000 | 0 | hlw4090 | `experiments` | `20260523-online-ppo-no4bit-200g-formal` |
| 0.290000 | True | 0.000000 | 0 | autodl4090 | `20260523-online-ppo-no4bit-200g-formal` | `20260523-online-ppo-no4bit-200g-formal` |
| 0.215000 | True | 0.000000 | 0 | hlw4090 | `experiments` | `20260526-expanded-action-proxy-200g` |
| 0.215000 | True | 0.000000 | 0 | autodl4090 | `20260526-expanded-action-proxy-200g` | `expanded_action_proxy` |
| 0.195000 | True | 0.000000 | 0 | hlw4090 | `experiments` | `20260526-action-only-no-match-bonus-200g` |

### Top Overall Rows

| win_rate | access | parse | illegal | server | experiment | variant |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1.000000 | True | 0.000000 | 0 | hlw4090 | `20260617-hlw4090-action-only-frozen-smoke-v2` | `action_only_proxy_frozen_20260523` |
| 0.800000 | True | 0.000000 | 0 | hlw4090 | `experiments` | `20260523-online-grpo-pilot-3u` |
| 0.800000 | True | 0.000000 | 0 | autodl4090 | `20260523-online-grpo-pilot-3u` | `20260523-online-grpo-pilot-3u` |
| 0.620000 | False | 0.013397 | 21 | hlw4090 | `experiments` | `20260524-scoped-hicra-200g` |
| 0.620000 | False | 0.013397 | 21 | autodl4090 | `20260524-scoped-hicra-200g` | `scoped_hicra` |
| 0.615000 | False | 0.000442 | 0 | hlw4090 | `experiments` | `20260524-controlled-best-action-proxy-200g` |
| 0.615000 | False | 0.000442 | 0 | autodl4090 | `20260524-controlled-best-action-proxy-200g` | `action_only_proxy_rerun_200g` |
| 0.610000 | True | 0.000000 | 0 | hlw4090 | `20260617-action-only-frozen-20260523-300g` | `action_only_proxy_frozen_20260523` |
| 0.600000 | False | 0.000000 | 1 | hlw4090 | `experiments` | `20260523-crc-gra-ablation-200g` |
| 0.600000 | False | 0.061828 | 64 | hlw4090 | `experiments` | `20260525-hicra-structural-variants-200g` |
| 0.600000 | False | 0.000000 | 1 | autodl4090 | `20260523-crc-gra-ablation-200g` | `action_only_proxy` |
| 0.600000 | False | 0.061828 | 64 | autodl4090 | `20260525-hicra-structural-variants-200g` | `scoped_hicra_reward_only` |

## Current Interpretation

1. The strongest reliable current baseline is `20260617-action-only-frozen-20260523-300g/action_only_proxy_frozen_20260523`: `win_rate=0.610000`, clean access, zero parse errors, zero illegal choices. Its adapter is the recovered `20260523-crc-gra-ablation-200g/action_only_proxy` final checkpoint.
2. Some older rows have slightly higher or pilot scores, but should not be treated as mainline: the 0.800 row is a tiny online-GRPO pilot, and 0.615/0.620 rows have `access_ok=False` or parse/illegal problems.
3. HICRA as direct token/reward shaping is not reliable in the recovered evidence. It remains useful as an audit/explanation signal, but the training variants using it directly were unstable or weaker.
4. Conservative candidate expansion and action-json repair are technically useful, especially for access/schema control, but did not improve win rate. The 20260620 eval-schema repair is clean (`win_rate=0.295000`) and should be kept as a schema-control reference, not as a performance baseline.
5. The project should continue from action-only CRC/proxy-guided alignment, treating HICRA/action-json as diagnostics and safety rails rather than the primary reward path.

## Preserved Data Priority

See `docs/recovery/data_preservation_manifest_20260710.md` and `.csv` for cloud upload. P0 items are the minimum set to preserve before doing more experiments.

P0 includes:

- best reliable action-only adapter from hlw4090;
- selected 20260522-20260527 experiment archive;
- 20260617 latest eval/result directories;
- 20260617-20260620 action-json experiment directories;
- recovered task2_1/task2_3 HICRA datasets;
- frozen value proxy model;
- autodl4090 root-level patch/bundle provenance.

## Suggested Next Plan

1. Upload P0 assets to cloud and verify hashes/byte sizes.
2. On a fresh GPU server, recreate the Python environment and mount/download Qwen/Qwen2.5-7B-Instruct separately.
3. Re-run formal eval for the recovered 20260523 action-only adapter with current code and `prompts/profiles/action_only.yaml`, first 300 games, then 500+ games if stable.
4. Use 20260620 action-json eval repair as a schema/access test harness, but do not optimize from it unless win rate improves.
5. Avoid spending the next cycle on direct HICRA token penalties or broad candidate expansion; recovered results suggest these paths are weak unless redesigned.

## Verification

Local verification after recovery merge:

- `python3 -m py_compile $(find liars_game_engine scripts tests -name '*.py' -not -path '*/__pycache__/*')`: passed.
- Lightweight unittest subset covering parsers, candidate expansion, schedules, scorecards, formal eval driver, HICRA offline audit, server scripts, and the recovered `local_backend` LoRA loading contract: 31 tests passed.
- `python3 -m unittest discover tests -v`: failed in this local macOS Python 3.14 environment because optional/runtime dependencies are absent (`yaml`, `torch`), plus torch-dependent tests are not runnable here. The two script regressions found by that run were fixed, and the local_backend expectation was updated to match hlw3090 recovered code.

A full test run should be repeated inside the intended conda/server environment with `PyYAML`, `torch`, `transformers`, `peft`, and GPU/base-model access.
