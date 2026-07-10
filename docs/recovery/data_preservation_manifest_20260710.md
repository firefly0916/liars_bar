# Data Preservation Manifest (2026-07-10)

建议云端目录：`liars_bar_recovery_20260710/`。优先上传 P0，再上传 P1；P2 可作为补充。

| priority | asset | size | local path | why save |
| --- | --- | ---: | --- | --- |
| P0 | best reliable action-only adapter | 87.95 MB | `recovery/hlw4090/experiments_checkpoints/experiments/20260523-crc-gra-ablation-200g/action_only_proxy/train/checkpoints/final` | Adapter used by the clean 20260617 300-game frozen eval, win_rate=0.610000. |
| P0 | selected 20260522-20260527 experiment archive | 1.94 GB | `recovery/hlw4090/archives/liars_bar_experiments_20260522_20260527_selected_20260616.tar.gz` | Largest compact archive for the core CRC-GRA / HICRA / PPO / GRPO experiment lineage. |
| P0 | 20260617 latest experiment results | 375.16 MB | `recovery/hlw4090/experiments_latest` | Contains the 20260617 clean action-only 300-game eval outputs and candidate-expansion regression outputs. |
| P0 | 20260617-20260620 action-json experiments | 563.45 MB | `recovery/hlw3090/experiments` | Contains action-scope/action-json adapters and scorecards, including 20260620 eval-schema repair provenance. |
| P0 | missing task2_1 HICRA train dataset | 91.20 KB | `recovery/autodl4090/data/task_n_hicra_preprocessed/20260504-task2_1-english/savi_alignment_train.jsonl` | Missing dataset recovered from autodl4090; referenced by early HICRA/protocol training summaries. |
| P0 | task2_3 full HICRA dataset | 1.17 MB | `recovery/autodl4090/data/task_n_hicra_preprocessed/20260504-task2_3-english/savi_alignment_full_v3.jsonl` | Full-state dataset also present on hlw4090/hlw3090; keep one canonical copy plus hashes. |
| P0 | frozen value proxy model | 13.77 KB | `recovery/hlw4090/models/proxy/value_proxy_mlp_distill.pt` | Small but important proxy model referenced by CRC/GRA and audit paths. |
| P0 | root patches and git bundles | 107.95 MB | `recovery/autodl4090/root_files` | Most complete patch/bundle provenance set from autodl4090 root. |
| P1 | 20260510 best-main adapter | 87.95 MB | `recovery/autodl4090/checkpoints/best_main_20260510_r2_delay_rerun3` | Recovered early protocol-mainline adapter; important historical baseline, not current best. |
| P1 | hlw4090 code snapshot archive | 106.16 MB | `recovery/hlw4090/archives/liars_bar_code_snapshot_20260616.tar.gz` | Verified 20260616 code snapshot archive. |
| P1 | hlw3090 recovered proxy copy | 13.77 KB | `recovery/hlw3090/proxy/value_proxy_mlp_distill.pt` | Duplicate proxy copy for redundancy. |
| P1 | legacy 20260429 datasets | 3.45 MB | `recovery/hlw4090/data/legacy_20260429` | Older alignment data for provenance fallback. |
| P1 | combined local scorecard/indexes | 22.80 KB | `recovery/combined_scorecard_summary.csv` | Machine-readable experiment index used by this report. |
| P2 | autodl4090 lightweight experiment summaries | 41.64 MB | `recovery/autodl4090/experiments_results` | Lightweight result files from 85 remote experiment dirs; mostly overlaps hlw4090 but useful provenance. |

## Upload Notes

- Git 仓库只保存代码、测试、openspec、报告和小型 CSV；不要把 `recovery/` 整体提交到 Git。
- Qwen/Qwen2.5-7B-Instruct 基座模型没有复制，本报告把它视为外部依赖；重现实验时需要在服务器重新下载或挂载。
- `*.safetensors`、`*.pt`、`*.tar.gz` 已加入 `.gitignore`，应走对象存储/网盘/云盘。
- 大文件上传后建议用本 CSV 的 `sha256` 字段或源目录自带 `SHA256SUMS.txt` 复验。
