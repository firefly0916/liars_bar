# 全员信用审计报告 (Global Credit Audit)

- game_id: `task-c-20260417-191531-065`
- source_csv: `logs/task_d_probe/audit/single_case_phi_full.csv`
- winner: `p3`

## 4人信用台账

| player_id | is_winner | phi_sum | target_value | residual_to_target |
| --- | --- | ---: | ---: | ---: |
| p1 | False | -0.015000 | -0.250000 | -0.235000 |
| p2 | False | -0.020000 | -0.250000 | -0.230000 |
| p3 | True | 0.030000 | 0.750000 | 0.720000 |
| p4 | False | 0.000000 | -0.250000 | -0.250000 |

## 全局效率与环境贡献

- phi_total_sum: `-0.005000`
- target_total_sum: `0.000000`
- efficiency_gap_total: `-0.005000`
- V_env: `0.005000`

## 单合法动作归一化（锚点）

- single_legal_turn_count: `4`
- mean_abs_phi: `0.000000`
- max_abs_phi: `0.000000`
