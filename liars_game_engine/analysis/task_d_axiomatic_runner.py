from __future__ import annotations

import asyncio
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path

from liars_game_engine.analysis.shapley_analyzer import (
    BASELINE_MODE_FORCE_ORIGINAL,
    BASELINE_MODE_RANDOM_ALL_LEGAL,
    LogIterator,
    ShapleyAnalyzer,
    ShapleyAttribution,
    _rollout_once,
)
from liars_game_engine.analysis.task_c_runner import generate_baseline_logs
from liars_game_engine.config.loader import load_settings
from liars_game_engine.config.schema import AppSettings


def _build_probe_experiment_settings(settings: AppSettings) -> AppSettings:
    """作用: 构建 Task D 实验配置（4 Mock + Null Probe 开关）。

    输入:
    - settings: 原始配置对象。

    返回:
    - AppSettings: 用于探测模式采样的配置。
    """
    raw = asdict(settings)
    runtime_raw = dict(raw.get("runtime", {}))
    runtime_raw["enable_null_player_probe"] = True
    raw["runtime"] = runtime_raw

    existing_players = raw.get("players", []) if isinstance(raw.get("players", []), list) else []
    normalized_players: list[dict[str, object]] = []

    for index in range(4):
        if index < len(existing_players) and isinstance(existing_players[index], dict):
            source = existing_players[index]
            normalized_players.append(
                {
                    "player_id": str(source.get("player_id", f"p{index + 1}")),
                    "name": str(source.get("name", f"Player{index + 1}")),
                    "agent_type": "mock",
                    "model": str(source.get("model", f"mock-model-{index + 1}")),
                    "prompt_profile": str(source.get("prompt_profile", "baseline")),
                    "temperature": float(source.get("temperature", 0.2)),
                }
            )
            continue

        normalized_players.append(
            {
                "player_id": f"p{index + 1}",
                "name": f"Player{index + 1}",
                "agent_type": "mock",
                "model": f"mock-model-{index + 1}",
                "prompt_profile": "baseline",
                "temperature": 0.2,
            }
        )

    raw["players"] = normalized_players
    return AppSettings.from_dict(raw)


def _iter_log_records(log_paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for log_path in log_paths:
        game_id = Path(log_path).stem
        for line in Path(log_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            records.append({**payload, "_game_id": game_id})
    return records


def count_probe_actions(log_paths: list[Path]) -> int:
    """作用: 统计日志中 Probe 动作数量。

    输入:
    - log_paths: 日志路径列表。

    返回:
    - int: `skill_category=Probe` 或 `skill_name=Null_Probe_Skill` 的条目数。
    """
    records = _iter_log_records(log_paths)
    count = 0
    for record in records:
        skill_name = str(record.get("skill_name", ""))
        skill_category = str(record.get("skill_category", ""))
        if skill_name == "Null_Probe_Skill" or skill_category == "Probe":
            count += 1
    return count


def compute_efficiency_error(
    attributions: list[ShapleyAttribution],
    log_paths: list[Path],
    initial_value: float = 0.25,
) -> dict[str, float | int]:
    """作用: 按“单局总和”计算有效性偏差 `sum(phi)-(Outcome-V_initial)`。

    输入:
    - attributions: 全量归因结果。
    - log_paths: 对应日志路径。
    - initial_value: 初始价值估计，四人局默认 0.25。

    返回:
    - dict[str, float | int]: 样本局数、平均绝对误差、平均符号误差与最大绝对误差。
    """
    games = LogIterator(log_paths).iter_games()

    phi_sum_by_game: dict[str, float] = {}
    for attribution in attributions:
        phi_sum_by_game[attribution.game_id] = phi_sum_by_game.get(attribution.game_id, 0.0) + float(attribution.phi)

    errors: list[float] = []
    for game in games:
        phi_sum = phi_sum_by_game.get(game.game_id, 0.0)
        outcome = 1.0 if game.winner else 0.0
        target = outcome - float(initial_value)
        errors.append(phi_sum - target)

    if not errors:
        return {
            "sample_count": 0,
            "mean_abs_error": 0.0,
            "mean_signed_error": 0.0,
            "max_abs_error": 0.0,
        }

    abs_errors = [abs(item) for item in errors]
    return {
        "sample_count": len(errors),
        "mean_abs_error": sum(abs_errors) / len(abs_errors),
        "mean_signed_error": sum(errors) / len(errors),
        "max_abs_error": max(abs_errors),
    }


def compute_symmetry_deviation(
    attributions: list[ShapleyAttribution],
    log_paths: list[Path],
) -> dict[str, float | int]:
    """作用: 计算等效动作组的 phi 差值，用于对称性偏离评估。

    输入:
    - attributions: 全量归因结果。
    - log_paths: 对应日志路径。

    返回:
    - dict[str, float | int]: 等效组数量、配对数量、平均差值与最大差值。
    """
    attr_by_key = {
        (item.game_id, int(item.turn), item.player_id): item
        for item in attributions
    }

    grouped_phi: dict[tuple[str, str, str, str, int], list[float]] = {}
    for record in _iter_log_records(log_paths):
        game_id = str(record.get("_game_id", ""))
        turn = int(record.get("turn", 0))
        player_id = str(record.get("player_id", ""))
        attr = attr_by_key.get((game_id, turn, player_id))
        if attr is None:
            continue

        action = record.get("action", {})
        if not isinstance(action, dict):
            continue

        cards = action.get("cards", [])
        card_count = len(cards) if isinstance(cards, list) else 0
        action_type = str(action.get("type", ""))
        claim_rank = str(action.get("claim_rank", ""))

        equiv_key = (
            attr.state_feature,
            attr.skill_name,
            action_type,
            claim_rank,
            card_count,
        )
        grouped_phi.setdefault(equiv_key, []).append(float(attr.phi))

    pair_diffs: list[float] = []
    equivalent_group_count = 0
    for values in grouped_phi.values():
        if len(values) < 2:
            continue
        equivalent_group_count += 1
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                pair_diffs.append(abs(values[left_index] - values[right_index]))

    if not pair_diffs:
        return {
            "equivalent_group_count": equivalent_group_count,
            "pair_count": 0,
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
        }

    return {
        "equivalent_group_count": equivalent_group_count,
        "pair_count": len(pair_diffs),
        "mean_abs_diff": sum(pair_diffs) / len(pair_diffs),
        "max_abs_diff": max(pair_diffs),
    }


def compute_force_original_alignment(
    settings: AppSettings,
    log_paths: list[Path],
    rollout_policy: str = "random",
    rollout_samples: int = 50,
    rollout_step_limit: int = 100,
    max_checked_turns: int = 20,
) -> dict[str, float | int]:
    """作用: 校验 `A_t == baseline` 时价值差是否趋近 0，排查 RNG/快照漂移。

    输入:
    - settings: 运行配置。
    - log_paths: 对应日志路径。
    - rollout_policy: 回放策略。
    - rollout_samples: 每个决策点采样数。
    - rollout_step_limit: 单路径最大步数。
    - max_checked_turns: 最多检查的决策点数量。

    返回:
    - dict[str, float | int]: 对齐检查统计结果。
    """
    iterator = LogIterator(log_paths)
    games = iterator.iter_games()
    settings_raw = asdict(settings)

    diffs: list[float] = []
    turns_checked = 0

    for game in games:
        for trajectory in game.turns:
            if trajectory.checkpoint_format != "pickle_base64_v1" or not trajectory.checkpoint_payload:
                continue

            action_payload = {
                "type": trajectory.action.type,
                "claim_rank": trajectory.action.claim_rank,
                "cards": list(trajectory.action.cards),
            }

            for sample_idx in range(max(1, int(rollout_samples))):
                sample_seed = int(settings.runtime.random_seed) + int(trajectory.turn) * 10_000 + sample_idx
                score_actual = _rollout_once(
                    settings_raw=settings_raw,
                    encoded_checkpoint=trajectory.checkpoint_payload,
                    initial_action_payload=action_payload,
                    target_player_id=trajectory.player_id,
                    sample_seed=sample_seed,
                    rollout_policy=rollout_policy,
                    counterfactual=False,
                    rollout_step_limit=rollout_step_limit,
                    baseline_mode=BASELINE_MODE_RANDOM_ALL_LEGAL,
                )
                score_force_original = _rollout_once(
                    settings_raw=settings_raw,
                    encoded_checkpoint=trajectory.checkpoint_payload,
                    initial_action_payload=action_payload,
                    target_player_id=trajectory.player_id,
                    sample_seed=sample_seed,
                    rollout_policy=rollout_policy,
                    counterfactual=True,
                    rollout_step_limit=rollout_step_limit,
                    baseline_mode=BASELINE_MODE_FORCE_ORIGINAL,
                )
                diffs.append(float(score_actual) - float(score_force_original))

            turns_checked += 1
            if turns_checked >= max_checked_turns:
                break
        if turns_checked >= max_checked_turns:
            break

    if not diffs:
        return {
            "turns_checked": turns_checked,
            "sample_count": 0,
            "mean_abs_delta": 0.0,
            "mean_signed_delta": 0.0,
            "max_abs_delta": 0.0,
        }

    abs_diffs = [abs(item) for item in diffs]
    return {
        "turns_checked": turns_checked,
        "sample_count": len(diffs),
        "mean_abs_delta": sum(abs_diffs) / len(abs_diffs),
        "mean_signed_delta": sum(diffs) / len(diffs),
        "max_abs_delta": max(abs_diffs),
    }


def export_single_game_phi_curve(
    attributions: list[ShapleyAttribution],
    output_path: Path | str,
) -> dict[str, object]:
    """作用: 导出单局回合级 phi 与累计和曲线，定位能量不守恒跳变。

    输入:
    - attributions: 全量归因结果。
    - output_path: 导出的 CSV 路径。

    返回:
    - dict[str, object]: 导出状态与目标对局摘要。
    """
    if not attributions:
        return {
            "exported": False,
            "reason": "empty_attributions",
        }

    winning_game_id = next((item.game_id for item in attributions if item.winner is not None), None)
    if winning_game_id is None:
        return {
            "exported": False,
            "reason": "no_winning_game",
        }

    selected = sorted(
        [item for item in attributions if item.game_id == winning_game_id],
        key=lambda item: (int(item.turn), item.player_id),
    )
    if not selected:
        return {
            "exported": False,
            "reason": "no_turns_for_selected_game",
        }

    curve_path = Path(output_path)
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    cumulative = 0.0
    max_jump = 0.0
    max_jump_turn = 0

    with curve_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["game_id", "turn", "player_id", "skill_name", "phi", "cumulative_phi"],
        )
        writer.writeheader()

        for item in selected:
            phi = float(item.phi)
            cumulative += phi
            jump = abs(phi)
            if jump > max_jump:
                max_jump = jump
                max_jump_turn = int(item.turn)

            writer.writerow(
                {
                    "game_id": item.game_id,
                    "turn": int(item.turn),
                    "player_id": item.player_id,
                    "skill_name": item.skill_name,
                    "phi": f"{phi:.6f}",
                    "cumulative_phi": f"{cumulative:.6f}",
                }
            )

    return {
        "exported": True,
        "game_id": winning_game_id,
        "turn_count": len(selected),
        "final_cumulative_phi": cumulative,
        "max_abs_jump": max_jump,
        "max_abs_jump_turn": max_jump_turn,
        "path": str(curve_path),
    }


def run_task_d_probe_pipeline(
    config_file: str | Path = "config/experiment.yaml",
    game_count: int = 100,
    rollout_samples: int = 100,
    output_dir: str | Path = "logs/task_d_probe",
    max_workers: int | None = None,
    initial_value: float = 0.25,
) -> dict[str, object]:
    """作用: 执行 Task D 探测模式采样与公理验证简报。

    输入:
    - config_file: 主配置路径。
    - game_count: 对战局数。
    - rollout_samples: 每决策点每组回放采样次数。
    - output_dir: 输出目录。
    - max_workers: 并行 worker 数，默认 CPU 核心数。
    - initial_value: 初始价值估计。

    返回:
    - dict[str, object]: 任务摘要与公理验证指标。
    """
    settings = load_settings(config_file=config_file)
    task_settings = _build_probe_experiment_settings(settings)

    output_path = Path(output_dir)
    baseline_dir = output_path / "probe_logs"
    probe_logs = asyncio.run(generate_baseline_logs(task_settings, game_count=game_count, log_dir=baseline_dir))

    probe_action_count = count_probe_actions(probe_logs)
    if probe_action_count == 0:
        raise RuntimeError("Probe logs contain zero Null_Probe_Skill actions; rerun with a different random seed.")

    analyzer_workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    analyzer = ShapleyAnalyzer(
        settings=task_settings,
        rollout_samples=rollout_samples,
        rollout_policy="random",
        max_workers=max(1, analyzer_workers),
        baseline_mode=BASELINE_MODE_RANDOM_ALL_LEGAL,
    )
    attributions, _ = analyzer.analyze_logs(probe_logs)

    probe_summary = ShapleyAnalyzer.summarize_probe_skill(attributions)
    efficiency_summary = compute_efficiency_error(
        attributions=attributions,
        log_paths=probe_logs,
        initial_value=initial_value,
    )
    symmetry_summary = compute_symmetry_deviation(attributions=attributions, log_paths=probe_logs)
    alignment_summary = compute_force_original_alignment(
        settings=task_settings,
        log_paths=probe_logs,
        rollout_policy="random",
        rollout_samples=min(rollout_samples, 50),
        rollout_step_limit=analyzer.rollout_step_limit,
        max_checked_turns=20,
    )
    phi_curve_summary = export_single_game_phi_curve(
        attributions=attributions,
        output_path=output_path / "single_game_phi_curve.csv",
    )

    brief = {
        "phi_probe_avg": float(probe_summary.get("phi_avg", 0.0)),
        "efficiency_total_bias": float(efficiency_summary.get("mean_abs_error", 0.0)),
        "symmetry_deviation": float(symmetry_summary.get("mean_abs_diff", 0.0)),
        "force_original_alignment": float(alignment_summary.get("mean_abs_delta", 0.0)),
        "thresholds": {
            "phi_probe_avg": 0.01,
            "efficiency_total_bias": 0.05,
        },
        "passed": {
            "phi_probe_avg": abs(float(probe_summary.get("phi_avg", 0.0))) < 0.01,
            "efficiency_total_bias": float(efficiency_summary.get("mean_abs_error", 0.0)) < 0.05,
        },
    }

    output_path.mkdir(parents=True, exist_ok=True)
    brief_path = output_path / "axiomatic_brief.json"
    details_path = output_path / "axiomatic_details.json"
    brief_path.write_text(json.dumps(brief, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path.write_text(
        json.dumps(
            {
                "probe_summary": probe_summary,
                "efficiency_summary": efficiency_summary,
                "symmetry_summary": symmetry_summary,
                "alignment_summary": alignment_summary,
                "phi_curve_summary": phi_curve_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "probe_game_count": len(probe_logs),
        "probe_action_count": probe_action_count,
        "probe_log_dir": str(baseline_dir),
        "attribution_count": len(attributions),
        "rollout_samples": rollout_samples,
        "max_workers": max(1, analyzer_workers),
        "axiomatic_brief": str(brief_path),
        "axiomatic_details": str(details_path),
        "single_game_phi_curve": str(output_path / "single_game_phi_curve.csv"),
        "brief": brief,
    }


def main() -> None:
    summary = run_task_d_probe_pipeline()
    print("Task D probe pipeline finished:", summary)


if __name__ == "__main__":
    main()
