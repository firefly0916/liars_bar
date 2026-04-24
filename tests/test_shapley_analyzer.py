import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from liars_game_engine.agents.factory import build_agents
from liars_game_engine.analysis.shapley_analyzer import (
    BASELINE_MODE_FORCE_ORIGINAL,
    BASELINE_MODE_RANDOM_ALL_LEGAL,
    BASELINE_MODE_RANDOM_LEGAL_AGENT,
    BASELINE_MODE_RANDOM_NON_ORIGINAL,
    LogIterator,
    ProxyValuePredictor,
    ShapleyAnalyzer,
    ShapleyAttribution,
    TurnTrajectory,
    _build_action_from_legal_template,
    _build_counterfactual_action,
    _rollout_once,
)
from liars_game_engine.analysis.train_value_proxy import (
    VALUE_PROXY_INPUT_DIM,
    VALUE_PROXY_TARGET_PHI,
    ValueProxyMLP,
    _build_feature_vector,
    build_value_proxy_feature_context,
    encode_value_proxy_features,
    load_value_samples,
)
from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.engine.game_state import ActionModel
from liars_game_engine.experiment.logger import ExperimentLogger
from liars_game_engine.experiment.orchestrator import GameOrchestrator


class ShapleyAnalyzerTest(unittest.IsolatedAsyncioTestCase):
    def _make_settings(self) -> AppSettings:
        """作用: 构造归因分析测试使用的最小可运行配置。

        输入:
        - 无。

        返回:
        - AppSettings: 可用于生成日志与离线回放的配置对象。
        """
        return AppSettings.from_dict(
            {
                "runtime": {"max_turns": 32, "random_seed": 19, "fallback_action": "challenge"},
                "parser": {"max_retries": 2, "allow_markdown_json": True, "allow_key_alias": True},
                "logging": {"run_log_dir": "logs/runs", "level": "INFO"},
                "players": [
                    {
                        "player_id": "p1",
                        "name": "Alice",
                        "agent_type": "mock",
                        "model": "m1",
                        "prompt_profile": "baseline",
                    },
                    {
                        "player_id": "p2",
                        "name": "Bob",
                        "agent_type": "mock",
                        "model": "m2",
                        "prompt_profile": "baseline",
                    },
                ],
                "rules": {
                    "deck_ranks": ["A", "K", "Q", "JOKER"],
                    "cards_per_player": 3,
                    "roulette_slots": 2,
                    "enable_items": False,
                },
            }
        )

    async def test_log_iterator_reads_checkpoint_and_infers_winner(self) -> None:
        """作用: 验证日志迭代器可提取 checkpoint 并识别赢家。

        输入:
        - 无（测试内部运行一局 mock 对局并读取日志）。

        返回:
        - 无。
        """
        settings = self._make_settings()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings.logging.run_log_dir = temp_dir
            env = GameEnvironment(settings)
            agents = build_agents(settings)
            logger = ExperimentLogger(base_dir=temp_dir, game_id="shapley-log-iter")
            orchestrator = GameOrchestrator(
                env=env,
                agents=agents,
                logger=logger,
                fallback_action=settings.runtime.fallback_action,
                max_turns=settings.runtime.max_turns,
            )

            summary = await orchestrator.run_game_loop()

            games = LogIterator([logger.log_file]).iter_games()

            self.assertEqual(len(games), 1)
            self.assertGreaterEqual(len(games[0].turns), 1)
            self.assertTrue(games[0].turns[0].checkpoint_payload)
            self.assertEqual(games[0].winner, summary["winner"])

    async def test_shapley_analyzer_produces_attribution_and_credit_ledger(self) -> None:
        """作用: 验证 ShapleyAnalyzer 能输出边际贡献和聚类台账。

        输入:
        - 无（测试内部运行一局 mock 对局并离线分析）。

        返回:
        - 无。
        """
        settings = self._make_settings()

        with tempfile.TemporaryDirectory() as temp_dir:
            settings.logging.run_log_dir = temp_dir
            env = GameEnvironment(settings)
            agents = build_agents(settings)
            logger = ExperimentLogger(base_dir=temp_dir, game_id="shapley-analyze")
            orchestrator = GameOrchestrator(
                env=env,
                agents=agents,
                logger=logger,
                fallback_action=settings.runtime.fallback_action,
                max_turns=settings.runtime.max_turns,
            )

            await orchestrator.run_game_loop()

            analyzer = ShapleyAnalyzer(settings=settings, rollout_samples=3, rollout_policy="random", max_workers=1)
            attributions, ledger = analyzer.analyze_logs([logger.log_file])

            self.assertGreaterEqual(len(attributions), 1)
            self.assertGreaterEqual(len(ledger.by_skill), 1)

            first = attributions[0]
            self.assertIsInstance(first.phi, float)
            self.assertGreaterEqual(first.value_action, 0.0)
            self.assertLessEqual(first.value_action, 1.0)
            self.assertGreaterEqual(first.value_counterfactual, 0.0)
            self.assertLessEqual(first.value_counterfactual, 1.0)

    def test_load_value_samples_ignores_games_shorter_than_three_turns(self) -> None:
        """作用: 验证 value proxy 训练样本会跳过少于 3 回合的对局日志。

        输入:
        - 无（测试内部构造 2 回合和 3 回合最小日志）。

        返回:
        - 无。
        """
        short_records = [
            {
                "turn": 1,
                "player_id": "p1",
                "state_features": {
                    "phase": "turn_start",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 2,
                    "hand_count": 3,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p1",
                    "phase": "turn_start",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["A", "K", "Q"],
                    "player_states": {"p1": {"death_probability": 0.0}},
                    "pending_claim": None,
                },
                "action": {"type": "play_claim", "cards": ["A"]},
                "step_result": {"events": []},
            },
            {
                "turn": 2,
                "player_id": "p2",
                "state_features": {
                    "phase": "response_window",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 2,
                    "hand_count": 3,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p2",
                    "phase": "response_window",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["A", "K", "Q"],
                    "player_states": {"p2": {"death_probability": 0.0}},
                    "pending_claim": {"declared_count": 1},
                },
                "action": {"type": "challenge", "cards": []},
                "step_result": {"events": []},
            },
        ]
        valid_records = short_records + [
            {
                "turn": 3,
                "player_id": "p1",
                "state_features": {
                    "phase": "resolution",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 1,
                    "hand_count": 2,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p1",
                    "phase": "resolution",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1"],
                    "private_hand": ["A", "K"],
                    "player_states": {"p1": {"death_probability": 0.0}},
                    "pending_claim": None,
                },
                "action": {"type": "pass", "cards": []},
                "step_result": {"events": []},
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            short_path = Path(temp_dir) / "short.jsonl"
            valid_path = Path(temp_dir) / "valid.jsonl"
            short_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in short_records),
                encoding="utf-8",
            )
            valid_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in valid_records),
                encoding="utf-8",
            )

            samples = load_value_samples(Path(temp_dir))

        self.assertEqual(len(samples), 3)
        self.assertEqual({sample.game_id for sample in samples}, {"valid"})

    def test_load_value_samples_prefers_phi_and_shapley_labels_over_winner_targets(self) -> None:
        """作用: 验证 value proxy 训练样本优先使用逐条 phi/shapley_value 标签。"""
        labeled_records = [
            {
                "turn": 1,
                "player_id": "p1",
                "phi": 0.7,
                "state_features": {
                    "phase": "turn_start",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 2,
                    "hand_count": 3,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p1",
                    "phase": "turn_start",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["A", "K", "Q"],
                    "player_states": {"p1": {"death_probability": 0.0}},
                    "pending_claim": None,
                },
                "action": {"type": "play_claim", "cards": ["A"]},
                "step_result": {"events": []},
            },
            {
                "turn": 2,
                "player_id": "p2",
                "shapley_value": 0.2,
                "state_features": {
                    "phase": "response_window",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 2,
                    "hand_count": 3,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p2",
                    "phase": "response_window",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["A", "K", "Q"],
                    "player_states": {"p2": {"death_probability": 0.0}},
                    "pending_claim": {"declared_count": 1},
                },
                "action": {"type": "challenge", "cards": []},
                "step_result": {"events": []},
            },
            {
                "turn": 3,
                "player_id": "p2",
                "phi": 0.4,
                "state_features": {
                    "phase": "resolution",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 1,
                    "hand_count": 2,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p2",
                    "phase": "resolution",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p2"],
                    "private_hand": ["A", "K"],
                    "player_states": {"p2": {"death_probability": 0.0}},
                    "pending_claim": None,
                },
                "action": {"type": "pass", "cards": []},
                "step_result": {"events": []},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "labeled.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in labeled_records),
                encoding="utf-8",
            )

            samples = load_value_samples(Path(temp_dir))

        self.assertEqual([sample.target for sample in samples], [0.7, 0.2, 0.4])

    def test_load_value_samples_in_phi_mode_skips_records_without_phi_labels(self) -> None:
        """作用: 验证严格 phi 模式不会对无标签记录回退到 winner target。"""
        unlabeled_records = [
            {
                "turn": 1,
                "player_id": "p1",
                "state_features": {
                    "phase": "turn_start",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 2,
                    "hand_count": 3,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p1",
                    "phase": "turn_start",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["A", "K", "Q"],
                    "player_states": {"p1": {"death_probability": 0.0}},
                    "pending_claim": None,
                },
                "action": {"type": "play_claim", "cards": ["A"]},
                "step_result": {"events": []},
            },
            {
                "turn": 2,
                "player_id": "p2",
                "state_features": {
                    "phase": "response_window",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 2,
                    "hand_count": 3,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p2",
                    "phase": "response_window",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p1", "p2"],
                    "private_hand": ["A", "K", "Q"],
                    "player_states": {"p2": {"death_probability": 0.0}},
                    "pending_claim": {"declared_count": 1},
                },
                "action": {"type": "challenge", "cards": []},
                "step_result": {"events": []},
            },
            {
                "turn": 3,
                "player_id": "p2",
                "state_features": {
                    "phase": "resolution",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_player_count": 1,
                    "hand_count": 2,
                    "death_probability": 0.0,
                },
                "observation": {
                    "player_id": "p2",
                    "phase": "resolution",
                    "table_type": "A",
                    "must_call_liar": False,
                    "alive_players": ["p2"],
                    "private_hand": ["A", "K"],
                    "player_states": {"p2": {"death_probability": 0.0}},
                    "pending_claim": None,
                },
                "action": {"type": "pass", "cards": []},
                "step_result": {"events": []},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "unlabeled.jsonl"
            log_path.write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in unlabeled_records),
                encoding="utf-8",
            )

            samples = load_value_samples(Path(temp_dir), target_mode=VALUE_PROXY_TARGET_PHI)

        self.assertEqual(samples, [])

    def test_counterfactual_action_excludes_logged_action(self) -> None:
        """作用: 验证反事实动作采样不会回退到原日志动作。

        输入:
        - 无（测试内部构造 checkpoint 环境并重复采样）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        first_player = env.get_current_player()
        first_hand = list(env.state.players[first_player].hand)
        self.assertGreaterEqual(len(first_hand), 1)

        opening_result = env.step(
            first_player,
            ActionModel(type="play_claim", claim_rank=env.state.table_type, cards=[first_hand[0]]),
        )
        self.assertTrue(opening_result.success)

        player_id = env.get_current_player()
        legal_actions = env.get_legal_actions(player_id)
        self.assertGreaterEqual(len(legal_actions), 2)

        hand = list(env.state.players[player_id].hand)
        self.assertGreaterEqual(len(hand), 1)
        logged_action = ActionModel(type="play_claim", claim_rank=env.state.table_type, cards=[hand[0]])

        for sample_idx in range(20):
            counter_action = _build_counterfactual_action(
                env,
                player_id,
                logged_action,
                sample_seed=sample_idx,
                baseline_mode=BASELINE_MODE_RANDOM_NON_ORIGINAL,
            )
            self.assertIsNotNone(counter_action)
            self.assertNotEqual(counter_action.type, "noop")
            self.assertFalse(
                counter_action.type == logged_action.type
                and counter_action.claim_rank == logged_action.claim_rank
                and counter_action.cards == logged_action.cards
            )

    def test_counterfactual_action_returns_none_when_only_one_legal_action(self) -> None:
        """作用: 验证仅有一个合法动作时，非原动作基线不可采样。

        输入:
        - 无（测试内部使用首手仅能 play_claim 的场景）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        player_id = env.get_current_player()
        hand = list(env.state.players[player_id].hand)
        self.assertGreaterEqual(len(hand), 1)

        logged_action = ActionModel(type="play_claim", claim_rank=env.state.table_type, cards=[hand[0]])
        counter_action = _build_counterfactual_action(
            env,
            player_id,
            logged_action,
            sample_seed=7,
            baseline_mode=BASELINE_MODE_RANDOM_NON_ORIGINAL,
        )

        self.assertIsNone(counter_action)

    def test_counterfactual_action_random_legal_agent_returns_original_when_single_legal_action(self) -> None:
        """作用: 验证 Random_Legal_Agent 在单合法动作场景回退到原动作（phi=0 锚点）。

        输入:
        - 无（测试内部使用首手仅能 play_claim 的场景）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        player_id = env.get_current_player()
        hand = list(env.state.players[player_id].hand)
        self.assertGreaterEqual(len(hand), 1)

        logged_action = ActionModel(type="play_claim", claim_rank=env.state.table_type, cards=[hand[0]])
        counter_action = _build_counterfactual_action(
            env,
            player_id,
            logged_action,
            sample_seed=7,
            baseline_mode=BASELINE_MODE_RANDOM_LEGAL_AGENT,
        )

        self.assertIsNotNone(counter_action)
        self.assertEqual(counter_action.type, logged_action.type)
        self.assertEqual(counter_action.claim_rank, logged_action.claim_rank)
        self.assertEqual(counter_action.cards, logged_action.cards)

    def test_rollout_once_returns_draw_when_step_limit_is_zero(self) -> None:
        """作用: 验证回放步数限制触发时按 0.5 平局计分。

        输入:
        - 无（测试内部构造起始 checkpoint 并将步数上限设为 0）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        player_id = env.get_current_player()
        hand = list(env.state.players[player_id].hand)
        checkpoint_payload = env.serialize_checkpoint(env.save_checkpoint())

        score = _rollout_once(
            settings_raw=asdict(settings),
            encoded_checkpoint=checkpoint_payload,
            initial_action_payload={"type": "play_claim", "claim_rank": env.state.table_type, "cards": [hand[0]]},
            target_player_id=player_id,
            sample_seed=7,
            rollout_policy="random",
            counterfactual=False,
            rollout_step_limit=0,
        )

        self.assertEqual(score, 0.5)

    def test_rollout_once_force_original_baseline_matches_actual_value(self) -> None:
        """作用: 验证 A_t = A_baseline 时两组回放价值一致（phi 应趋近 0）。

        输入:
        - 无（测试内部使用同一 checkpoint 和同一采样种子）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        player_id = env.get_current_player()
        hand = list(env.state.players[player_id].hand)
        checkpoint_payload = env.serialize_checkpoint(env.save_checkpoint())
        action_payload = {"type": "play_claim", "claim_rank": env.state.table_type, "cards": [hand[0]]}

        actual_score = _rollout_once(
            settings_raw=asdict(settings),
            encoded_checkpoint=checkpoint_payload,
            initial_action_payload=action_payload,
            target_player_id=player_id,
            sample_seed=23,
            rollout_policy="random",
            counterfactual=False,
            rollout_step_limit=100,
        )

        same_baseline_score = _rollout_once(
            settings_raw=asdict(settings),
            encoded_checkpoint=checkpoint_payload,
            initial_action_payload=action_payload,
            target_player_id=player_id,
            sample_seed=23,
            rollout_policy="random",
            counterfactual=True,
            rollout_step_limit=100,
            baseline_mode=BASELINE_MODE_FORCE_ORIGINAL,
        )

        self.assertEqual(actual_score, same_baseline_score)

    def test_rollout_once_all_legal_baseline_matches_actual_with_single_legal_action(self) -> None:
        """作用: 验证仅有一个合法动作时，全合法动作基线与原动作价值一致。

        输入:
        - 无（测试内部使用首手仅能 play_claim 的场景）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        player_id = env.get_current_player()
        hand = list(env.state.players[player_id].hand)
        checkpoint_payload = env.serialize_checkpoint(env.save_checkpoint())
        action_payload = {"type": "play_claim", "claim_rank": env.state.table_type, "cards": [hand[0]]}

        actual_score = _rollout_once(
            settings_raw=asdict(settings),
            encoded_checkpoint=checkpoint_payload,
            initial_action_payload=action_payload,
            target_player_id=player_id,
            sample_seed=23,
            rollout_policy="random",
            counterfactual=False,
            rollout_step_limit=100,
        )

        baseline_score = _rollout_once(
            settings_raw=asdict(settings),
            encoded_checkpoint=checkpoint_payload,
            initial_action_payload=action_payload,
            target_player_id=player_id,
            sample_seed=23,
            rollout_policy="random",
            counterfactual=True,
            rollout_step_limit=100,
            baseline_mode=BASELINE_MODE_RANDOM_ALL_LEGAL,
        )

        self.assertEqual(actual_score, baseline_score)

    def test_export_credit_report_writes_required_columns(self) -> None:
        """作用: 验证 credit_report.csv 包含任务要求字段。

        输入:
        - 无（测试内部构造 attribution 并写入临时 CSV）。

        返回:
        - 无。
        """
        attributions = [
            ShapleyAttribution(
                game_id="g1",
                turn=1,
                player_id="p1",
                skill_name="Truthful_Action",
                state_feature="phase=response_window|table=A|risk=low_risk|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=1.0,
                value_counterfactual=0.4,
                phi=0.6,
                rollout_samples=50,
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "credit_report.csv"
            ShapleyAnalyzer.export_credit_report(attributions, report_path)

            with report_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(
            set(rows[0].keys()),
            {"skill_name", "death_prob_bucket", "avg_shapley_value", "sample_count"},
        )

    def test_summarize_probe_skill_returns_only_null_probe_stats(self) -> None:
        """作用: 验证分析器可专门汇总 Null_Probe_Skill 的 phi 统计。

        输入:
        - 无（测试内构造 attribution 列表）。

        返回:
        - 无。
        """
        attributions = [
            ShapleyAttribution(
                game_id="g1",
                turn=1,
                player_id="p1",
                skill_name="Null_Probe_Skill",
                state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=0.5,
                value_counterfactual=0.5,
                phi=0.0,
                rollout_samples=50,
            ),
            ShapleyAttribution(
                game_id="g1",
                turn=2,
                player_id="p1",
                skill_name="Truthful_Action",
                state_feature="phase=response_window|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=1.0,
                value_counterfactual=0.4,
                phi=0.6,
                rollout_samples=50,
            ),
        ]

        summary = ShapleyAnalyzer.summarize_probe_skill(attributions)

        self.assertEqual(summary["skill_name"], "Null_Probe_Skill")
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["phi_sum"], 0.0)

    def test_value_proxy_encoder_builds_8d_features_for_play_claim_and_challenge(self) -> None:
        """作用: 验证价值代理编码器输出 8D 特征，并正确填充新增维度。

        输入:
        - 无（测试内部构造 play_claim 与 challenge 两类上下文）。

        返回:
        - 无。
        """
        play_context = build_value_proxy_feature_context(
            observation={
                "phase": "response_window",
                "table_type": "K",
                "must_call_liar": True,
                "alive_players": ["p1", "p2", "p3"],
                "private_hand": ["K", "JOKER", "A", "Q"],
                "pending_claim": {"declared_count": 2},
                "player_states": {"p1": {"death_probability": 0.25}},
            },
            player_id="p1",
            action={"type": "play_claim", "cards": ["K", "A", "JOKER"]},
        )
        challenge_context = build_value_proxy_feature_context(
            observation={
                "phase": "response_window",
                "table_type": "K",
                "must_call_liar": True,
                "alive_players": ["p1", "p2", "p3"],
                "private_hand": ["K", "JOKER", "A", "Q"],
                "pending_claim": {"declared_count": 3},
                "player_states": {"p1": {"death_probability": 0.25}},
            },
            player_id="p1",
            action={"type": "challenge"},
        )

        encoded_play = encode_value_proxy_features(play_context)
        encoded_challenge = encode_value_proxy_features(challenge_context)

        self.assertEqual(len(encoded_play), VALUE_PROXY_INPUT_DIM)
        self.assertEqual(len(encoded_challenge), VALUE_PROXY_INPUT_DIM)
        self.assertEqual(
            encoded_play[:6],
            [
                2.0 / 3.0,
                0.5,
                1.0,
                0.75,
                0.5,
                0.25,
            ],
        )
        self.assertAlmostEqual(encoded_play[6], 0.5, places=6)
        self.assertAlmostEqual(encoded_play[7], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(encoded_challenge[6], 0.5, places=6)
        self.assertAlmostEqual(encoded_challenge[7], 5.0 / 8.0, places=6)

    def test_proxy_value_predictor_reuses_training_encoder(self) -> None:
        """作用: 验证 ProxyValuePredictor 直接复用训练模块中的编码器。

        输入:
        - 无（测试内部保存零权重 MLP 并验证编码函数调用链）。

        返回:
        - 无。
        """
        model = ValueProxyMLP(input_dim=VALUE_PROXY_INPUT_DIM, hidden_dim=64)
        for parameter in model.parameters():
            parameter.data.zero_()

        feature_context = build_value_proxy_feature_context(
            observation={
                "phase": "response_window",
                "table_type": "K",
                "must_call_liar": True,
                "alive_players": ["p1", "p2", "p3"],
                "private_hand": ["K", "JOKER", "A", "Q"],
                "pending_claim": {"declared_count": 2},
                "player_states": {"p1": {"death_probability": 0.25}},
            },
            player_id="p1",
            action={"type": "play_claim", "cards": ["K", "A", "JOKER"]},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "value_proxy_mlp.pt"
            torch.save(model.state_dict(), model_path)

            predictor = ProxyValuePredictor(model_path=model_path)
            with patch(
                "liars_game_engine.analysis.shapley_analyzer.encode_value_proxy_features",
                wraps=encode_value_proxy_features,
            ) as encode_mock:
                encoded = predictor.encode_state_features(feature_context)
                prediction = predictor.predict_state_features(feature_context)

        self.assertGreaterEqual(encode_mock.call_count, 2)
        self.assertEqual(encoded, _build_feature_vector(feature_context))
        self.assertEqual(len(encoded), VALUE_PROXY_INPUT_DIM)
        self.assertAlmostEqual(prediction, 0.5, places=6)

    def test_proxy_attribution_uses_legal_action_average(self) -> None:
        """作用: 验证 proxy 归因按原动作得分减去合法动作平均得分计算。

        输入:
        - 无（测试内部构造 response_window 场景与可追踪的假预测器）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        env = GameEnvironment(settings)
        opening_player = env.get_current_player()
        opening_hand = list(env.state.players[opening_player].hand)
        self.assertGreaterEqual(len(opening_hand), 1)

        opening_result = env.step(
            opening_player,
            ActionModel(type="play_claim", claim_rank=env.state.table_type, cards=[opening_hand[0]]),
        )
        self.assertTrue(opening_result.success)

        player_id = env.get_current_player()
        observation = env.get_observation_for(player_id)
        challenge_action = ActionModel(type="challenge")
        trajectory = TurnTrajectory(
            game_id="g-proxy",
            turn=2,
            player_id=player_id,
            observation=observation,
            action=challenge_action,
            skill_name="Challenge",
            skill_parameters={},
            checkpoint_format="pickle_base64_v1",
            checkpoint_payload=env.serialize_checkpoint(env.save_checkpoint()),
        )

        class FakePredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                encoded = encode_value_proxy_features(state_features)
                return round(sum(encoded) / len(encoded), 6)

        analyzer = ShapleyAnalyzer(settings=settings, rollout_samples=3, rollout_policy="random", max_workers=1)
        predictor = FakePredictor()
        attribution = analyzer.attribute_step_proxy(trajectory=trajectory, winner=None, predictor=predictor)

        self.assertIsNotNone(attribution)

        expected_values: list[float] = []
        expected_action = None
        legal_actions = analyzer._build_proxy_legal_actions(trajectory)
        self.assertGreaterEqual(len(legal_actions), 2)
        for action in legal_actions:
            feature_context = build_value_proxy_feature_context(
                observation=observation,
                player_id=player_id,
                action={
                    "type": action.type,
                    "claim_rank": action.claim_rank,
                    "cards": list(action.cards),
                },
            )
            predicted = predictor.predict_state_features(feature_context)
            expected_values.append(predicted)
            if action.type == challenge_action.type and action.claim_rank == challenge_action.claim_rank:
                expected_action = predicted

        self.assertIsNotNone(expected_action)
        expected_average = sum(expected_values) / len(expected_values)

        self.assertAlmostEqual(attribution.value_action, expected_action, places=6)
        self.assertAlmostEqual(attribution.value_counterfactual, expected_average, places=6)
        self.assertAlmostEqual(attribution.phi, expected_action - expected_average, places=6)

    def test_proxy_alignment_report_contains_required_metrics(self) -> None:
        """作用: 验证 proxy alignment 报告包含相关性、误差与速度字段。

        输入:
        - 无（测试内部用桩归因结果驱动对齐报告）。

        返回:
        - 无。
        """
        settings = self._make_settings()
        analyzer = ShapleyAnalyzer(settings=settings, rollout_samples=3, rollout_policy="random", max_workers=1)
        predictor = object()
        sampled = [
            TurnTrajectory(
                game_id="g1",
                turn=index + 1,
                player_id="p1",
                observation={"player_id": "p1"},
                action=ActionModel(type="challenge"),
                skill_name="Skill",
                skill_parameters={},
                checkpoint_format="pickle_base64_v1",
                checkpoint_payload="payload",
            )
            for index in range(3)
        ]

        def _make_attr(turn: int, phi: float) -> ShapleyAttribution:
            return ShapleyAttribution(
                game_id="g1",
                turn=turn,
                player_id="p1",
                skill_name="Skill",
                state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=0.5 + phi,
                value_counterfactual=0.5,
                phi=phi,
                rollout_samples=3,
            )

        with patch.object(ShapleyAnalyzer, "_sample_alignment_trajectories", return_value=sampled), patch.object(
            ShapleyAnalyzer,
            "attribute_step_rollout",
            side_effect=[_make_attr(1, 0.10), _make_attr(2, 0.20), _make_attr(3, 0.30)],
        ), patch.object(
            ShapleyAnalyzer,
            "attribute_step_proxy",
            side_effect=[_make_attr(1, 0.12), _make_attr(2, 0.18), _make_attr(3, 0.33)],
        ), patch(
            "liars_game_engine.analysis.shapley_analyzer.time.perf_counter",
            side_effect=[0.0, 6.0, 10.0, 11.0],
        ):
            report = analyzer.run_proxy_alignment(
                log_paths=[Path("logs/task_d_probe/probe_logs/task-c-20260417-174210-001.jsonl")],
                predictor=predictor,
                sample_size=3,
                sample_seed=7,
            )

        self.assertEqual(report["sample_size"], 3)
        self.assertIn("pearson_correlation", report)
        self.assertIn("mae", report)
        self.assertIn("speedup_ratio", report)
        self.assertIn("alignment_passed", report)
        self.assertGreater(report["speedup_ratio"], 1.0)
        json.dumps(report, ensure_ascii=False)

    def test_run_direct_phi_alignment_compares_predicted_phi_to_rollout_phi(self) -> None:
        settings = self._make_settings()
        analyzer = ShapleyAnalyzer(settings=settings, rollout_samples=3, rollout_policy="random", max_workers=1)

        sampled = [
            TurnTrajectory(
                game_id="g1",
                turn=index + 1,
                player_id="p1",
                observation={"player_id": "p1"},
                action=ActionModel(type="challenge"),
                skill_name="Skill",
                skill_parameters={},
                checkpoint_format="pickle_base64_v1",
                checkpoint_payload="payload",
            )
            for index in range(3)
        ]

        def _make_attr(turn: int, phi: float) -> ShapleyAttribution:
            return ShapleyAttribution(
                game_id="g1",
                turn=turn,
                player_id="p1",
                skill_name="Skill",
                state_feature="phase=turn_start|table=A|risk=0-1/6|must_call_liar=False",
                death_prob_bucket="0-1/6",
                winner="p1",
                value_action=0.0,
                value_counterfactual=0.0,
                phi=phi,
                rollout_samples=3,
            )

        class FakePredictor:
            def predict_state_features(self, state_features: dict[str, object]) -> float:
                return float(state_features["predicted_phi"])

        with patch.object(ShapleyAnalyzer, "_sample_alignment_trajectories", return_value=sampled), patch.object(
            ShapleyAnalyzer,
            "attribute_step_rollout",
            side_effect=[_make_attr(1, 0.20), _make_attr(2, -0.10), _make_attr(3, 0.40)],
        ), patch.object(
            ShapleyAnalyzer,
            "_build_proxy_feature_context",
            side_effect=[
                {"predicted_phi": 0.22},
                {"predicted_phi": -0.12},
                {"predicted_phi": 0.35},
            ],
        ), patch(
            "liars_game_engine.analysis.shapley_analyzer.LogIterator.iter_games",
            return_value=[],
        ), patch(
            "liars_game_engine.analysis.shapley_analyzer.time.perf_counter",
            side_effect=[0.0, 6.0, 10.0, 11.0],
        ):
            report = analyzer.run_direct_phi_alignment(
                log_paths=[Path("logs/task_d_probe/probe_logs/task-c-20260417-174210-001.jsonl")],
                predictor=FakePredictor(),
                sample_size=3,
                sample_seed=7,
            )

        self.assertEqual(report["sample_size"], 3)
        self.assertIn("pearson_correlation", report)
        self.assertIn("mae", report)
        self.assertGreater(report["pearson_correlation"], 0.9)

    def test_load_value_samples_prefers_phi_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "g1.jsonl"
            records = []
            for turn in (1, 2, 3):
                records.append(
                    {
                        "turn": turn,
                        "player_id": "p2",
                        "observation": {
                            "player_id": "p2",
                            "phase": "response_window",
                            "alive_players": ["p1", "p2"],
                            "private_hand": ["K", "Q"],
                            "pending_claim": {"declared_count": 1},
                            "table_type": "K",
                            "player_states": {"p2": {"death_probability": 0.25}},
                        },
                        "action": {"type": "challenge", "cards": []},
                        "phi": -0.375,
                    }
                )
            log_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            samples = load_value_samples(Path(temp_dir))

        self.assertEqual(len(samples), 3)
        self.assertEqual([round(sample.target, 6) for sample in samples], [-0.375, -0.375, -0.375])


if __name__ == "__main__":
    unittest.main()
