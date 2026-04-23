import unittest

from liars_game_engine.config.schema import AppSettings
from liars_game_engine.engine.environment import GameEnvironment
from liars_game_engine.engine.game_state import ActionModel, ClaimState, GamePhase, REVOLVER_BLANK, REVOLVER_LETHAL


class EnvironmentPipelineTest(unittest.TestCase):
    def _make_settings(self) -> AppSettings:
        """作用: 构造环境规则测试所需的基础配置。

        输入:
        - 无。

        返回:
        - AppSettings: 可直接初始化 GameEnvironment 的配置对象。
        """
        return AppSettings.from_dict(
            {
                "api": {},
                "runtime": {"max_turns": 40, "random_seed": 7, "fallback_action": "challenge"},
                "parser": {"max_retries": 3, "allow_markdown_json": True, "allow_key_alias": True},
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
                    {
                        "player_id": "p3",
                        "name": "Carol",
                        "agent_type": "mock",
                        "model": "m3",
                        "prompt_profile": "baseline",
                    },
                ],
                "rules": {
                    "deck_ranks": ["A", "K", "Q", "JOKER"],
                    "cards_per_player": 5,
                    "roulette_slots": 1,
                    "enable_items": False,
                },
            }
        )

    def test_response_window_allows_play_instead_of_challenge(self) -> None:
        """作用: 验证响应窗口允许继续出牌而非只能 challenge。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["K", "A"]
        env.state.players["p2"].hand = ["Q", "A"]
        env.state.players["p3"].hand = ["A"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        first_play = env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["K"]))
        second_play = env.step("p2", ActionModel(type="play_claim", claim_rank="A", cards=["Q"]))

        self.assertTrue(first_play.success)
        self.assertTrue(second_play.success)
        self.assertEqual(env.state.phase, GamePhase.RESPONSE_WINDOW)
        self.assertEqual(env.state.current_player_id, "p3")
        self.assertEqual(env.state.pending_claim.actor_id, "p2")

    def test_play_claim_rejects_more_than_three_cards(self) -> None:
        """作用: 验证单次出牌超过 3 张会被拒绝。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["A", "K", "Q", "A"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        result = env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["A", "K", "Q", "A"]))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "E_ACTION_RULE_VIOLATION")
        self.assertIn("1 to 3", result.error_reason or "")

    def test_challenge_considers_table_type_and_joker_as_innocent(self) -> None:
        """作用: 验证 `table_type + JOKER` 作为 Innocent 的判定逻辑。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["A", "JOKER"]
        env.state.players["p2"].hand = ["Q"]
        env.state.players["p3"].hand = ["K"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["A", "JOKER"]))
        result = env.step("p2", ActionModel(type="challenge"))

        self.assertTrue(result.success)
        self.assertFalse(env.state.players["p1"].eliminated)
        self.assertTrue(env.state.players["p2"].eliminated)

    def test_round_restarts_after_liar_resolution_and_redeals(self) -> None:
        """作用: 验证 challenge 结算后会开启新轮并重发手牌。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["A"]
        env.state.players["p2"].hand = ["K"]
        env.state.players["p3"].hand = ["Q"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["A"]))
        env.step("p2", ActionModel(type="challenge"))

        self.assertEqual(env.state.phase, GamePhase.TURN_START)
        self.assertEqual(env.state.current_player_id, "p3")
        self.assertEqual(len(env.state.players["p1"].hand), 5)
        self.assertEqual(len(env.state.players["p3"].hand), 5)

    def test_only_player_with_cards_must_call_liar(self) -> None:
        """作用: 验证仅剩唯一持牌者时必须 call LIAR。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.phase = GamePhase.RESPONSE_WINDOW
        env.state.current_player_id = "p2"
        env.state.pending_claim = ClaimState(actor_id="p1", claim_rank="A", cards=["K"], declared_count=1)
        env.state.players["p1"].hand = []
        env.state.players["p2"].hand = ["A"]
        env.state.players["p3"].hand = []

        result = env.step("p2", ActionModel(type="play_claim", claim_rank="A", cards=["A"]))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "E_ACTION_RULE_VIOLATION")
        self.assertIn("must call liar", (result.error_reason or "").lower())

    def test_play_claim_rejects_claim_rank_not_matching_table_type(self) -> None:
        """作用: 验证声明牌型不等于桌面牌型时会被拒绝。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["A", "K"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        result = env.step("p1", ActionModel(type="play_claim", claim_rank="K", cards=["A"]))

        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "E_ACTION_RULE_VIOLATION")
        self.assertIn("table type", (result.error_reason or "").lower())

    def test_pile_history_accumulates_play_sequence_before_challenge(self) -> None:
        """作用: 验证同一轮内多次出牌会累积到 pile_history。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["K", "A"]
        env.state.players["p2"].hand = ["Q", "A"]
        env.state.players["p3"].hand = ["A"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["K"]))
        env.step("p2", ActionModel(type="play_claim", claim_rank="A", cards=["Q"]))

        self.assertEqual(len(env.state.pile_history), 2)
        self.assertEqual(env.state.pile_history[0].actor_id, "p1")
        self.assertEqual(env.state.pile_history[1].actor_id, "p2")

    def test_save_and_load_checkpoint_restore_same_future_transition(self) -> None:
        """作用: 验证加载快照后可复现同一步转移结果。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["K", "A"]
        env.state.players["p2"].hand = ["Q", "A"]
        env.state.players["p3"].hand = ["A"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        checkpoint = env.save_checkpoint()
        first = env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["K"]))

        env.load_checkpoint(checkpoint)
        replay = env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["K"]))

        self.assertTrue(first.success)
        self.assertTrue(replay.success)
        self.assertEqual(first.events, replay.events)
        self.assertEqual(env.state.current_player_id, "p2")

    def test_checkpoint_can_be_serialized_and_restored_from_base64(self) -> None:
        """作用: 验证 checkpoint 可序列化到 Base64 并还原后继续运行。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["K", "A"]
        env.state.players["p2"].hand = ["Q", "A"]
        env.state.players["p3"].hand = ["A"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        checkpoint = env.save_checkpoint()
        encoded = env.serialize_checkpoint(checkpoint)
        decoded = env.deserialize_checkpoint(encoded)

        env.load_checkpoint(decoded)
        result = env.step("p1", ActionModel(type="play_claim", claim_rank="A", cards=["K"]))

        self.assertTrue(result.success)
        self.assertEqual(env.state.current_player_id, "p2")

    def test_get_legal_actions_first_turn_excludes_challenge(self) -> None:
        """作用: 验证首手阶段合法动作不包含 challenge。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.players["p1"].hand = ["A", "K"]
        env.state.current_player_id = "p1"
        env.state.phase = GamePhase.TURN_START

        legal = env.get_legal_actions("p1")
        legal_types = {item["type"] for item in legal}

        self.assertIn("play_claim", legal_types)
        self.assertNotIn("challenge", legal_types)

    def test_get_legal_actions_forced_call_lists_only_challenge(self) -> None:
        """作用: 验证强制质疑时合法动作仅剩 challenge。

        输入:
        - 无（测试内部构造状态与动作）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        env.state.turn_order = ["p1", "p2", "p3"]
        env.state.table_type = "A"
        env.state.phase = GamePhase.RESPONSE_WINDOW
        env.state.current_player_id = "p2"
        env.state.pending_claim = ClaimState(actor_id="p1", claim_rank="A", cards=["K"], declared_count=1)
        env.state.players["p1"].hand = []
        env.state.players["p2"].hand = ["A"]
        env.state.players["p3"].hand = []

        legal = env.get_legal_actions("p2")

        self.assertEqual([item["type"] for item in legal], ["challenge"])

    def test_player_runtime_readonly_flags_for_safety_and_death_probability(self) -> None:
        """作用: 验证玩家运行态暴露 is_safe 与 death_probability 两个只读指标。

        输入:
        - 无（测试内部构造状态）。

        返回:
        - 无。
        """
        env = GameEnvironment(self._make_settings())
        player = env.state.players["p1"]

        player.hand = ["A"]
        player.revolver_deck = [REVOLVER_BLANK, REVOLVER_BLANK, REVOLVER_LETHAL, REVOLVER_BLANK]
        self.assertFalse(player.is_safe)
        self.assertAlmostEqual(player.death_probability, 0.25)

        player.hand = []
        self.assertTrue(player.is_safe)


if __name__ == "__main__":
    unittest.main()
