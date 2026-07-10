import unittest

from liars_game_engine.agents.parsers import parse_agent_output, parse_planner_output


class ActionParserTest(unittest.TestCase):
    def test_parser_accepts_markdown_json_block(self) -> None:
        """作用: 验证解析器可从 Markdown JSON 代码块中提取动作。

        输入:
        - 无（测试内构造 raw_output）。

        返回:
        - 无。
        """
        raw_output = (
            "```json\n"
            "{\"thought\":\"I can bluff\",\"action\":{\"type\":\"play_claim\",\"claim_rank\":\"A\",\"cards\":[\"K\"]}}\n"
            "```"
        )

        result = parse_agent_output(raw_output)

        self.assertTrue(result.ok)
        self.assertEqual(result.action.type, "play_claim")
        self.assertEqual(result.action.claim_rank, "A")
        self.assertEqual(result.action.cards, ["K"])

    def test_parser_maps_alias_keys(self) -> None:
        """作用: 验证别名字段 `reasoning/act` 能被归一化解析。

        输入:
        - 无（测试内构造 raw_output）。

        返回:
        - 无。
        """
        raw_output = '{"reasoning":"test","act":{"type":"challenge"}}'

        result = parse_agent_output(raw_output)

        self.assertTrue(result.ok)
        self.assertEqual(result.action.type, "challenge")

    def test_parser_treats_agent_output_keys_case_insensitively(self) -> None:
        raw_output = '{"REASONING":"risk high","ACTION":{"TYPE":"ChAlLeNgE","CARDS":[]}}'

        result = parse_agent_output(raw_output)

        self.assertTrue(result.ok)
        self.assertEqual(result.thought, "risk high")
        self.assertEqual(result.action.type, "challenge")

    def test_parser_returns_structured_error_on_invalid_output(self) -> None:
        """作用: 验证非法文本会返回结构化解析错误。

        输入:
        - 无（测试内构造无效输出）。

        返回:
        - 无。
        """
        result = parse_agent_output("I will just think silently")

        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "E_AGENT_FORMAT_INVALID")

    def test_parse_agent_output_extracts_embedded_json_payload(self) -> None:
        """作用: 验证解析器可从前后裹有说明文字的文本中提取 JSON 动作。

        输入:
        - 无（测试内构造带说明文字的 raw_output）。

        返回:
        - 无。
        """
        raw_output = (
            "我会先分析局势，再给最终答案。\n"
            '{"Reasoning":"轮盘风险偏高，优先质疑更稳。","Action":{"type":"challenge"}}\n'
            "以上是最终决策。"
        )

        result = parse_agent_output(raw_output)

        self.assertTrue(result.ok)
        self.assertEqual(result.thought, "轮盘风险偏高，优先质疑更稳。")
        self.assertEqual(result.action.type, "challenge")

    def test_planner_parser_accepts_selected_skill_schema(self) -> None:
        """作用: 验证 Planner 解析器支持 selected_skill + skill_parameters 架构。

        输入:
        - 无（测试内构造 raw_output）。

        返回:
        - 无。
        """
        raw_output = (
            '{"thought":"risk is medium","selected_skill":"Strategic_Drain",'
            '"skill_parameters":{"bluff_ratio":0.5,"intended_total_cards":2}}'
        )

        result = parse_planner_output(raw_output)

        self.assertTrue(result.ok)
        self.assertEqual(result.selected_skill, "Strategic_Drain")
        self.assertEqual(result.skill_parameters["intended_total_cards"], 2)

    def test_planner_parser_accepts_parameter_alias(self) -> None:
        """作用: 验证 Planner 解析器兼容历史字段 parameter。

        输入:
        - 无（测试内构造 raw_output）。

        返回:
        - 无。
        """
        raw_output = '{"thought":"x","selected_skill":"Calculated_Bluff","parameter":{"num_cards":1}}'

        result = parse_planner_output(raw_output)

        self.assertTrue(result.ok)
        self.assertEqual(result.selected_skill, "Calculated_Bluff")
        self.assertEqual(result.skill_parameters["num_cards"], 1)


if __name__ == "__main__":
    unittest.main()
