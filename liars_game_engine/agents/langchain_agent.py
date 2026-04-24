from __future__ import annotations

from liars_game_engine.agents.base_agent import AgentDecision, BaseAgent
from liars_game_engine.agents.liar_planner import LiarPlanner
from liars_game_engine.agents.parsers import parse_planner_output
from liars_game_engine.agents.prompts import load_prompt_profile
from liars_game_engine.engine.game_state import ActionModel, ParseError


class LangChainAgent(BaseAgent):
    def __init__(
        self,
        player_id: str,
        model: str,
        prompt_profile: str,
        temperature: float,
        api_key: str,
        base_url: str,
        max_retries: int,
        enable_null_player_probe: bool = False,
    ) -> None:
        """作用: 初始化 LangChainAgent 及其运行参数。

        输入:
        - player_id: 玩家 ID。
        - model: 目标模型名。
        - prompt_profile: Prompt profile 名称。
        - temperature: 推理温度。
        - api_key: OpenAI-compatible API key。
        - base_url: OpenAI-compatible base URL。
        - max_retries: 输出解析失败后的最大重试次数。

        返回:
        - 无。
        """
        super().__init__(player_id=player_id, model=model, prompt_profile=prompt_profile, temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.profile = load_prompt_profile(prompt_profile)
        self.planner = LiarPlanner(enable_null_player_probe=enable_null_player_probe)

    def _build_planner_prompt(self, observation: dict[str, object]) -> str:
        """作用: 拼接含 Skill 规则与观察叙述的 Planner Prompt。

        输入:
        - observation: 当前玩家观测。

        返回:
        - str: 用于 LLM 规划的完整提示词。
        """
        observation_text = self.planner.observation_parser.parse(observation)
        skill_block = self.planner.build_skill_prompt_block()

        return (
            f"SYSTEM:\n{self.profile['system']}\n\n"
            f"SKILL_SYSTEM:\n{skill_block}\n\n"
            f"INSTRUCTION:\n{self.profile['instruction']}\n\n"
            f"OBSERVATION_TEXT:\n{observation_text}\n\n"
            f"OBSERVATION_STRUCT:\n{observation}\n\n"
            "请严格输出 JSON，不要输出额外文本。"
        )

    async def act(self, observation: dict[str, object]) -> AgentDecision:
        """作用: 调用 LangChain 模型并将文本输出解析为结构化动作。

        输入:
        - observation: 当前玩家可见观察信息。

        返回:
        - AgentDecision: 成功时返回解析后的动作；失败时返回带 parse_error 的降级动作。
        """
        if not self.api_key:
            return AgentDecision(
                thought="Missing API key, fallback to challenge.",
                action=ActionModel(type="challenge"),
                selected_skill="Logical_Skepticism",
                skill_parameters={},
                parse_error=ParseError(
                    code="E_AGENT_PROVIDER_UNAVAILABLE",
                    message="An OpenAI-compatible API key is required for LangChainAgent",
                    raw_output="",
                ),
            )

        prompt = self._build_planner_prompt(observation)

        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return AgentDecision(
                thought="LangChain dependency unavailable, fallback to challenge.",
                action=ActionModel(type="challenge"),
                selected_skill="Logical_Skepticism",
                skill_parameters={},
                parse_error=ParseError(
                    code="E_AGENT_PROVIDER_UNAVAILABLE",
                    message="langchain_openai is not installed",
                    raw_output="",
                ),
            )

        client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
        )

        last_raw = ""
        for _ in range(self.max_retries):
            response = await client.ainvoke(prompt)
            content = str(getattr(response, "content", ""))
            last_raw = content
            parsed = parse_planner_output(content)
            if parsed.ok:
                outcome = self.planner.resolve_outcome(
                    thought=parsed.thought,
                    selected_skill=parsed.selected_skill,
                    skill_parameters=parsed.skill_parameters,
                    observation=observation,
                )
                return AgentDecision(
                    thought=outcome.thought,
                    action=outcome.action,
                    raw_output=content,
                    selected_skill=outcome.selected_skill,
                    skill_parameters=outcome.skill_parameters,
                    decision_bias=outcome.decision_bias,
                )

            available_skills = "|".join(self.planner.available_skills.keys())
            prompt = (
                "Previous output invalid. You must return JSON in this schema only: "
                "{\"thought\": string, \"selected_skill\": "
                f"\"{available_skills}\", "
                "\"skill_parameters\": object}. "
                f"Error: {parsed.error.code if parsed.error else 'unknown'} - "
                f"{parsed.error.message if parsed.error else 'unknown'}."
            )

        return AgentDecision(
            thought="Parser retries exceeded, fallback to challenge.",
            action=ActionModel(type="challenge"),
            raw_output=last_raw,
            selected_skill="Logical_Skepticism",
            skill_parameters={},
            parse_error=ParseError(
                code="E_AGENT_FORMAT_INVALID",
                message="Failed to parse model output after retries",
                raw_output=last_raw,
            ),
        )
