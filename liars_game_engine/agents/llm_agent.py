from __future__ import annotations

from dataclasses import asdict

from liars_game_engine.agents.action_resolver import resolve_action_from_intent
from liars_game_engine.agents.base_agent import AgentDecision, BaseAgent
from liars_game_engine.agents.local_backend import LocalBackendUnavailableError, generate_local_chat_completion
from liars_game_engine.agents.parsers import parse_agent_output
from liars_game_engine.agents.prompts import build_openai_messages, load_prompt_profile
from liars_game_engine.engine.game_state import ActionModel, ParseError

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - exercised via fallback branch when dependency is absent.
    AsyncOpenAI = None


class LlmAgent(BaseAgent):
    def __init__(
        self,
        player_id: str,
        model: str,
        prompt_profile: str,
        temperature: float,
        api_key: str,
        base_url: str,
    ) -> None:
        """作用: 初始化单轮 OpenAI-compatible LLM Agent。

        输入:
        - player_id: 玩家 ID。
        - model: 目标模型名。
        - prompt_profile: Prompt profile 名称。
        - temperature: 采样温度。
        - api_key: OpenAI-compatible API key。
        - base_url: OpenAI-compatible base URL。

        返回:
        - 无。
        """
        super().__init__(player_id=player_id, model=model, prompt_profile=prompt_profile, temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url
        self.profile = load_prompt_profile(prompt_profile)

    async def act(self, observation: dict[str, object]) -> AgentDecision:
        """作用: 发起单次 OpenAI-compatible 对话调用并解析动作 JSON。

        输入:
        - observation: 当前玩家可见观察信息。

        返回:
        - AgentDecision: 成功时返回解析后的动作；失败时返回 challenge 降级动作。
        """
        messages = build_openai_messages(self.profile, observation)
        raw_output = await self._request_raw_output(messages)
        if isinstance(raw_output, AgentDecision):
            return raw_output

        parsed = parse_agent_output(raw_output)
        if parsed.ok and parsed.action_intent is not None:
            resolved = resolve_action_from_intent(
                observation=observation,
                action_type=parsed.action_intent.type,
                play_count=parsed.action_intent.play_count,
                true_card_count=parsed.action_intent.true_card_count,
                cards=list(parsed.action_intent.cards),
            )
            return AgentDecision(
                thought=parsed.thought,
                action=resolved.action,
                raw_output=raw_output,
                action_intent=asdict(parsed.action_intent),
                resolution_reason=resolved.resolution_reason,
            )
        if parsed.ok and parsed.action is not None:
            return AgentDecision(thought=parsed.thought, action=parsed.action, raw_output=raw_output)

        return AgentDecision(
            thought="Model output invalid, fallback to challenge.",
            action=ActionModel(type="challenge"),
            raw_output=raw_output,
            parse_error=parsed.error,
        )

    async def _request_raw_output(self, messages: list[dict[str, str]]) -> str | AgentDecision:
        if self.base_url.startswith("local://"):
            try:
                return await generate_local_chat_completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
            except LocalBackendUnavailableError as error:
                return self._provider_unavailable_decision(str(error))
            except Exception as error:
                return self._provider_unavailable_decision(str(error))

        if not self.api_key:
            return self._provider_unavailable_decision("An OpenAI-compatible API key is required for LlmAgent")

        if AsyncOpenAI is None:
            return self._provider_unavailable_decision("openai is not installed")

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        response = await client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=messages,
        )
        return _extract_message_content(response)

    def _provider_unavailable_decision(self, message: str) -> AgentDecision:
        return AgentDecision(
            thought="LLM provider unavailable, fallback to challenge.",
            action=ActionModel(type="challenge"),
            parse_error=ParseError(
                code="E_AGENT_PROVIDER_UNAVAILABLE",
                message=message,
                raw_output="",
            ),
        )


def _extract_message_content(response: object) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(getattr(item, "text", "")))
        return "".join(text_parts)

    return str(content)
