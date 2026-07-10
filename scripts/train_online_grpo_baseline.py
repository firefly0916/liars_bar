from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_online_ppo_baseline import (  # noqa: E402
    build_safe_legal_action,
    compute_decision_reward,
    _extract_prompt_text,
    _normalize_action,
    _resolve_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an online GRPO-style LoRA baseline in the Liar's Bar environment.")
    parser.add_argument("--config", default="config/experiment.yaml")
    parser.add_argument("--policy-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--rollout-games-per-update", type=int, default=2)
    parser.add_argument("--train-max-turns", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--win-reward", type=float, default=1.0)
    parser.add_argument("--loss-reward", type=float, default=0.0)
    parser.add_argument("--parse-error-penalty", type=float, default=-0.5)
    parser.add_argument("--fallback-used-penalty", type=float, default=-0.35)
    parser.add_argument("--illegal-action-penalty", type=float, default=-0.75)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="fp16")
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--llm-player-id", default="p1")
    parser.add_argument("--save-final-adapter", action="store_true")
    return parser.parse_args()


def assign_group_advantages(records: list[dict[str, Any]], group_size: int | None = None) -> None:
    if not records:
        return
    if group_size is not None and int(group_size) > 0:
        for index, record in enumerate(records):
            record.setdefault("group_id", f"group-{index // int(group_size)}")

    groups: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        group_id = str(record.get("group_id", f"group-{index // max(1, int(group_size or len(records)))}"))
        record["group_id"] = group_id
        groups.setdefault(group_id, []).append(record)

    for group_records in groups.values():
        rewards = [float(record.get("reward", 0.0) or 0.0) for record in group_records]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        for record, reward in zip(group_records, rewards):
            record["advantage"] = float(reward - mean_reward)
            record["group_reward_mean"] = float(mean_reward)


@dataclass
class GRPODecision:
    decision: Any
    record: dict[str, Any]


class OnlineGRPOPolicy:
    def __init__(self, args: argparse.Namespace, player_id: str, prompt_profile: str) -> None:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from liars_game_engine.agents.prompts import load_prompt_profile

        self.args = args
        self.player_id = player_id
        self.profile = load_prompt_profile(prompt_profile)
        self.device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.policy_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = _resolve_torch_dtype(args.torch_dtype)
        load_kwargs: dict[str, Any] = {"local_files_only": True, "trust_remote_code": True}
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
        if args.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif dtype != "auto":
            load_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(args.policy_model_path, **load_kwargs)
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(model, lora_config)
        self.model.train()

    def trainable_parameters(self) -> list[Any]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def _sequence_logprob(self, input_ids, attention_mask, prompt_len: int):
        import torch
        import torch.nn.functional as F

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        response_mask = torch.zeros_like(gathered, dtype=torch.float32)
        response_start = max(0, int(prompt_len) - 1)
        response_mask[:, response_start:] = attention_mask[:, 1:][:, response_start:].to(dtype=torch.float32)
        denom = torch.clamp(response_mask.sum(dim=1), min=1.0)
        return (gathered * response_mask).sum(dim=1) / denom

    def generate_decision(self, observation: dict[str, object]) -> GRPODecision:
        import torch
        from liars_game_engine.agents.action_resolver import resolve_action_from_intent
        from liars_game_engine.agents.base_agent import AgentDecision
        from liars_game_engine.agents.parsers import parse_agent_output
        from liars_game_engine.agents.prompts import build_openai_messages

        messages = build_openai_messages(self.profile, observation)
        prompt_text = _extract_prompt_text(self.tokenizer, messages)
        encoded_prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded_prompt["input_ids"].to(self.device)
        attention_mask = encoded_prompt.get("attention_mask").to(self.device)
        if input_ids.shape[1] > int(self.args.max_seq_len):
            input_ids = input_ids[:, -int(self.args.max_seq_len) :]
            attention_mask = attention_mask[:, -int(self.args.max_seq_len) :]
        prompt_len = int(input_ids.shape[1])

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=float(self.args.temperature),
                top_p=float(self.args.top_p),
                max_new_tokens=int(self.args.max_new_tokens),
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=int(self.tokenizer.eos_token_id),
            )
            full_attention = torch.ones_like(generated, device=generated.device)
            old_logprob = self._sequence_logprob(generated, full_attention, prompt_len=prompt_len)

        completion_ids = generated[0, prompt_len:]
        raw_output = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        parsed = parse_agent_output(raw_output)
        if parsed.ok and parsed.action_intent is not None:
            resolved = resolve_action_from_intent(
                observation=observation,
                action_type=parsed.action_intent.type,
                play_count=parsed.action_intent.play_count,
                true_card_count=parsed.action_intent.true_card_count,
                cards=list(parsed.action_intent.cards),
            )
            decision = AgentDecision(
                thought=parsed.thought,
                action=resolved.action,
                raw_output=raw_output,
                action_intent=parsed.action_intent.__dict__,
                resolution_reason=resolved.resolution_reason,
            )
            parser_error = None
        elif parsed.ok and parsed.action is not None:
            decision = AgentDecision(thought=parsed.thought, action=parsed.action, raw_output=raw_output)
            parser_error = None
        else:
            decision = AgentDecision(
                thought="Model output invalid, fallback to a legal action.",
                action=build_safe_legal_action(observation),
                raw_output=raw_output,
                parse_error=parsed.error,
            )
            parser_error = {"code": parsed.error.code, "message": parsed.error.message} if parsed.error else {"code": "E_PARSE"}

        record = {
            "prompt_text": prompt_text,
            "input_ids": [int(value) for value in generated[0].detach().cpu().tolist()],
            "prompt_len": prompt_len,
            "old_logprob": float(old_logprob.detach().cpu().item()),
            "parser_error": parser_error,
            "raw_output": raw_output,
            "action": _normalize_action(decision.action),
            "observation": observation,
        }
        return GRPODecision(decision=decision, record=record)

    def save(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)


async def collect_rollout_games(
    *,
    policy: OnlineGRPOPolicy,
    games: int,
    base_seed: int,
    llm_player_id: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from liars_game_engine.agents.mock_agent import MockAgent
    from liars_game_engine.config.loader import load_settings
    from liars_game_engine.engine.environment import GameEnvironment
    from liars_game_engine.engine.game_state import ActionModel

    all_records: list[dict[str, Any]] = []
    game_summaries: list[dict[str, Any]] = []
    for game_index in range(max(1, int(games))):
        game_settings = load_settings(config_file=args.config)
        game_settings.runtime.random_seed = int(base_seed) + int(game_index)
        if int(args.train_max_turns) > 0:
            game_settings.runtime.max_turns = int(args.train_max_turns)
        env = GameEnvironment(game_settings)
        mock_agents = {
            player.player_id: MockAgent(
                player_id=player.player_id,
                model=player.model,
                prompt_profile=player.prompt_profile,
                temperature=player.temperature,
                seed=int(base_seed) + int(game_index) + int(offset) + 17,
            )
            for offset, player in enumerate(game_settings.players)
            if player.player_id != llm_player_id
        }
        turns = 0
        game_records: list[dict[str, Any]] = []
        while turns < int(game_settings.runtime.max_turns) and not env.is_game_over():
            player_id = env.get_current_player()
            observation = env.get_observation_for(player_id)
            if player_id == llm_player_id:
                generated = policy.generate_decision(observation)
                decision = generated.decision
                record = generated.record
            else:
                decision = await mock_agents[player_id].act(observation)
                record = {}

            step_result = env.step(player_id, decision.action)
            original_step_success = bool(step_result.success)
            fallback_used = False
            fallback_reason = None
            if not step_result.success:
                fallback_used = True
                fallback_reason = step_result.error_reason
                step_result = env.step(player_id, ActionModel(type=game_settings.runtime.fallback_action))
                if not step_result.success:
                    step_result = env.step(player_id, build_safe_legal_action(observation))

            if player_id == llm_player_id:
                record["fallback_used"] = bool(fallback_used)
                record["fallback_reason"] = fallback_reason
                record["step_success"] = bool(original_step_success)
                record["turn"] = turns + 1
                record["game_index"] = game_index + 1
                record["reward"] = compute_decision_reward(
                    terminal_reward=0.0,
                    parser_error=record.get("parser_error"),
                    fallback_used=fallback_used,
                    step_success=original_step_success,
                    parse_error_penalty=args.parse_error_penalty,
                    fallback_used_penalty=args.fallback_used_penalty,
                    illegal_action_penalty=args.illegal_action_penalty,
                )
                game_records.append(record)

            turns += 1

        alive_players = [pid for pid, runtime in env.state.players.items() if not runtime.eliminated]
        winner = alive_players[0] if len(alive_players) == 1 else None
        terminal_reward = float(args.win_reward) if winner == llm_player_id else float(args.loss_reward)
        if game_records:
            game_records[-1]["reward"] = float(game_records[-1].get("reward", 0.0) or 0.0) + terminal_reward
            all_records.extend(game_records)
        game_summaries.append({"game_index": game_index + 1, "turns": turns, "winner": winner, "p1_decisions": len(game_records)})
    return all_records, game_summaries


def _collate_records(records: list[dict[str, Any]], device: str) -> dict[str, Any]:
    import torch

    max_len = max(len(record["input_ids"]) for record in records)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    old_logprobs: list[float] = []
    advantages: list[float] = []
    prompt_lens: list[int] = []
    for record in records:
        ids = [int(value) for value in record["input_ids"]]
        pad_count = max_len - len(ids)
        input_rows.append(ids + [0] * pad_count)
        attention_rows.append([1] * len(ids) + [0] * pad_count)
        old_logprobs.append(float(record["old_logprob"]))
        advantages.append(float(record["advantage"]))
        prompt_lens.append(int(record["prompt_len"]))
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32, device=device),
        "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        "prompt_lens": prompt_lens,
    }


def grpo_update(policy: OnlineGRPOPolicy, optimizer: Any, records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, float]:
    import torch

    if not records:
        return {"loss": 0.0, "policy_loss": 0.0, "entropy_proxy": 0.0}
    batch_size = len(records)
    mini_batch_size = max(1, int(args.mini_batch_size))
    indices = list(range(batch_size))
    losses: list[float] = []
    policy_losses: list[float] = []
    entropies: list[float] = []
    for _ in range(max(1, int(args.grpo_epochs))):
        random.shuffle(indices)
        for start in range(0, batch_size, mini_batch_size):
            selected = [records[index] for index in indices[start : start + mini_batch_size]]
            batch = _collate_records(selected, device=policy.device)
            new_logprob_values: list[Any] = []
            for row_index, prompt_len in enumerate(batch["prompt_lens"]):
                new_logprob = policy._sequence_logprob(
                    batch["input_ids"][row_index : row_index + 1],
                    batch["attention_mask"][row_index : row_index + 1],
                    prompt_len=prompt_len,
                )
                new_logprob_values.append(new_logprob.squeeze(0))
            new_logprobs = torch.stack(new_logprob_values)
            advantages = batch["advantages"]
            ratio = torch.exp(torch.clamp(new_logprobs - batch["old_logprobs"], min=-20.0, max=20.0))
            clipped_ratio = torch.clamp(ratio, 1.0 - float(args.clip_range), 1.0 + float(args.clip_range))
            policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            entropy_proxy = -new_logprobs.mean()
            loss = policy_loss - float(args.entropy_coef) * entropy_proxy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            policy_losses.append(float(policy_loss.detach().cpu().item()))
            entropies.append(float(entropy_proxy.detach().cpu().item()))
    return {
        "loss": sum(losses) / len(losses) if losses else 0.0,
        "policy_loss": sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
        "entropy_proxy": sum(entropies) / len(entropies) if entropies else 0.0,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from liars_game_engine.config.loader import load_settings

    random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(config_file=args.config)
    llm_player = next((player for player in settings.players if player.player_id == args.llm_player_id), settings.players[0])
    policy = OnlineGRPOPolicy(args=args, player_id=args.llm_player_id, prompt_profile=llm_player.prompt_profile)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=float(args.learning_rate))

    update_summaries: list[dict[str, Any]] = []
    for update_index in range(1, max(1, int(args.updates)) + 1):
        records, game_summaries = asyncio.run(
            collect_rollout_games(
                policy=policy,
                games=int(args.rollout_games_per_update),
                base_seed=int(args.random_seed) + (update_index * 1000),
                llm_player_id=str(args.llm_player_id),
                args=args,
            )
        )
        assign_group_advantages(records, group_size=int(args.group_size))
        metrics = grpo_update(policy=policy, optimizer=optimizer, records=records, args=args)
        win_count = sum(1 for item in game_summaries if item.get("winner") == args.llm_player_id)
        summary = {
            "update": update_index,
            "rollout_game_count": len(game_summaries),
            "decision_count": len(records),
            "group_size": int(args.group_size),
            "win_count": win_count,
            "win_rate": win_count / len(game_summaries) if game_summaries else 0.0,
            "mean_reward": (
                sum(float(record.get("reward", 0.0) or 0.0) for record in records) / len(records)
                if records
                else 0.0
            ),
            "mean_advantage": (
                sum(float(record.get("advantage", 0.0) or 0.0) for record in records) / len(records)
                if records
                else 0.0
            ),
            **metrics,
        }
        update_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)

    final_dir = checkpoint_dir / "final"
    if args.save_final_adapter:
        policy.save(final_dir)
    train_summary = {
        "algorithm": "online_grpo_lora_baseline",
        "policy_model_path": str(args.policy_model_path),
        "output_dir": str(output_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "final_adapter_path": str(final_dir),
        "updates": int(args.updates),
        "rollout_games_per_update": int(args.rollout_games_per_update),
        "group_size": int(args.group_size),
        "reward_source": "terminal_outcome_plus_protocol_penalty",
        "uses_crc_or_proxy_reward": False,
        "uses_value_head": False,
        "update_summaries": update_summaries,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
    return train_summary


def main() -> int:
    args = parse_args()
    summary = train(args)
    print("Online GRPO baseline training finished:", json.dumps(summary, ensure_ascii=True)[:1000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
