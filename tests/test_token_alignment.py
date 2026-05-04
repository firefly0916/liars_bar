import importlib
import json
import unittest


class _FakeQwenTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        input_ids = list(range(len(text)))
        payload = {"input_ids": input_ids}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload


class TokenAlignmentTest(unittest.TestCase):
    def test_build_alignment_metadata_accounts_for_rendered_template_offset(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.token_alignment")
        sample = {
            "thought": "我怀疑对手在撒谎，所以选择挑战。",
            "action": {"type": "challenge", "claim_rank": "", "cards": []},
            "strategic_tokens": [
                {
                    "label": "skepticism",
                    "token": "怀疑",
                    "start": 1,
                    "end": 3,
                    "weight": 1.4,
                    "penalty_signal": -0.4,
                }
            ],
        }
        tokenizer = _FakeQwenTokenizer()
        messages = [
            {"role": "system", "content": "You are a strategic Liar's Bar player."},
            {"role": "user", "content": "State report omitted for test."},
        ]

        aligned = module.align_sample_to_tokens(
            sample=sample,
            tokenizer=tokenizer,
            messages=messages,
            weight_distribution_strategy="equal",
        )

        metadata = aligned["alignment_metadata"]
        mask = aligned["token_weight_mask"]
        self.assertGreater(metadata["assistant_rendered_span"]["start"], 0)
        self.assertGreater(
            metadata["reasoning_rendered_span"]["start"],
            metadata["assistant_rendered_span"]["start"],
        )
        self.assertTrue(metadata["mapping_rules"]["chat_template_offset_applied"])
        self.assertGreaterEqual(len(metadata["special_token_spans"]), 2)
        alignment = metadata["strategic_token_alignments"][0]
        self.assertEqual(alignment["token"], "怀疑")
        self.assertEqual(alignment["allocation_strategy"], "equal")
        self.assertEqual(alignment["raw_char_span"], {"start": 1, "end": 3})
        self.assertEqual(len(alignment["token_indices"]), 2)
        self.assertEqual(alignment["distributed_weights"], [-0.28, -0.28])
        self.assertAlmostEqual(sum(mask), -0.56, places=6)
        for index in alignment["token_indices"]:
            self.assertAlmostEqual(mask[index], -0.28, places=6)

    def test_build_alignment_metadata_preserves_rendering_and_indices(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.token_alignment")
        sample = {
            "thought": "风险很高，但我想试探一下。",
            "action": {"type": "play_claim", "claim_rank": "A", "cards": ["Q"]},
            "strategic_tokens": [
                {
                    "label": "risk",
                    "token": "风险",
                    "start": 0,
                    "end": 2,
                    "weight": 1.5,
                    "penalty_signal": -0.2,
                }
            ],
        }
        tokenizer = _FakeQwenTokenizer()
        assistant_text = module.build_assistant_response_text(sample)
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "Dummy prompt"},
                {"role": "assistant", "content": assistant_text},
            ]
        )

        metadata = module.build_alignment_metadata(
            sample=sample,
            tokenizer=tokenizer,
            messages=[
                {"role": "user", "content": "Dummy prompt"},
                {"role": "assistant", "content": assistant_text},
            ],
            assistant_text=assistant_text,
            weight_distribution_strategy="equal",
        )

        self.assertEqual(metadata["assistant_text"], assistant_text)
        self.assertEqual(metadata["rendered_text"], rendered)
        self.assertEqual(metadata["reasoning_text"], sample["thought"])
        self.assertEqual(metadata["token_count"], len(rendered))
        self.assertEqual(metadata["mapping_rules"]["span_overlap_rule"], "token_end > span_start and token_start < span_end")

    def test_build_assistant_response_text_uses_action_first_intent_schema(self) -> None:
        module = importlib.import_module("liars_game_engine.analysis.token_alignment")
        sample = {
            "thought": "我想诚实试探一下。",
            "action": {"type": "play_claim", "claim_rank": "A", "cards": ["A", "Q"]},
            "observation": {"table_type": "A"},
        }

        assistant_text = module.build_assistant_response_text(sample)
        payload = json.loads(assistant_text)

        self.assertEqual(sorted(payload.keys()), ["Action", "Reasoning"])
        self.assertEqual(payload["Reasoning"], sample["thought"])
        self.assertEqual(payload["Action"]["type"], "play_claim")
        self.assertEqual(payload["Action"]["play_count"], 2)
        self.assertEqual(payload["Action"]["true_card_count"], 1)
        self.assertNotIn("claim_rank", payload["Action"])
        self.assertNotIn("cards", payload["Action"])


if __name__ == "__main__":
    unittest.main()
