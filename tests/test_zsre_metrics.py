import unittest
from horen_opd.evaluation import evaluate_edits, token_agreement


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens if not skip_special_tokens or token != 1)


class FakeEngine:
    tokenizer = CharTokenizer()

    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.calls = []

    def generate_tokens(self, prompt, max_new_tokens, sample=False):
        self.calls.append((prompt, max_new_tokens, sample))
        continuation = self.tokenizer.encode(self.outputs.get(prompt, ""), add_special_tokens=False)[
            :max_new_tokens
        ]
        return self.tokenizer.encode(prompt) + continuation

    def generate(self, prompt, max_new_tokens, sample=False):
        self.calls.append((prompt, max_new_tokens, sample))
        return self.outputs.get(prompt, "#### 42")


class EditEvaluationTests(unittest.TestCase):
    def test_matches_upstream_autoregressive_token_metric(self):
        engine = FakeEngine({"edit": " car", "paraphrase": "cat", "neighbor": " dog"})
        result = evaluate_edits(engine, [request()])
        self.assertAlmostEqual(result["metrics"]["rewrite"], 2 / 3)
        self.assertEqual(result["metrics"]["rephrase"], 1)
        self.assertIsNone(result["metrics"]["locality"])
        self.assertEqual(result["original_locality"]["0"]["neighborhood"], [[100, 111, 103]])
        self.assertEqual(engine.calls[0], ("edit", 4, False))
        self.assertIsNone(result["metrics"]["portability"])

    def test_early_stop_is_zero_for_rewrite_and_full_tail_for_locality(self):
        result = evaluate_edits(FakeEngine({"edit": "c", "neighbor": "d"}), [request()])
        self.assertEqual(result["metrics"]["rewrite"], 0)
        self.assertEqual(result["original_locality"]["0"]["neighborhood"], [[ord("o"), ord("r"), ord("d")]])

    def test_original_locality_agreement_and_missing_reference(self):
        engine = FakeEngine({"neighbor": " dog"})
        baseline = evaluate_edits(engine, [request()])["original_locality"]
        engine.outputs["neighbor"] = " dig"
        result = evaluate_edits(engine, [request()], baseline)
        self.assertAlmostEqual(result["metrics"]["locality"], 2 / 3)
        with self.assertRaisesRegex(ValueError, "Missing original"):
            evaluate_edits(engine, [request()], {})

    def test_length_agreement_and_validation(self):
        self.assertEqual(token_agreement([], []), 1)
        self.assertEqual(token_agreement([1, 2], [1]), 0.5)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate_edits(FakeEngine(), [request(), request()])
        invalid = request()
        invalid["prompt"] = ["a", "b"]
        invalid["target_new"] = ["c"]
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            evaluate_edits(FakeEngine(), [invalid])


def request(case_id=0):
    return {
        "case_id": case_id,
        "prompt": "edit",
        "target_new": "cat",
        "rephrase_prompt": "paraphrase",
        "locality": {"neighborhood": {"prompt": "neighbor", "ground_truth": "dog"}},
        "portability": {},
    }
