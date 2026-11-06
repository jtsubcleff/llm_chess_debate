import json
import tempfile
import unittest
from unittest.mock import AsyncMock

from core.chess.protocols.direct_qa import run_direct_qa
from core.chess.protocols.interaction import interaction_messages, run_interaction
from core.chess.word_limits import PUBLIC_WORD_LIMITS, limit_public_response
from tests.test_chess import puzzle, response


def argument(count, final=None):
    return " ".join(
        ["word"] * (count - (final is not None)) + ([final] if final else [])
    )


def public_arguments(result):
    return [
        "".join(e["text"] for e in turn["events"])
        for turn in json.loads(result["transcript"])
    ]


class WordLimitTests(unittest.TestCase):
    def test_opening_reserves_final_word_and_counts_arbitrary_whitespace(self):
        original = "\n\t".join([str(i) for i in range(400)] + ["**e4**"])
        capped, metrics = limit_public_response(
            original, PUBLIC_WORD_LIMITS["opening"], True
        )
        self.assertEqual(capped.split(), [str(i) for i in range(224)] + ["**e4**"])
        self.assertEqual(metrics["public_words_original"], 401)
        self.assertEqual(metrics["public_words"], 225)
        self.assertTrue(metrics["public_truncated"])

    def test_exact_limit_is_unchanged(self):
        for kind, limits in PUBLIC_WORD_LIMITS.items():
            text = argument(limits["maximum"]).replace(" ", "\n")
            bounded, metrics = limit_public_response(text, limits)
            self.assertEqual(bounded, text)
            self.assertFalse(metrics["public_truncated"])

    def test_prompts_request_concision_with_explicit_word_limits(self):
        prior = [
            dict(
                round=1, role="proposer", events=[dict(type="text", text="Opening e4")]
            )
        ]
        for tools in (True, False):
            opening = [
                interaction_messages(mode, "proposer", puzzle(), [], "", tools)
                for mode in ("debate", "consultancy")
            ]
            self.assertEqual(*opening)
            self.assertIn("Be concise", opening[0][1]["content"])
            for mode, role in [
                ("debate", "proposer"),
                ("debate", "critic"),
                ("consultancy", "proposer"),
            ]:
                messages = interaction_messages(
                    mode, role, puzzle(), prior, "e4", tools
                )
                self.assertIn("Be concise", messages[0]["content"])
                self.assertIn(
                    f"{100 if mode == 'debate' else 200} words", json.dumps(messages)
                )
            questioner = interaction_messages(
                "consultancy", "questioner", puzzle(), prior, "e4", tools
            )
            self.assertIn("Be concise", questioner[0]["content"])
            for prompt in [*opening, questioner]:
                self.assertIn("200 words", json.dumps(prompt))


class WordLimitAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_debate_public_limits_apply_before_next_agent_and_preserve_raw_cache(
        self,
    ):
        prompts = []
        answers = iter(
            [argument(500, "e4"), argument(250), argument(230), argument(200)]
        )

        async def answer(model, messages, **kwargs):
            prompts.append(json.dumps(messages))
            return [response(next(answers))]

        with tempfile.TemporaryDirectory() as folder:
            result = await run_interaction(
                AsyncMock(side_effect=answer),
                "gpt-4",
                puzzle(),
                folder,
                "debate",
                2,
                use_tools=False,
            )
            self.assertEqual(
                [len(a.split()) for a in public_arguments(result)], [225, 125, 125, 125]
            )
            self.assertEqual(result["final_move"], "e4")
            self.assertTrue(result["correct"])
            self.assertEqual(result["public_truncated_turns"], 4)
            metrics = json.loads(result["turn_metrics"])
            self.assertEqual(
                [m["public_words_original"] for m in metrics], [500, 250, 230, 200]
            )
            self.assertEqual(
                [m["prompt_word_limit"] for m in metrics], [200, 100, 100, 100]
            )
            trace = json.loads(result["tool_trace"])
            self.assertEqual(len(trace[0]["completion"].split()), 500)
            self.assertIn(
                public_arguments(result)[0], json.loads(prompts[1])[1]["content"]
            )
            self.assertNotIn(argument(500, "e4"), json.loads(prompts[1])[1]["content"])
            replay = await run_interaction(
                AsyncMock(side_effect=AssertionError("Use cache")),
                "gpt-4",
                puzzle(),
                folder,
                "debate",
                2,
                use_tools=False,
            )
            self.assertEqual(result, replay)

    async def test_consultant_and_questioner_are_capped_before_sharing_and_on_replay(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            api = AsyncMock(
                side_effect=[
                    [response(argument(500, "e4"))],
                    [response(argument(900))],
                    [response(argument(500))],
                ]
            )
            result = await run_interaction(
                api,
                "gpt-4",
                puzzle(),
                folder,
                "consultancy",
                2,
                questioner_model="gpt-3.5-turbo",
                use_tools=False,
            )
            self.assertIn(argument(225), api.call_args_list[2].args[1][1]["content"])
            self.assertNotIn(argument(900), api.call_args_list[2].args[1][1]["content"])
            replay = await run_interaction(
                AsyncMock(side_effect=AssertionError("Use cache")),
                "gpt-4",
                puzzle(),
                folder,
                "consultancy",
                2,
                questioner_model="gpt-3.5-turbo",
                use_tools=False,
            )
            self.assertEqual(result, replay)
        self.assertEqual(
            [len(a.split()) for a in public_arguments(result)], [225, 225, 225]
        )
        self.assertEqual(
            [m["public_word_limit"] for m in json.loads(result["turn_metrics"])],
            [225, 225, 225],
        )
        questioner_metrics = json.loads(result["turn_metrics"])[1]
        self.assertEqual(questioner_metrics["prompt_word_limit"], 200)
        self.assertEqual(questioner_metrics["public_words_original"], 900)
        self.assertTrue(questioner_metrics["public_truncated"])

    async def test_direct_qa_both_prompts_and_limits_keep_original_scoring_and_replay(
        self,
    ):
        for tools in (True, False):
            for final, correct in [("e4", True), ("d4", False), ("e4.", False)]:
                with self.subTest(
                    tools=tools, final=final
                ), tempfile.TemporaryDirectory() as folder:
                    api = AsyncMock(return_value=[response(argument(500, final))])
                    result = await run_direct_qa(
                        api, "gpt-4", puzzle(), folder, use_tools=tools
                    )
                    self.assertEqual(len(result["model_response"].split()), 225)
                    self.assertEqual(result["model_response"].split()[-1], final)
                    self.assertEqual(result["final_move"], final)
                    self.assertEqual(result["correct"], correct)
                    self.assertTrue(result["public_truncated"])
                    self.assertIn("Be concise", api.call_args.args[1][0]["content"])
                    self.assertIn("200 words", json.dumps(api.call_args.args[1]))
                    replay = await run_direct_qa(
                        AsyncMock(side_effect=AssertionError("Use cache")),
                        "gpt-4",
                        puzzle(),
                        folder,
                        use_tools=tools,
                    )
                    self.assertEqual(result, replay)

    async def test_private_thinking_does_not_spend_public_word_allowance(self):
        text = "<thinking>" + argument(600) + "</thinking>\n" + argument(225, "e4")
        with tempfile.TemporaryDirectory() as folder:
            result = await run_interaction(
                AsyncMock(return_value=[response(text)]),
                "gpt-4",
                puzzle(),
                folder,
                "consultancy",
                1,
                questioner_model="gpt-4",
                use_tools=False,
            )
        self.assertEqual(len(public_arguments(result)[0].split()), 225)
        self.assertFalse(json.loads(result["turn_metrics"])[0]["public_truncated"])


if __name__ == "__main__":
    unittest.main()
