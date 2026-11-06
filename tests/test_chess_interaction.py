import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
from attrs import evolve

from core.chess.interaction import interaction_models
from core.chess.judge import load_transcripts, run_judging
from core.chess.protocols.interaction import (
    format_transcript,
    interaction_messages,
    public_turn,
    run_interaction,
    turn_schedule,
)
from core.chess.rescore import rescore_experiment
from core.chess.run import run_experiment
from tests.test_chess import puzzle, response
from tests.test_chess_judge import config


def tool(identifier="c1", moves='["e4"]'):
    return dict(
        id=identifier,
        type="function",
        function=dict(
            name="validate_move_sequence",
            arguments='{"moves":' + moves + "}",
        ),
    )


class PublicTranscriptTests(unittest.TestCase):
    def test_reused_call_ids_do_not_expose_private_results(self):
        call = tool("same_id")
        events, _ = public_turn(
            [
                response("Public", [call], "tool_calls").to_dict(),
                dict(tool_call=call, result="PUBLIC RESULT"),
                response("<thinking>", [call], "tool_calls").to_dict(),
                dict(tool_call=call, result="SECRET RESULT"),
                response("</thinking>Done e4").to_dict(),
            ]
        )
        self.assertNotIn("PUBLIC RESULT", json.dumps(events))
        self.assertNotIn("SECRET RESULT", json.dumps(events))

    def test_private_tools_spanning_responses_and_unclosed_thinking(self):
        private, public = tool("private"), tool("public")
        trace = [
            response(
                "Public start <thinking>SECRET", [private], "tool_calls"
            ).to_dict(),
            dict(tool_call=private, result=dict(secret="PRIVATE RESULT")),
            response(
                "more secret</thinking>Public end", [public], "tool_calls"
            ).to_dict(),
            dict(tool_call=public, result=dict(legal=True)),
            response("Best e4<thinking>unclosed secret").to_dict(),
        ]
        events, final = public_turn(trace)
        text = json.dumps(events)
        for hidden in (
            "SECRET",
            "PRIVATE RESULT",
            "private",
            "unclosed secret",
            "more secret",
        ):
            self.assertNotIn(hidden, text)
        self.assertNotIn('"public"', text)
        self.assertNotIn('"legal": true', text)
        self.assertEqual(final, "Best e4")

    def test_native_order_and_provider_reasoning(self):
        hidden, shared = tool("hidden"), tool("shared")
        native = [
            dict(type="thinking", thinking="NATIVE SECRET", signature="opaque"),
            dict(type="text", text="<thinking>PRIVATE"),
            dict(type="tool_use", id="hidden", name="validate_move_sequence", input={}),
            dict(type="text", text="</thinking>Before"),
            dict(type="tool_use", id="shared", name="validate_move_sequence", input={}),
            dict(type="text", text="After"),
        ]
        r = evolve(
            response("<thinking>PRIVATE</thinking>BeforeAfter", [hidden, shared]),
            response_items=native,
            stop_reason="tool_calls",
        )
        events, _ = public_turn(
            [
                r.to_dict(),
                dict(tool_call=hidden, result="HIDDEN RESULT"),
                dict(tool_call=shared, result="SHARED RESULT"),
            ]
        )
        self.assertEqual([e["type"] for e in events], ["text", "text"])
        self.assertEqual(events[0]["text"], "Before")
        self.assertEqual(events[1]["text"], "After")
        self.assertNotIn("SECRET", json.dumps(events))
        self.assertNotIn("PRIVATE", json.dumps(events))

    def test_legacy_tool_events_are_not_rendered_or_forwarded(self):
        transcript = [
            dict(
                round=1,
                role="proposer",
                model="gpt-4",
                events=[
                    dict(type="text", text="Public argument e4"),
                    dict(
                        type="tool",
                        tool_call=tool("LEGACY_TOOL"),
                        result="LEGACY_RESULT",
                    ),
                ],
            )
        ]
        formatted = format_transcript(transcript)
        self.assertIn("Public argument e4", formatted)
        self.assertNotIn("LEGACY", formatted)
        for mode, role in [
            ("debate", "critic"),
            ("debate", "proposer"),
            ("consultancy", "questioner"),
            ("consultancy", "proposer"),
        ]:
            messages = interaction_messages(
                mode, role, puzzle(), transcript, "e4", True
            )
            self.assertIn("Public argument e4", messages[1]["content"])
            self.assertNotIn("LEGACY", json.dumps(messages))

    def test_round_counts_and_validation(self):
        self.assertEqual(turn_schedule("debate", 1), [(1, "proposer"), (1, "critic")])
        self.assertEqual(turn_schedule("consultancy", 1), [(1, "proposer")])
        self.assertEqual(
            turn_schedule("consultancy", 2),
            [
                (1, "proposer"),
                (1, "questioner"),
                (2, "proposer"),
            ],
        )
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                turn_schedule("debate", invalid)


class InteractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_debate_lock_full_public_history_and_cached_replay(self):
        prompts = []
        answers = iter(
            [
                response("<thinking>SECRET</thinking>Opening defense e4"),
                response("Critic objection"),
                response("Revised argument d4"),
                response("Critic last word"),
            ]
        )

        async def answer(model, messages, **kwargs):
            prompts.append((model, json.dumps(messages)))
            return [next(answers)]

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
            self.assertTrue(result["complete"])
            self.assertTrue(result["correct"])
            self.assertEqual(result["final_move"], "e4")
            self.assertEqual(result["api_calls"], 4)
            self.assertEqual(result["input_tokens"], 40)
            self.assertAlmostEqual(result["cost"], 0.04)
            self.assertEqual(
                [t["role"] for t in json.loads(result["transcript"])],
                ["proposer", "critic", "proposer", "critic"],
            )
            self.assertIn("Opening defense", prompts[3][1])
            self.assertIn("Critic objection", prompts[3][1])
            self.assertIn("Revised argument", prompts[3][1])
            self.assertNotIn("SECRET", result["model_response"])
            self.assertNotIn("SECRET", "".join(p[1] for p in prompts))
            self.assertIn("SECRET", result["tool_trace"])
            again = await run_interaction(
                AsyncMock(side_effect=AssertionError("cache")),
                "gpt-4",
                puzzle(),
                folder,
                "debate",
                2,
                use_tools=False,
            )
            self.assertEqual(result, again)

    async def test_consultancy_models_tools_and_privacy(self):
        prompts = []
        answers = iter(
            [
                response("<thinking>SECRET", [tool("hidden")], "tool_calls"),
                response(
                    "</thinking>Public", [tool("proposer_tool_sentinel")], "tool_calls"
                ),
                response("Defense e4"),
                response(
                    "Checking a question", [tool("questioner_tool")], "tool_calls"
                ),
                response("What about the reply?"),
                response("Answer to question d4"),
            ]
        )

        async def answer(model, messages, **kwargs):
            prompts.append((model, json.dumps(messages), kwargs))
            return [next(answers)]

        with tempfile.TemporaryDirectory() as folder:
            result = await run_interaction(
                AsyncMock(side_effect=answer),
                "gpt-4",
                puzzle(),
                folder,
                "consultancy",
                2,
                questioner_model="gpt-3.5-turbo",
                max_tool_rounds=2,
            )
        self.assertEqual(
            [p[0] for p in prompts],
            ["gpt-4", "gpt-4", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-4"],
        )
        shared = prompts[3][1]
        self.assertNotIn("SECRET", shared)
        self.assertNotIn("hidden", shared)
        self.assertNotIn("proposer_tool_sentinel", shared)
        self.assertNotIn('"legal"', shared)
        self.assertIn("What about the reply?", prompts[5][1])
        self.assertIn("questioner_tool", prompts[4][1])
        self.assertNotIn("questioner_tool", prompts[5][1])
        self.assertIn(
            "proposer_tool_sentinel", prompts[2][1]
        )  # Own tool history remains available.
        self.assertNotIn("proposer_tool_sentinel", result["model_response"])
        self.assertNotIn("proposer_tool_sentinel", result["transcript"])
        self.assertIn("proposer_tool_sentinel", result["tool_trace"])
        self.assertTrue(all(len(p[2]["tools"]) == 1 for p in prompts))
        self.assertEqual(result["validate_move_sequence_calls"], 3)
        self.assertEqual(result["final_move"], "e4")
        self.assertTrue(result["correct"])

    async def test_mid_debate_failure_resume_keeps_opening_and_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            result = await run_interaction(
                AsyncMock(
                    side_effect=[
                        [response("Opening e4")],
                        [response("Critic", [tool()], "tool_calls")],
                        RuntimeError("transient"),
                    ]
                ),
                "gpt-4",
                puzzle(),
                folder,
                "debate",
                2,
            )
            self.assertFalse(result["complete"])
            self.assertEqual(result["final_move"], "e4")
            self.assertIn("transient", result["error"])
            self.assertTrue(result["correct"])
            retry = AsyncMock(
                side_effect=[
                    [response("Critic conclusion")],
                    [response("Defense d4")],
                    [response("Last critic")],
                ]
            )
            result = await run_interaction(
                retry, "gpt-4", puzzle(), folder, "debate", 2
            )
            self.assertTrue(result["complete"])
            self.assertEqual(retry.await_count, 3)
            self.assertEqual(result["api_calls"], 5)
            self.assertEqual(result["validate_move_sequence_calls"], 1)
            self.assertEqual(result["final_move"], "e4")

    async def test_opening_refusal_truncation_and_no_public_answer(self):
        for text, reason in [
            ("e4", "length"),
            ("e4", "refusal"),
            ("<thinking>e4</thinking>", "stop"),
        ]:
            with tempfile.TemporaryDirectory() as folder:
                api = AsyncMock(
                    side_effect=[[response(text, reason=reason)], [response("critic")]]
                )
                result = await run_interaction(
                    api, "gpt-4", puzzle(), folder, "debate", 1
                )
                self.assertTrue(result["complete"])
                self.assertFalse(result["correct"])


class InteractionRunnerTests(unittest.IsolatedAsyncioTestCase):
    @patch(
        "builtins.input",
        side_effect=AssertionError("Runs must not prompt for a cost estimate"),
    )
    async def test_pairs_judging_rescoring_manifest_and_resume(self, _):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            dataset = Path(folder) / "dataset.csv"
            with dataset.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
                for i, rating in enumerate((1000, 2000, 2500)):
                    writer.writerow([f"00{i}", puzzle().FEN, "e2e4", rating])
            cfg = config(
                folder,
                "chess",
                [
                    "protocol=consultancy",
                    "num_rounds=2",
                    "use_tools=false",
                    "limit=3",
                    "models=[[gpt-4,gpt-3.5-turbo],[gpt-4,gpt-4]]",
                    f"dataset_path={dataset}",
                ],
            )

            async def answer(model, messages, **kwargs):
                return [response("<thinking>SECRET</thinking>Public argument e4")]

            api = AsyncMock(side_effect=answer)
            frames = await run_experiment(cfg, api)
            self.assertEqual(api.await_count, 18)
            self.assertEqual(len(frames), 2)
            summary = pd.read_csv(Path(cfg.exp_dir) / "results.csv")
            self.assertEqual(len(summary), 2)
            sources = load_transcripts(cfg.exp_dir)
            self.assertEqual(len(sources), 6)
            self.assertTrue(all("SECRET" not in json.dumps(s) for s in sources))
            judge_cfg = config(
                folder, overrides=[f"transcripts_dir={cfg.exp_dir}", "models=[gpt-4]"]
            )
            judge_cfg.exp_dir = str(Path(folder) / "final_judge")
            judged = await run_judging(
                judge_cfg, AsyncMock(return_value=[response("Yes")])
            )
            self.assertEqual(len(judged), 2)
            await run_experiment(
                cfg, AsyncMock(side_effect=AssertionError("completed"))
            )
            reports = rescore_experiment(cfg.exp_dir, protocol="consultancy")
            self.assertTrue(
                all(r["changed"] == 0 and r["rescored"] == 3 for r in reports)
            )
            debate_cfg = config(
                folder,
                "chess",
                [
                    "protocol=debate",
                    "num_rounds=1",
                    "models=[gpt-4]",
                    "use_tools=false",
                    "limit=3",
                    f"dataset_path={dataset}",
                    f"exp_dir={folder}/debate_run",
                ],
            )
            debate_api = AsyncMock(side_effect=answer)
            await run_experiment(debate_cfg, debate_api)
            self.assertEqual(debate_api.await_count, 6)
            self.assertEqual(len(load_transcripts(debate_cfg.exp_dir)), 3)
            self.assertEqual(
                rescore_experiment(debate_cfg.exp_dir, protocol="debate")[0]["changed"],
                0,
            )
            cfg.num_rounds = 3
            with self.assertRaisesRegex(ValueError, "settings or sample changed"):
                await run_experiment(cfg, api)

    async def test_loading_legacy_interactions_strips_tools_before_judging(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "debate" / "old" / "puzzles.csv"
            path.parent.mkdir(parents=True)
            transcript = [
                dict(
                    round=1,
                    role="proposer",
                    model="gpt-4",
                    events=[
                        dict(type="text", text="Public argument e4"),
                        dict(
                            type="tool",
                            tool_call=tool("LEGACY_TOOL"),
                            result="LEGACY_RESULT",
                        ),
                    ],
                )
            ]
            pd.DataFrame(
                [
                    dict(
                        **puzzle().to_dict(),
                        model="gpt-4",
                        complete=True,
                        correct=True,
                        final_move="e4",
                        model_response="Public argument e4 LEGACY_RESULT",
                        transcript=json.dumps(transcript),
                        tool_trace="AUDIT_ONLY",
                    )
                ]
            ).to_csv(path, index=False)
            sources = load_transcripts(folder)
            self.assertIn("Public argument e4", sources[0]["model_response"])
            self.assertNotIn("LEGACY", json.dumps(sources))
            self.assertNotIn("AUDIT_ONLY", json.dumps(sources))
            cfg = config(
                folder, overrides=[f"transcripts_dir={folder}", "models=[gpt-4]"]
            )
            api = AsyncMock(return_value=[response("Yes")])
            await run_judging(cfg, api)
            self.assertIn("Public argument e4", api.call_args.args[1][1]["content"])
            self.assertNotIn("LEGACY", api.call_args.args[1][1]["content"])
            self.assertNotIn("AUDIT_ONLY", api.call_args.args[1][1]["content"])

    async def test_invalid_pairs_and_model_resolution(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = config(
                folder,
                "chess",
                ["protocol=consultancy", "models=[gpt-4]", "num_rounds=2"],
            )
            with self.assertRaisesRegex(ValueError, "pairs"):
                interaction_models(cfg)
            cfg.models = [["gpt-4", "gpt-3.5-turbo"]]
            specs = next(iter(interaction_models(cfg).values()))
            self.assertEqual(specs[0]["id"], "gpt-4")
            self.assertEqual(specs[1]["id"], "gpt-3.5-turbo")


if __name__ == "__main__":
    unittest.main()
