import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import chess
import pandas as pd
from hydra import compose, initialize_config_dir

from core.chess.judge import load_transcripts, run_judging
from core.chess.protocols.judge import run_judge, score_judgement
from core.chess.results import write_csv
from core.chess.run import run_experiment
from core.llm_api.base_llm import LLMResponse
from core.load.chess import ChessPuzzle


def response(text, calls=None, reason="stop"):
    return LLMResponse(
        "gpt-4",
        text,
        reason,
        0.01,
        input_tokens=10,
        output_tokens=5,
        tool_calls=calls or [],
    )


def config(folder, name="chess_judge", overrides=()):
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("core/config").resolve())
    ):
        return compose(
            config_name=name, overrides=[f"exp_dir={folder}/judged", *overrides]
        )


def save_source(folder, model="gpt-4"):
    path = Path(folder) / "direct_qa" / model / "puzzles.csv"
    write_csv(
        pd.DataFrame(
            [
                dict(
                    puzzleId=f"00{i}",
                    FEN=chess.STARTING_FEN,
                    moves="e2e4",
                    rating=rating,
                    rating_bucket=bucket,
                    model=model,
                    complete=True,
                    correct=correct,
                    final_move=move,
                    model_response=f"My analysis. {move}",
                    tool_trace="[]",
                )
                for i, (rating, bucket, correct, move) in enumerate(
                    [
                        (1000, "Low", True, "e4"),
                        (2000, "Medium", False, "a3"),
                        (2500, "High", False, "h3"),
                    ]
                )
            ]
        ),
        path,
    )
    return path


class JudgeScoringTests(unittest.TestCase):
    def test_binary_scoring_contract(self):
        for correct, text, expected in [
            (True, "Reasoning Yes", True),
            (False, "Reasoning No", True),
            (True, "No", False),
            (False, "Yes", False),
            (False, "No.", True),
            (True, "**Yes**", True),
            (False, "No**", True),
            (True, "Yes!!!", True),
            (False, "**No.**", True),
            (True, "Verdict: ‘Yes’…", True),
            (False, "No . **", True),
            (False, "Yes/No.", False),
            (False, "No. More explanation.", False),
            (False, "", False),
            (False, "Yes then no", False),
        ]:
            with self.subTest(text=text, correct=correct):
                self.assertEqual(score_judgement(text, correct)[1], expected)


class JudgeRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_tools_prompt_truncation_and_cache(self):
        puzzle = ChessPuzzle("001", chess.STARTING_FEN, "e2e4", 1000, "Low")
        proposal = dict(
            final_move="a3",
            model_response="Try this argument a3",
            proposer_correct=False,
            tool_trace='[{"signature":"RAW_TRACE_SENTINEL"}]',
        )
        calls = [
            dict(
                id="v",
                type="function",
                function=dict(
                    name="validate_move_sequence", arguments='{"moves":["a3","e5"]}'
                ),
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            api = AsyncMock(
                side_effect=[
                    [response("Check replies", calls)],
                    [response("It fails. No")],
                ]
            )
            result = await run_judge(
                api, "gpt-4", puzzle, proposal, folder, max_tool_rounds=1
            )
            self.assertTrue(result["correct"])
            self.assertEqual(result["judgement"], "No")
            self.assertEqual(result["validate_move_sequence_calls"], 1)
            # AsyncMock retains the mutable conversation, including tool results.
            messages = api.call_args_list[0].args[1]
            user = messages[1]["content"]
            self.assertIn("solver's public explanation", messages[0]["content"])
            self.assertIn("Proposed solution: a3", user)
            self.assertIn(proposal["model_response"], user)
            self.assertNotIn("RAW_TRACE_SENTINEL", user)
            self.assertNotIn("Tool transcript (responses, calls and results)", user)
            self.assertNotIn("e2e4", user)
            self.assertNotIn("proposer_correct", user)
            self.assertEqual(api.call_args_list[-1].kwargs["tool_choice"], "none")
            self.assertEqual(
                result,
                await run_judge(
                    AsyncMock(side_effect=AssertionError("cache")),
                    "gpt-4",
                    puzzle,
                    proposal,
                    folder,
                    max_tool_rounds=1,
                ),
            )
        for reason in ("length", "refusal"):
            with tempfile.TemporaryDirectory() as folder:
                result = await run_judge(
                    AsyncMock(return_value=[response("No", reason=reason)]),
                    "gpt-4",
                    puzzle,
                    proposal,
                    folder,
                )
                self.assertTrue(result["complete"])
                self.assertFalse(result["correct"])

    @patch(
        "builtins.input",
        side_effect=AssertionError("Runs must not prompt for a cost estimate"),
    )
    async def test_standalone_accuracy_resume_and_source_pinning(self, _):
        with tempfile.TemporaryDirectory() as folder:
            source = save_source(folder)
            cfg = config(
                folder, overrides=[f"transcripts_dir={folder}", "models=[gpt-4]"]
            )
            api = AsyncMock(
                side_effect=[[response("Yes")], [response("No")], [response("Yes")]]
            )
            with self.assertLogs("core.chess.judge", level="INFO") as logs:
                await run_judging(cfg, api)
            self.assertEqual(len(logs.output), 3)
            for puzzle_id in ("000", "001", "002"):
                self.assertTrue(
                    any(
                        f"gpt-4 puzzle {puzzle_id}: complete=True correct=" in line
                        for line in logs.output
                    )
                )
            summary = pd.read_csv(Path(cfg.exp_dir) / "results.csv")
            self.assertAlmostEqual(summary.accuracy.iloc[0], 200 / 3)
            self.assertEqual(summary.completed.iloc[0], 3)
            with self.assertNoLogs("core.chess.judge", level="INFO"):
                await run_judging(
                    cfg, AsyncMock(side_effect=AssertionError("completed"))
                )
            frame = pd.read_csv(source, keep_default_na=False, dtype={"puzzleId": str})
            frame.at[0, "model_response"] = "changed"
            write_csv(frame, source)
            with self.assertRaisesRegex(ValueError, "transcripts changed"):
                await run_judging(cfg, api)

    @patch(
        "builtins.input",
        side_effect=AssertionError("Runs must not prompt for a cost estimate"),
    )
    async def test_pair_pipeline_and_aggregate_multiple_proposers(self, _):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "dataset.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
                for i, rating in enumerate((1000, 2000, 2500)):
                    writer.writerow([f"00{i}", chess.STARTING_FEN, "e2e4", rating])
            cfg = config(
                folder,
                "chess",
                [
                    "protocol=proposer_judge",
                    "models=[[gpt-4,gpt-4],[gpt-3.5-turbo,gpt-4]]",
                    f"dataset_path={source}",
                    "limit=3",
                    "use_tools=false",
                ],
            )

            async def answer(model, messages, **kwargs):
                return [
                    response("Yes" if "judging" in messages[0]["content"] else "e4")
                ]

            api = AsyncMock(side_effect=answer)
            await run_experiment(cfg, api)
            self.assertEqual(api.await_count, 12)
            summary = pd.read_csv(Path(cfg.exp_dir) / "judgements" / "results.csv")
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.completed.iloc[0], 6)
            self.assertEqual(summary.accuracy.iloc[0], 100)
            for call in api.call_args_list:
                if "judging" in call.args[1][0]["content"]:
                    self.assertEqual(len(call.kwargs["tools"]), 1)
                else:
                    self.assertNotIn("tools", call.kwargs)
            await run_experiment(cfg, AsyncMock(side_effect=AssertionError("resume")))
            cfg.models = ["gpt-4"]
            with self.assertRaisesRegex(ValueError, "pairs"):
                await run_experiment(cfg, api)

    @patch(
        "builtins.input",
        side_effect=AssertionError("Runs must not prompt for a cost estimate"),
    )
    async def test_failure_and_resume(self, _):
        with tempfile.TemporaryDirectory() as folder:
            source = save_source(folder)
            original = source.read_bytes()
            cfg = config(
                folder, overrides=[f"transcripts_dir={folder}", "models=[gpt-4]"]
            )
            api = AsyncMock(
                side_effect=[
                    RuntimeError("provider unavailable"),
                    [response("No")],
                    [response("No")],
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "Rerun to resume"):
                await run_judging(cfg, api)
            summary = pd.read_csv(Path(cfg.exp_dir) / "results.csv")
            self.assertEqual(summary.completed.iloc[0], 2)
            retry = AsyncMock(return_value=[response("Yes")])
            await run_judging(cfg, retry)
            self.assertEqual(retry.await_count, 1)
            self.assertEqual(source.read_bytes(), original)
            summary = pd.read_csv(Path(cfg.exp_dir) / "results.csv")
            self.assertEqual(summary.accuracy.iloc[0], 100)

    @patch(
        "builtins.input",
        side_effect=AssertionError("Runs must not prompt for a cost estimate"),
    )
    async def test_pairs_do_not_form_cartesian_product(self, _):
        with tempfile.TemporaryDirectory() as folder:
            save_source(folder, "gpt-4")
            save_source(folder, "gpt-3.5-turbo")
            cfg = config(
                folder,
                overrides=[f"transcripts_dir={folder}", "models=[gpt-4,gpt-3.5-turbo]"],
            )
            pairs = {("gpt-4", "gpt-3.5-turbo"), ("gpt-3.5-turbo", "gpt-4")}
            api = AsyncMock(return_value=[response("No")])
            frames = await run_judging(cfg, api, pairs=pairs)
            self.assertEqual(api.await_count, 6)
            self.assertEqual(len(frames), 2)
            for (judge, source), frame in frames.items():
                self.assertIn((frame.proposer.iloc[0], judge), pairs)

    async def test_empty_and_incomplete_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "No completed"):
                load_transcripts(folder)
            path = save_source(folder)
            frame = pd.read_csv(path)
            frame["complete"] = False
            write_csv(frame, path)
            with self.assertRaisesRegex(ValueError, "No completed"):
                load_transcripts(folder)


if __name__ == "__main__":
    unittest.main()
