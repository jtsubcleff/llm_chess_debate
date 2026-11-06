import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import yaml

from core.chess.protocols.interaction import interaction_messages, run_interaction
from core.chess.run import run_experiment
from core.chess.suite import saved_rows, summarize_suite
from core.llm_api.registry import CatalogLoader
from core.prompts import load_template
from tests.test_chess import puzzle, response
from tests.test_chess_interaction import tool
from tests.test_chess_judge import config


def suite_config(folder, pairs="[[gpt-4,gpt-3.5-turbo]]", extra=()):
    dataset = Path(folder) / "dataset.csv"
    with dataset.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
        for index, rating in enumerate((1000, 2000, 2500)):
            writer.writerow([f"00{index}", puzzle().FEN, "e2e4", rating])
    return config(
        folder,
        "chess",
        [
            "protocol=research_suite",
            "num_rounds=2",
            "use_tools=false",
            "limit=3",
            f"models={pairs}",
            f"dataset_path={dataset}",
            *extra,
        ],
    )


class SuiteTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompts_are_loaded_once_per_interaction(self):
        template = load_template("chess_interaction")
        with tempfile.TemporaryDirectory() as folder:
            with patch(
                "core.chess.protocols.interaction.load_template",
                side_effect=[template, AssertionError("Prompt reread during a run")],
            ) as loader:
                api = AsyncMock(
                    side_effect=[[response("Opening e4")], [response("Critic audit")]]
                )
                result = await run_interaction(
                    api, "gpt-4", puzzle(), folder, "debate", 1, use_tools=False
                )
                self.assertTrue(result["complete"])
                self.assertEqual(loader.call_count, 1)

    def test_summary_preserves_class_counts_invalid_verdicts_and_matched_cohort(self):
        conditions = ["single_consultancy", "multi_consultancy"]
        manifest = dict(
            conditions=conditions,
            models=[dict(proposer=dict(id="p"), judge=dict(id="j"))],
            puzzles=[dict(puzzleId=str(i), rating_bucket="Low") for i in range(3)],
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for condition in conditions:
                proposals = [
                    dict(
                        puzzleId=str(i),
                        model="p",
                        target_judge="j",
                        correct=i != 1,
                        final_move="e4",
                        proposal_valid=True,
                        interaction_valid=True,
                        proposal_id=str(i),
                    )
                    for i in range(3)
                ]
                verdicts = [
                    dict(
                        puzzleId=str(i),
                        proposer="p",
                        model="j",
                        complete=True,
                        judgement="Yes" if i < 2 else "No",
                        verdict_valid=i != 2,
                        correct=i == 0,
                    )
                    for i in range(3)
                ]
                if condition.startswith("multi"):
                    verdicts[1]["complete"] = False
                    verdicts[2]["verdict_valid"] = True
                paths = [
                    (
                        root
                        / "conditions"
                        / condition
                        / "consultancy/pair/puzzles.csv",
                        proposals,
                    ),
                    (
                        root
                        / "conditions"
                        / condition
                        / "judgements/judge/pair/puzzles.csv",
                        verdicts,
                    ),
                ]
                for path, rows in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(path, index=False)
            summarize_suite(root, manifest)
            rows = pd.read_csv(root / "results.csv").set_index("condition")
            single = rows.loc["single_consultancy"]
            self.assertEqual(single.positives, 2)
            self.assertEqual(single.negatives, 1)
            self.assertEqual(single.tp, 1)
            self.assertEqual(single.fp, 1)
            self.assertEqual(single.invalid_verdicts, 1)
            self.assertEqual(single.balanced_accuracy, 25)
            self.assertEqual(single.matched_count, 2)
            self.assertEqual(single.matched_accuracy, 50)
            self.assertEqual(rows.loc["multi_consultancy"].unjudged, 1)

    def test_catalog_rejects_duplicate_keys_without_breaking_yaml_merges(self):
        with self.assertRaisesRegex(ValueError, "Duplicate model catalog key 'judge'"):
            yaml.load(
                "roles:\n  judge: stronger\n  judge: weaker\n", Loader=CatalogLoader
            )
        data = yaml.load(
            "base: &base {temperature: 0.2}\nprofile: {<<: *base, temperature: 0.3}\n",
            Loader=CatalogLoader,
        )
        self.assertEqual(data["profile"]["temperature"], 0.3)

    def test_optional_values_have_stable_hash_inputs_after_csv_roundtrip(self):
        initial = pd.DataFrame([dict(cost=None, request_id=None, final_move="e4")])
        replay = pd.DataFrame([dict(cost="", request_id="", final_move="e4")])
        self.assertEqual(saved_rows(initial), saved_rows(replay))
        partial = saved_rows(
            pd.DataFrame(
                [
                    dict(complete="True", verdict_valid="True"),
                    dict(complete="False", verdict_valid=""),
                ]
            )
        )
        self.assertIs(partial[0]["complete"], True)
        self.assertIs(partial[0]["verdict_valid"], True)

    @patch(
        "builtins.input", side_effect=AssertionError("No interactive budget prompts")
    )
    async def test_all_conditions_share_openings_prefixes_and_route_judges(self, _):
        prompts = []
        openings = 0

        async def answer(model, messages, **kwargs):
            nonlocal openings
            system, user = messages[0]["content"], messages[1]["content"]
            prompts.append((model, system, user))
            self.assertNotIn("gpt-4", user)
            self.assertNotIn("gpt-3.5-turbo", user)
            if "judging a proposed solution" in system:
                self.assertIn("<transcript>", user)
                if "— critic:" in user:
                    self.assertIn("critic's objections", system)
                else:
                    self.assertIn("your conversation with the proposer", system)
                return [response("Yes" if "Proposed solution: e4" in user else "No")]
            if "Find the best move in this position:" in user:
                openings += 1
                return [
                    response("Opening argument " + ("e4" if openings == 1 else "d4"))
                ]
            return [response("A concise new argument or question.")]

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg = suite_config(folder, "[[gpt-4,gpt-3.5-turbo],[gpt-4,gpt-4]]")
            api = AsyncMock(side_effect=answer)
            result = await run_experiment(cfg, api)
            self.assertEqual(openings, 3)
            self.assertEqual(api.await_count, 48)
            self.assertEqual(len(result), 8)
            self.assertTrue(result.completed.eq(3).all())
            self.assertTrue(result.matched_count.eq(3).all())
            self.assertTrue(result.accuracy.eq(100).all())
            root = Path(cfg.exp_dir)
            records = pd.read_csv(root / "evaluations.csv", dtype={"puzzleId": str})
            self.assertTrue(
                records.groupby("puzzleId").proposal_id.nunique().eq(1).all()
            )
            self.assertEqual(pd.read_csv(root / "costs.csv").api_calls.sum(), 48)
            self.assertEqual(
                sum("judging a proposed solution" in p[1] for p in prompts), 24
            )
            # A weak questioner receives no strong auditor and the final judge
            # is the exact model assigned to that questioner's source.
            multi = root / "conditions/multi_consultancy/judgements/judge"
            for path in multi.glob("*/puzzles.csv"):
                frame = pd.read_csv(path)
                self.assertTrue(frame.model.eq(frame.questioner).all())
            single_paths = sorted(
                (root / "conditions/single_debate/debate").glob("*/puzzles.csv")
            )
            multi_paths = sorted(
                (root / "conditions/multi_debate/debate").glob("*/puzzles.csv")
            )
            for single_path, multi_path in zip(single_paths, multi_paths):
                single, multiple = pd.read_csv(single_path), pd.read_csv(multi_path)
                for left, right in zip(single.transcript, multiple.transcript):
                    self.assertEqual(json.loads(left), json.loads(right)[:2])
            await run_experiment(
                cfg,
                AsyncMock(side_effect=AssertionError("Resume must use saved results")),
            )
            cfg.max_tokens += 1
            with self.assertRaisesRegex(ValueError, "Suite inputs"):
                await run_experiment(cfg, api)

    async def test_failure_coverage_and_resume_without_new_proposals(self):
        failed = False
        openings = 0

        async def answer(model, messages, **kwargs):
            nonlocal failed, openings
            system, user = messages[0]["content"], messages[1]["content"]
            if "Find the best move in this position:" in user:
                openings += 1
                return [response("Opening e4")]
            if "You are the critic" in system and not failed:
                failed = True
                raise RuntimeError("transient critic outage")
            return [
                response(
                    "Yes" if "judging a proposed solution" in system else "Argument"
                )
            ]

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg = suite_config(folder)
            api = AsyncMock(side_effect=answer)
            with self.assertRaisesRegex(RuntimeError, "Rerun the same command"):
                await run_experiment(cfg, api)
            partial = pd.read_csv(Path(cfg.exp_dir) / "results.csv")
            self.assertTrue(partial.completed.eq(0).all())
            self.assertTrue(partial.planned.eq(3).all())
            self.assertEqual(openings, 3)
            result = await run_experiment(cfg, api)
            self.assertEqual(openings, 3)
            self.assertTrue(result.completed.eq(3).all())
            self.assertEqual(
                pd.read_csv(Path(cfg.exp_dir) / "costs.csv").api_calls.sum(), 30
            )

    async def test_invalid_openings_are_counted_without_futile_followups_or_judging(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg = suite_config(folder)
            api = AsyncMock(return_value=[response("e4", reason="length")])
            result = await run_experiment(cfg, api)
            self.assertEqual(api.await_count, 3)
            self.assertTrue(result.invalid_proposals.eq(3).all())
            self.assertTrue(result.completed.eq(0).all())
            self.assertTrue(result.matched_count.eq(0).all())
            self.assertTrue(result.accuracy.isna().all())

    async def test_single_condition_end_to_end_and_validation(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg = suite_config(folder, extra=["suite_conditions=[single_consultancy]"])

            async def answer(model, messages, **kwargs):
                return [
                    response(
                        "Yes"
                        if "judging a proposed solution" in messages[0]["content"]
                        else "Opening e4"
                    )
                ]

            api = AsyncMock(side_effect=answer)
            result = await run_experiment(cfg, api)
            self.assertEqual(api.await_count, 6)
            self.assertEqual(result.accuracy.iloc[0], 100)
            cfg.suite_conditions = ["multi_debate"]
            cfg.num_rounds = 1
            with self.assertRaisesRegex(ValueError, "num_rounds >= 2"):
                await run_experiment(cfg, api)

    async def test_only_final_public_argument_is_forwarded(self):
        prompts = []
        values = iter(
            [
                response("SCRATCH COMMENTARY", [tool()], "tool_calls"),
                response("Final public defense e4"),
                response("Critic argument"),
            ]
        )

        async def answer(model, messages, **kwargs):
            prompts.append(json.dumps(messages))
            return [next(values)]

        with tempfile.TemporaryDirectory() as folder:
            result = await run_interaction(
                AsyncMock(side_effect=answer), "gpt-4", puzzle(), folder, "debate", 1
            )
        self.assertNotIn("SCRATCH", result["model_response"])
        self.assertNotIn("SCRATCH", prompts[-1])
        self.assertIn("SCRATCH", result["tool_trace"])
        self.assertIn("Final public defense", prompts[-1])

    def test_opening_is_identical_across_modes_and_later_roles_are_not_solver_prompts(
        self,
    ):
        for tools in (True, False):
            left = interaction_messages("debate", "proposer", puzzle(), [], "", tools)
            right = interaction_messages(
                "consultancy", "proposer", puzzle(), [], "", tools
            )
            self.assertEqual(left, right)
            transcript = [
                dict(
                    round=1,
                    role="proposer",
                    model="SECRET_MODEL",
                    events=[dict(type="text", text="Opening e4")],
                )
            ]
            later = interaction_messages(
                "consultancy", "questioner", puzzle(), transcript, "e4", tools
            )
            self.assertNotIn("SECRET_MODEL", json.dumps(later))
            self.assertNotIn("find the best move", later[0]["content"].lower())
            if not tools:
                self.assertNotIn("validate_move_sequence", later[0]["content"])


if __name__ == "__main__":
    unittest.main()
