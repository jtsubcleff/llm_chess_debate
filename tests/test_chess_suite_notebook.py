"""Check the notebook's research denominators without requiring a Jupyter kernel."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / "scripts/plot_chess_suite.ipynb"
LABELS = {
    "single_consultancy": "Single consultancy",
    "multi_consultancy": "Multi consultancy",
    "single_debate": "Single debate",
    "multi_debate": "Multi debate",
}


class SuiteNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.helpers = dict(pd=pd, np=np, json=json, Path=Path, CONDITION_LABELS=LABELS)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"notebook-cell-{index}", "exec")
            if "def bool_column" in source:
                exec(source, cls.helpers)

    def write_fixture(self, directory, pending=False):
        manifest = dict(
            conditions=["single_consultancy", "multi_consultancy"],
            models=[dict(proposer=dict(id="strong"), judge=dict(id="weak"))],
            puzzles=[dict(puzzleId=p, rating_bucket="Low") for p in ["001", "002"]],
        )
        directory.joinpath("chess_suite.json").write_text(json.dumps(manifest))
        rows = []
        for condition in manifest["conditions"]:
            for puzzle_id in ["001", "002"]:
                complete = not pending and not (
                    condition == "multi_consultancy" and puzzle_id == "002"
                )
                truth = puzzle_id == "001"
                rows.append(
                    dict(
                        condition=condition,
                        proposer="strong",
                        judge="weak",
                        puzzleId=puzzle_id,
                        complete=complete,
                        correct=complete and truth,
                        generated=not pending,
                        proposal_valid=not pending,
                        interaction_valid=not pending,
                        verdict_valid=complete,
                        proposer_correct=truth if not pending else "",
                        proposal_id=puzzle_id if not pending else "",
                        proposed_move="e4" if not pending else "",
                        judgement="Yes" if complete else "",
                    )
                )
        pd.DataFrame(rows).to_csv(directory / "evaluations.csv", index=False)
        return manifest

    def test_partial_results_use_common_cohort_and_correct_class_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.write_fixture(directory)
            manifest, data, costs = self.helpers["load_suite"](directory)
            self.assertEqual(set(data.puzzleId), {"001", "002"})
            self.assertTrue(costs.empty)
            summary = self.helpers["summarize"](data, manifest["conditions"])
            single = summary[summary.condition.eq("single_consultancy")].iloc[0]
            self.assertEqual(single.accuracy, 50)
            self.assertEqual(single.matched_accuracy, 100)
            self.assertEqual(single.matched_count, 1)
            self.assertEqual(single.false_accept_rate, 100)
            self.assertEqual(single.balanced_accuracy, 50)
            changes = self.helpers["paired_changes"](data)
            self.assertEqual(changes.puzzles.iloc[0], 1)
            self.assertEqual(changes.delta_pp.iloc[0], 0)

    def test_unjudged_and_missing_plan_rows_do_not_become_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest = self.write_fixture(directory, pending=True)
            path = directory / "evaluations.csv"
            pd.read_csv(path, dtype={"puzzleId": str}).iloc[:1].to_csv(
                path, index=False
            )
            _, data, _ = self.helpers["load_suite"](directory)
            summary = self.helpers["summarize"](data, manifest["conditions"])
            self.assertEqual(len(data), 4)
            self.assertTrue(summary.planned.eq(2).all())
            self.assertTrue(summary.accuracy.isna().all())
            self.assertTrue(summary.false_accept_rate.isna().all())

    def test_changed_proposals_are_rejected_as_unmatched(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.write_fixture(directory)
            path = directory / "evaluations.csv"
            data = pd.read_csv(path, dtype={"puzzleId": str, "proposal_id": str})
            data.loc[0, "proposal_id"] = "different-opening"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "disagree on proposal_id"):
                self.helpers["load_suite"](directory)

    def test_opening_accuracy_counts_invalid_attempts_without_condition_duplicates(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest = self.write_fixture(directory)
            # Sharing a solver across judges must not count its openings twice.
            manifest["models"].append(
                dict(proposer=dict(id="strong"), judge=dict(id="other"))
            )
            manifest["puzzles"].append(dict(puzzleId="003", rating_bucket="High"))
            directory.joinpath("chess_suite.json").write_text(json.dumps(manifest))
            checkpoint = directory / "_work/proposals/consultancy/strong__strong"
            checkpoint.mkdir(parents=True)
            pd.DataFrame(
                [
                    dict(
                        model="strong",
                        puzzleId="001",
                        complete=True,
                        correct=True,
                        proposal_valid=True,
                    ),
                    dict(
                        model="strong",
                        puzzleId="002",
                        complete=True,
                        correct=False,
                        proposal_valid=False,
                    ),
                ]
            ).to_csv(checkpoint / "puzzles.csv", index=False)
            summary = self.helpers["load_opening_accuracy"](directory)
            overall = summary[summary.difficulty.eq("All")].iloc[0]
            self.assertEqual(overall.planned, 3)
            self.assertEqual(overall.finished, 2)
            self.assertEqual(overall.correct, 1)
            self.assertEqual(overall.invalid, 1)
            self.assertEqual(overall.accuracy, 50)
            high = summary[summary.difficulty.eq("High")].iloc[0]
            self.assertEqual(high.planned, 1)
            self.assertEqual(high.finished, 0)
            self.assertTrue(pd.isna(high.accuracy))

    def test_opening_accuracy_before_checkpoints_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.write_fixture(directory, pending=True)
            summary = self.helpers["load_opening_accuracy"](directory)
            overall = summary[summary.difficulty.eq("All")].iloc[0]
            self.assertEqual(overall.planned, 2)
            self.assertEqual(overall.finished, 0)
            self.assertTrue(pd.isna(overall.accuracy))

    def test_comparison_witnesses_filter_answers_cohort_and_model_pair(self):
        rows = []
        for puzzle_id, answers, truth, matched, valid in [
            ("001", ("Yes", "No"), True, True, True),
            ("002", ("No", "Yes"), True, True, True),
            ("003", ("Yes", "Yes"), True, True, True),
            ("004", ("Yes", "No"), True, False, True),
            ("005", ("", "No"), True, True, False),
        ]:
            for condition, answer in zip(
                ["multi_debate", "multi_consultancy"], answers
            ):
                rows.append(
                    dict(
                        proposer="solver",
                        judge="judge",
                        puzzleId=puzzle_id,
                        condition=condition,
                        judgement=answer,
                        matched=matched,
                        complete=True,
                        verdict_valid=valid,
                        correct=(answer == "Yes") == truth,
                    )
                )
        # A second pair must not leak into the selected pair's dropdown.
        rows.extend([dict(row, proposer="other", puzzleId="999") for row in rows[:2]])
        data = pd.DataFrame(rows)
        witnesses = self.helpers["comparison_witnesses"](
            data, "solver", "judge", "multi_debate", "multi_consultancy"
        )
        self.assertEqual(witnesses.puzzleId.tolist(), ["001", "002"])
        self.assertEqual(witnesses.change.tolist(), ["Improved", "Worsened"])
        self.assertEqual(witnesses.baseline_answer.tolist(), ["No", "Yes"])
        self.assertEqual(witnesses.treatment_answer.tolist(), ["Yes", "No"])
        empty = self.helpers["comparison_witnesses"](
            data, "solver", "judge", "single_debate", "single_consultancy"
        )
        self.assertTrue(empty.empty)


if __name__ == "__main__":
    unittest.main()
