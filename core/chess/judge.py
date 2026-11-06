"""Judge saved chess transcripts: python -m core.chess.judge transcripts_dir=... models='[judge]'."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from core.chess.protocols.interaction import format_transcript
from core.chess.protocols.judge import run_judge
from core.chess.results import aggregate_results, write_csv, write_json
from core.chess.tools import CHESS_TOOLS
from core.llm_api.registry import resolve_model
from core.load.chess import ChessPuzzle
from core.prompts import load_template

LOGGER = logging.getLogger(__name__)


def load_transcripts(directory, allow_empty=False):
    if directory is None:
        raise ValueError("Specify transcripts_dir for chess judging")
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Transcript directory does not exist: {directory}")
    roots = [
        directory / name
        for name in ("direct_qa", "debate", "consultancy")
        if (directory / name).is_dir()
    ]
    if len(roots) > 1:
        raise ValueError(
            "Multiple chess protocols found; specify one protocol directory"
        )
    root = roots[0] if roots else directory
    paths = (
        [root / "puzzles.csv"]
        if (root / "puzzles.csv").exists()
        else sorted(root.glob("*/puzzles.csv"))
    )
    sources = []
    for path in paths:
        frame = pd.read_csv(
            path, keep_default_na=False, dtype={"puzzleId": str}
        ).astype(object)
        required = {
            "puzzleId",
            "FEN",
            "moves",
            "rating",
            "rating_bucket",
            "model",
            "complete",
            "correct",
            "final_move",
            "model_response",
        }
        if not required.issubset(frame.columns):
            raise ValueError(
                f"Missing transcript columns in {path}: {sorted(required - set(frame.columns))}"
            )
        if frame.puzzleId.duplicated().any():
            raise ValueError(f"Duplicate puzzle IDs in {path}")
        for _, row in frame.iterrows():
            if str(row.complete).lower() != "true":
                continue
            label = str(row.correct).lower()
            if label not in ("true", "false"):
                raise ValueError(f"Invalid correctness label in {path}")
            transcript = json.loads(row.transcript) if "transcript" in frame else None
            transcript_protocol = "direct_qa"
            if transcript is not None:
                is_consultancy = (
                    row.get("condition") in ("single_consultancy", "multi_consultancy")
                    or bool(row.get("questioner"))
                    or "consultancy" in (path.parent.name, path.parent.parent.name)
                    or any(turn["role"] == "questioner" for turn in transcript)
                )
                transcript_protocol = "consultancy" if is_consultancy else "debate"
            sources.append(
                dict(
                    source=str(path.resolve()),
                    puzzleId=row.puzzleId,
                    FEN=row.FEN,
                    moves=row.moves,
                    rating=int(row.rating),
                    rating_bucket=row.rating_bucket,
                    proposer=row.model,
                    proposer_correct=label == "true",
                    final_move=row.final_move,
                    transcript_protocol=transcript_protocol,
                    **{
                        key: row[key]
                        for key in (
                            "target_judge",
                            "condition",
                            "num_rounds",
                            "questioner",
                            "proposal_id",
                            "proposal_valid",
                            "interaction_valid",
                        )
                        if key in frame
                    },
                    # Re-render older interaction transcripts to exclude legacy tool events.
                    model_response=format_transcript(transcript)
                    if transcript is not None
                    else row.model_response,
                )
            )
    if not sources and not allow_empty:
        raise ValueError(f"No completed chess transcripts found in {directory}")
    return sources


async def run_judging(cfg, api, sources=None, pairs=None, prompt_template=None):
    # Import lazily: the experiment runner also dispatches to this module.
    from core.chess.run import model_directory

    if cfg.max_tokens <= 0 or cfg.max_tool_rounds < 0 or cfg.num_threads <= 0:
        raise ValueError(
            "Require max_tokens > 0, max_tool_rounds >= 0, num_threads > 0"
        )
    sources = load_transcripts(cfg.transcripts_dir) if sources is None else sources
    names = [cfg.models] if isinstance(cfg.models, str) else list(cfg.models)
    judges = list(dict.fromkeys(resolve_model(name)["id"] for name in names))
    if not judges:
        raise ValueError("Specify at least one judge model")
    specs = {name: resolve_model(name) for name in judges}
    if any(s["endpoint"] == "completions" for s in specs.values()):
        raise ValueError("Chess judging requires Chat, Responses, or Messages models")
    settings = dict(
        protocol="judge",
        transcript_content="final_public_argument_anonymous_v2",
        use_tools=True,
        max_tokens=cfg.max_tokens,
        max_tool_rounds=cfg.max_tool_rounds,
        temperature=cfg.temperature,
        prompt=prompt_template
        if prompt_template is not None
        else load_template("chess_judge"),
        tool=CHESS_TOOLS,
        sources=sources,
    )
    exp_dir = Path(cfg.exp_dir)
    manifest = exp_dir / "chess_judge_run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Judge settings or transcripts changed; choose a new exp_dir")
    frames, files, pending, profiles = {}, {}, [], []
    for model in judges:
        selected = [
            s
            for s in sources
            if (pairs is None or (s["proposer"], model) in pairs)
            and s.get("target_judge", model) == model
        ]
        # Keep different proposers' caches and records separate, even for the same puzzle.
        for source in dict.fromkeys(s["source"] for s in selected):
            proposals = [s for s in selected if s["source"] == source]
            folder = (
                exp_dir
                / "judge"
                / (
                    model_directory(model)
                    + "_"
                    + hashlib.sha256(source.encode()).hexdigest()[:12]
                )
            )
            key = (model, source)
            path = folder / "puzzles.csv"
            profile = folder / "model.json"
            if profile.exists() and json.loads(profile.read_text()) != specs[model]:
                raise ValueError(
                    f"Judge profile changed for {model}; choose a new exp_dir"
                )
            profiles.append((profile, specs[model]))
            puzzles = [
                ChessPuzzle(
                    s["puzzleId"], s["FEN"], s["moves"], s["rating"], s["rating_bucket"]
                )
                for s in proposals
            ]
            frame = (
                pd.read_csv(
                    path, keep_default_na=False, dtype={"puzzleId": str}
                ).astype(object)
                if path.exists()
                else pd.DataFrame(
                    [
                        dict(
                            **p.to_dict(),
                            model=model,
                            proposer=s["proposer"],
                            proposer_correct=s["proposer_correct"],
                            proposed_solution=s["final_move"],
                            source=source,
                            **{
                                key: s[key]
                                for key in (
                                    "condition",
                                    "num_rounds",
                                    "questioner",
                                    "proposal_id",
                                    "proposal_valid",
                                    "interaction_valid",
                                )
                                if key in s
                            },
                            complete=False,
                            correct=False,
                            judgement="",
                            model_response="",
                            error="",
                        )
                        for p, s in zip(puzzles, proposals)
                    ]
                ).astype(object)
            )
            if frame.puzzleId.tolist() != [p.puzzleId for p in puzzles]:
                raise ValueError("Saved judge sample differs; choose a new exp_dir")
            frames[key], files[key] = frame, path
            pending.extend(
                (key, i, p, s, folder / "cache")
                for i, (p, s) in enumerate(zip(puzzles, proposals))
                if str(frame.at[i, "complete"]).lower() != "true"
            )
    write_json(manifest, settings)
    for path, spec in profiles:
        write_json(path, spec)
    for key, frame in frames.items():
        write_csv(frame, files[key])
    aggregate_results(exp_dir, "judge")
    semaphore = asyncio.Semaphore(cfg.num_threads)

    async def evaluate(key, index, puzzle, proposal, cache):
        async with semaphore:
            result = await run_judge(
                api,
                key[0],
                puzzle,
                proposal,
                cache,
                cfg.max_tokens,
                cfg.max_tool_rounds,
                cfg.temperature,
                prompt_template=settings["prompt"],
            )
            frame = frames[key]
            for field, value in result.items():
                if field not in frame:
                    frame[field] = pd.Series([None] * len(frame), dtype=object)
                frame.at[index, field] = value
            write_csv(frame, files[key])
            aggregate_results(exp_dir, "judge")
            LOGGER.info(
                "%s puzzle %s: complete=%s correct=%s",
                key[0],
                puzzle.puzzleId,
                result["complete"],
                result["correct"],
            )
            return result

    results = await asyncio.gather(*(evaluate(*item) for item in pending))
    if any(not r["complete"] for r in results):
        raise RuntimeError(
            "Chess judgements failed; details saved in puzzles.csv. Rerun to resume."
        )
    return frames


@hydra.main(version_base=None, config_path="../config", config_name="chess_judge")
def main(cfg: DictConfig):
    from core.chess.run import async_main

    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
