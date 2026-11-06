"""Sequential proposer/critic debate and proposer/questioner consultancy.

Each agent turn reuses Direct QA's native tool loop and caches. Only public
text crosses agent boundaries; tool traces and private reasoning stay local.
"""

import json
from pathlib import Path

from core.chess.protocols.direct_qa import run_chess_turns, score_response
from core.chess.public_text import public_turn
from core.chess.tools import TOOL_COUNT_FIELDS
from core.chess.word_limits import limit_public_response, turn_word_limits
from core.prompts import load_template, render


def turn_schedule(mode, num_rounds):
    if mode not in ("debate", "consultancy"):
        raise ValueError(f"Unknown interaction mode: {mode}")
    if (
        isinstance(num_rounds, bool)
        or not isinstance(num_rounds, int)
        or num_rounds < 1
    ):
        raise ValueError("num_rounds must be a positive integer")
    schedule = []
    for round_index in range(num_rounds):
        schedule.append((round_index + 1, "proposer"))
        if mode == "debate" or round_index < num_rounds - 1:
            schedule.append(
                (round_index + 1, "critic" if mode == "debate" else "questioner")
            )
    return schedule


def turn_cache(cache_dir, index, role):
    return Path(cache_dir) / f"{index:03d}_{role}"


def format_transcript(transcript):
    """Render text only, including when reading older transcripts with tool events."""
    sections = []
    for turn in transcript:
        events = []
        for event in turn["events"]:
            if event["type"] == "text":
                events.append(event["text"])
        sections.append(
            f"Round {turn['round']} — {turn['role']}:\n" + "\n".join(events)
        )
    return "\n\n".join(sections)


def score_proposal(puzzle, trace):
    """Score only the opening proposer's final public answer."""
    _, text = public_turn(trace)
    move, correct = score_response(puzzle, text)
    responses = [entry for entry in trace if "completion" in entry]
    normal_stop = responses and responses[-1].get("stop_reason") in (
        "STOP_SEQUENCE",
        "stop",
        "end_turn",
        "completed",
    )
    return move, bool(correct and normal_stop)


def interaction_messages(
    mode, role, puzzle, transcript, move, use_tools, instructions=None
):
    opening = not transcript
    limits = turn_word_limits(mode, role, opening)
    instructions = (
        instructions if instructions is not None else load_template("chess_interaction")
    )
    system = instructions["context"] + "\n" + instructions["privacy"]
    if use_tools:
        system += "\n" + instructions["tools"]
    key = "opening" if opening else role + "_" + mode
    variables = {"WORD_LIMIT": limits["prompt"]}
    system += "\n" + render(instructions[key], variables)
    messages = [dict(role="system", content=system), dict(role="user", content="")]
    if opening:
        messages[1][
            "content"
        ] = f"Find the best move in this position:\nFEN: {puzzle.FEN}\n" + render(
            instructions["opening_answer"], variables
        )
    else:
        request = "questions." if role == "questioner" else "argument."
        messages[1]["content"] = (
            f"Original puzzle FEN: {puzzle.FEN}\nLocked proposed move: {move}\n\n"
            f"Full public transcript:\n{format_transcript(transcript)}\n\n"
            f"Provide your next {request}"
        )
    return messages


METRIC_FIELDS = (*TOOL_COUNT_FIELDS, "input_tokens", "output_tokens", "api_calls")


def interaction_result(
    transcript,
    trace,
    turns,
    puzzle,
    move,
    correct,
    use_tools,
    num_rounds,
    expected_turns,
):
    """Build full or prefix results without duplicating generation or accounting."""
    opening_responses = [
        e
        for e in trace
        if e.get("round") == 1 and e.get("role") == "proposer" and "completion" in e
    ]
    stop = opening_responses[-1]["stop_reason"] if opening_responses else ""
    return dict(
        **{field: sum(t[field] for t in turns) for field in METRIC_FIELDS},
        cost=None
        if any(t["cost"] is None for t in turns)
        else sum(t["cost"] for t in turns),
        use_tools=use_tools,
        num_rounds=num_rounds,
        final_move=move,
        expected_move=puzzle.expected_move,
        correct=bool(correct),
        proposal_valid=bool(move) and stop == "STOP_SEQUENCE",
        interaction_valid=all(t["valid_response"] for t in turns),
        complete=len(turns) == expected_turns and bool(turns[-1]["complete"]),
        error=turns[-1]["error"],
        request_id=turns[-1]["request_id"],
        model_response=format_transcript(transcript),
        transcript=json.dumps(transcript),
        tool_trace=json.dumps(trace),
        turn_metrics=json.dumps(turns),
        public_truncated_turns=sum(t.get("public_truncated", False) for t in turns),
    )


def interaction_prefix(result, puzzle, mode, num_rounds):
    """A shorter condition is an exact prefix of an unannounced longer run."""
    count = len(turn_schedule(mode, num_rounds))
    transcript = json.loads(result["transcript"])[:count]
    for turn, (round_number, role) in zip(transcript, turn_schedule(mode, num_rounds)):
        if (turn["round"], turn["role"]) != (round_number, role):
            raise ValueError("Transcript does not match the requested interaction mode")
    turns = json.loads(result["turn_metrics"])[:count]
    slots = {(t["round"], t["role"]) for t in transcript}
    trace = [
        e for e in json.loads(result["tool_trace"]) if (e["round"], e["role"]) in slots
    ]
    return interaction_result(
        transcript,
        trace,
        turns,
        puzzle,
        result["final_move"],
        result["correct"],
        result["use_tools"],
        num_rounds,
        count,
    )


async def run_interaction(
    api,
    model,
    puzzle,
    cache_dir,
    mode,
    num_rounds,
    questioner_model=None,
    max_tokens=8192,
    max_tool_rounds=10,
    temperature=0.2,
    use_tools=True,
    opening_result=None,
    instructions=None,
):
    schedule = turn_schedule(mode, num_rounds)
    if mode == "consultancy" and not questioner_model:
        raise ValueError("Consultancy requires an explicit questioner model")
    transcript, trace, turns = [], [], []
    move, correct = "", False
    instructions = (
        instructions if instructions is not None else load_template("chess_interaction")
    )
    direct_opening = opening_result is not None and "transcript" not in opening_result
    if opening_result is not None and not direct_opening:
        opening_transcript = json.loads(opening_result["transcript"])
        if len(opening_transcript) != 1 or opening_transcript[0]["role"] != "proposer":
            raise ValueError("A reused opening must contain exactly one proposer turn")
    for index, (round_number, role) in enumerate(schedule):
        agent_model = questioner_model if role == "questioner" else model
        messages = interaction_messages(
            mode, role, puzzle, transcript, move, use_tools, instructions
        )
        if index == 0 and opening_result is not None:
            result = dict(opening_result)
            result["tool_trace"] = json.dumps(
                [
                    {k: v for k, v in e.items() if k not in ("round", "role", "model")}
                    for e in json.loads(result["tool_trace"])
                ]
            )
        else:
            result = await run_chess_turns(
                api,
                agent_model,
                puzzle,
                turn_cache(cache_dir, index, role),
                messages,
                lambda text: ("", False),
                "Tool budget exhausted. Finish your response now."
                + (" End with only your selected SAN move." if index == 0 else ""),
                max_tokens,
                max_tool_rounds,
                temperature,
                use_tools,
            )
        raw = json.loads(result["tool_trace"])
        _, public_text = public_turn(raw)
        limits = turn_word_limits(mode, role, opening=index == 0)
        public_text, length_metrics = limit_public_response(
            public_text, limits, preserve_final_token=index == 0
        )
        if index == 0 and direct_opening:
            public_text = result.get("model_response", public_text)
        responses = [e for e in raw if "completion" in e]
        has_final = bool(responses) and not responses[-1].get("tool_calls")
        # Tool-loop preambles and scratch commentary are local. Share one final
        # self-contained argument/question per agent turn, never tool chatter.
        events = (
            [dict(type="text", text=public_text)]
            if has_final and public_text.strip()
            else []
        )
        turns.append(
            dict(
                **{
                    field: result[field]
                    for field in (
                        *METRIC_FIELDS,
                        "cost",
                        "complete",
                        "error",
                        "request_id",
                    )
                },
                valid_response=has_final
                and bool(public_text.strip())
                and responses[-1]["stop_reason"] == "STOP_SEQUENCE",
                reused=index == 0 and opening_result is not None,
                stop_reason=responses[-1]["stop_reason"] if responses else "",
                **length_metrics,
            )
        )
        transcript.append(
            dict(
                round=round_number,
                role=role,
                model=agent_model,
                complete=result["complete"],
                events=events,
            )
        )
        trace.extend(
            dict(round=round_number, role=role, model=agent_model, **e) for e in raw
        )
        if index == 0 and result["complete"]:
            if direct_opening:
                move, correct = result["final_move"], result["correct"]
            else:
                move, correct = score_proposal(puzzle, raw)
        if not result["complete"]:
            break
    return interaction_result(
        transcript,
        trace,
        turns,
        puzzle,
        move,
        correct,
        use_tools,
        num_rounds,
        len(schedule),
    )
