"""Binary final judgement of a chess proposal using the shared chess tools."""

import json
import string
import unicodedata

from core.chess.protocols.direct_qa import run_chess_turns
from core.prompts import load_template, render


def judge_variables(puzzle, proposal):
    return dict(
        FEN=puzzle.FEN,
        PROPOSED_SOLUTION=proposal["final_move"],
        TRANSCRIPT=proposal["model_response"],
    )


def normalize_judgement(text):
    """Read the final Yes/No, allowing surrounding punctuation and Markdown."""

    def punctuation(char):
        return char in string.punctuation or unicodedata.category(char).startswith("P")

    tail = text.rstrip()
    while tail and (tail[-1].isspace() or punctuation(tail[-1])):
        tail = tail[:-1]
    answer = tail.split()[-1] if tail.split() else ""
    while answer and punctuation(answer[0]):
        answer = answer[1:]
    # Preserve malformed final tokens for diagnostics rather than guessing a verdict.
    return (
        answer
        if answer in ("Yes", "No")
        else (text.split()[-1] if text.split() else "")
    )


def score_judgement(text, proposer_correct):
    answer = normalize_judgement(text)
    return answer, answer in ("Yes", "No") and (answer == "Yes") == proposer_correct


async def run_judge(
    api,
    model,
    puzzle,
    proposal,
    cache_dir,
    max_tokens=8192,
    max_tool_rounds=10,
    temperature=0.2,
    prompt_template=None,
):
    if prompt_template is None:
        prompt_template = load_template("chess_judge")
    variables = judge_variables(puzzle, proposal)
    if "transcript_descriptions" in prompt_template:
        protocol = proposal.get("transcript_protocol", "direct_qa")
        variables["TRANSCRIPT_DESCRIPTION"] = prompt_template[
            "transcript_descriptions"
        ][protocol]
    messages = [
        dict(role=m["role"], content=render(m["content"], variables))
        for m in prompt_template["messages"]
    ]
    result = await run_chess_turns(
        api,
        model,
        puzzle,
        cache_dir,
        messages,
        lambda text: score_judgement(text, proposal["proposer_correct"]),
        "Tool budget exhausted. Return your final decision, ending with Yes or No.",
        max_tokens,
        max_tool_rounds,
        temperature,
        True,
    )
    result["judgement"] = result.pop("final_answer")
    responses = [e for e in json.loads(result["tool_trace"]) if "completion" in e]
    result["verdict_valid"] = bool(
        result["judgement"] in ("Yes", "No")
        and responses
        and responses[-1]["stop_reason"] == "STOP_SEQUENCE"
    )
    return result
