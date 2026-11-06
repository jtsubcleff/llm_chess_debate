"""Direct QA with bounded native tool use and resumable per-turn caching."""

import hashlib
import json
from pathlib import Path

import chess

from core.chess.public_text import public_turn
from core.chess.results import write_json
from core.chess.tools import CHESS_TOOLS, TOOL_COUNT_FIELDS, validate_move_sequence
from core.chess.word_limits import PUBLIC_WORD_LIMITS, limit_public_response
from core.llm_api.base_llm import LLMResponse, StopReason
from core.llm_api.errors import request_id_from_error
from core.llm_api.tool_calls import assistant_message
from core.prompts import load_template, render


def prompt_template_name(use_tools=True):
    return "chess_direct_qa" if use_tools else "chess_direct_qa_no_tools"


def score_response(puzzle, response):
    answer = response.split()[-1] if response.split() else ""
    # Models sometimes format an otherwise valid final SAN move as **Bxf6**
    # or `Bxf6`. Peel matching Markdown wrappers (including nested ones)
    # before scoring, while preserving SAN check/mate suffixes and punctuation.
    while answer:
        for marker in ("***", "___", "**", "__", "~~", "`", "*", "_"):
            if (
                len(answer) > 2 * len(marker)
                and answer.startswith(marker)
                and answer.endswith(marker)
            ):
                answer = answer[len(marker) : -len(marker)]
                break
        else:
            break
    if answer == puzzle.expected_move:
        return answer, True
    if answer.endswith("#"):
        board = chess.Board(puzzle.FEN)
        try:
            move = board.parse_san(answer)
            # Keep the final-answer SAN contract; parse_san also accepts UCI.
            if move in board.legal_moves and board.san(move) == answer:
                board.push(move)
                return answer, board.is_checkmate()
        except ValueError:
            pass
    return answer, False


async def run_direct_qa(
    api,
    model,
    puzzle,
    cache_dir,
    max_tokens=8192,
    max_tool_rounds=10,
    temperature=0.2,
    use_tools=True,
):
    messages = [
        dict(
            role=m["role"],
            content=render(
                m["content"],
                {
                    "FEN": puzzle.FEN,
                    "WORD_LIMIT": PUBLIC_WORD_LIMITS["opening"]["prompt"],
                },
            ),
        )
        for m in load_template(prompt_template_name(use_tools))["messages"]
    ]
    result = await run_chess_turns(
        api,
        model,
        puzzle,
        cache_dir,
        messages,
        lambda text: score_response(puzzle, text),
        "Tool budget exhausted. Return your final SAN move now.",
        max_tokens,
        max_tool_rounds,
        temperature,
        use_tools,
    )
    result["final_move"] = result.pop("final_answer")
    result["expected_move"] = puzzle.expected_move
    trace = json.loads(result["tool_trace"])
    responses = [e for e in trace if "completion" in e]
    _, public_text = public_turn(trace)
    if not responses or responses[-1].get("tool_calls"):
        public_text = ""
    # Preserve the original final token for scoring; expose only the bounded
    # final public solution to later judges, not tool-loop scratch commentary.
    result["model_response"], length_metrics = limit_public_response(
        public_text, PUBLIC_WORD_LIMITS["opening"], preserve_final_token=True
    )
    result.update(length_metrics)
    return result


async def run_chess_turns(
    api,
    model,
    puzzle,
    cache_dir,
    messages,
    scorer,
    final_instruction,
    max_tokens=8192,
    max_tool_rounds=10,
    temperature=0.2,
    use_tools=True,
):
    """Shared cached conversation and chess tool execution for final answers."""
    texts, trace = [], []
    tool_counts = dict.fromkeys(TOOL_COUNT_FIELDS, 0)
    input_tokens = output_tokens = 0
    cost = 0.0
    unknown_cost = False
    complete, error, answer, correct = False, "", "", False
    request_id = None
    tool_rounds = max_tool_rounds if use_tools else 0
    try:
        for turn in range(tool_rounds + 1):
            cache_key = hashlib.sha256(puzzle.puzzleId.encode()).hexdigest()
            cache = Path(cache_dir) / f"{cache_key}_{turn}.json"
            if cache.exists():
                data = json.loads(cache.read_text())
                # Enum names differ from provider finish reasons.
                data["stop_reason"] = {
                    "MAX_TOKENS": "max_tokens",
                    "STOP_SEQUENCE": "stop",
                    "TOOL_CALL": "tool_calls",
                    "REFUSAL": "refusal",
                }[data["stop_reason"]]
                response = LLMResponse(**data)
            else:
                kwargs = dict(temperature=temperature)
                if use_tools:
                    kwargs["tools"] = CHESS_TOOLS
                if use_tools and turn == tool_rounds:
                    # Give a final answer turn after all permitted tool rounds.
                    messages.append(
                        dict(
                            role="user",
                            content=final_instruction,
                        )
                    )
                    kwargs["tool_choice"] = "none"
                response = (
                    await api(model, messages, max_tokens=max_tokens, **kwargs)
                )[0]
                write_json(cache, response.to_dict(include_opaque=True))
            request_id = response.request_id
            input_tokens += response.input_tokens or 0
            output_tokens += response.output_tokens or 0
            unknown_cost |= response.cost is None
            cost += response.cost or 0
            texts.append(response.completion)
            trace.append(response.to_dict())
            if not response.tool_calls:
                answer, correct = scorer(response.completion)
                complete = True
                # A truncated or refused answer is never a successful solution.
                correct &= response.stop_reason == StopReason.STOP_SEQUENCE
                break
            if not use_tools:
                raise RuntimeError(
                    "Model requested tools while use_tools=false; no tools were executed"
                )
            if turn == tool_rounds:
                raise RuntimeError(
                    "Model requested tools after exhausting the tool budget"
                )
            messages.append(assistant_message(response))
            for call in response.tool_calls:
                try:
                    name = call["function"]["name"]
                    if name != "validate_move_sequence":
                        raise ValueError("Unknown tool")
                    tool_counts["validate_move_sequence_calls"] += 1
                    arguments = json.loads(call["function"]["arguments"])
                    result = validate_move_sequence(puzzle.FEN, **arguments)
                    if result["legal"] is False:
                        tool_counts["validate_move_sequence_illegal_calls"] += 1
                except (ValueError, TypeError) as exc:
                    result = dict(error=str(exc))
                trace.append(dict(tool_call=call, result=result))
                messages.append(
                    dict(
                        role="tool", tool_call_id=call["id"], content=json.dumps(result)
                    )
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        request_id = request_id_from_error(exc)
        trace.append(dict(error=error, request_id=request_id))
    return dict(
        **tool_counts,
        use_tools=use_tools,
        model_response="\n\n".join(texts),
        final_answer=answer,
        correct=bool(correct),
        complete=complete,
        error=error,
        request_id=request_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=None if unknown_cost else cost,
        tool_trace=json.dumps(trace),
        api_calls=sum("completion" in entry for entry in trace),
    )
