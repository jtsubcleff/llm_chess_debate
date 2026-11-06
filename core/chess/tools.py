"""Chess board utilities, with no engine or solution access."""

import chess

VALIDATE_MOVE_SEQUENCE = {
    "type": "function",
    "function": {
        "name": "validate_move_sequence",
        "description": "Validate one or more SAN or UCI moves from the ORIGINAL puzzle FEN. Each call starts afresh. Returns legal, fen (only if all moves are legal), and first_illegal_move.",
        "parameters": {
            "type": "object",
            "properties": {
                "moves": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            },
            "required": ["moves"],
            "additionalProperties": False,
        },
    },
}


def validate_move_sequence(fen: str, moves: list[str]) -> dict:
    if (
        not isinstance(moves, list)
        or not moves
        or not all(isinstance(m, str) for m in moves)
    ):
        raise ValueError("moves must be a nonempty list of SAN or UCI strings")
    board = chess.Board(fen)
    for token in moves:
        try:
            # parse_san also accepts some UCI strings; explicitly reject null moves.
            try:
                move = board.parse_san(token)
            except ValueError:
                move = board.parse_uci(token)
            if move not in board.legal_moves:
                raise ValueError("Illegal move")
            board.push(move)
        except ValueError:
            return dict(legal=False, fen=None, first_illegal_move=token)
    return dict(legal=True, fen=board.fen(), first_illegal_move=None)


CHESS_TOOLS = [VALIDATE_MOVE_SEQUENCE]
TOOL_COUNT_FIELDS = (
    "validate_move_sequence_calls",
    "validate_move_sequence_illegal_calls",
)
