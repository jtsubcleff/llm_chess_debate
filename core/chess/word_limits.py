"""Shared limits on public chess evidence, independent of reasoning/token budgets."""

PUBLIC_WORD_LIMITS = {
    "opening": {"prompt": 200, "maximum": 225},
    "debate": {"prompt": 100, "maximum": 125},
    "consultancy": {"prompt": 200, "maximum": 225},
    "questioner": {"prompt": 200, "maximum": 225},
}


def turn_word_limits(mode, role, opening=False):
    if opening:
        return PUBLIC_WORD_LIMITS["opening"]
    if role == "questioner":
        return PUBLIC_WORD_LIMITS["questioner"]
    return PUBLIC_WORD_LIMITS[mode]


def limit_public_response(text, limits, preserve_final_token=False):
    """Count whitespace-separated words; never manufacture or repair an answer.

    Openings reserve their last allowed word for the original final token,
    including when it is malformed or incorrect. Scoring uses the original
    response, so truncation cannot turn a bad final answer into a good one.
    """
    words = text.split()
    maximum = limits["maximum"]
    truncated = maximum is not None and len(words) > maximum
    if truncated:
        retained = (
            words[: maximum - 1] + words[-1:]
            if preserve_final_token
            else words[:maximum]
        )
        text = " ".join(retained)
    return text, dict(
        prompt_word_limit=limits["prompt"],
        public_word_limit=maximum,
        public_words_original=len(words),
        public_words=len(text.split()),
        public_truncated=truncated,
    )
