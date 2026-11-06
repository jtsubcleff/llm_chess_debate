# Chess experiments

This repository contains experiments testing LLM chess puzzle solving abilities, in various evaluations protocols. Puzzles are sourced from [Lichess Database](https://database.lichess.org/#puzzles)

## Setup

Use Python 3.11 and install the repository's `requirements.txt`. API credentials
use the existing `SECRETS` setup described in [README.md](README.md). Model names,
roles, and API IDs resolve through [core/config/models.yaml](core/config/models.yaml).
Use modern Chat Completions, Responses, or Claude Messages profiles supporting
function tools. Historical Completions profiles are rejected.

The experimental dataset should already be present at `chess_data/lichess_puzzles.csv`. To load more puzzles that adhere to the format required by this repository, use `scripts/prepare_lichess_puzzles.py`

## Run Direct QA

Direct QA asks a model for the best first move from a FEN. It can reason and
call `validate_move_sequence` to check candidate lines with python-chess. No
engine evaluation or solution is exposed to the model.

From the repository root with the virtual environment activated:

```bash
# One model: 30 puzzles total, ten per bucket
python -m core.chess.run exp_dir=./exp/chess_direct_qa models='[judge]' limit=30

# Several models on the same 300 puzzles, 100 per bucket
python -m core.chess.run exp_dir=./exp/chess_comparison \
  models='[openai_small,anthropic_small,openrouter_gpt_oss_20b]' limit=300 seed=42

# Disable tools: one model call per puzzle, using a prompt without tool references
python -m core.chess.run exp_dir=./exp/chess_no_tools models='[judge]' \
  limit=30 use_tools=false

# Adjust concurrency and the per-puzzle tool/per-call token budgets
python -m core.chess.run exp_dir=./exp/chess_custom models='[judge]' \
  limit=30 num_threads=5 max_tokens=8192 max_tool_rounds=10 temperature=0.2
```

`limit` is the **total per model**, must be positive and divisible by three,
and fails before any API calls if a bucket has insufficient puzzles. Selection
is seeded random sampling without replacement; all models receive the same
sample. Default buckets are Low `<1300`, Medium `1300–2300` inclusive, and
High `>2300`.


The default input is `chess_data/lichess_puzzles.csv`; override `dataset_path`
for another prepared CSV. Required columns are `PuzzleId,FEN,Moves,Rating`.
**FEN must already be the solver's position**, and `Moves` must start with the
solution in UCI. The repository's preparation script has already played and
removed the opponent's setup move. Do not pass a raw Lichess database directly.
The loader validates positions and solution sequences.

## Matched end-to-end research suite

Run all four research conditions on the same puzzles and the same opening proposals:

```bash
python -m core.chess.run protocol=research_suite \
  models='[[openrouter_gpt_oss_120b,openrouter_gpt_oss_20b]]' \
  num_rounds=3 limit=30 seed=42 exp_dir=./exp/chess_research
```

Every entry is an explicit `[proposer, judge]` pair. The critic uses the proposer's
exact profile; in multi-round consultancy the questioner uses the final judge's
exact profile in a separate conversation. This command runs generation, judging,
and scoring. To run just one condition end-to-end, add, for example,
`suite_conditions='[single_debate]'`. Available conditions are:

| Condition | Transcript assessed by final judge |
| --- | --- |
| `single_consultancy` | One proposal and its supporting argument |
| `multi_consultancy` | `num_rounds` proposer responses; `num_rounds - 1` questions |
| `single_debate` | One proposal followed by one critic audit |
| `multi_debate` | `num_rounds` proposer arguments and `num_rounds` critic audits |

Multi-round conditions require `num_rounds >= 2`. One mode-neutral opening is
generated per proposer/puzzle and reused across judges and conditions. Shorter
conditions are exact prefixes of longer interactions; agents are not told the
future round count. Final judging is a fresh call for each condition. This removes
variation in the proposed move and first argument as a confound.

Shared evidence contains only each agent turn's final public response. Tool-loop
commentary, calls/results, private reasoning, and participant model names are
excluded. Audit traces and model identities remain available in saved records.
The opening prompt is identical in all conditions. Role prompts ask for concise
new evidence; public responses use the word limits below, with truncation recorded
in per-turn metrics. The final judge receives
only the original FEN, locked move, and public transcript, not the reference move
or correctness label.

Outputs under the suite directory:

- `chess_suite.json`: exact sample, resolved profiles, prompts, settings, tool
  definitions, sharing policy, source-code hashes, and Python/library versions. Relevant changes require
  a new `exp_dir`. Prompts are frozen in memory for the duration of the run.
- `_work/`: reusable proposal and longest-interaction checkpoints/caches.
- `conditions/<condition>/`: public transcripts, full audit traces, and separate
  final judgements. Per-turn metrics mark reused openings.
- `evaluations.csv`: one row per planned condition/proposer/judge/puzzle, including
  missing judgements and a shared `proposal_id` for matching.
- `results.csv`: per-condition/model-pair coverage, accuracy, matched-cohort
  accuracy, class counts, TP/TN/FP/FN, invalid outputs, and balanced accuracy.
  Percentages use 0–100. Balanced accuracy is undefined when either class is absent.
- `costs.csv`: recorded usage/cost from each unique functional response cache,
  counted once despite proposal/prefix reuse; unknown-cost calls are explicit.

Rerun the same command to resume without regenerating successful calls. A transient
failure leaves incomplete work visible in the planned denominator. Matched accuracy
uses only puzzle IDs judged in every selected condition for that model pair.
Refused, truncated, or missing opening answers are counted as invalid proposals;
they do not trigger follow-ups or misleading final-judge labels. Malformed/truncated
final judgements still count as incorrect completed judge evaluations. Truncated
intermediate arguments retain the existing completion behavior and are flagged
in interaction validity metrics. Per-condition usage includes its logical shared
opening; use suite `costs.csv` for total recorded expenditure. Provider retries
without returned usage may incur additional unrecorded costs.

The original independent protocol commands below remain available. Use the suite
for matched comparisons. See [the design review](reports/chess_experiment_review.md)
for the changes, validation, and interpretation limits.

## Sequential debate and consultancy

Run interactions separately from final judging:

```bash
# Each model plays both proposer and critic. Three turns each; critic ends.
python -m core.chess.run protocol=debate num_rounds=3 \
  models='[openai_small,anthropic_small]' limit=30 exp_dir=./exp/chess_debate

# Explicit [proposer, questioner] pairs. Three proposer turns and two questions.
python -m core.chess.run protocol=consultancy num_rounds=3 \
  models='[[openai_small,anthropic_small],[openai_small,openai_small]]' \
  limit=30 exp_dir=./exp/chess_consultancy

# Make a final decision in a new run, using only the saved public transcript.
python -m core.chess.judge transcripts_dir=./exp/chess_consultancy \
  models='[anthropic_small]' exp_dir=./exp/chess_consultancy_judged

# Preview/reapply proposal scoring without model requests.
python -m core.chess.rescore ./exp/chess_debate --protocol debate
python -m core.chess.rescore ./exp/chess_debate --protocol debate --write
```

`num_rounds` must be a positive integer. In debate, each round is a proposer
argument followed by a critic argument, using the same model profile in separate
agent turns. The opening proposer finds and defends its best move; the critic's
task is to help a judge inspect that solution for errors in calculation,
reasoning, or move selection. Later turns receive the complete public transcript
and address the preceding argument. There is no cross-examiner. The critic has
the last word, for `2 * num_rounds` total turns.

In consultancy, the opening proposer solves the puzzle as in Direct QA. The
questioning judge seeks truth by asking about the proposed move, then the
proposer answers. The proposer has the last word: `num_rounds` proposer turns
and `num_rounds - 1` questioner turns. `num_rounds=1` produces only the opening
proposal. Questioner selection requires explicit pairs, including when both
agents use the same model. Different pairs have separate caches and result rows;
the proposer opening is generated independently for each pair in these standalone
commands (the research suite shares it). Questioning does
not make a final Yes/No decision; final judging uses the existing separate command.

The proposed move is locked when the opening proposer's tool-assisted turn
finishes. `final_move` and `correct` always describe that opening answer; later
changes of opinion cannot overwrite it. No agent receives the reference move or
correctness label. Final judging measures the judge's decision about that locked
move, separately from proposal accuracy.

Both modes rely on native provider reasoning for private thinking. Visible
final responses are public arguments or questions shared with other participants and
the final judge; the prompts do not ask models to generate thinking tags.
Structured tool calls and tool results stay private to the model's own tool
conversation. Agents should describe useful findings in their public arguments
when another participant needs them. As a defensive fallback, the transcript
filter still excludes any `<thinking>` content emitted by a model, including
unclosed tags spanning tool responses.

Public word limits are shared across the suite and standalone chess protocols:

| Response | Prompt instruction | Enforced maximum |
| --- | ---: | ---: |
| Initial solve in any format, including Direct QA | 200 words | 225 words |
| Later debate proposer arguments and every critic argument | 100 words | 125 words |
| Later consultancy proposer responses | 200 words | 225 words |
| Consultancy questioner | 200 words | 225 words |

Limits count whitespace-separated words in the final public response, excluding
private thinking and tool-loop commentary. For an overlong opening, retain the
first 224 words and reappend the original final token as word 225. This preserves
the selected move without changing the original response's correctness or fixing
malformed answers. Later arguments are cut at their hard maximum. There is no
extra truncation marker consuming words. Shorter responses retain their formatting.

The policy is set in `core/chess/word_limits.py` and pinned in run manifests.
Prompts request concision and include the corresponding suggested word limits. `turn_metrics` records
`prompt_word_limit`, `public_word_limit`, `public_words_original`, `public_words`,
and `public_truncated`; `public_truncated_turns` summarizes each interaction.
Direct QA records the same length fields directly in its result row and shares
only its bounded final public solution with later judges. Full original responses
remain in audit traces and functional caches. Intentional word truncation does
not make a response invalid; provider token truncation retains its existing
invalidity/scoring rules. Existing run directories must not be reused after this
policy change. The word caps affect public evidence, not the generation token
budget or already incurred API usage.

Each agent turn has its own `max_tool_rounds`, with `max_tokens` per API call.
Both agents receive the move validator unless `use_tools=false`.
Successful calls are cached by puzzle and agent turn. A mid-interaction API/tool-loop failure saves a partial transcript,
keeps the opening move, and leaves the puzzle incomplete; rerunning the same
command replays cached calls and resumes. As in Direct QA, a non-tool final
response that is truncated or refused counts as a completed turn. An opening
truncation/refusal is scored incorrect. Completed interactions alone enter the
aggregate accuracy denominator.

Outputs live under `exp_dir/debate/<model>/puzzles.csv` or
`exp_dir/consultancy/<proposer>__<questioner>/puzzles.csv`. They retain Direct QA's
usage, cost, request ID, tool counts, error, move, and correctness fields, summed
over all agent turns. `transcript` is structured public JSON; `model_response` is
its readable, role-labelled text-only form. `tool_trace` retains the
full audit history, including tagged private text/tools, with role/round/model
annotations and the existing opaque-payload redaction. Functional caches retain
full provider replay state. Private audit traces are excluded when these outputs
are loaded for final judging. Consultancy summaries group by proposer/questioner
pair; debate summaries group by proposer model. Changing rounds, prompts, tools,
generation settings, or model profiles requires a new experiment directory.
New transcripts omit model names from the rendered evidence and omit intermediate
tool-loop commentary; the raw audit trace retains it. The current prompt builder
uses explicit context/tool/role sections instead of splitting Direct QA prose.

The protocol is in `core/chess/protocols/interaction.py`, and shared private-text
filtering is in `core/chess/public_text.py`; experiment orchestration
are in `core/chess/interaction.py`; role prompts are in
`core/config/prompts/chess_interaction.yaml`. Opening-move scoring reuses Direct
QA's SAN rules. The existing judge and rescoring commands consume saved outputs
without rerunning the interaction.

## Judge saved proposals or run proposer/judge pairs

The chess judge makes a binary final judgement about a proposed first move.
It receives puzzle instructions, the solver-position FEN, the proposed SAN move,
and the proposer's response text (or the text-only public interaction). Tool
calls, results, and audit traces are never included. It can challenge the argument
and explore alternative lines with the move validator. The reference solution and
saved correctness label are never supplied to the judge. The prompt lives in
`core/config/prompts/chess_judge.yaml`.

```bash
# Judge completed transcripts from an existing Direct QA experiment
python -m core.chess.judge transcripts_dir=./exp/chess_direct_qa \
  exp_dir=./exp/chess_judged models='[judge]'

# Equivalent entry through the chess experiment runner
python -m core.chess.run protocol=judge \
  transcripts_dir=./exp/chess_direct_qa exp_dir=./exp/chess_judged models='[judge]'

# End to end: each entry is [proposer, judge]
python -m core.chess.run protocol=proposer_judge exp_dir=./exp/chess_pairs \
  models='[[openai_small,anthropic_small],[anthropic_small,openai_small]]' \
  limit=30 seed=42
```

`transcripts_dir` accepts an experiment root, its `direct_qa` directory, or one
model's directory containing `puzzles.csv`. All completed rows in that directory
are judged; `limit` and `dataset_path` do not select or resample saved transcripts.
Incomplete proposer rows are skipped, and an empty selection fails before requests.
Judge any number of models by passing a flat `models` list in standalone mode.

The judge is prompted to finish with `Yes` or `No`. Scoring tolerates surrounding
punctuation and Markdown, including `No.`, `**Yes**`, and `No**`, and stores the
normalized verdict. A verdict followed by further prose is still invalid. A judge is correct
when `Yes` matches a true proposer `correct` label, or `No` matches a false label.
These labels use the existing Direct QA scoring contract, including alternative
mates in one. If scoring rules changed, rescore the source experiment first.
Malformed, truncated, or refused judge answers count as incorrect completed
judgements. API/tool-loop failures remain incomplete for retry.

Standalone judgements are saved under
`exp_dir/judge/<judge-and-source-id>/puzzles.csv`, including `proposer`,
`proposed_solution`, `proposer_correct`, `judgement`, judge `model_response`,
`correct`, tool traces, usage, and costs. `exp_dir/results.csv` groups all judged
sources by judge model; `accuracy` is the percentage of correct decisions (0–100).
The existing `Low`, `Medium`, `High`, and `all` columns remain fractions (0–1).
Only completed judgements enter these denominators. `pair_results.csv` additionally
separates proposer/judge pairs and interaction conditions when that metadata is
available. Research-suite summaries contain class and matched-cohort metrics.

Paired runs solve each distinct proposer once on the shared balanced sample and
judge only the requested pairs. Proposer results stay in the usual locations;
judge outputs and summaries live under `exp_dir/judgements/`. Both stages use
`max_tokens`, `max_tool_rounds`, and `temperature`. `use_tools=false` disables
proposer tools only; the judge always receives the move validator. Each stage
starts automatically after validation, with proposals checkpointed before judging.

Both stages checkpoint and resume with the same command. Judge manifests pin
the exact input transcripts, prompt, tools, and generation settings; model
profiles are pinned separately. Changing source responses or labels requires a
new judge output directory. Source CSVs are read without modification.
The text-only sharing policy is pinned in new judge and interaction manifests;
use a new `exp_dir` for runs created before this policy. Older interaction
transcripts are re-rendered without their tool events when loaded for judging.

## Tools and scoring

Example tool arguments: `{"moves": ["Nf3", "d5"]}` or
`{"moves": ["g1f3", "d7d5"]}`. SAN and UCI may be mixed. Every invocation
starts from the original puzzle FEN, independently of earlier calls and other
puzzles. Null moves are illegal. Results have `legal` (boolean), `fen` (new FEN
if fully legal, otherwise null), and `first_illegal_move` (null if legal).
Malformed arguments return an error to the model so it can correct them.

The model's final whitespace-separated token must exactly match canonical SAN
for the first solution move, including check/mate/promotion notation. As an
exception, a SAN move ending in `#` is also correct if python-chess confirms
that it is legal from the original puzzle FEN and immediately delivers checkmate,
even when it differs from the listed solution. This allows alternative mates in one.
Matching Markdown wrappers around the final token (such as `**Bxf6**` or
inline backticks) are removed before scoring, including nested wrappers.
Extra punctuation or a UCI final answer is incorrect. Truncated and refused
responses count as incorrect. The prompt is in
[core/config/prompts/chess_direct_qa.yaml](core/config/prompts/chess_direct_qa.yaml).

`use_tools` defaults to `true`. With `use_tools=false`, the runner uses
[the no-tools prompt](core/config/prompts/chess_direct_qa_no_tools.yaml), sends
no tool definitions or tool-choice parameters, and makes one API call per puzzle.
`max_tool_rounds` is ignored in this mode; `max_tokens` is the output budget for
the single call. Any unexpected structured tool request is recorded as an error
and is never executed. Use a separate `exp_dir` when switching modes.

With tools enabled, there are at most `max_tool_rounds` tool rounds, followed by one final API call
with tools disabled if needed. A round can contain several independent tool
calls. `max_tokens` applies to each API call (including provider reasoning tokens
where applicable), rather than the whole puzzle. Provider adapters retain
native tool/reasoning history and their existing rate limits, retries, and cost
handling. Actual input/output usage is summed over all calls for each puzzle.

## Results and resuming

Each model has `exp_dir/direct_qa/<safe-model-name>_<hash>/puzzles.csv` with:

- `puzzleId`, `rating`, `rating_bucket`, `FEN`, `moves`, and `model`;
- `model_response`, `final_move`, `expected_move`, `correct`, `complete`, and `error`;
- `request_id`: the last API turn's provider request ID, or the terminal API
  error's request ID when available. Blank means the provider/old cache did not
  supply one; connection failures and timeouts may have no ID. Each response in
  `tool_trace`, turn caches, and prompt history also includes its own
  `request_id`. Retried API errors log their IDs when available. IDs are provider
  correlation IDs, not local prompt-history filenames. Old caches remain readable
  but cannot recover IDs that were not originally saved;
- `use_tools`, `input_tokens`, `output_tokens`, `cost` (USD; blank if unknown), `api_calls`,
  and `tool_trace` (JSON containing responses, tool calls, and results);
- `validate_move_sequence_calls` and
  `validate_move_sequence_illegal_calls` for each puzzle solve.

Tool counts include each dispatched call, including malformed arguments. The
illegal-sequence count includes only validations returning `legal=false`; argument
errors are not classified as illegal sequences. Counts are reconstructed from
cached turns when resuming, so replay does not double-count. All counts are zero
with `use_tools=false`.

Opaque provider reasoning payloads (`signature`, `encrypted_content`, and
encrypted/redacted reasoning data blocks) are replaced with a short marker in
CSV tool traces and prompt-history logs. Readable response text, tool arguments,
results, usage, and request IDs remain available. Full opaque payloads are kept
only in functional turn caches and live requests, where providers may require
them to continue tool conversations or resume an interrupted run. Existing files
are not rewritten by this logging change.

Rows are checkpointed atomically after each completed or failed puzzle, and
successful API turns are cached under the model's `cache/` directory. Run the
same command to resume: completed puzzles are skipped, and cached turns are
reused without billing again. API failures stay incomplete and can be retried;
an incorrect answer is a completed evaluation.

`exp_dir/results.csv` updates after every puzzle. One row per model contains
success fractions in `Low`, `Medium`, `High`, and `all`, corresponding counts,
and planned/completed totals. Only completed evaluations enter the denominator;
partial-run buckets may be unbalanced until all planned puzzles finish.

`chess_run.json` pins the sample, prompt, tool schema, and generation settings;
per-model `model.json` pins model profiles. Changed settings require a new
`exp_dir`. Additional models can be added to the same experiment. Avoid running
two processes writing the same experiment directory concurrently.

## Rescore saved results

To apply updated scoring to saved responses without API calls, preview with:

```bash
python -m core.chess.rescore exp/chess_comparison
python -m core.chess.rescore exp/chess_comparison --write

# Apply judge scoring updates to saved judgements, with backups
python -m core.chess.rescore exp/chess_judged_haiku --protocol judge --write
```

## Timeouts and Reasoning Effort


OpenRouter calls default to a 600-second timeout per attempt (OpenAI calls
retain 120 seconds). Override it in a chess command with
`openrouter_timeout=300`, for example. The timeout applies to both the local
deadline and SDK request, and must be positive. Existing retry counts still
apply, so repeated failures can take longer to finish. This setting can change
when resuming the same `exp_dir` without invalidating caches.

For OpenRouter models, optionally add `reasoning_effort` to the model entry in
`core/config/models.yaml` (or the catalog selected by `LLM_MODELS_CONFIG`):

```yaml
  openrouter_gpt_oss_20b:
    id: openai/gpt-oss-20b
    profile: openrouter_chat
    reasoning_effort: low
    # Keep the entry's existing prices and other metadata.
```

GPT-OSS supports `low`, `medium`, and `high`. The gateway also accepts `none`,
`minimal`, `xhigh`, and `max` for models that support those levels; model/provider
support varies. Omitting the field or setting it to `null` sends no effort
override. The adapter sends this as OpenRouter's `reasoning.effort`, preserving
configured routing and usage reporting. This convenience field currently applies
only to OpenRouter; other providers retain their existing `request_defaults`.
Changing effort changes the pinned model profile: use a new `exp_dir` so the
runner does not mix previous cached answers with a different reasoning setting.
See [OpenRouter reasoning controls](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

## Visualize

For a full research suite, open [scripts/plot_chess_suite.ipynb](scripts/plot_chess_suite.ipynb),
set `EXP_DIR` to the directory containing `chess_suite.json`, and run all cells.
It compares the four conditions separately for each proposer/judge pair, including
matched-cohort accuracy, coverage, paired improvements/regressions, false accepts
and rejects, difficulty buckets, and recorded costs without double-counting reused
work. Its puzzle viewer shows the board, public interaction, and final judgement
for each condition, preferring the exact transcript pinned for judging. Partial
runs and absent classes display unavailable rates as N/A. Set `SAVE_FIGURES=True`
to export PNGs under `<EXP_DIR>/figures/suite/`; loading and analysis never make
model requests or modify result files.

For judge evaluations, open [scripts/plot_chess_judge.ipynb](scripts/plot_chess_judge.ipynb),
set `EXP_DIR`, and run all cells. It accepts standalone judge or end-to-end
experiment directories and creates a chart of false-positive, false-negative,
true-positive, and true-negative rates per proposer/judge pairing. Positives
mean accepting the proposal; rates use their standard ground-truth class
denominators. The notebook shows class counts and excluded invalid/incomplete
decisions, and saves `<EXP_DIR>/figures/judge_decision_rates.png`.
The initial table and chart labels also show overall correctness as correct
judgements divided by puzzles seen, counting invalid and failed attempts as
incorrect and excluding untouched rows. A second chart uses the same accuracy
definition for Low/Medium/High difficulty buckets, with correct/seen counts on
each bar, and saves `<EXP_DIR>/figures/judge_accuracy_by_difficulty.png`.
Its puzzle-ID widget also displays the board and the full proposer output followed
by the judge output for every matching pairing, using the pinned proposer
transcript when available.

Open [scripts/plot_chess.ipynb](scripts/plot_chess.ipynb), set `EXP_DIR`, and run
all cells. It uses the same pandas/matplotlib/seaborn stack as `plot_minimal.ipynb`
and saves PNGs to `<EXP_DIR>/figures/`:

1. Overall correctness by model.
2. Correctness by model and rating bucket.
3. Success versus puzzle rating, with one ordinary least squares line per model.


Regression fits per-puzzle binary correctness; points show observed success in
100-rating intervals. Linear predictions can fall outside 0–100%.

## Extending chess protocols

`core/load/chess.py` owns puzzle loading and balanced sampling;
`core/chess/tools.py` owns board validation; `core/chess/results.py` owns CSV
storage and summaries. Direct QA lives in `core/chess/protocols/direct_qa.py`; binary judging lives in
`core/chess/protocols/judge.py`. Both reuse `run_chess_turns` for native tools,
turn caches, traces, usage, and error handling. `core/chess/judge.py` loads saved
transcripts and checkpoints judge evaluations; `core/chess/run.py` dispatches
standalone protocols and orchestrates proposer/judge pairs. Register future
solving protocols in `core/chess/run.py:PROTOCOLS` and add their prompt templates. The runner passes the selected protocol to result discovery.
