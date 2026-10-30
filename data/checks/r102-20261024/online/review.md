# R102 bounded online Wardens validation

Verdict: **PASS**

- Run: `validation/20261024-204102_toughness_interactive`
- Hard limit: 8 chat / 6 embedding requests
- Actual traffic: 3 chat / 0 embedding requests; all three requests returned HTTP 200; no retries
- First turn: two SQL calls followed by one no-tool Gemini final call
- Follow-up: zero SQL, zero embedding and zero Gemini calls; the existing structured evidence was formatted locally

## Evidence and answer checks

The second SQL result from the first turn contains the exact `Wardens of Ultramar` datasheet row, complete model
profile groups, publication and edition. A separate read-only database query confirmed the same values:

- Ancient Gadriel and Veteran Sergeant Metaurus: T4
- Gaius Silva, Aemelia Minervas, Dainal Kornelius and Lucia Vestha: T3
- Publication: *Space Marines*
- Edition: 11

The follow-up answer reproduces all six names and both Toughness values, with `[DB: datasheets]`,
`[DB: datasheets_models]` and `[DB: source]` attribution. It contains no model category or relationship absent from
the retrieved rows.

## Execution-path checks

The trace segment after the Wardens human message consists of one AI message. Its
`additional_kwargs.answer_format` is `structured_database_fields`; it has no tool calls and no usage metadata. No
new model call appears after the follow-up. The acceptance report likewise records zero follow-up tool calls. This
proves that the follow-up used the R102 deterministic final-answer path rather than Gemini free generation.

`turn-02/console.txt` contains the complete formatted answer and ends with `No tool was called for the last answer.`
There was no repeated SQL after the complete Wardens evidence had already been stored.

## Scope and attribution

This was the single authorized R102 scenario. It did not run Advance, the complete six-scenario Stage C suite,
RAGAS, or an index rebuild. It does not create a new RAGAS score or a current-system 6/6 claim. The saved Round 3
scores and historical six-case result remain attributable only to R94.

After the documentation update, `tests/test_docs.py` passed 12 tests, Ruff lint and format checks passed, and
`git diff --check` passed. SHA-256 comparison confirmed that all 133 files under the preserved R94, R96 and
R98–R101 result directories were unchanged.
