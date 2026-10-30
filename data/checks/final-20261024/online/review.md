# Final current-system acceptance and Round 4 evaluation

Verdict: **COMPLETE**

The evaluation is tied to commit `2f3d3529533528501a35a3950128bf38d4a27f40`, `gemini-3.8-flash` with effective temperature omitted (`null`), `gemini-embedding-001`, `TOP_K = 8`, and the hashes in `freeze.json` and `data/eval/runs/round4-final/inputs.json`.

## Stage C

All six current-system scenarios passed semantic and functional review. The suite used 17 chat and 6 embedding requests, all HTTP 200 with no retry:

- Toughness plus Wardens follow-up: complete first-turn structured evidence, zero follow-up SQL/Gemini, correct deterministic six-profile T4/T3 answer and source metadata.
- Standalone Wardens: all six profiles, publication and edition correct.
- Stratagem: regular and additional CP gain, payment, reuse, card fields, and Battle-shock restrictions present.
- Scoring: Only War scoring and all ordered winner branches kept separate.
- Grenades: one complete dual-source search followed by the existing no-tool final route; source-version differences preserved.
- Unit rules plus Advance: database prohibition and Core Rules uncertainty kept separate under the four-state evidence contract.

The detailed case review is in `stage-c-review.md`.

## Round 4 RAGAS

All ten saved answers and all 50 metric records are complete. The same ten questions and five metrics as Round 3 were used, but the current implementation, prompts, retrieval behavior, answer routing, evaluator sampling configuration, and judge executions differ. The comparison is observational, not a single-variable causal experiment.

| Metric | R94 Round 3 | Current Round 4 | Valid / expected |
|---|---:|---:|---:|
| Faithfulness | 0.861 | 0.839 | 10/10 |
| Answer Relevancy | 0.897 | 0.897 | 10/10 |
| Context Precision | 0.802 | 0.863 | 10/10 |
| Context Recall | 0.900 | 1.000 | 10/10 |
| Factual Correctness | 0.196 | 0.257 | 10/10 |

Every metric has zero failed and zero missing samples. The original 47 score lines, all ten answer files, and `inputs.json` are byte-identical to the pre-resume backup. Exactly three q011 records were appended, producing 50 unique question/metric pairs with no duplicates.

## Resume and API accounting

Answer generation used 32 chat and 32 embedding requests. RAGAS scoring used 192 chat attempts and 20 embedding requests across the original run and resumes: 182 chat responses were HTTP 200, seven temporary HTTP 503 high-demand responses were retried within the hard cap, and three HTTP 402 responses from the depleted original key stopped their invocations without appending a score.

Before the final resume, the effective key was verified to come from `.env` with no process-environment override. A model metadata request returned HTTP 200 for `models/gemini-3.8-flash`; the final scoring resume then used 20 chat requests and appended only q011 Context Precision (1.000), Context Recall (1.000), and Factual Correctness (0.000). Changing the credential did not change the model name, metric configuration, temperature, references, answers, contexts, or score fingerprints.

Two bounded generation probes timed out during key diagnosis and one metadata GET was sent. Including those probes, Stage C, answer generation, and scoring, the final campaign used 243 chat and 58 embedding attempts plus one model-metadata request, within the authorized 332/130 request ceiling.

## Interpretation limits

- q006 Faithfulness is 0.600; its closing prose includes an unsupported situational statement about exposing units after Falling Back, so the lower groundedness score identifies a real answer defect.
- q008 and q011 receive Factual Correctness 0.000 although their answers contain the short reference's core rule and add cited conditions or exceptions. These cases show the sensitivity of claim-based scoring to short synthetic references and fuller answers; they remain real saved scores and are not discarded.
- The ten-question set is suitable for this course evaluation but does not estimate accuracy for every Warhammer 40,000 question.

Historical R94, R96, and R98–R102 artifacts remain separate. Round 3 scores remain attributable only to R94; Round 4 scores apply only to the frozen current implementation above.

## Offline completion checks

- `uv --no-cache run pytest -p no:cacheprovider --basetemp data/checks/final-20261024/pytest-tmp`: 375 passed, 61 deselected.
- `uv --no-cache run ruff check .`: passed.
- `uv --no-cache run ruff format --check .`: 82 files already formatted.
- `git diff --check`: passed.

The warnings were limited to existing RAGAS import deprecations. No source code, prompt, retrieval setting, index, or evaluation configuration was changed while completing the checkpoint and delivery documentation.
