# Round 4 / Final RAGAS status

Status: **INCOMPLETE — 47/50 metric values saved**

## Frozen inputs

- Commit: `2f3d3529533528501a35a3950128bf38d4a27f40`
- Chat model: `gemini-3.8-flash`; effective temperature omitted (`null`)
- Embedding model: `gemini-embedding-001`
- Test set: the existing 10-question `data/eval/testset.jsonl`
- The saved `data/eval/runs/round4-final/inputs.json` matches the test-set, chunks, embeddings, clean database, source, schema, and lock hashes captured in `freeze.json`.

## Answer phase

All 10 expected IDs (`q001`–`q004`, `q006`–`q011`) have complete saved answer artifacts. The phase used 32 chat and 32 embedding requests, within its 60/60 cap. No answer was regenerated after scoring began.

## Scoring phase

Nine questions have all five metrics. q011 has Faithfulness and Answer Relevancy, leaving three metrics unfinished:

- `q011/llm_context_precision_with_reference`
- `q011/context_recall`
- `q011/factual_correctness`

The first scoring process stopped after Google returned HTTP 402 `RESOURCE_EXHAUSTED`: the project's prepayment credits were depleted. Two resumption attempts were safe—47 matching input fingerprints were skipped—but each first unsaved metric request received the same 402. Completed values remain intact in `scores.jsonl`.

Partial aggregates must not be reported as final Round 4 scores:

| Metric | Saved / expected | Partial mean |
|---|---:|---:|
| Faithfulness | 10/10 | 0.839 |
| Answer Relevancy | 10/10 | 0.897 |
| Context Precision | 9/10 | 0.847 |
| Context Recall | 9/10 | 1.000 |
| Factual Correctness | 9/10 | 0.286 |

Actual scoring traffic, including the failed resumption probes, is 172 chat and 20 embedding requests. Across Stage C, answer generation, and scoring, total traffic is 221 chat and 58 embedding requests, within the authorized 332/130 overall ceiling.

## Observed interpretation limits

- q008 received Factual Correctness 0.000 even though its answer states the reference's general Advance restrictions and adds the cited Assault-weapon exception. This illustrates that a short reference answer and a broader supported answer can produce a severe precision penalty.
- q006 Faithfulness is 0.600. Its closing tactical prose includes an unsupported situational claim about exposing units after Falling Back; the lower groundedness score is therefore materially informative.
- Round 4 uses the same questions and five metric names as R94, but the implementation, prompts, retrieval behavior, model configuration, and evaluation code are not a controlled single-variable experiment. Any eventual comparison is observational.

## Safe resumption

After billing is restored, rerun the same `score` command for `round4-final` with no file deletion or reset. The scorer will reuse only values whose score fingerprints match and compute the three missing q011 metrics. Use no more than the remaining 68 chat and 10 embedding requests under the original total cap; do not regenerate answers or indexes.
