# Evaluation Report

## 1. Scope

This report summarizes the saved evaluation of the Warhammer 40,000 multi-source RAG chatbot. The
system answers from the 10th-edition Core Rules PDF and a datasheet SQLite database.

The final evidence has two complementary parts:

1. a six-scenario semantic acceptance suite that checks complete system behaviour; and
2. a ten-question RAGAS evaluation with five metrics per question.

The acceptance suite tests concrete product requirements. RAGAS supplies repeatable diagnostic
metrics, but its LLM-judged results are not treated as a direct percentage of correct answers.

## 2. Final evaluation configuration

The saved Round 4 evaluation and the six-scenario acceptance suite in Section 3 were run on the same version of the project, using gemini-3.8-flash. The final submission later added a fresh database lookup for the Wardens follow-up, strengthened the instructions against drawing conclusions from missing evidence, and made application-initiated database lookups visible in the source notes. These changes did not alter the underlying rules corpus, database, embeddings or evaluation set. The saved answers, scores, traces and input fingerprints provide the record of those runs. The saved Round 4 run used the following configuration.

| Setting | Value |
|---|---|
| Answer and judge model | `gemini-3.8-flash` |
| Effective chat temperature | omitted (`None`) |
| Embedding model | `gemini-embedding-001` |
| RAGAS version | 0.4.3 |
| Retrieval depth | `TOP_K = 8` |
| RRF weighting | 1:1 lexical/vector |
| Questions | 10 curated questions: q001–q004 and q006–q011 |
| Metrics | Faithfulness, Answer Relevancy, Context Precision, Context Recall, Factual Correctness |
| Factual Correctness | claim-level `mode="f1"`, `atomicity="low"`, `coverage="low"` |

Each question was answered in a new conversation. Faithfulness used the successful tool outputs the
agent actually received. Context Precision and Context Recall used a separate top-eight retrieval of
the original question. Answer Relevancy used the question and answer; Factual Correctness compared
the answer with its saved reference.

RAGAS generated 12 candidate questions before manual curation. Two were removed before evaluation:

- q005 asked how targets are selected for both ranged and melee attacks, but its reference answered
  the melee half with the order of the Fight steps rather than the target-selection rule; the
  generation material also omitted the passage needed to repair that reference.
- q012 covered the same Advance distance and shooting/charging restrictions as the earlier q004 and
  was removed as a near duplicate under the pre-answer duplicate criterion.

The ten retained questions cover the Command phase and Command points, Battle-shock, movement and
Advancing, Desperate Escape tests, shooting and charge eligibility, and Stratagems. They do not
directly cover the Fight phase, objective markers, Only War mission scoring or the `MAKE RANGED
ATTACKS` procedure, and Battle-shock is represented more often than those other topics.

`data/eval/runs/round4-final/inputs.json` fingerprints the test set, chunks, embeddings, clean
database, schema, source modules and lock file. This prevents a resumed run from silently combining
scores from incompatible inputs or settings.

The saved numerical results belong to the archived run. Because generation and judging are stochastic, a new run may produce different answers and metrics.

## 3. End-to-end acceptance

All six scenarios ran once. Their saved answers and tool traces were reviewed for semantic
correctness and source use.

| Scenario | Chat / embedding requests | Verified result |
|---|---:|---|
| Toughness + Wardens follow-up | 3 / 0 | The first turn obtained complete structured evidence. The follow-up made no SQL, embedding or Gemini call and returned all six model profiles, correct T4/T3 values, *Space Marines*, edition 11 and database attribution without unsupported classifications. |
| Standalone Wardens | 2 / 0 | One SQL call returned all six model names and Toughness values with publication, edition and source tables. |
| Stratagem | 3 / 2 | The answer covered regular and additional CP generation, payment, same-phase reuse, card fields and the Battle-shock restriction with citations. |
| Scoring | 2 / 1 | The answer separated Only War victory-point scoring from objective control and preserved the ordered destroyed-army, VP and draw branches. |
| Grenades | 2 / 1 | One rules search supplied complete Core Rules and exact database evidence; the answer kept their Advance/Fall Back and measurement differences separate. |
| Unit rules + Advance follow-up | 5 / 2 | Database evidence supported its explicit prohibition after Advancing, while Core Rules silence was reported as undetermined rather than permission; the 1CP cost was preserved. |

Result: 6/6 PASS, using 17 chat and 6 embedding requests. Every request returned HTTP 200
without retry. One successful run per scenario is regression evidence, not a statistical reliability
estimate.

Later checks of the submitted version with gemini-3.6-flash covered eight selected questions and scenarios, and all eight runs completed. Some answers still over-interpreted missing evidence or omitted rule details. These behaviours remain a limitation for rules-sensitive use.

## 4. Round 4 RAGAS results

| Question | Faithfulness | Answer relevancy | Context precision | Context recall | Factual correctness |
|---|---:|---:|---:|---:|---:|
| `q001` | 0.750 | 0.804 | 1.000 | 1.000 | 0.180 |
| `q002` | 0.895 | 0.933 | 1.000 | 1.000 | 0.530 |
| `q003` | 1.000 | 0.880 | 0.250 | 1.000 | 0.150 |
| `q004` | 0.926 | 0.920 | 1.000 | 1.000 | 0.460 |
| `q006` | 0.600 | 0.907 | 0.867 | 1.000 | 0.120 |
| `q007` | 0.882 | 0.883 | 0.643 | 1.000 | 0.750 |
| `q008` | 0.857 | 0.895 | 0.867 | 1.000 | 0.000 |
| `q009` | 0.846 | 0.952 | 1.000 | 1.000 | 0.180 |
| `q010` | 0.636 | 0.857 | 1.000 | 1.000 | 0.200 |
| `q011` | 1.000 | 0.940 | 1.000 | 1.000 | 0.000 |
| **Mean** | **0.839** | **0.897** | **0.863** | **1.000** | **0.257** |
| **Valid / expected** | **10 / 10** | **10 / 10** | **10 / 10** | **10 / 10** | **10 / 10** |
| **Failed / missing** | **0 / 0** | **0 / 0** | **0 / 0** | **0 / 0** | **0 / 0** |

Because the retrieval depth was chosen using known saved-query misses and the same ten questions
were retained across the later rounds, Round 4 is an in-sample regression result. Context Recall
`1.000` means the RAGAS judge found every saved reference claim attributable to these ten questions'
top-eight retrieval contexts. It does not estimate recall on unseen questions or establish that
eight contexts is optimal outside this set.

All ten answer files are complete and `scores.jsonl` contains exactly 50 unique question/metric
records. Answer generation used 32 chat and 32 embedding requests. Scoring used 192 chat and 20
embedding requests across checkpointed runs.

## 5. Factual Correctness qualitative analysis

The mean Factual Correctness score is **0.257**. This is RAGAS's reference-based claim-level F1, not
the percentage of answers that are factually correct. The references are generally short, while the
answers often include supported conditions and exceptions beyond the minimum reference.

The table below compares the saved questions, references, answers and contexts. It supplements the
scalar metric with a manual reading of answer quality.

| Question | Score | Manual classification | Evidence from the saved answer and reference |
|---|---:|---|---|
| `q001` | 0.180 | Reference-response asymmetry | The answer includes both players gaining 1CP at each Command phase and spending CP on Stratagems, matching the reference, then adds the supported additional-CP cap, costs and restrictions. |
| `q002` | 0.530 | Core answer complete; fuller than reference | It gives the 2D6 Leadership threshold, duration, OC 0, Stratagem restriction and Desperate Escape consequence required by the reference, with extra supported detail about individual test outcomes. |
| `q003` | 0.150 | Reference-response asymmetry | Normal Move, Advance, Remain Stationary and engaged-unit Fall Back choices are present. The response also covers Transports and Reinforcements, far beyond the short reference. |
| `q004` | 0.460 | Core answer complete; expanded scope | The required D6 distance, M+D6 movement, Engagement Range, shooting and charging restrictions are explicit. The answer adds movement geometry, Fly/Aircraft, Assault, Transport and re-roll rules. |
| `q006` | 0.120 | Real unsupported addition | The required no-charge rule and per-model Desperate Escape tests are correct. The closing claim that Falling Back frees the enemy to be shot while leaving the retreating unit exposed is situational tactical prose not established by the cited evidence. |
| `q007` | 0.750 | Substantially aligned | The answer reproduces the 12-inch requirement and every listed disqualifier. Its Assault clarification is relevant and supported. |
| `q008` | 0.000 | Zero-score anomaly | The answer explicitly states both reference facts: an Advanced unit normally cannot shoot and cannot charge. It additionally gives the supported Assault-weapon shooting exception. The final zero cannot be explained from the saved scalar alone. |
| `q009` | 0.180 | Core answer complete; expanded source comparison | The controlling-player Stratagem prohibition is stated directly. The answer then distinguishes printed and database versions of Insane Bravery, substantially increasing the claim set beyond the reference. |
| `q010` | 0.200 | Over-general wording | The same-Stratagem once-per-phase rule and both players' CP gain are correct. Saying there is “no overall limit” on different Stratagems is broader than the evidence; the safer conclusion is that the cited core rule gives no separate numeric cap, subject to CP, timing, targets and restrictions. |
| `q011` | 0.000 | Zero-score anomaly | The first sentence exactly supplies the reference fact: every model in a Battle-shocked unit must take a Desperate Escape test before Falling Back. The additional roll outcome, movement and post-Fall-Back restrictions are supported. The final zero cannot identify which judge step failed. |

Manual inspection found no systematic error in the central rule values, phases or conditions across
the ten answers. It identified an unsupported tactical addition in q006 and overly absolute wording
in q010. The zero scores for q008 and q011 cannot be explained from the saved scalar values because
the intermediate claim decomposition and NLI decisions were not retained. These findings support
reading Factual Correctness alongside Faithfulness and manual review rather than treating `0.257` as
the percentage of correct answers.

## 6. Historical comparison

The previous complete ten-question evaluation is retained as a historical baseline:

| Metric | Historical Round 3 | Final Round 4 |
|---|---:|---:|
| Faithfulness | 0.861 | 0.839 |
| Answer Relevancy | 0.897 | 0.897 |
| Context Precision | 0.802 | 0.863 |
| Context Recall | 0.900 | 1.000 |
| Factual Correctness | 0.196 | 0.257 |

These differences are observational. Retrieval behaviour, prompts, answer routing, evaluator
sampling and judge executions changed between the two snapshots, so the table is not evidence that
one isolated change caused an improvement or regression.

## 7. Evidence and reproducibility

The supporting evaluation archive is available on Google Drive as
[`evaluation-attachments.zip`](https://drive.google.com/file/d/1zBxryCG6mMU_OohWUiv7skjWeMZLk_M1/view?usp=drive_link).


Inside the extracted archive, the most relevant paths are:

| Path | Contents |
|---|---|
| `data/eval/runs/round4-final/inputs.json` | Evaluation input and configuration fingerprints |
| `data/eval/runs/round4-final/answers/` | Ten saved answers, contexts, agent traces and answer request counts |
| `data/eval/runs/round4-final/scores.jsonl` | Fifty final score records |
| `data/checks/final-20261024/online/stage-c-review.md` | Six-scenario semantic review |


## 8. Limitations

- Ten questions and one run per acceptance scenario do not measure system-wide reliability or generation variability.
- `TOP_K = 8` was tuned against known saved queries, so Round 4 retrieval metrics are in-sample;
  Context Recall `1.000` is not evidence of recall on new questions.
- The answer model and RAGAS judge used the exact same model, `gemini-3.8-flash`, rather than an
  independent and preferably stronger judge.
- Short references can reward concise overlap and penalize fuller answers, even when added details
  are supported.
- RAGAS saved the final scalar scores but not its intermediate claim decomposition or NLI decisions,
  so the exact rationale for an individual judge score cannot always be reconstructed.
