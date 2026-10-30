# Final Stage C acceptance

Verdict: **6/6 PASS**

The six scenarios ran once against commit `2f3d3529533528501a35a3950128bf38d4a27f40`. All 17 chat and 6 embedding requests returned HTTP 200, with no retries. Each verdict below checks the saved answer and tool trace rather than HTTP success alone.

| Scenario | Chat | Embedding | Tool behavior | Semantic verdict |
|---|---:|---:|---|---|
| Toughness + Wardens follow-up | 3 | 0 | Two first-turn SQL calls; zero follow-up SQL, embedding, or Gemini calls | PASS: the deterministic `structured_database_fields` follow-up gives all six profiles, T4/T3, *Space Marines*, edition 11, and database attribution without unsupported classifications |
| Wardens | 2 | 0 | One SQL call followed by final answer | PASS: all six names, Toughness values, publication, edition, and source tables are correct |
| Stratagem | 3 | 2 | Two focused rule searches | PASS: answer covers regular and additional CP gain, CP payment, same-phase reuse, card fields, and the Battle-shock restriction with citations |
| Scoring | 2 | 1 | One rule search | PASS: answer separates Only War scoring from objective control and the ordered destroyed-army / VP / draw winner branches |
| Grenades | 2 | 1 | One `search_rules` call, then deterministic no-tool final | PASS: complete Core Rules and exact database record are kept separate, including the Advance/Fall Back and measurement differences |
| Unit rules + Advance follow-up | 5 | 2 | First turn uses rule search plus two SQL calls; follow-up performs one focused rule search | PASS: database evidence supports prohibition after advancing; Core Rules silence is reported as undetermined rather than permission; 1 CP cost is preserved |

The suite remained within its 32 chat / 40 embedding cap. It did not run RAGAS, rebuild the index, modify code, or overwrite R94–R102 artifacts.

The acceptance result is an end-to-end functional check of the current frozen implementation. It is separate from both the historical R94 RAGAS scores and the incomplete Round 4 RAGAS run.
