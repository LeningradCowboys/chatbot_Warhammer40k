# Warhammer 40k RAG Chatbot
*Group members: Zhexi WANG, Zhiyi CHEN*

## Overview

This project is a multi-source retrieval-augmented generation (RAG) chatbot for Warhammer 40,000.
It answers questions from two complementary sources:

- the 10th-edition Core Rules PDF, cited by page; and
- a SQLite datasheet database, cited by table and retaining publication and edition metadata.

The command-line chat can search either source, combine their evidence, keep short multi-turn
conversations and explain which tools supplied an answer. The implementation uses Gemini through
Google AI Studio for generation and embeddings.

## Quick start

### Requirements

- **Windows x64**, or an **Apple Silicon Mac** with macOS 14 or newer. Intel Macs are not supported.
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the locked dependencies.
- A Google AI Studio API key for indexing, chat and online evaluation. Parsing, chunking and offline
  tests do not need a key.

Install `uv`:

- Windows PowerShell:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Then clone and install the project:

```text
git clone https://github.com/LeningradCowboys/chatbot_Warhammer40k.git chatbot
cd chatbot
uv sync --locked
```

Install the evaluation group before running RAGAS or the complete offline test suite:

```text
uv sync --locked --group eval
```

With only the base `uv sync`, `pytest` skips `test_evaluate.py`,
`test_evaluation_testset.py` and `test_ragas_adapters.py` because RAGAS is unavailable.
The first complete test run also downloads tiktoken's `cl100k_base` and `o200k_base` encoding files.
After they are cached the suite runs offline; an empty-cache machine must allow that first download or
pre-provision both files.

### Data

The licensed source files are not committed. Download them from the course
[Google Drive folder](https://drive.google.com/drive/folders/1IM1lNxWGK6Ll3m5_GhhpV6wpAHxs8X9j?usp=drive_link)
and place them under `data/raw/` with exactly these names:

| Source | Path | Size | SHA-256 |
|---|---|---:|---|
| Core Rules PDF, 10th edition, May 2023 | `data/raw/Core Rules.pdf` | 20,700,229 bytes | `4d0e8019cbfddd6f46781d5b4ed31d46fb21eb2d0d10a0f6fabefac0ce054364` |
| Datasheet database | `data/raw/wahadb.sqlite` | 9,699,328 bytes | `364f412c3c1802d24ccc97eb74ffac78e6a5894a312b2fde685045945595fbc5` |
| Database dictionary, reference only | `data/raw/SQL_AGENT_DATA_DICTIONARY.md` | 15,934 bytes | `5a0e527b430ab5538d6ecad0aa05b515eca61214ad10705aae9e3b946c04be4a` |

Create the directory with `mkdir data\raw` on Windows or `mkdir -p data/raw` on macOS. A different
PDF version changes page positions and causes real-data tests to fail.

The parsed rulebook is committed so that most users can skip the expensive parse:

| File | Size | SHA-256 of canonical LF bytes |
|---|---:|---|
| `data/processed/core_rules.md` | 155,312 bytes | `612cc2c7a5921e667bdd60f0f7c104d00257d7dd1a4dd9b9686d46d55bbf90e7` |
| `data/processed/core_rules.json` | 1,368,144 bytes | `71f9cc80ba100b398f73ffe3dd687275f6a7421686a1d325ee071172c1ebdc11` |

They were produced with Docling 2.132.0, docling-core 2.99.0 and pypdfium2 5.13.0 on Windows 11
x64. Parsing on another platform can produce small floating-point layout differences; the committed
files are the canonical input to chunking.

The repository includes the derived artifacts needed to run and review the project: the canonical
parsed rulebook, structural chunks and embedding index, frozen test set and saved Round 4 results,
final validation records, and the sanitized evaluation archive. Licensed source inputs and the
generated clean SQLite database are not committed; `prepare_db` recreates the clean database from
the separately supplied original.

### API key

Create a key in [Google AI Studio](https://aistudio.google.com), copy `.env.example` to `.env`, and
set:

```text
GOOGLE_API_KEY=your-key-here
```

The default models are `gemini-3.8-flash` and `gemini-embedding-001`. `CHAT_TEMPERATURE` is optional. API limits and billing belong to the Google Cloud project associated with the key, so check the current quota before running online commands. Failed requests still count against any declared request budget. For the fallback procedure, see [Chat](#chat).

### Prepare the derived data

The parsed rulebook, structural chunks and embedding index are already included in the repository.
For normal chat and evaluation, place the source database under `data/raw/` and run only step 3
(`prepare_db`). Steps 1, 2 and 4 are provided only to reproduce the committed derived artifacts.

Run commands from the repository root. Every writer refuses to overwrite its input or write inside
`data/raw/`.

1. Parse the PDF only if the committed parse is unavailable or must be reproduced:

   ```text
   uv run python -m src.scripts.parse "data/raw/Core Rules.pdf"
   ```

2. Create structural rule chunks:

   ```text
   uv run python -m src.scripts.chunk
   ```

   This produces `data/index/rules_chunks.jsonl`. The reference run contains 255 chunks and retains
   page numbers, section labels, table structure, figure labels and page footers.

3. Prepare the SQLite database:

   ```text
   uv run python -m src.scripts.prepare_db
   ```

   This writes `data/processed/wahadb_clean.sqlite`, repairing text encoding and Stratagem card
   labels while leaving the source database unchanged.

4. Build the embedding index:

   ```text
   uv run python -m src.scripts.build_index
   ```

   The command embeds all chunks with `gemini-embedding-001` and stores the vectors together with
   their input fingerprint in `data/index/rules_embeddings.npz`. The default request budget is 18
   embedding HTTP attempts, including failed attempts and SDK retries. The reference set contains
   255 chunks, sent in three batches of at most 100; the default therefore allows up to six attempts
   per batch. Use `--budget N` to choose a different safety limit

## Usage and testing

### Chat

Start the interactive chat after preparing the database and index:

```text
uv run python -m src.scripts.chat
```

Useful commands inside the chat:

- `/sources` shows the rule query or SQL used by the last answer.
- `/reset` clears the current conversation and its retained evidence.
- `/quit` exits.

Run one non-interactive question and redirect the answer to a file:

```text
uv run python -m src.scripts.chat --question "How do grenades work?" > answer.txt
```

Answers cite evidence as `[Core Rules p. N]` or `[DB: table_name]`. Separate source notes report
database publication/edition values, mission labels, query errors and truncation. They describe the
retrieved evidence and do not independently verify every generated claim.

**NOTE**: If the terminal remains silent for an unusually long time, press `Ctrl+C` rather than submitting the
question repeatedly. If `gemini-3.8-flash` repeatedly hangs or returns HTTP 503, set the following
in `.env`, then **restart the chat**:

```text
CHAT_MODEL=gemini-3.6-flash
```

The application does not switch models automatically.


### Testing

| Command | Purpose | External requirements |
|---|---|---|
| `uv run pytest` | Complete fast offline suite | `uv sync --group eval`; no API key; tiktoken files already cached or first-download access |
| `uv run pytest -m slow` | Real PDF, committed parse and clean-database integration tests | Source PDF and prepared database |
| `uv run ruff check .` | Python lint | None |
| `uv run ruff format --check .` | Formatting check | None |


### Evaluation

This workflow reruns the complete ten-question frozen test set against the real Gemini API. The
saved Round 4 answers and scores are already included in the repository; use a new run name for any
new evaluation so that the historical results are not overwritten.

Install the evaluation dependency group, prepare the database and index, and replace `N` and `M`
below with positive integer safety limits for the available chat and embedding quota. The commands
are templates; do not enter the letters `N` and `M` literally:

```text
uv run python -m src.scripts.evaluate answer --run my-run --chat-budget N --embedding-budget M
uv run python -m src.scripts.evaluate score --run my-run --chat-budget N --embedding-budget M
```

`evaluate answer` saves each generated answer, its retrieved contexts and its trace. `evaluate score` calculates the five RAGAS metrics from the saved answers without regenerating them. A run
manifest fingerprints the test set, code, corpus, vectors, database, schema, models and lock file;
continuation is refused if those inputs change.

The historical Round 4 answer step used 32 chat and 32 embedding requests. Its checkpointed scoring
runs used a further 192 chat and 20 embedding attempts. A free-tier project limited to 20 chat
requests per day cannot complete the entire workflow in one day.

Request limits apply separately to each command invocation, and failed attempts and SDK retries
count against them. Completed answers and scores are checkpointed, so rerunning with the same
compatible run name continues the unfinished work. Local budget exhaustion exits with status 2;
provider errors such as HTTP 429 or 503 can also stop a run, while previously completed checkpoints
remain saved.

## Architecture and design choices

```text
Core Rules PDF ── Docling/PDFium ── structural chunks ── Gemini embeddings ─┐
                                                                            ├─ search_rules ─┐
Question ── LangGraph agent ────────────────────────────────────────────────┤                ├─ cited answer
                                                                            ├─ run_sql ──────┘
Datasheet SQLite ── encoding cleanup ── read-only query interface ──────────┘

Saved answers + contexts ── RAGAS ── faithfulness / relevance / retrieval / factual matching
```


| Component | Implementation | Rationale |
|---|---|---|
| Rulebook ingestion | Docling, PDFium and structural chunking | Preserve multi-column layout, tables, page locations, vector dice, figure labels and rule-card structure without sending the source PDF to another model. |
| Rule retrieval | BM25, Gemini embeddings and equal-weight RRF with `TOP_K = 8` | Combine exact terminology with semantic retrieval while avoiding direct comparison of incompatible raw scores. |
| Spelling handling | Conservative correction protected by explicit database entity names | Correct observed misspellings such as `Movment`, `attaks` and `eligable` without expanding known proper names as similar corpus words. |
| Structured retrieval | Read-only SQLite queries | Query unit profiles, model attributes and publication metadata as structured facts rather than embedding them as prose. |
| Agent orchestration | LangGraph | Coordinate rule search and SQL, preserve short multi-turn state, enforce tool-round limits and expose a separate no-tool final-answer path. |
| Grounding safeguards | Structured evidence reuse and deterministic formatting for narrow database facts | Reduce redundant calls and unsupported classifications when complete structured evidence is available. |
| Evaluation | RAGAS plus human review | Measure retrieval, relevance, faithfulness and reference agreement while retaining manual checks for semantic correctness and source boundaries. |

The rules retriever uses a fixed 1:1 reciprocal-rank fusion and returns eight chunks. SQLite accepts
only read-only queries and limits results to 50 rows, three seconds of virtual-machine execution and
64 KiB. Exact named Core Stratagem results can attach their corresponding database card and closely
related rule sections. Early finalization is allowed only when required evidence is complete,
unambiguous and not truncated.

## Evaluation results

The final RAGAS evaluation completed all 10/10 frozen questions and produced 50/50 valid
metric scores. The separate end-to-end acceptance suite passed 6/6 scenarios.

| Final Round 4 metric | Mean |
|---|---:|
| Faithfulness | 0.839 |
| Answer Relevancy | 0.897 |
| Context Precision | 0.863 |
| Context Recall | 1.000 |
| Factual Correctness | 0.257 |

The saved Round 4 results and their supporting artifacts are included with the project. Because
model generation and judging are stochastic, a new evaluation run is not expected to reproduce the
saved scores exactly.

Detailed settings, per-question results, coverage limitations and qualitative analysis are available
in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

The supporting evaluation archive is included at
`data/delivery/evaluation-attachments-v2.zip` and mirrored on
[Google Drive](https://drive.google.com/file/d/1zBxryCG6mMU_OohWUiv7skjWeMZLk_M1/view?usp=drive_link).


## Limitations

- **Source inputs:** the original PDF and database are supplied separately by the course and are not included in this repository. A new user must obtain the matching files separately.
- **PDF extraction:** standard vector dice and known layout structures are recovered, but the parser is not a general image-understanding system. Unrecognized graphics are not indexed, and some sidebars can still appear under neighbouring headings.
- **Spelling tolerance:** protection applies to known entities found in explicit database name fields. An unknown proper noun may still be mistaken for a similar term in the indexed text.
- **Database provenance:** the supplied database combines records from multiple publications and edition labels. Answers report the metadata stored for each record rather than assuming one edition for the whole database.
- **Conversation size:** in-memory history is intended for short local sessions and is not automatically summarized. Use `/reset` between topics.
- **Model variability:** generated answers can vary and may occasionally over-interpret missing information. For rules-sensitive decisions, users should verify the cited sources.


## Troubleshooting

| Symptom | Resolution |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run scripts as modules from the repository root, for example `uv run python -m src.scripts.chat`. |
| A required input is not found | Put the exact source files under `data/raw/`, then run the preparation steps above. |
| The first parse appears stuck | Docling downloads its model files on first use; check access to `huggingface.co`. |
| `GOOGLE_API_KEY is not set` | Copy `.env.example` to `.env` and set a non-empty key. |
| Repeated chat-model timeout or HTTP 503 | Stop repeated retries. Set `CHAT_MODEL=gemini-3.6-flash` in `.env`, restart the command, and record the actual model in any evaluation or acceptance report. The application does not fall back automatically. |
| `429 RESOURCE_EXHAUSTED` | The project quota is exhausted. Check Google AI Studio usage and wait for the relevant limit to reset; do not repeatedly retry. |
| `402` or a billing/prepayment error | Check the billing project associated with the actual API key. Completed evaluation checkpoints remain safe to resume. |
| macOS reports no `x86_64` wheel | Use a native Apple Silicon terminal (`uname -m` must print `arm64`), remove `.venv` and run `uv sync` again. |
| `403 PERMISSION_DENIED` | Verify the effective API key, AI Studio project status and key restrictions. Do not retry until the permission problem is resolved. |


