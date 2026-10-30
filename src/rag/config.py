"""Project paths, project settings and runtime settings: the single place where they are defined."""

from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/rag/config.py -> chatbot/
SOURCE_DIR = PROJECT_ROOT / "src"
LOCK_FILE = PROJECT_ROOT / "uv.lock"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # the original inputs: never written (AGENTS.md)
RAW_PDF = RAW_DIR / "Core Rules.pdf"
RAW_DB = RAW_DIR / "wahadb.sqlite"
SQL_SCHEMA = PROJECT_ROOT / "src" / "rag" / "storage" / "sql_schema.md"  # the model's schema note
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DB = PROCESSED_DIR / "wahadb_clean.sqlite"  # the database with its text repaired
RULES_MARKDOWN = PROCESSED_DIR / "core_rules.md"
RULES_JSON = PROCESSED_DIR / "core_rules.json"  # the parsed document that chunk pictures refer to
TABLES_DIR = DATA_DIR / "tables"
RULES_CHUNKS = DATA_DIR / "index" / "rules_chunks.jsonl"
RULES_EMBEDDINGS = DATA_DIR / "index" / "rules_embeddings.npz"  # one vector per chunk + fingerprint
ONLINE_CHECK_DIR = DATA_DIR / "checks"  # the signature reports of the online compatibility check
CHAT_ACCEPTANCE_DIR = ONLINE_CHECK_DIR / "acceptance"  # the answers of the online chat acceptance
EVAL_DIR = DATA_DIR / "eval"  # the RAGAS evaluation (plan §6.9)
KNOWLEDGE_GRAPH = EVAL_DIR / "knowledge_graph.json"  # RAGAS's graph of the rulebook's sections
KNOWLEDGE_GRAPH_PROGRESS = EVAL_DIR / "knowledge_graph_progress.json"  # its steps done
TESTSET = EVAL_DIR / "testset.jsonl"  # the generated questions, checked by a person
EVAL_RUNS_DIR = EVAL_DIR / "runs"  # one folder per run of the chat on the test set: answers, scores

CHUNK_SIZE = 1500  # characters (~350 tokens): most rule sections stay whole (§16.5)
CHUNK_OVERLAP = 200  # characters shared by two consecutive pieces of a long section
# pages left out of the index: p. 4 duplicates pp. 5, 7, 8 (§4); p. 27 is a picture of weapons
# whose labels carry no rule (§14)
SKIP_PAGES = frozenset({4, 27})
EMBEDDING_DIM = 768  # dimensions of each embedding: a small index, little loss (plan §4.4)
TOP_K = 8  # includes the observed scoring/allocation misses at ranks 7–8 (plan R74)
SQL_MAX_ROWS = 50  # rows returned by one query; more are reported as truncated
SQL_MAX_SECONDS = 3.0
SQL_MAX_BYTES = 65536  # serialized tool result and SQLite value/row allocation limit
MAX_TOOL_ROUNDS = 5  # tool rounds for one question, then a final answer without tools
# The sections RAGAS writes the test set from (plan §6.9, R64, R65), for the second round: the
# parts of a battle round the first left out (commanding, moving, shooting, charging, fighting),
# the Stratagems and the scoring of the Core Rules mission. A group of names is one document, so
# that a rule keeps the context it belongs to. The first round used the five steps of making an
# attack (R51; its material is in data/eval/round1/). The chat still searches the whole rulebook.
TESTSET_SECTIONS = (
    "1. COMMAND",
    "2. BATTLE-SHOCK",
    "1. MOVE UNITS",
    "ADVANCE MOVES",
    ("FALL BACK MOVES", "Desperate Escape Tests"),
    "SHOOTING PHASE > SELECT TARGETS",
    "MAKE RANGED ATTACKS",
    ("CHARGE BONUS", "CHARGING WITH A UNIT"),
    "FIGHT PHASE",
    ("FIGHT", "1. PILE IN", "2. MAKE MELEE ATTACKS", "3. CONSOLIDATE"),
    "STRATAGEMS",
    "OBJECTIVE MARKERS",
    ("ONLY WAR", "CAPTURE AND CONTROL"),
)
TESTSET_SIZE = 10  # questions RAGAS is asked for; it may write a few more


class Settings(BaseSettings):
    """Values that differ between people or machines, read from the environment and `.env`.

    Created once by each script that calls Gemini and passed to the library; importing this module
    never needs a key. A key that is missing, empty or only spaces raises a `ValidationError` on
    `google_api_key`.
    """

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", frozen=True, extra="ignore")

    google_api_key: SecretStr
    chat_model: str = "gemini-3.8-flash"
    chat_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    embedding_model: str = "gemini-embedding-001"

    @field_validator("google_api_key")
    @classmethod
    def _key_is_set(cls, key: SecretStr) -> SecretStr:
        stripped = key.get_secret_value().strip()
        if not stripped:
            raise ValueError("GOOGLE_API_KEY is empty")
        return SecretStr(stripped)


def key_not_set(error: ValidationError) -> bool:
    """Whether a `Settings` error is the API key missing or empty, rather than another setting."""
    return any(tuple(detail["loc"]) == ("google_api_key",) for detail in error.errors())
