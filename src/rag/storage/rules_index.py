"""Save and load the embeddings of the rule chunks, with the fingerprint of their source.

The file holds one vector per chunk (row i = chunk i) and a fingerprint: the SHA-256 of the
embedding model, `EMBEDDING_DIM` and every chunk text. Loading recomputes it from the current
chunks and settings and refuses an index built from anything else, which would otherwise make
search return the wrong chunk for each vector without any error (plan §6.6).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from src.rag.chunker.markdown_chunker import Chunk
from src.rag.config import EMBEDDING_DIM


def index_fingerprint(chunks: list[Chunk], embedding_model: str) -> str:
    """The SHA-256 of the embedding model, the dimension and every chunk text, in order."""
    content = json.dumps([embedding_model, EMBEDDING_DIM, [chunk.text for chunk in chunks]])
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_embeddings(
    path: Path, vectors: np.ndarray, chunks: list[Chunk], embedding_model: str
) -> None:
    """Write the vectors of `chunks` and their fingerprint to one `.npz` file.

    Args:
        path: The index file.
        vectors: One row per chunk, in the order of `chunks`.
        chunks: The chunks the vectors were computed from.
        embedding_model: The model that computed them.

    Raises:
        ValueError: The vectors do not match the chunks in number, or the dimension.
    """
    if vectors.shape != (len(chunks), EMBEDDING_DIM):
        raise ValueError(f"{vectors.shape} vectors for {len(chunks)} chunks of {EMBEDDING_DIM}")
    with path.open("wb") as file:  # a file object keeps the name as given (no ".npz" added)
        np.savez(
            file,
            vectors=vectors.astype(np.float32),
            fingerprint=np.array(index_fingerprint(chunks, embedding_model)),
        )


def load_embeddings(path: Path, chunks: list[Chunk], embedding_model: str) -> np.ndarray:
    """Read the vectors back, if they were built from exactly these chunks and this model.

    Args:
        path: The index file.
        chunks: The current chunks.
        embedding_model: The current embedding model.

    Returns:
        The float32 vectors, row i for chunk i.

    Raises:
        ValueError: The index was built from other chunks, another model or another dimension.
    """
    with np.load(path, allow_pickle=False) as index:
        if str(index["fingerprint"]) != index_fingerprint(chunks, embedding_model):
            raise ValueError(
                f"{path} was built from other chunks or another embedding model: "
                "re-run src.scripts.build_index"
            )
        return index["vectors"]
