"""Retrieval: cut a contract into chunks and find the few that matter for each clause.

Two search methods over the same chunks, so they can be compared on the same eval:
  - "bm25":   keyword search. Matches words. Free, instant, strong on formulaic clauses.
  - "vector": meaning search. Turns text into numbers (embeddings) and finds the nearest.
              Runs locally on the laptop with a small embedding model; no API quota.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNK_CHARS = 1500      # about 350-400 tokens
OVERLAP_CHARS = 200     # so a sentence cut at a boundary still appears whole in one chunk
VECTOR_CHUNK_CHARS = 1500   # must fit inside the embedding model's 256-token window

# EMBED_MODEL = "all-MiniLM-L6-v2"      # small, fast, runs on CPU; 384 numbers per text
EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # retrieval-tuned, 512-token window, 768 numbers per text
CACHE_DIR = Path("cache/embeddings")  # embeddings are computed once per contract and kept

# Keyword queries: rare, specific words that appear near the clause. (BM25 only.)
QUERIES = {
    "governing_law": "governing law governed by the laws of the state jurisdiction construed in accordance",
    "cap_on_liability": "limitation of liability liable damages exceed aggregate consequential indirect "
                        "incidental special punitive remedies limited in no event "
                        "sole exclusive remedy entire liability",
    "non_compete": "compete competing competitive non-compete restrict restriction business engage "
                   "solicit territory during the term "
                   "non-competition competition restricted period covenant similar business",
}

# Meaning queries: a plain description of the clause, as a sentence. (Vector search only.)
# An embedding model compares meanings, so a description beats a pile of keywords.
DESCRIPTIONS = {
    "governing_law": "This agreement shall be governed by and construed in accordance with the laws "
                     "of a particular state or country.",
    "cap_on_liability": "In no event shall either party's total liability under this Agreement exceed the "
                        "fees paid in the twelve months preceding the claim, and neither party shall be "
                        "liable for any indirect, incidental, consequential or punitive damages.",
    "non_compete": "During the term of this Agreement and for two years thereafter, Distributor shall "
                   "not, directly or indirectly, engage in or assist any business that competes with "
                   "Company's products in the Territory.",
}


@dataclass
class Chunk:
    id: int
    start: int          # character offset in the original text
    end: int
    text: str


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    """Sliding window over the text. Simple and robust; smarter chunking comes later."""
    chunks = []
    step = size - overlap
    for i, start in enumerate(range(0, max(len(text), 1), step)):
        piece = text[start:start + size]
        if not piece.strip():
            continue
        chunks.append(Chunk(id=i, start=start, end=start + len(piece), text=piece))
        if start + size >= len(text):
            break
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase words only. Good enough for keyword search."""
    return re.findall(r"[a-z]+", text.lower())


# --- embeddings: text -> a list of numbers that is close for texts with close meanings ---
_model = None


def embed(texts: list[str]) -> np.ndarray:
    """One row of numbers per text. Rows are scaled to length 1, so a dot product is a cosine."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer   # imported here: slow import, only when needed
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


class ContractIndex:
    """Chunks of one contract plus a BM25 index over them. Built once, queried many times."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        """The k chunks that best match the query words, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in order[:k]]

    def retrieve_for_clause(self, clause: str, k: int = 3) -> list[Chunk]:
        return self.search(QUERIES[clause], k)


class VectorIndex:
    """Chunks of one contract plus one embedding per chunk. Embeddings are cached on disk."""

    def __init__(self, text: str, cache_key: str | None = None):
        self.chunks = chunk_text(text, size=VECTOR_CHUNK_CHARS)
        cache_file = CACHE_DIR / f"{cache_key}_{EMBED_MODEL.split('/')[-1]}_c{VECTOR_CHUNK_CHARS}.npy" if cache_key else None
        if cache_file and cache_file.exists():
            self.vectors = np.load(cache_file)
        else:
            self.vectors = embed([c.text for c in self.chunks])
            if cache_file:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, self.vectors)

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        """The k chunks whose meaning is closest to the query, best first."""
        q = embed([query])[0]
        scores = self.vectors @ q                    # cosine similarity to every chunk, in one line
        order = np.argsort(-scores)                  # highest first
        return [self.chunks[i] for i in order[:k]]

    def retrieve_for_clause(self, clause: str, k: int = 3) -> list[Chunk]:
        return self.search(DESCRIPTIONS[clause], k)


def build_index(text: str, method: str, cache_key: str | None = None):
    """One place that knows how to build either kind of index."""
    if method == "bm25":
        return ContractIndex(text)
    if method == "vector":
        return VectorIndex(text, cache_key=cache_key)
    raise ValueError(f"unknown method: {method} (use 'bm25' or 'vector')")