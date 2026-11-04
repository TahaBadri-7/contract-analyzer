"""Retrieval: cut a contract into chunks and find the few that matter for each clause.

Chunkers (how the contract is cut):
  - "window":    fixed 1,500-character windows with 200 of overlap. Simple; cuts mid-sentence.
  - "sentences": whole sentences grouped up to 1,500 characters. Never cuts a sentence.

Search methods (how chunks are ranked), all over the same chunks so they compare fairly:
  - "bm25":   keyword search. Matches words.
  - "vector": meaning search with a local embedding model.
  - "merge":  bm25's top-k plus vector's top-k, both sent (up to 2k chunks). Recall can only rise; cost doubles.
  - "fuse":   both rankings combined into one with reciprocal rank fusion, then top-k. Same cost as one method.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

CHUNKER = "window"          # "window" or "sentences"
CHUNK_CHARS = 1500          # about 350-400 tokens; both search methods use the same chunks
OVERLAP_CHARS = 200         # window chunker only

EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # retrieval-tuned, 512-token window, 768 numbers per text
CACHE_DIR = Path("cache/embeddings")    # embeddings are computed once per contract and kept
RRF_K = 60                              # the standard constant in reciprocal rank fusion

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

# Meaning queries: text that looks like the clause we want. (Vector search only.)
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
    """Sliding window over the text. Simple and robust; cuts mid-sentence."""
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


# A sentence ends at . ; or : followed by whitespace and then a capital, a digit, or an opening bracket.
# Not perfect for legal text ("Inc.", "Sec. 6.2(e)"), but good enough to measure.
SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")


def chunk_text_sentences(text: str, size: int = CHUNK_CHARS) -> list[Chunk]:
    """Group whole sentences into chunks of about `size` characters. Never cut a sentence;
    only a single sentence longer than `size` ends up alone in its own chunk."""
    chunks, buf, buf_start, pos = [], [], 0, 0
    for sentence in SENTENCE_END.split(text):
        if buf and sum(len(s) for s in buf) + len(sentence) > size:
            piece = " ".join(buf)
            chunks.append(Chunk(id=len(chunks), start=buf_start, end=buf_start + len(piece), text=piece))
            buf, buf_start = [], pos
        buf.append(sentence)
        pos += len(sentence) + 1
    if buf:
        piece = " ".join(buf)
        chunks.append(Chunk(id=len(chunks), start=buf_start, end=buf_start + len(piece), text=piece))
    return chunks


def make_chunks(text: str) -> list[Chunk]:
    """One place that knows which chunker is in use."""
    if CHUNKER == "sentences":
        return chunk_text_sentences(text)
    return chunk_text(text)


def tokenize(text: str) -> list[str]:
    """Lowercase words only. Good enough for keyword search."""
    return re.findall(r"[a-z]+", text.lower())


# --- embeddings: text -> a list of numbers that is close for texts with close meanings ---
_model = None


def embed(texts: list[str]) -> np.ndarray:
    """One row of numbers per text. Rows are scaled to length 1, so a dot product is a cosine."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer   # slow import, only when needed
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"embedding model: {EMBED_MODEL}, max tokens {_model.max_seq_length}, "
              f"chunker={CHUNKER}, chunk={CHUNK_CHARS} chars", flush=True)
    return _model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


class ContractIndex:
    """Chunks of one contract plus a BM25 index over them."""

    def __init__(self, text: str, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else make_chunks(text)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in order[:k]]

    def retrieve_for_clause(self, clause: str, k: int = 3) -> list[Chunk]:
        return self.search(QUERIES[clause], k)


class VectorIndex:
    """Chunks of one contract plus one embedding per chunk. Embeddings are cached on disk."""

    def __init__(self, text: str, cache_key: str | None = None, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else make_chunks(text)
        tag = f"{EMBED_MODEL.split('/')[-1]}_{CHUNKER}_c{CHUNK_CHARS}"   # every setting that changes the numbers
        cache_file = CACHE_DIR / f"{cache_key}_{tag}.npy" if cache_key else None
        if cache_file and cache_file.exists():
            self.vectors = np.load(cache_file)
        else:
            self.vectors = embed([c.text for c in self.chunks])
            if cache_file:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, self.vectors)

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        q = embed([query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)
        return [self.chunks[i] for i in order[:k]]

    def retrieve_for_clause(self, clause: str, k: int = 3) -> list[Chunk]:
        return self.search(DESCRIPTIONS[clause], k)


class HybridIndex:
    """Both searches over the same chunks, combined either by merging or by fusing."""

    def __init__(self, text: str, cache_key: str | None = None, mode: str = "fuse"):
        self.chunks = make_chunks(text)                      # one chunk list, shared by both
        self.bm25 = ContractIndex(text, chunks=self.chunks)
        self.vector = VectorIndex(text, cache_key=cache_key, chunks=self.chunks)
        self.mode = mode

    def retrieve_for_clause(self, clause: str, k: int = 3) -> list[Chunk]:
        if self.mode == "merge":
            # Send both lists. Keyword's k come first; vector's add only chunks not already there.
            seen, out = set(), []
            for c in self.bm25.retrieve_for_clause(clause, k) + self.vector.retrieve_for_clause(clause, k):
                if c.id not in seen:
                    seen.add(c.id)
                    out.append(c)
            return out
        # Fuse: take a deeper list from each, give every chunk 1/(60 + rank) from each list, add up, keep top-k.
        depth = max(10, 3 * k)
        scores: dict[int, float] = {}
        for ranked in (self.bm25.retrieve_for_clause(clause, depth),
                       self.vector.retrieve_for_clause(clause, depth)):
            for rank, c in enumerate(ranked, start=1):
                scores[c.id] = scores.get(c.id, 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k]
        return [self.chunks[cid] for cid in best]


def build_index(text: str, method: str, cache_key: str | None = None):
    """One place that knows how to build any kind of index."""
    if method == "bm25":
        return ContractIndex(text)
    if method == "vector":
        return VectorIndex(text, cache_key=cache_key)
    if method in ("merge", "fuse"):
        return HybridIndex(text, cache_key=cache_key, mode=method)
    raise ValueError(f"unknown method: {method} (use bm25, vector, merge or fuse)")