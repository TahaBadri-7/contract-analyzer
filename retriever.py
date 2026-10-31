"""Retrieval, part 1: cut a contract into chunks and find the few that matter for each clause.

Keyword search (BM25) first: it is what Elasticsearch does, it is free, it needs no model,
and legal clauses carry strong keywords. Vector search comes in a later lesson and is
compared against this on the same eval.
"""
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

CHUNK_CHARS = 1500      # about 350-400 tokens
OVERLAP_CHARS = 200     # so a sentence cut at a boundary still appears whole in one chunk

# What to search for, per clause. These are tuned on dev contracts only.
QUERIES = {
    "governing_law": "governing law governed by the laws of the state jurisdiction construed in accordance",
    "cap_on_liability": "limitation of liability liable damages exceed aggregate consequential indirect "
                        "incidental special punitive remedies limited in no event "
                        "sole exclusive remedy entire liability",
    "non_compete": "compete competing competitive non-compete restrict restriction business engage "
                   "solicit territory during the term "
                   "non-competition competition restricted period covenant similar business",
}


@dataclass
class Chunk:
    id: int
    start: int          # character offset in the original text
    end: int
    text: str

# chunk_text: the tearing-into-slips step. 
# Input a contract, output a list of pieces, each remembering where in the original it came from.
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


# ContractIndex builds the BM25 index once per contract and answers search(query, k). 
# BM25 scores a chunk higher when the query’s words appear in it often, and higher still when those words are rare across the document. 
# “Governed” is rare in a contract; “agreement” is everywhere; BM25 knows which one to trust.

# Its like building the back-of-book index over the slips.
class ContractIndex:
    """Chunks of one contract plus a BM25 index over them. Built once, queried many times."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        """The k chunks that best match the query, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in order[:k]]

    def retrieve_for_clause(self, clause: str, k: int = 3) -> list[Chunk]:
        return self.search(QUERIES[clause], k)