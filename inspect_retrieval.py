"""Debugging one contract: where did each search put the chunk that holds the lawyers' sentence?

The question that decides what to fix: is the right chunk ranked 4th (close, tune the query or k)
or ranked 30th (far, the method or the model is wrong for this text)?

Usage:
    python inspect_retrieval.py EdietsComInc governing_law
"""
import sys

import numpy as np

from data import CLAUSES, highlights, load_cuad
from retriever import DESCRIPTIONS, QUERIES, ContractIndex, VectorIndex, embed, tokenize

NAME = sys.argv[1] if len(sys.argv) > 1 else "EdietsComInc"
CLAUSE = sys.argv[2] if len(sys.argv) > 2 else "governing_law"


def normalize(text) -> str:
    return " ".join(str(text).split())


df = load_cuad()
row = df[df["Filename"].str.contains(NAME)].iloc[0]
text = row["txt_path"].read_text(errors="ignore")
text_col, _answer_col = CLAUSES[CLAUSE]
probes = [normalize(h)[:150] for h in highlights(row[text_col])]
print(f"{row['Filename'][:70]}\nclause={CLAUSE}: {len(probes)} highlighted sentence(s)\n")

# --- vector search: score every chunk against the description, then rank ---
vi = VectorIndex(text)
q = embed([DESCRIPTIONS[CLAUSE]])[0]
scores = vi.vectors @ q
order = np.argsort(-scores)
rank_of = {int(i): r + 1 for r, i in enumerate(order)}

print(f"VECTOR ({len(vi.chunks)} chunks) top 5:")
for i in order[:5]:
    print(f"  rank {rank_of[int(i)]:>2}  score {scores[i]:.3f}  {normalize(vi.chunks[i].text)[:90]}...")

print("\nVECTOR rank of the chunk(s) holding the lawyers' sentence:")
for p in probes:
    holders = [c.id for c in vi.chunks if p in normalize(c.text)]
    for cid in holders:
        print(f"  chunk {cid:>3}: rank {rank_of[cid]:>2} of {len(vi.chunks)}  score {scores[cid]:.3f}   '{p[:60]}...'")
    if not holders:
        print(f"  not inside any chunk: '{p[:60]}...'")

# --- keyword search on the same contract, for comparison (its chunks are 1,500 chars, so ids differ) ---
bi = ContractIndex(text)
bscores = bi.bm25.get_scores(tokenize(QUERIES[CLAUSE]))
border = np.argsort(-bscores)
brank = {int(i): r + 1 for r, i in enumerate(border)}

print(f"\nBM25 ({len(bi.chunks)} chunks) rank of the chunk(s) holding the lawyers' sentence:")
for p in probes:
    for c in bi.chunks:
        if p in normalize(c.text):
            print(f"  chunk {c.id:>3}: rank {brank[c.id]:>2} of {len(bi.chunks)}  score {bscores[c.id]:.2f}")