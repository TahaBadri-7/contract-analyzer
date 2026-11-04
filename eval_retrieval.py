"""Retrieval eval: did the chunks we would send the model contain the lawyers' sentence?

No model API calls. Reports recall@k per clause, and the average number of chunks actually
sent at each k (merge sends up to 2k, so its recall must be read against its cost).

Usage:
    python eval_retrieval.py dev bm25
    python eval_retrieval.py dev vector
    python eval_retrieval.py dev merge
    python eval_retrieval.py dev fuse
    python eval_retrieval.py test bm25      # only when you are done tuning
"""
import sys
import time
from pathlib import Path

import retriever
from data import CLAUSES, highlights, load_cuad
from retriever import build_index

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
METHOD = sys.argv[2] if len(sys.argv) > 2 else "bm25"
KS = [3, 5, 8]
PROBE_CHARS = 150       # a hit = the first 150 chars of the highlight appear in a retrieved chunk


def normalize(text) -> str:
    return " ".join(str(text).split())


df = load_cuad()
subset = df[df["split"] == SPLIT]
print(f"split={SPLIT}  method={METHOD}  chunker={retriever.CHUNKER}: {len(subset)} contracts\n")

counts = {c: {k: [0, 0] for k in KS} for c in CLAUSES}      # [hits, total]
sent = {k: [0, 0] for k in KS}                                # [chunks sent, retrievals]
chunks_per_contract = []
misses = []
start = time.perf_counter()

for n_done, (_, row) in enumerate(subset.iterrows(), start=1):
    text = row["txt_path"].read_text(errors="ignore")
    index = build_index(text, METHOD, cache_key=Path(row["Filename"]).stem)
    chunks_per_contract.append(len(index.chunks))
    for clause, (text_col, _answer_col) in CLAUSES.items():
        hs = highlights(row[text_col])
        if not hs:
            continue
        for k in KS:
            retrieved = index.retrieve_for_clause(clause, k)
            sent[k][0] += len(retrieved)
            sent[k][1] += 1
            for h in hs:
                probe = normalize(h)[:PROBE_CHARS]
                if not probe:
                    continue
                hit = any(probe in normalize(c.text) for c in retrieved)
                counts[clause][k][0] += int(hit)
                counts[clause][k][1] += 1
                if k == max(KS) and not hit:
                    misses.append((row["Filename"], clause, h[:120]))
    if n_done % 50 == 0:
        print(f"  {n_done}/{len(subset)} contracts, {time.perf_counter() - start:.0f}s", flush=True)

print(f"\navg chunks per contract: {sum(chunks_per_contract) / len(chunks_per_contract):.0f}")
print(f"total time: {time.perf_counter() - start:.0f}s\n")
print(f"{'clause':18}" + "".join(f"{'recall@' + str(k):>12}" for k in KS) + f"{'n':>8}")
for clause in CLAUSES:
    line = f"{clause:18}"
    for k in KS:
        hits, total = counts[clause][k]
        line += f"{(hits / total if total else 0):>12.0%}"
    line += f"{counts[clause][KS[0]][1]:>8}"
    print(line)
print(f"{'chunks sent':18}" + "".join(f"{(sent[k][0] / sent[k][1] if sent[k][1] else 0):>12.1f}" for k in KS))

print(f"\n{len(misses)} highlights missed at k={max(KS)}. First five:")
for file, clause, h in misses[:5]:
    print(f"  [{clause}] {file[:40]}: {h}...")