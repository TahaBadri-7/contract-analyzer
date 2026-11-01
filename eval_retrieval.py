"""Retrieval eval: did the chunks we would send the model contain the lawyers' sentence?

No model API calls. The vector method runs a small embedding model locally; the first vector
run embeds every dev contract (a few minutes on a laptop) and caches the result, so later
runs are fast.

The number it reports is recall@k: of all the clauses the lawyers highlighted, what fraction
had their highlighted sentence inside the top-k retrieved chunks. The extractor can never
score higher than this on retrieved text: if the sentence is not sent, the model cannot find it.

Usage:
    python eval_retrieval.py                 # dev split, bm25
    python eval_retrieval.py dev vector      # dev split, vector search
    python eval_retrieval.py test bm25       # only when you are done tuning
"""
import sys
import time
from pathlib import Path

from data import CLAUSES, highlights, load_cuad
from retriever import build_index

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
METHOD = sys.argv[2] if len(sys.argv) > 2 else "bm25"
KS = [3, 5]
PROBE_CHARS = 150       # a hit = the first 150 chars of the highlight appear in a retrieved chunk


def normalize(text) -> str:
    return " ".join(str(text).split())


df = load_cuad()
subset = df[df["split"] == SPLIT]
print(f"split={SPLIT}  method={METHOD}: {len(subset)} contracts\n")

# counts[clause][k] = [hits, total]
counts = {c: {k: [0, 0] for k in KS} for c in CLAUSES}
chunks_per_contract = []
misses = []     # (file, clause, highlight) for the k=max case, to inspect later
start = time.perf_counter()

for n_done, (_, row) in enumerate(subset.iterrows(), start=1):
    text = row["txt_path"].read_text(errors="ignore")
    index = build_index(text, METHOD, cache_key=Path(row["Filename"]).stem)
    chunks_per_contract.append(len(index.chunks))
    for clause, (text_col, _answer_col) in CLAUSES.items():
        for h in highlights(row[text_col]):
            probe = normalize(h)[:PROBE_CHARS]
            if not probe:
                continue
            for k in KS:
                retrieved = index.retrieve_for_clause(clause, k)
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

print(f"\n{len(misses)} highlights missed at k={max(KS)}. First five:")
for file, clause, h in misses[:5]:
    print(f"  [{clause}] {file[:40]}: {h}...")