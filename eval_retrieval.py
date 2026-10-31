"""Retrieval eval: did the chunks we would send the model contain the lawyers' sentence?

No model calls. Runs on every dev contract in seconds, so it can be run after every change.

The number it reports is recall@k: of all the clauses the lawyers highlighted, what fraction
had their highlighted sentence inside the top-k retrieved chunks. The extractor can never
score higher than this on retrieved text: if the sentence is not sent, the model cannot find it.

Usage:
    python eval_retrieval.py            # dev split, k=3 and k=5
    python eval_retrieval.py test       # only when you are done tuning
"""

# eval_retrieval.py: the ruler. 
# 1. For every dev contract and every clause the lawyers highlighted, it asks: do the top-k 
#    chunks contain the first 150 characters of the highlighted sentence? 
# 2. The fraction that do is recall@k. It prints recall@3 and recall@5 per clause, and the 
#    first five misses so you can look at them. No model, no quota, runs in seconds.


# It is the test for the search engine. 
# For every contract in the dev pile, and for every clause the lawyers highlighted in it: 
# tear into slips -> build the index -> search with that clause’s phrase, take the top 3 -> and check one thing: 
# does the lawyers’ highlighted sentence appear inside any of those 3 slips? Yes or no. Count the yeses, divide by the total. 
# That percentage is called recall@3: of all the sentences the lawyers marked, what fraction did our top-3 search actually catch. 
# It does the same for top-5. Then it prints the first five misses so you can see what kind of sentence the search failed on.


import sys

from data import CLAUSES, highlights, load_cuad
from retriever import ContractIndex

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
KS = [3, 5]
PROBE_CHARS = 150       # a hit = the first 150 chars of the highlight appear in a retrieved chunk


def normalize(text) -> str:
    return " ".join(str(text).split())


df = load_cuad()
subset = df[df["split"] == SPLIT]
print(f"split={SPLIT}: {len(subset)} contracts\n")

# counts[clause][k] = [hits, total]
counts = {c: {k: [0, 0] for k in KS} for c in CLAUSES}
chunks_per_contract = []
misses = []     # (file, clause, highlight) for the k=max case, to inspect later

for _, row in subset.iterrows():
    text = row["txt_path"].read_text(errors="ignore")
    index = ContractIndex(text)
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

print(f"avg chunks per contract: {sum(chunks_per_contract) / len(chunks_per_contract):.0f}\n")
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