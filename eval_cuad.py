"""Eval: run the extractor over labeled CUAD contracts and score it against the lawyers.

This file knows nothing about prompts or models. It imports extract() from extractor.py,
so the service and the eval can never disagree about what the model was asked.

Usage:
    python eval_cuad.py            # small mode: balanced 12 contracts under 40 KB
    python eval_cuad.py large      # large mode: balanced 10 contracts of 100-190 KB
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

from extractor import extract

# --- settings ---
MODE = sys.argv[1] if len(sys.argv) > 1 else "small"   # sys.argv is process.argv
RUN = "run6"                                            # bump this every run

DATA = Path("data/CUAD_v1")
CSV = DATA / "master_clauses.csv"
TXT_DIR = DATA / "full_contract_txt"
OUT = Path(f"results_{MODE}_{RUN}.jsonl")   # one file per run and mode; keep them all
SEED = 42                                   # fixed seed: the same contracts every run
PAUSE_SECONDS = 12                           # breathing room between calls; rate limits are real

GOV_COL = "Governing Law-Answer"
CAP_COL = "Cap On Liability-Answer"
NC_COL = "Non-Compete-Answer"


# --- grading helpers ---
def normalize(text) -> str:
    return " ".join(str(text).split())


def label_yes(cell) -> bool:
    """Turn the lawyers' 'Yes'/'No' into True/False."""
    return str(cell).strip().lower() == "yes"


def gov_law_correct(label, found: bool, value: str | None) -> bool:
    """Lenient match: any part of the label ('England, United Kingdom') inside the model's value."""
    if pd.isna(label) or not str(label).strip():
        return not found                             # nothing labeled: model should say absent
    if not found:
        return False
    value = normalize(value or "").lower()
    parts = [p.strip().lower() for chunk in str(label).split(";") for p in chunk.split(",")]
    return any(p and p in value for p in parts)


# --- choosing contracts: by labels, with size as a filter ---
def choose_contracts(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a balanced sample (half with a cap, half without) from a size band."""
    pool = df.dropna(subset=["size"])
    if mode == "small":
        pool = pool[pool["size"] < 40_000]
        per_group = 6
    elif mode == "large":
        pool = pool[(pool["size"] > 100_000) & (pool["size"] < 190_000)]   # under the API's 200k-char cap
        per_group = 5
    else:
        raise SystemExit(f"unknown mode: {mode} (use 'small' or 'large')")
    cap_yes = pool[pool[CAP_COL].str.strip().str.lower() == "yes"]
    cap_no = pool[pool[CAP_COL].str.strip().str.lower() == "no"]
    return pd.concat([cap_yes.sample(per_group, random_state=SEED),
                      cap_no.sample(per_group, random_state=SEED)])


# --- load the answer key and choose ---
df = pd.read_csv(CSV)
df["txt_path"] = df["Filename"].map(lambda f: TXT_DIR / (Path(f).stem + ".txt"))
df["size"] = df["txt_path"].map(lambda p: p.stat().st_size if p.exists() else None)
sample = choose_contracts(df, MODE)
print(f"mode={MODE}  {len(sample)} contracts selected:\n")
print(sample[["Filename", "size", GOV_COL, CAP_COL, NC_COL]].to_string())
print()

# --- run the eval, writing each result the moment it exists ---
results = []
failures = []
out_file = OUT.open("a")             # "a" = append: an interrupted run keeps what finished

for _, row in sample.iterrows():
    contract = row["txt_path"].read_text(errors="ignore")
    try:
        res = extract(contract)
    except Exception as e:
        print(f"FAILED {row['Filename'][:50]}: {e}")
        failures.append(row["Filename"])
        time.sleep(PAUSE_SECONDS)
        continue

    clauses = [res.governing_law, res.cap_on_liability, res.non_compete]
    found = [c for c in clauses if c.found]

    r = {
        "file": row["Filename"],
        "size_bytes": int(row["size"]),
        "input_tokens": res.input_tokens,
        "seconds": res.seconds,
        "gov_correct": gov_law_correct(row[GOV_COL], res.governing_law.found, res.governing_law.value),
        "cap_correct": res.cap_on_liability.found == label_yes(row[CAP_COL]),
        "nc_correct": res.non_compete.found == label_yes(row[NC_COL]),
        "found_count": len(found),
        "evidence_verified": sum(c.evidence_verified for c in found),
        "labels": {"gov": row[GOV_COL], "cap": row[CAP_COL], "nc": row[NC_COL]},
        "model_output": res.model_dump(),
    }
    results.append(r)
    out_file.write(json.dumps(r, default=str) + "\n")
    out_file.flush()
    print(f"{r['file'][:42]:42} gov={r['gov_correct']!s:5} cap={r['cap_correct']!s:5} "
          f"nc={r['nc_correct']!s:5} ev={r['evidence_verified']}/{r['found_count']} "
          f"{r['input_tokens']:>7} tok {r['seconds']:>6}s")
    time.sleep(PAUSE_SECONDS)

out_file.close()

# --- score: the denominator is every contract we chose, failures count as wrong ---
n_total = len(sample)
print(f"\nSCORE mode={MODE} over {n_total} contracts ({len(failures)} failed calls counted as wrong)")
for key in ["gov_correct", "cap_correct", "nc_correct"]:
    print(f"  {key:12} {sum(r[key] for r in results):>2}/{n_total}")
found_total = sum(r["found_count"] for r in results)
verified_total = sum(r["evidence_verified"] for r in results)
print(f"  evidence     {verified_total}/{found_total} of found clauses verified")
if results:
    avg_tokens = sum(r["input_tokens"] for r in results) / len(results)
    avg_seconds = sum(r["seconds"] for r in results) / len(results)
    print(f"  avg input tokens {avg_tokens:,.0f}   avg seconds {avg_seconds:.1f}")