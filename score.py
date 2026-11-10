"""Score any results file without calling the model again.

Running and scoring are two different jobs. Running spends quota and is slow. Scoring is
instant, free, and can be changed any time, because every results file keeps the model's
full output and the lawyers' labels. So new metrics can be computed from old runs.

Per clause this prints four counts and three rates:
  found right  = the clause is there and the model found it
  missed       = the clause is there and the model said "not found"
  false alarm  = the clause is not there and the model reported one
  absent right = the clause is not there and the model said "not found"
  recall       = found right / (found right + missed)        -> how much of what is there did we catch
  precision    = found right / (found right + false alarm)   -> how much of what we reported is real
  accuracy     = (found right + absent right) / all          -> the single number that hides the other two

Usage:
    python score.py results_large_whole_run6.jsonl
    python score.py results_large_whole_run6.jsonl results_large_retrieved_run7.jsonl    # side by side
"""
import json
import math
import sys

CLAUSES = ["governing_law", "cap_on_liability", "non_compete"]
LABEL_KEY = {"governing_law": "gov", "cap_on_liability": "cap", "non_compete": "nc"}


def normalize(text) -> str:
    return " ".join(str(text).split())


def is_blank(label) -> bool:
    """A missing label arrives as NaN (a float) or an empty string."""
    return label is None or (isinstance(label, float) and math.isnan(label)) or not str(label).strip() \
        or str(label).strip().lower() in ("nan", "none")


def clause_present(clause: str, label) -> bool:
    """Did the lawyers say this clause exists? Governing law: any text. Others: 'Yes'."""
    if clause == "governing_law":
        return not is_blank(label)
    return str(label).strip().lower() == "yes"


def gov_value_right(label, value) -> bool:
    """Lenient match: any part of the label inside the model's value."""
    value = normalize(value or "").lower()
    parts = [p.strip().lower() for chunk in str(label).split(";") for p in chunk.split(",")]
    return any(p and p in value for p in parts)


def read_records(path: str) -> list[dict]:
    """Read every JSON object in the file, whatever the layout: one per line (JSONL), or
    pretty-printed across many lines (what an editor's "format document" produces)."""
    text = open(path).read()
    decoder = json.JSONDecoder()
    records, pos = [], 0
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":    # skip blank lines, indentation, stray commas
            pos += 1
        if pos >= len(text):
            return records
        obj, pos = decoder.raw_decode(text, pos)              # read one whole object, wherever it ends
        records.extend(obj if isinstance(obj, list) else [obj])


def score_file(path: str) -> dict:
    rows = read_records(path)
    out = {"file": path, "n": len(rows)}
    for clause in CLAUSES:
        found_right = missed = false_alarm = absent_right = 0
        for r in rows:
            m = r["model_output"][clause]
            label = r["labels"][LABEL_KEY[clause]]
            present = clause_present(clause, label)
            found = bool(m["found"])
            if clause == "governing_law" and present and found and not gov_value_right(label, m["value"]):
                found = False          # found something, but the wrong place: counts as a miss
            if present and found:
                found_right += 1
            elif present:
                missed += 1
            elif found:
                false_alarm += 1
            else:
                absent_right += 1
        n = found_right + missed + false_alarm + absent_right
        out[clause] = {
            "found right": found_right, "missed": missed, "false alarm": false_alarm, "absent right": absent_right,
            "recall": found_right / (found_right + missed) if found_right + missed else None,
            "precision": found_right / (found_right + false_alarm) if found_right + false_alarm else None,
            "accuracy": (found_right + absent_right) / n if n else None,
            "baseline": max(found_right + missed, false_alarm + absent_right) / n if n else None,
        }
    found_total = sum(r["found_count"] for r in rows)
    out["evidence"] = (sum(r["evidence_verified"] for r in rows), found_total)
    out["avg tokens"] = sum(r["input_tokens"] for r in rows) / len(rows) if rows else 0
    out["avg seconds"] = sum(r["seconds"] for r in rows) / len(rows) if rows else 0
    return out


def pct(x) -> str:
    return "   -" if x is None else f"{x:4.0%}"


def print_scores(scores: list[dict]) -> None:
    width = 26
    print(f"{'':20}" + "".join(f"{s['file'][:width]:>{width + 2}}" for s in scores))
    print(f"{'contracts':20}" + "".join(f"{s['n']:>{width + 2}}" for s in scores))
    for clause in CLAUSES:
        print(f"\n{clause}")
        for key in ["found right", "missed", "false alarm", "absent right"]:
            print(f"  {key:18}" + "".join(f"{s[clause][key]:>{width + 2}}" for s in scores))
        for key in ["recall", "precision", "accuracy", "baseline"]:
            print(f"  {key:18}" + "".join(f"{pct(s[clause][key]):>{width + 2}}" for s in scores))
    print(f"\n{'evidence verified':20}" + "".join(f"{s['evidence'][0]}/{s['evidence'][1]}".rjust(width + 2) for s in scores))
    print(f"{'avg input tokens':20}" + "".join(f"{s['avg tokens']:>{width + 2},.0f}" for s in scores))
    print(f"{'avg seconds':20}" + "".join(f"{s['avg seconds']:>{width + 2}.1f}" for s in scores))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python score.py results_a.jsonl [results_b.jsonl ...]")
    print_scores([score_file(p) for p in sys.argv[1:]])