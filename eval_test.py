"""The honest number: run the extractor on the test pile and save raw results for score.py.

The test pile (see data.py) was set aside in Lesson 7 and never used for tuning anything.
This is the first time the extractor sees it. Freeze the settings, run once, score once,
and do not change prompts or k values because of what it shows.

Built for the free tier: it stops after MAX_CALLS model calls and skips contracts already
saved in the results file, so running it once a day finishes the sample in a few days.
A job that picks up where it stopped is called resumable; crashes and Ctrl+C lose nothing.

This script only runs and saves. score.py does all the scoring (Lesson 10, part 1).

Usage:
    python eval_test.py                 # retrieved mode, the product setting
    python eval_test.py whole           # whole-document mode, for comparison (about 5x the tokens)
Then, once it says complete:
    python score.py results_test_retrieved.jsonl
"""
import json
import sys
import time
from pathlib import Path

from data import load_cuad
from extractor import extract
from score import read_records

READ_MODE = sys.argv[1] if len(sys.argv) > 1 else "retrieved"
N_CONTRACTS = 40        # a random sample of the test pile; raise it when billing is on
MAX_CALLS = 18          # per run; the free tier allows 20 calls a day
PAUSE_SECONDS = 12      # the free tier allows 5 calls a minute
SAMPLE_SEED = 11        # fixed: the same 40 contracts every time the script runs
OUT = Path(f"results_test_{READ_MODE}.jsonl")

# A random sample, not a balanced one: the honest number should reflect the real mix of
# contracts, the way a user would meet them.
df = load_cuad()
test = df[df["split"] == "test"]
sample = test.sample(min(N_CONTRACTS, len(test)), random_state=SAMPLE_SEED)

done = {r["file"] for r in read_records(str(OUT))} if OUT.exists() else set()
todo = sample[~sample["Filename"].isin(done)]
print(f"test pile {len(test)} contracts | sample {len(sample)} | already saved {len(done)} | "
      f"to do {len(todo)} | this run: up to {MAX_CALLS} calls, mode={READ_MODE}\n")

calls = 0
with OUT.open("a") as out_file:
    for _, row in todo.iterrows():
        if calls >= MAX_CALLS:
            print(f"\nstopped after {MAX_CALLS} calls. Run the same command tomorrow to continue.")
            break
        contract = row["txt_path"].read_text(errors="ignore")
        calls += 1
        try:
            res = extract(contract, mode=READ_MODE)
        except Exception as e:
            print(f"FAILED {row['Filename'][:50]}: {type(e).__name__} (not saved; retried next run)")
            time.sleep(PAUSE_SECONDS)
            continue

        clauses = [res.governing_law, res.cap_on_liability, res.non_compete]
        found = [c for c in clauses if c.found]
        r = {
            "file": row["Filename"],
            "size_bytes": int(row["size"]),
            "mode": res.mode,
            "chunks_sent": res.chunks_sent,
            "input_tokens": res.input_tokens,
            "seconds": res.seconds,
            "found_count": len(found),
            "evidence_verified": sum(c.evidence_verified for c in found),
            "labels": {"gov": row["Governing Law-Answer"],
                       "cap": row["Cap On Liability-Answer"],
                       "nc": row["Non-Compete-Answer"]},
            "model_output": res.model_dump(),
        }
        out_file.write(json.dumps(r, default=str) + "\n")
        out_file.flush()                                  # saved the moment it exists
        done.add(row["Filename"])
        print(f"{len(done):>3}/{len(sample)}  {row['Filename'][:45]:45} {res.chunks_sent:>2} chunks "
              f"{res.input_tokens:>6} tok {res.seconds:>6}s  ev {r['evidence_verified']}/{r['found_count']}")
        time.sleep(PAUSE_SECONDS)

print(f"\n{len(done)} of {len(sample)} sampled test contracts saved in {OUT}.")
if len(done) >= len(sample):
    print(f"complete. Score it:  python score.py {OUT}")