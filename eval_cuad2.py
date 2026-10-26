import json
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from google import genai

# --- settings ---
DATA = Path("data/CUAD_v1")
CSV = DATA / "master_clauses.csv"
TXT_DIR = DATA / "full_contract_txt"
OUT = Path("results_run4.jsonl")   # one file per run; keep them all
MODEL = "gemini-3.5-flash"
MAX_BYTES = 40_000                  # size is a filter for speed, not the selector
PER_GROUP = 6                       # contracts with a cap, and contracts without
SEED = 42                           # fixed seed: the same contracts every run
PAUSE_SECONDS = 2                   # breathing room between calls; rate limits are real
TIMEOUT_MS = 60_000                 # a hung call must fail, not freeze
MAX_OUTPUT_TOKENS = 4000            # must also cover hidden thinking tokens

GOV_COL = "Governing Law-Answer"
CAP_COL = "Cap On Liability-Answer"
NC_COL = "Non-Compete-Answer"


# --- the form the model fills in ---
class Clause(BaseModel):
    found: bool
    value: str | None
    evidence: str | None


class Extraction(BaseModel):
    governing_law: Clause
    cap_on_liability: Clause
    non_compete: Clause


# --- the prompt (definitions match the CUAD labelers' definitions) ---
PROMPT_TEMPLATE = """You are reviewing a commercial contract. Extract the clauses below.
For each: if present, set found=true, put the key fact in value, and copy one exact
sentence from the contract into evidence, word for word. If absent, set found=false
and leave value and evidence null. Do not infer or guess.

governing_law: the state or country whose law governs the agreement. value = that name only.

cap_on_liability: any clause that limits a party's liability in any way: a maximum amount,
a formula, excluding certain types of damages (indirect, consequential, special, punitive),
limiting remedies (e.g. to repair or replacement), or a time limit to bring claims.

non_compete: a clause that stops a party from competing with the other party, selling
competing products, or operating in a certain market or area. An exclusivity clause
(one party agrees to deal only with the other) is NOT a non-compete.

CONTRACT:
{contract}"""


# --- helpers ---
def normalize(text) -> str:
    """Collapse every run of whitespace into one space so quotes compare fairly."""
    return " ".join(str(text).split())


def label_yes(cell) -> bool:
    """Turn the lawyers' 'Yes'/'No' into True/False."""
    return str(cell).strip().lower() == "yes"


def gov_law_correct(label, clause: Clause) -> bool:
    """Lenient match: any part of the label ('England, United Kingdom') inside the model's value."""
    if pd.isna(label) or not str(label).strip():
        return not clause.found                      # nothing labeled: model should say absent
    if not clause.found:
        return False
    value = normalize(clause.value or "").lower()
    parts = [p.strip().lower() for chunk in str(label).split(";") for p in chunk.split(",")]
    return any(p and p in value for p in parts)


# --- load the answer key ---
client = genai.Client(http_options={"timeout": TIMEOUT_MS})
df = pd.read_csv(CSV)

# --- choose contracts: by labels, not by size ---
df["txt_path"] = df["Filename"].map(lambda f: TXT_DIR / (Path(f).stem + ".txt"))
df["size"] = df["txt_path"].map(lambda p: p.stat().st_size if p.exists() else None)
pool = df.dropna(subset=["size"])
pool = pool[pool["size"] < MAX_BYTES]
cap_yes = pool[pool[CAP_COL].str.strip().str.lower() == "yes"]
cap_no = pool[pool[CAP_COL].str.strip().str.lower() == "no"]
sample = pd.concat([cap_yes.sample(PER_GROUP, random_state=SEED),
                    cap_no.sample(PER_GROUP, random_state=SEED)])
print(f"{len(sample)} contracts selected:\n")
print(sample[["Filename", "size", GOV_COL, CAP_COL, NC_COL]].to_string())
print()

# --- run the eval, writing each result the moment it exists ---
results = []
failures = []
out_file = OUT.open("a")             # "a" = append: an interrupted run keeps what finished

for _, row in sample.iterrows():
    contract = row["txt_path"].read_text(errors="ignore")
    start = time.perf_counter()
    try:
        interaction = client.interactions.create(
            model=MODEL,
            input=PROMPT_TEMPLATE.format(contract=contract),
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": Extraction.model_json_schema()},
            generation_config={"thinking_level": "low", "max_output_tokens": MAX_OUTPUT_TOKENS},
        )
        out = Extraction.model_validate_json(interaction.output_text)
    except Exception as e:
        print(f"FAILED {row['Filename'][:50]}: {e}")
        failures.append(row["Filename"])
        time.sleep(PAUSE_SECONDS)
        continue
    elapsed = time.perf_counter() - start
    norm_contract = normalize(contract)
    found = [c for c in [out.governing_law, out.cap_on_liability, out.non_compete] if c.found]

    r = {
        "file": row["Filename"],
        "input_tokens": interaction.usage.total_input_tokens,
        "seconds": round(elapsed, 1),
        "gov_correct": gov_law_correct(row[GOV_COL], out.governing_law),
        "cap_correct": out.cap_on_liability.found == label_yes(row[CAP_COL]),
        "nc_correct": out.non_compete.found == label_yes(row[NC_COL]),
        "found_count": len(found),
        "evidence_verified": sum(normalize(c.evidence or "") in norm_contract for c in found),
        "labels": {"gov": row[GOV_COL], "cap": row[CAP_COL], "nc": row[NC_COL]},
        "model_output": out.model_dump(),
    }
    results.append(r)
    out_file.write(json.dumps(r, default=str) + "\n")
    out_file.flush()
    print(f"{r['file'][:42]:42} gov={r['gov_correct']!s:5} cap={r['cap_correct']!s:5} "
          f"nc={r['nc_correct']!s:5} ev={r['evidence_verified']}/{r['found_count']} "
          f"{r['input_tokens']:>6} tok {r['seconds']}s")
    time.sleep(PAUSE_SECONDS)

out_file.close()

# --- score: the denominator is every contract we chose, failures count as wrong ---
n_total = len(sample)
print(f"\nSCORE over {n_total} contracts ({len(failures)} failed calls counted as wrong)")
for key in ["gov_correct", "cap_correct", "nc_correct"]:
    print(f"  {key:12} {sum(r[key] for r in results):>2}/{n_total}")
found_total = sum(r["found_count"] for r in results)
verified_total = sum(r["evidence_verified"] for r in results)
print(f"  evidence     {verified_total}/{found_total} of found clauses verified")