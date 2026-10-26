import json
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from google import genai

DATA = Path("data/CUAD_v1")
CSV = DATA / "master_clauses.csv"
TXT_DIR = DATA / "full_contract_txt"
N_CONTRACTS = 10
MODEL = "gemini-3.5-flash"

GOV_COL = "Governing Law-Answer"
CAP_COL = "Cap On Liability-Answer"
NC_COL = "Non-Compete-Answer"


class Clause(BaseModel):
    found: bool
    value: str | None
    evidence: str | None


class Extraction(BaseModel):
    governing_law: Clause
    cap_on_liability: Clause
    non_compete: Clause


PROMPT_TEMPLATE = """You are reviewing a commercial contract. Extract the clauses below.
For each: if present, set found=true, put the key fact in value, and copy one exact
sentence from the contract into evidence, word for word. If absent, set found=false
and leave value and evidence null. Do not infer or guess.

governing_law: the state or country whose law governs the agreement. value = that name only.
cap_on_liability: any clause limiting a party's liability to a fixed amount or formula.
non_compete: any clause restricting a party from competing with the other party.

CONTRACT:
{contract}"""


def normalize(text) -> str:
    return " ".join(str(text).split())


def label_yes(cell) -> bool:
    return str(cell).strip().lower() == "yes"


def gov_law_correct(label, clause: Clause) -> bool:
    if pd.isna(label) or not str(label).strip():
        return not clause.found
    if not clause.found:
        return False
    value = normalize(clause.value or "").lower()
    parts = [p.strip().lower() for chunk in str(label).split(";") for p in chunk.split(",")]
    return any(p and p in value for p in parts)


client = genai.Client(http_options={"timeout": 60_000})
df = pd.read_csv(CSV)

# Map each row to its .txt file, then take the shortest N so the first run is fast.
df["txt_path"] = df["Filename"].map(lambda f: TXT_DIR / (Path(f).stem + ".txt"))
df["size"] = df["txt_path"].map(lambda p: p.stat().st_size if p.exists() else None)
sample = df.dropna(subset=["size"]).nsmallest(N_CONTRACTS, "size")
print(f"{len(sample)} contracts selected\n")

results = []
for _, row in sample.iterrows():
    contract = row["txt_path"].read_text(errors="ignore")
    start = time.perf_counter()
    try:
        interaction = client.interactions.create(
            model=MODEL,
            input=PROMPT_TEMPLATE.format(contract=contract),
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": Extraction.model_json_schema()},
            generation_config={"thinking_level": "low", "max_output_tokens": 1000},
        )
        out = Extraction.model_validate_json(interaction.output_text)
    except Exception as e:
        print(f"FAILED {row['Filename'][:50]}: {e}")
        continue
    elapsed = time.perf_counter() - start
    norm_contract = normalize(contract)

    r = {
        "file": row["Filename"],
        "input_tokens": interaction.usage.total_input_tokens,
        "seconds": round(elapsed, 1),
        "gov_correct": gov_law_correct(row[GOV_COL], out.governing_law),
        "cap_correct": out.cap_on_liability.found == label_yes(row[CAP_COL]),
        "nc_correct": out.non_compete.found == label_yes(row[NC_COL]),
        "evidence_ok": all(
            normalize(c.evidence or "") in norm_contract
            for c in [out.governing_law, out.cap_on_liability, out.non_compete]
            if c.found
        ),
        "labels": {"gov": row[GOV_COL], "cap": row[CAP_COL], "nc": row[NC_COL]},
        "model_output": out.model_dump(),
    }
    results.append(r)
    print(f"{r['file'][:42]:42} gov={r['gov_correct']!s:5} cap={r['cap_correct']!s:5} "
          f"nc={r['nc_correct']!s:5} ev={r['evidence_ok']!s:5} {r['input_tokens']:>6} tok {r['seconds']}s")

Path("results.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in results))

n = len(results)
print(f"\nSCORE over {n} contracts")
for key in ["gov_correct", "cap_correct", "nc_correct", "evidence_ok"]:
    print(f"  {key:12} {sum(r[key] for r in results):>2}/{n}")