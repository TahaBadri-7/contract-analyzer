"""Loading CUAD and splitting it into dev and test.

dev  = contracts you may look at and tune on (prompts, definitions, retrieval queries).
test = contracts you never tune on and only ever score on. The honest number lives here.

Both evals import from here so they agree on the split.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/CUAD_v1")
CSV = DATA / "master_clauses.csv"
TXT_DIR = DATA / "full_contract_txt"
SPLIT_SEED = 7          # different from the sampling seed on purpose
DEV_FRACTION = 0.7

# Our three clause types -> the CSV columns holding the lawyers' highlighted text and answer.
CLAUSES = {
    "governing_law":    ("Governing Law",    "Governing Law-Answer"),
    "cap_on_liability": ("Cap On Liability", "Cap On Liability-Answer"),
    "non_compete":      ("Non-Compete",      "Non-Compete-Answer"),
}

# load_cuad() loads the CSV -> attaches text paths and sizes, and assigns every contract to dev or test with a fixed seed, 70/30. 
# Dev is where you look and tune. 
# Test is where you only ever score. 
# Both evals import this so they can’t disagree
def load_cuad() -> pd.DataFrame:
    """The answer key with a text path, a size, and a dev/test split per contract."""
    df = pd.read_csv(CSV)
    df["txt_path"] = df["Filename"].map(lambda f: TXT_DIR / (Path(f).stem + ".txt"))
    df["size"] = df["txt_path"].map(lambda p: p.stat().st_size if p.exists() else None)
    df = df.dropna(subset=["size"]).reset_index(drop=True)
    rng = np.random.default_rng(SPLIT_SEED)
    df["split"] = np.where(rng.random(len(df)) < DEV_FRACTION, "dev", "test")
    return df

# highlights() parses the lawyers’ highlighted sentences, which the CSV stores as a stringified list like ['...', '...']; 
# ast.literal_eval safely turns that string back into a list.
def highlights(cell) -> list[str]:
    """The lawyers' highlighted sentences for one clause: the CSV stores them as "['...', '...']"."""
    if pd.isna(cell) or not str(cell).strip():
        return []
    try:
        parsed = ast.literal_eval(str(cell))
        return [str(s) for s in parsed] if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [str(cell)]