"""Is our PDF text good enough? A free check: no model calls.

For dev contracts that have a PDF, extract the text with documents.py and check whether the
lawyers' highlighted sentences can be found in it. A highlight missing from our text is lost
before retrieval or the model ever run, so this number is a ceiling on everything after it.
The same check against CUAD's own .txt files is printed beside it, as the reference.

Usage:
    python eval_pdf.py          # first 50 dev contracts
    python eval_pdf.py 343      # all of them
"""
import sys
import time
import unicodedata

from data import CLAUSES, DATA, highlights, load_cuad
from documents import NoTextLayer, pdf_to_document

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
PROBE_CHARS = 80


def loose(text) -> str:
    """Compare text the forgiving way: ligatures, curly quotes, spacing and case don't count."""
    text = unicodedata.normalize("NFKC", str(text))
    for curly, straight in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        text = text.replace(curly, straight)
    return " ".join(text.split()).lower()


pdfs = {p.name.lower(): p for p in (DATA / "full_contract_pdf").rglob("*") if p.suffix.lower() == ".pdf"}
df = load_cuad()
dev = df[df["split"] == "dev"].head(N)
print(f"{len(pdfs)} PDFs found under full_contract_pdf/ | checking {len(dev)} dev contracts\n")

parsed, scanned, no_pdf, seconds = 0, [], [], 0.0
total = found_pdf = found_txt = 0
misses = []

for _, row in dev.iterrows():
    path = pdfs.get(row["Filename"].lower())
    if path is None:
        no_pdf.append(row["Filename"])
        continue
    start = time.perf_counter()
    try:
        doc = pdf_to_document(path.read_bytes())
    except NoTextLayer:
        scanned.append(row["Filename"])
        continue
    except Exception as e:
        print(f"  could not read {row['Filename'][:50]}: {type(e).__name__}")
        continue
    seconds += time.perf_counter() - start
    parsed += 1
    ours = loose(doc.text)
    reference = loose(row["txt_path"].read_text(errors="ignore"))
    for clause, (text_col, _answer_col) in CLAUSES.items():
        for h in highlights(row[text_col]):
            probe = loose(h.split("<omitted>")[0])[:PROBE_CHARS]
            if not probe:
                continue
            total += 1
            found_pdf += probe in ours
            found_txt += probe in reference
            if probe not in ours:
                misses.append((row["Filename"], clause, h[:100]))

print(f"parsed {parsed} | scanned (refused) {len(scanned)} | no PDF found {len(no_pdf)} | "
      f"avg {seconds / max(parsed, 1):.2f}s per PDF\n")
print(f"lawyers' highlights found in OUR pdf text:   {found_pdf}/{total} = {found_pdf / max(total, 1):.0%}")
print(f"lawyers' highlights found in CUAD's .txt:    {found_txt}/{total} = {found_txt / max(total, 1):.0%}")
print(f"\n{len(misses)} highlights missing from our text. First five:")
for file, clause, h in misses[:5]:
    print(f"  [{clause}] {file[:40]}: {h}...")
if no_pdf:
    print(f"\nno PDF matched these names (first three): {no_pdf[:3]}")