"""HTTP layer. Knows nothing about prompts or models; it calls extractor.extract().

Run:  python -m uvicorn api:app --reload                  (real model)
      FAKE_MODEL=1 python -m uvicorn api:app --reload     (canned answer, no quota used)
Docs: http://localhost:8000/docs
"""
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from documents import NoTextLayer, page_of_quote, pdf_to_document
from extractor import ExtractionResult, extract

MAX_CHARS = 200_000          # whole mode: ~50k tokens. Retrieved mode reads excerpts, so it can take more.
MAX_PDF_BYTES = 10_000_000   # 10 MB upload cap, checked while reading (like burnbox's body cap)

app = FastAPI(title="Contract Analyzer", version="0.3")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    mode: Literal["whole", "retrieved"] = "whole"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResult)
def extract_endpoint(req: ExtractRequest):
    try:
        return extract(req.text, mode=req.mode)
    except Exception as e:
        # The model failed: timeout, rate limit, cut-off JSON. Say so with a 503.
        raise HTTPException(status_code=503, detail=f"extraction failed: {type(e).__name__}")


@app.post("/extract-pdf", response_model=ExtractionResult)
def extract_pdf_endpoint(file: UploadFile = File(...), mode: Literal["whole", "retrieved"] = "retrieved"):
    # A plain `def` (not async): FastAPI runs it in a worker thread, so the slow model call
    # does not freeze the server for everyone else.
    data = file.file.read(MAX_PDF_BYTES + 1)           # read at most one byte past the cap
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail=f"PDF larger than {MAX_PDF_BYTES // 1_000_000} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="not a PDF file")
    try:
        doc = pdf_to_document(data)
    except NoTextLayer as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"could not read the PDF: {type(e).__name__}")
    if mode == "whole" and len(doc.text) > MAX_CHARS:
        raise HTTPException(status_code=422, detail="contract too long for whole mode; use mode=retrieved")

    try:
        res = extract(doc.text, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"extraction failed: {type(e).__name__}")

    for clause in (res.governing_law, res.cap_on_liability, res.non_compete):
        clause.page = page_of_quote(doc, clause.evidence) if clause.found else None
    res.pages = len(doc.pages)
    return res