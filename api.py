"""HTTP layer. Knows nothing about prompts or models; it calls extractor.extract().

Run:  python -m uvicorn api:app --reload                  (real model)
      FAKE_MODEL=1 python -m uvicorn api:app --reload     (canned answer, no quota used)
Page: http://localhost:8000/        (upload a PDF in the browser)
Docs: http://localhost:8000/docs
"""
import os
import time
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from documents import NoTextLayer, page_of_quote, pdf_to_document
from extractor import FAKE, ExtractionResult, extract

MAX_CHARS = 200_000          # whole mode: ~50k tokens. Retrieved mode reads excerpts, so it can take more.
MAX_PDF_BYTES = 10_000_000   # 10 MB upload cap, checked while reading (like burnbox's body cap)
PAGE = Path(__file__).parent / "static" / "index.html"

# Guards for a public demo that spends a real, limited model quota.
DAILY_MODEL_CALLS = int(os.environ.get("DAILY_MODEL_CALLS", "15"))   # stay under the provider's free daily quota
PER_ADDRESS_CALLS = 3                                                # per visitor address...
PER_ADDRESS_WINDOW = 600                                             # ...per 10 minutes

app = FastAPI(title="Contract Analyzer", version="0.4")
log = logging.getLogger("uvicorn.error")
app.mount("/static", StaticFiles(directory=PAGE.parent), name="static")   # the sample contract and its saved result

_today = {"day": "", "calls": 0}
_recent: dict[str, deque] = defaultdict(deque)


def visitor_address(request: Request) -> str:
    """Behind a host's proxy, the visitor's address arrives in X-Forwarded-For. A client can put
    its own value at the front of that header, so this limit is best effort; the daily cap is
    the guard that cannot be dodged."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def spend_one_call(request: Request) -> None:
    """Admit one model call or refuse with 429 before spending anything.
    The counters live in memory, so they reset when the server restarts. Fine for a demo;
    a real service would keep them in a shared store such as Redis."""
    if FAKE:
        return                                      # fake answers cost nothing
    day = time.strftime("%Y-%m-%d", time.gmtime())
    if _today["day"] != day:
        _today.update(day=day, calls=0)
    if _today["calls"] >= DAILY_MODEL_CALLS:
        raise HTTPException(status_code=429, detail="daily demo limit reached")
    now = time.time()
    recent = _recent[visitor_address(request)]
    while recent and now - recent[0] > PER_ADDRESS_WINDOW:
        recent.popleft()
    if len(recent) >= PER_ADDRESS_CALLS:
        raise HTTPException(status_code=429, detail="too many requests from this address")
    recent.append(now)
    _today["calls"] += 1


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    mode: Literal["whole", "retrieved"] = "whole"


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def home():
    """The upload page. Served by the same app, so the page and the API share one address."""
    return FileResponse(PAGE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResult)
def extract_endpoint(req: ExtractRequest, request: Request):
    spend_one_call(request)
    try:
        return extract(req.text, mode=req.mode)
    except Exception as e:
        # The model failed: timeout, rate limit, cut-off JSON. Say so with a 503.
        log.exception("extraction failed")
        raise HTTPException(status_code=503, detail=f"extraction failed: {type(e).__name__}")


@app.post("/extract-pdf", response_model=ExtractionResult)
def extract_pdf_endpoint(request: Request, file: UploadFile = File(...),
                         mode: Literal["whole", "retrieved"] = "retrieved"):
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
        log.exception("extraction failed")
        raise HTTPException(status_code=422, detail=f"could not read the PDF: {type(e).__name__}")
    if mode == "whole" and len(doc.text) > MAX_CHARS:
        raise HTTPException(status_code=422, detail="contract too long for whole mode; use mode=retrieved")

    spend_one_call(request)          # only after the file passed every check: bad files cost nothing
    try:
        res = extract(doc.text, mode=mode)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"extraction failed: {type(e).__name__}")

    for clause in (res.governing_law, res.cap_on_liability, res.non_compete):
        clause.page = page_of_quote(doc, clause.evidence) if clause.found else None
    res.pages = len(doc.pages)
    return res
