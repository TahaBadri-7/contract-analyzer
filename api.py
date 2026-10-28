"""HTTP layer. Knows nothing about prompts or models; it calls extractor.extract().

Run:  uvicorn api:app --reload          (real model)
      FAKE_MODEL=1 uvicorn api:app --reload   (canned answer, no quota used)
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from extractor import ExtractionResult, extract

MAX_CHARS = 200_000   # ~50k tokens. Bigger documents need retrieval (Phase 2), not a bigger prompt.

app = FastAPI(title="Contract Analyzer", version="0.1")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResult)
def extract_endpoint(req: ExtractRequest):
    try:
        return extract(req.text)
    except Exception as e:
        # The model failed: timeout, rate limit, cut-off JSON. Say so with a 503.
        # Never return 200 with empty fields: a caller would read that as "no clauses found".
        raise HTTPException(status_code=503, detail=f"extraction failed: {type(e).__name__}")