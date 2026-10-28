"""Core of the contract analyzer: schema, prompt, model call, evidence check.

Everything else (the API, the eval) imports from here, so there is exactly one
place where the prompt and the schema live.
"""
import os
import time

from google import genai
from pydantic import BaseModel

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
TIMEOUT_MS = 60_000                 # a hung call must fail, not freeze
MAX_OUTPUT_TOKENS = 4000            # must also cover hidden thinking tokens
FAKE = os.environ.get("FAKE_MODEL") == "1"   # plumbing tests without spending quota


# --- the form the model fills in ---
class Clause(BaseModel):
    found: bool
    value: str | None
    evidence: str | None


class Extraction(BaseModel):
    governing_law: Clause
    cap_on_liability: Clause
    non_compete: Clause


# --- what we return to callers: the same form, plus proof and cost ---
class ClauseResult(Clause):
    evidence_verified: bool         # True only if the quote is really in the contract


class ExtractionResult(BaseModel):
    governing_law: ClauseResult
    cap_on_liability: ClauseResult
    non_compete: ClauseResult
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


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


def normalize(text) -> str:
    """Collapse every run of whitespace into one space so quotes compare fairly."""
    return " ".join(str(text).split())


_client = None


def client() -> genai.Client:
    """Create the API client once and reuse it."""
    global _client
    if _client is None:
        _client = genai.Client(http_options={"timeout": TIMEOUT_MS})
    return _client


def call_model(contract: str) -> tuple[Extraction, int, int]:
    """Send the contract to the model. Returns (filled form, input tokens, output tokens)."""
    if FAKE:
        fake = Extraction(
            governing_law=Clause(
                found=True,
                value="Delaware",
                evidence="This Agreement is governed by the laws of the State of Delaware.",
            ),
            cap_on_liability=Clause(found=False, value=None, evidence=None),
            non_compete=Clause(found=False, value=None, evidence=None),
        )
        return fake, 0, 0

    interaction = client().interactions.create(
        model=MODEL,
        input=PROMPT_TEMPLATE.format(contract=contract),
        response_format={"type": "text", "mime_type": "application/json",
                         "schema": Extraction.model_json_schema()},
        generation_config={"thinking_level": "low", "max_output_tokens": MAX_OUTPUT_TOKENS},
    )
    out = Extraction.model_validate_json(interaction.output_text)
    return out, interaction.usage.total_input_tokens, interaction.usage.total_output_tokens


def extract(contract: str) -> ExtractionResult:
    """The one function everything calls: run the model, verify every quote, report cost."""
    start = time.perf_counter()
    out, tokens_in, tokens_out = call_model(contract)
    seconds = round(time.perf_counter() - start, 2)
    norm_contract = normalize(contract)

    def check(c: Clause) -> ClauseResult:
        verified = bool(c.found and c.evidence and normalize(c.evidence) in norm_contract)
        return ClauseResult(**c.model_dump(), evidence_verified=verified)

    return ExtractionResult(
        governing_law=check(out.governing_law),
        cap_on_liability=check(out.cap_on_liability),
        non_compete=check(out.non_compete),
        model="fake" if FAKE else MODEL,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        seconds=seconds,
    )