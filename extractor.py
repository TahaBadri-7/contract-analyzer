"""Core of the contract analyzer: schema, prompt, model call, evidence check.

Two ways to read a contract:
  - mode="whole":     the full text goes into the prompt. Finds everything; costs every token.
  - mode="retrieved": keyword search picks the best chunks per clause and only those go in,
                      as numbered excerpts. Cheaper and faster; can miss a clause the search missed.
Everything else (schema, honest exit, evidence receipt) is the same in both modes.
"""
import os
import time

from google import genai
from pydantic import BaseModel

from retriever import Chunk, ContractIndex

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
MODE = os.environ.get("MODE", "whole")          # default until the numbers say otherwise
TIMEOUT_MS = 60_000
MAX_OUTPUT_TOKENS = 4000
FAKE = os.environ.get("FAKE_MODEL") == "1"      # plumbing tests without spending quota

# How many chunks to retrieve per clause. From the retrieval eval: governing law is 99% at 5,
# so 3 is plenty; cap and non-compete keep climbing to 8.
K_PER_CLAUSE = {"governing_law": 3, "cap_on_liability": 8, "non_compete": 8}


class Clause(BaseModel):
    found: bool
    value: str | None
    evidence: str | None


class Extraction(BaseModel):
    governing_law: Clause
    cap_on_liability: Clause
    non_compete: Clause


class ClauseResult(Clause):
    evidence_verified: bool         # True only if the quote is really in the text the model was given


class ExtractionResult(BaseModel):
    governing_law: ClauseResult
    cap_on_liability: ClauseResult
    non_compete: ClauseResult
    model: str
    mode: str                       # "whole" or "retrieved"
    chunks_sent: int                # 0 in whole mode
    input_tokens: int
    output_tokens: int
    seconds: float


PROMPT_TEMPLATE = """You are reviewing a commercial contract. Extract the clauses below.
For each: if present, set found=true, put the key fact in value, and copy one exact
sentence from the contract into evidence, word for word. If absent, set found=false
and leave value and evidence null. Do not infer or guess.
{excerpt_note}
governing_law: the state or country whose law governs the agreement. value = that name only.

cap_on_liability: any clause that limits a party's liability in any way: a maximum amount,
a formula, excluding certain types of damages (indirect, consequential, special, punitive),
limiting remedies (e.g. to repair or replacement), or a time limit to bring claims.

non_compete: a clause that stops a party from competing with the other party, selling
competing products, or operating in a certain market or area. An exclusivity clause
(one party agrees to deal only with the other) is NOT a non-compete.

CONTRACT:
{contract}"""

EXCERPT_NOTE = ("\nYou are shown numbered EXCERPTS from the contract, not the whole document. "
                "Only report a clause if it appears in these excerpts; if it does not, set found=false.\n")


def normalize(text) -> str:
    """Collapse every run of whitespace into one space so quotes compare fairly."""
    return " ".join(str(text).split())


def retrieve_context(contract: str) -> tuple[str, list[Chunk]]:
    """Pick the best chunks per clause, pool them, merge neighbours into continuous spans,
    and lay the spans out as numbered excerpts of the original text."""
    index = ContractIndex(contract)
    picked: dict[int, Chunk] = {}
    for clause, k in K_PER_CLAUSE.items():
        for c in index.retrieve_for_clause(clause, k):
            picked[c.id] = c
    chunks = sorted(picked.values(), key=lambda c: c.start)

    spans: list[list[int]] = []                              # [start, end] in the original text
    for c in chunks:
        if spans and c.start <= spans[-1][1]:                # touches or overlaps the previous span
            spans[-1][1] = max(spans[-1][1], c.end)          # extend it
        else:
            spans.append([c.start, c.end])
    text = "\n\n".join(f"[EXCERPT {i + 1}, characters {s}-{e}]\n{contract[s:e]}"
                       for i, (s, e) in enumerate(spans))
    return text, chunks


_client = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(http_options={"timeout": TIMEOUT_MS})
    return _client


def call_model(prompt: str) -> tuple[Extraction, int, int]:
    """Send the prompt to the model. Returns (filled form, input tokens, output tokens)."""
    if FAKE:
        fake = Extraction(
            governing_law=Clause(found=True, value="Delaware",
                                 evidence="This Agreement is governed by the laws of the State of Delaware."),
            cap_on_liability=Clause(found=False, value=None, evidence=None),
            non_compete=Clause(found=False, value=None, evidence=None),
        )
        return fake, 0, 0
    interaction = client().interactions.create(
        model=MODEL,
        input=prompt,
        response_format={"type": "text", "mime_type": "application/json",
                         "schema": Extraction.model_json_schema()},
        generation_config={"thinking_level": "low", "max_output_tokens": MAX_OUTPUT_TOKENS},
    )
    out = Extraction.model_validate_json(interaction.output_text)
    return out, interaction.usage.total_input_tokens, interaction.usage.total_output_tokens


def extract(contract: str, mode: str = MODE) -> ExtractionResult:
    """The one function everything calls: choose what the model reads, run it, verify every quote."""
    if mode == "retrieved":
        shown, chunks = retrieve_context(contract)
        note = EXCERPT_NOTE
    elif mode == "whole":
        shown, chunks, note = contract, [], ""
    else:
        raise ValueError(f"unknown mode: {mode} (use 'whole' or 'retrieved')")

    start = time.perf_counter()
    out, tokens_in, tokens_out = call_model(PROMPT_TEMPLATE.format(excerpt_note=note, contract=shown))
    seconds = round(time.perf_counter() - start, 2)
    norm_shown = normalize(shown)       # the quote must come from what the model was actually given

    def check(c: Clause) -> ClauseResult:
        verified = bool(c.found and c.evidence and normalize(c.evidence) in norm_shown)
        return ClauseResult(**c.model_dump(), evidence_verified=verified)

    return ExtractionResult(
        governing_law=check(out.governing_law),
        cap_on_liability=check(out.cap_on_liability),
        non_compete=check(out.non_compete),
        model="fake" if FAKE else MODEL,
        mode=mode,
        chunks_sent=len(chunks),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        seconds=seconds,
    )