import json
from pydantic import BaseModel
from google import genai


class Clause(BaseModel):
    found: bool
    value: str | None        # None is Python's null
    evidence: str | None     # the exact sentence from the contract, word for word


class Extraction(BaseModel):
    liability_cap: Clause
    governing_law: Clause
    termination_notice_days: Clause
    non_compete: Clause


CONTRACT = """
SERVICES AGREEMENT

1. Services. Provider shall deliver the software services described in Exhibit A.

2. Fees. Client shall pay Provider $12,000 per month, invoiced in advance.

3. Term and Termination. This Agreement begins on the Effective Date and continues
for twelve months. Either party may terminate this Agreement for convenience upon
sixty (60) days' written notice to the other party.

4. Limitation of Liability. Except for breaches of confidentiality, neither party's
aggregate liability under this Agreement shall exceed $250,000.

5. Non-Solicitation. During the term and for twelve months after, Client shall not
solicit for employment any Provider personnel who worked on the services.

6. Governing Law. This Agreement is governed by the laws of the State of Delaware.
"""

PROMPT = f"""Extract the requested clauses from the contract below.
For each clause: if present, set found=true, put the key fact in value, and copy the
exact sentence it came from into evidence, word for word. If the clause is absent,
set found=false and leave value and evidence as null. Do not infer or guess.

CONTRACT:
{CONTRACT}"""

client = genai.Client(http_options={"timeout": 120_000})

interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input=PROMPT,
    response_format={
        "type": "text",
        "mime_type": "application/json",
        "schema": Extraction.model_json_schema(),
    },
    generation_config={"thinking_level": "low", "max_output_tokens": 800},
)

result = Extraction.model_validate_json(interaction.output_text)
print(result.model_dump_json(indent=2))

print("\nEvidence check:")
for name, clause in result:
    if clause.found:
        ok = clause.evidence is not None and clause.evidence.strip() in CONTRACT
        print(f"  {name}: {'verified' if ok else 'NOT IN CONTRACT'}")
    else:
        print(f"  {name}: reported absent")