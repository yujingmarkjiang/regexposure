"""Prompts and output schemas for the three LLM-guided steps (all run on GPT-5.6 Terra).

Each prompt has a version hash that is stored with every result, so any output can be
matched to the exact instructions that produced it.
"""
from .textutil import short_hash

# =========================================================================================
# 1. Locating and cleaning Item 1 (Business) and Item 1A (Risk Factors)
# =========================================================================================
EXTRACT_INSTRUCTIONS = """\
You locate two sections in a U.S. SEC Form 10-K annual report: Item 1 (Business) and
Item 1A (Risk Factors).

You receive (a) HEADING CANDIDATES: numbered lines from the filing that may be headings,
as "line_index<TAB>text", and (b) REPEATED LINES: short lines that occur several times,
as "count<TAB>text".

For each section return:
- status: "present" if the section's text is in this filing; "not_applicable" if the filing
  only states that the item is not applicable or not required (for example, a smaller
  reporting company omitting Item 1A); "incorporated_by_reference" if the filing says the
  section's content is incorporated from another document; "absent" if the item does not
  appear at all (Item 1A is not required for fiscal years ending before December 1, 2005).
- start_line: the line index of the heading where the section itself begins in the body
  of the report. Never choose a table-of-contents entry or a cross-reference such as
  "see Item 1A". For "not_applicable", point to the heading followed by that statement.
- end_line: the line index of the heading of the next item (usually Item 1A for Item 1,
  and Item 1B or Item 2 for Item 1A). end_line must be greater than start_line.
Use -1 for start_line and end_line when the status is "absent" or "incorporated_by_reference".

Also return "boilerplate": the REPEATED LINES that are running page headers, page footers,
page numbers or table-of-contents links, copied exactly. Return an empty list if none.
"""

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["present", "not_applicable", "incorporated_by_reference", "absent"]},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
    },
    "required": ["status", "start_line", "end_line"],
    "additionalProperties": False,
}
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "item_1": SECTION_SCHEMA,
        "item_1a": SECTION_SCHEMA,
        "boilerplate": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["item_1", "item_1a", "boilerplate"],
    "additionalProperties": False,
}
EXTRACT_VERSION = short_hash(EXTRACT_INSTRUCTIONS + str(EXTRACT_SCHEMA))

# =========================================================================================
# 2. Naming the clusters (data-driven domains)
# =========================================================================================
LABEL_INSTRUCTIONS = """\
You name one cluster of documents. The cluster was formed by unsupervised clustering of
text embeddings of U.S. Federal Register rules and 10-K Business and Risk Factors sections.
Read the example texts from the cluster and write:
- label: a short summary label of 2 to 5 words naming the regulatory subject the texts share.
  Use "&" rather than "and".
- description: one sentence (at most 20 words) listing what the cluster covers.
Describe what the texts are about, not their document type. {avoid}
"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}, "description": {"type": "string"}},
    "required": ["label", "description"],
    "additionalProperties": False,
}
LABEL_VERSION = short_hash(LABEL_INSTRUCTIONS + str(LABEL_SCHEMA))

# =========================================================================================
# 3. Rating firms and rules on the 20 domains  (identical instructions for every document)
# =========================================================================================
SCORE_INSTRUCTIONS = """\
You rate one document against a fixed set of {k} regulatory domains. For every domain give
an integer score from 0 to 100 using these bands:
  0       no connection
  1-20    incidental
  21-40   indirect
  41-60   meaningful
  61-80   direct
  81-100  central to the document

Rate only what the document says. If the document gives no evidence for a domain, the score
is 0. Score each domain independently.

If the document is a section of a company's 10-K annual report (Item 1, Business, or
Item 1A, Risk Factors): rate the activities and risks the firm actually describes, not how
often a word appears or what the firm's industry usually faces.

If the document is a Federal Register rule: read its full text and rate what the rule
actually requires or proposes. The issuing agency's name alone is not evidence: an
Environmental Protection Agency rule about office procedures is not an environmental rule.

If the text you receive is one part of a longer document, rate that part.

Domains (key | name: coverage):
{domains}
"""


def domain_keys(k: int) -> list[str]:
    return [f"d{i + 1:02d}" for i in range(k)]


def score_schema(k: int) -> dict:
    keys = domain_keys(k)
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {key: {"type": "integer"} for key in keys},
                "required": keys,
                "additionalProperties": False,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def score_instructions(labels: list[str], descriptions: list[str]) -> str:
    lines = [f"{key} | {lab}: {desc}" for key, lab, desc in zip(domain_keys(len(labels)), labels, descriptions)]
    return SCORE_INSTRUCTIONS.format(k=len(labels), domains="\n".join(lines))


def score_version(instructions: str, k: int) -> str:
    return short_hash(instructions + str(score_schema(k)))
