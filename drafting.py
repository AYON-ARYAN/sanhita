"""Document drafting from the indexed corpus.

The corpus is not only a question-answering source: data/csvs holds 397
clause types with real drafted text (indemnification, arbitration, governing
law, assignment, termination...). That is a clause bank, so a draft can be
assembled from language the corpus actually contains rather than invented by
the model.

A draft is built in three steps:
  1. pick a document type, from the user's words or an explicit choice;
  2. retrieve real clause text for each section that type needs, plus any
     statutory requirements the request mentions;
  3. hand the model a skeleton plus that retrieved language and ask it to
     draft, marking anything it does not know as a [PLACEHOLDER].

The model is told never to invent a citation, and every draft carries a
header saying it is a draft requiring review. This is a research aid, not a
substitute for a lawyer settling the document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import config


@dataclass
class DocType:
    key: str
    name: str
    blurb: str
    sections: List[str]
    clause_queries: List[str] = field(default_factory=list)
    statute_queries: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)


DOC_TYPES: Dict[str, DocType] = {
    "notice_138": DocType(
        key="notice_138", name="Legal notice — cheque dishonour (s.138 NI Act)",
        blurb="Statutory demand notice after a dishonoured cheque.",
        sections=["Heading and parties", "Facts and the transaction",
                  "Particulars of the cheque and its dishonour",
                  "Statutory demand under Section 138", "Consequences of non-payment",
                  "Limitation and service"],
        clause_queries=["notice", "payments", "default"],
        statute_queries=["Section 138 Negotiable Instruments Act dishonour of cheque",
                         "notice demand fifteen days payment cheque",
                         "Section 139 presumption Negotiable Instruments Act"],
        triggers=["138", "cheque", "cheque bounce", "dishonour", "dishonor",
                  "negotiable instrument"]),
    "legal_notice": DocType(
        key="legal_notice", name="Legal notice (general)",
        blurb="Pre-litigation demand notice.",
        sections=["Heading and parties", "Facts", "Breach complained of",
                  "Relief demanded and time to comply",
                  "Consequences of non-compliance"],
        clause_queries=["notice", "remedies", "default", "termination"],
        statute_queries=["legal notice requirements limitation"],
        triggers=["legal notice", "notice to", "demand notice", "pre-litigation"]),
    "nda": DocType(
        key="nda", name="Non-disclosure agreement",
        blurb="Mutual or one-way confidentiality agreement.",
        sections=["Parties and recitals", "Definition of Confidential Information",
                  "Obligations of the receiving party", "Exclusions",
                  "Term and survival", "Return or destruction of materials",
                  "Remedies and injunctive relief", "Governing law and jurisdiction",
                  "Execution"],
        clause_queries=["confidentiality", "definitions", "remedies",
                        "governing-law-jurisdiction", "term", "survival",
                        "injunctive relief"],
        statute_queries=["Indian Contract Act agreement enforceable"],
        triggers=["nda", "non-disclosure", "confidentiality agreement", "secrecy"]),
    "services": DocType(
        key="services", name="Services / consultancy agreement",
        blurb="Agreement for the supply of services.",
        sections=["Parties and recitals", "Scope of services", "Term",
                  "Fees and payment", "Taxes", "Confidentiality",
                  "Intellectual property", "Indemnification",
                  "Limitation of liability", "Termination",
                  "Dispute resolution and arbitration",
                  "Governing law and jurisdiction", "Notices", "Execution"],
        clause_queries=["services", "payments", "fees", "taxes", "confidentiality",
                        "intellectual-property", "indemnification",
                        "limitation-of-liability", "termination", "arbitration",
                        "governing-law-jurisdiction", "notices", "assignment"],
        statute_queries=["Indian Contract Act consideration lawful agreement"],
        triggers=["service agreement", "consultancy", "consulting agreement",
                  "contract for services", "msa"]),
    "employment": DocType(
        key="employment", name="Employment agreement",
        blurb="Offer and terms of employment.",
        sections=["Parties", "Appointment and role", "Term and probation",
                  "Remuneration and benefits", "Duties and place of work",
                  "Confidentiality", "Intellectual property",
                  "Non-solicitation", "Termination and notice period",
                  "Governing law and jurisdiction", "Execution"],
        clause_queries=["employment", "compensation", "benefits", "confidentiality",
                        "intellectual-property", "non-solicitation", "termination",
                        "governing-law-jurisdiction"],
        statute_queries=["employment termination notice period"],
        triggers=["employment agreement", "offer letter", "appointment letter",
                  "employment contract"]),
    "lease": DocType(
        key="lease", name="Lease / rent agreement",
        blurb="Agreement to let immovable property.",
        sections=["Parties and property", "Term and commencement",
                  "Rent, escalation and deposit", "Use of the premises",
                  "Maintenance and repairs", "Utilities and outgoings",
                  "Lessor's and lessee's covenants", "Termination and lock-in",
                  "Dispute resolution", "Governing law", "Execution"],
        clause_queries=["lease", "rent", "security-deposit", "maintenance",
                        "termination", "use", "governing-law-jurisdiction"],
        statute_queries=["Transfer of Property Act lease immovable property"],
        triggers=["lease", "rent agreement", "leave and licence", "tenancy"]),
    "bail": DocType(
        key="bail", name="Bail application (outline)",
        blurb="Outline of an application for bail.",
        sections=["Cause title", "Facts of the prosecution case",
                  "Grounds for bail", "Period already undergone in custody",
                  "Statutory provisions relied on", "Precedent relied on",
                  "Undertakings offered", "Prayer"],
        clause_queries=[],
        statute_queries=["bail Section 37 NDPS Act prolonged incarceration",
                         "Article 21 speedy trial personal liberty",
                         "bail conditions Code of Criminal Procedure"],
        triggers=["bail", "bail application", "anticipatory bail"]),
    "reply_notice": DocType(
        key="reply_notice", name="Reply to a legal notice",
        blurb="Response denying or answering a notice received.",
        sections=["Heading and reference", "Preliminary objections",
                  "Paragraph-wise reply", "Counter-facts",
                  "Denial of liability", "Reservation of rights"],
        clause_queries=["notice", "remedies"],
        statute_queries=["reply legal notice denial liability"],
        triggers=["reply to notice", "reply to legal notice", "rejoinder",
                  "respond to notice"]),
}


def list_types() -> List[Dict[str, str]]:
    return [{"key": d.key, "name": d.name, "blurb": d.blurb,
             "sections": d.sections} for d in DOC_TYPES.values()]


def infer_type(request: str) -> DocType:
    """Pick a document type from the user's words; services agreement is the
    fallback because it is the most structurally general."""
    low = (request or "").lower()
    best, score = None, 0
    for d in DOC_TYPES.values():
        hits = sum(1 for t in d.triggers if t in low)
        if hits > score:
            best, score = d, hits
    return best or DOC_TYPES["services"]


def _trim(text: str, words: int = 110) -> str:
    parts = (text or "").split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def gather_material(doctype: DocType, request: str, retriever,
                    per_query: int = 2) -> Dict:
    """Retrieve real clause language and statutory text for this document."""
    clauses: List[Dict[str, str]] = []
    seen: set[str] = set()

    for q in doctype.clause_queries:
        label = q.replace("-", " ")
        for h in retriever.retrieve(f"{label} clause", top_k=per_query):
            key = h.text[:90]
            if key in seen:
                continue
            seen.add(key)
            clauses.append({"topic": label,
                            "source": h.metadata.get("source_file", "?"),
                            "text": _trim(h.text)})

    statutes: List[Dict[str, str]] = []
    for q in doctype.statute_queries + [request]:
        for h in retriever.retrieve(q, top_k=per_query):
            key = h.text[:90]
            if key in seen:
                continue
            seen.add(key)
            statutes.append({"source": h.metadata.get("source_file", "?"),
                             "text": _trim(h.text, 90)})

    return {"clauses": clauses[:18], "statutes": statutes[:8]}


DRAFT_SYSTEM = (
    "You are a careful Indian legal drafter. You draft documents in standard "
    "Indian legal style. You never invent a case citation, a statutory section "
    "number or a fact. Anything you do not know becomes a [PLACEHOLDER]."
)

DRAFT_PROMPT = """Draft the document described below. Write it out in full, as a lawyer would file it.

DOCUMENT TYPE: {name}

WHAT THE USER ASKED FOR:
{request}

SECTIONS THIS DOCUMENT MUST HAVE:
{sections}

CLAUSE LANGUAGE FROM THE CLAUSE BANK (real drafted wording — adapt it):
{clauses}

STATUTORY AND CASE MATERIAL RETRIEVED FOR THIS REQUEST:
{statutes}

HOW TO WRITE IT

Write continuous legal prose under every heading. Full sentences, numbered
clauses, the register of a real document. A section that contains only
placeholders is a failed section.

Square brackets are ONLY for facts that are specific to these parties and
genuinely unknown to you:
    [NAME OF ADDRESSEE]  [ADDRESS]  [DATE]  [CHEQUE NO.]  [BANK AND BRANCH]
Everything else you write out properly — the statutory language, the demand,
the consequences of non-compliance, the limitation position, the covenants.
Those are standard and you know them from the material above.

Use any fact the user did give you. If they state an amount, a period or a
provision, put it in the document rather than bracketing it.

Here is the register expected. Note that only the party-specific facts are
bracketed:

    ## 4. Statutory demand
    4.1 My client hereby calls upon you to pay the said sum of Rs. 4,50,000/-
        (Rupees Four Lakh Fifty Thousand only) within fifteen (15) days of
        receipt of this notice, as required by the proviso (c) to Section 138
        of the Negotiable Instruments Act, 1881.
    4.2 Payment shall be made by demand draft drawn in favour of
        [NAME OF PAYEE] and delivered at the address stated above.

Other rules:
- Number clauses 1, 2, 3 with sub-clauses 1.1, 1.2 or (a), (b).
- Cite a section or case only if it appears in the material above.
- Markdown: `#` for the document title, `##` for each section heading.
- Open with one italic line noting this is a draft requiring review.

Draft:"""


def build_prompt(doctype: DocType, request: str, material: Dict) -> str:
    clauses = "\n\n".join(
        f"[{c['topic']}] (from {c['source']})\n{c['text']}"
        for c in material["clauses"]) or "No clause-bank material matched."
    statutes = "\n\n".join(
        f"(from {s['source']})\n{s['text']}" for s in material["statutes"]
    ) or "No statutory material matched."
    return DRAFT_PROMPT.format(
        name=doctype.name, request=request.strip(),
        sections="\n".join(f"{i+1}. {x}" for i, x in enumerate(doctype.sections)),
        clauses=clauses, statutes=statutes)
