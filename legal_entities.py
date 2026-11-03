"""Deterministic legal entity + relation extraction.

LLM triple extraction (the approach in AYON-ARYAN/graph-rag) is accurate but
costs one generation per chunk. On this corpus that is 121k generations, which
is days of local inference. Legal text, however, cites its entities in highly
regular forms -- "Section 138 of the Negotiable Instruments Act", "Article 21",
"X vs Y", "High Court of Delhi" -- so a large, high-precision slice of the
graph can be extracted by pattern with no model at all.

This module is the fast path, used for the whole corpus. The LLM extractor in
`graph_rag.py` is the deep path, used for uploads and any subset worth the
inference cost. Both emit the same triple shape, so they populate one graph.

Triple shape matches the upstream project:
    {head, head_type, relation, tail, tail_type}
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

# --- known statutes, with the short forms Indian legal writing actually uses
ACTS: Dict[str, List[str]] = {
    "Indian Penal Code": [r"Indian Penal Code", r"\bIPC\b", r"\bI\.P\.C\.?"],
    "Code of Criminal Procedure": [r"Code of Criminal Procedure", r"\bCrPC\b",
                                   r"\bCr\.?P\.?C\.?"],
    "Code of Civil Procedure": [r"Code of Civil Procedure", r"\bCPC\b"],
    "Constitution of India": [r"Constitution of India", r"\bConstitution\b"],
    "Negotiable Instruments Act": [r"Negotiable Instruments Act", r"\bNI Act\b",
                                   r"\bN\.I\.? Act\b"],
    "Narcotic Drugs and Psychotropic Substances Act":
        [r"Narcotic Drugs and Psychotropic Substances Act", r"\bNDPS Act\b",
         r"\bNDPS\b"],
    "Indian Evidence Act": [r"Indian Evidence Act", r"\bEvidence Act\b"],
    "Income Tax Act": [r"Income[- ]Tax Act", r"\bIT Act\b"],
    "Customs Act": [r"Customs Act"],
    "Companies Act": [r"Companies Act"],
    "Arbitration and Conciliation Act": [r"Arbitration and Conciliation Act"],
    "Goods and Services Tax Act": [r"\bGST Act\b", r"Goods and Services Tax Act"],
    "Prevention of Money Laundering Act": [r"Prevention of Money[- ]Laundering Act",
                                           r"\bPMLA\b"],
    "Contract Act": [r"Indian Contract Act", r"\bContract Act\b"],
    "Specific Relief Act": [r"Specific Relief Act"],
    "Transfer of Property Act": [r"Transfer of Property Act"],
    "Hindu Marriage Act": [r"Hindu Marriage Act"],
    "Motor Vehicles Act": [r"Motor Vehicles Act"],
    "Consumer Protection Act": [r"Consumer Protection Act"],
    "Information Technology Act": [r"Information Technology Act", r"\bIT Act, 2000\b"],
    # Goa, Daman and Diu inherited Portuguese civil law; judgments there cite
    # Civil Code articles in the thousands, which are NOT Constitution articles.
    "Portuguese Civil Code": [r"Portuguese Civil Code", r"\bCivil Code\b"],
}

_ACT_RE = {
    canonical: re.compile("|".join(pats), re.IGNORECASE)
    for canonical, pats in ACTS.items()
}

# "Section 138 of the Negotiable Instruments Act" / "u/s 420 IPC" / "Sec. 37 NDPS"
_SECTION_OF_ACT = re.compile(
    r"(?:[Ss]ection|[Ss]ec\.?|u/s|U/[Ss])\s*"
    r"(\d+[A-Z]{0,2}(?:\s*\(\s*\d+\s*\))?)"
    r"(?:\s*(?:of|,)?\s*(?:the\s+)?)?"
    r"([A-Z][A-Za-z.\s&]{2,60}?(?:Code|Act|C\.?P\.?C\.?|I\.?P\.?C\.?|NDPS|NI Act))?",
)

# "Article 21", "Articles 14 and 19"
_ARTICLE = re.compile(r"\bArticle[s]?\s+(\d+[A-Z]?)\b")

# "Solomon Ogbe vs Narcotics Control Bureau"
# A party name is a run of capitalised words. Initials like "K.A." survive,
# but a period followed by space+capital ends the name, so
# "New Delhi. Solomon Ogbe" yields "Solomon Ogbe", not the location.
_PARTY = (r"(?:[A-Z][A-Za-z&'\-]*\.?"
          r"(?:\s+(?!vs?\.?\s|versus\s)[A-Z][A-Za-z&'\-]*\.?){0,7})")
_CASE = re.compile(rf"({_PARTY})\s+(?:vs?\.?|versus)\s+({_PARTY})")

_COURT = re.compile(
    r"\b((?:Hon'?ble\s+)?(?:Supreme Court(?:\s+of India)?"
    r"|High Court(?:\s+(?:of|at)\s+[A-Z][a-z]+)?"
    r"|District Court|Sessions Court|Family Court"
    r"|National Company Law Tribunal|NCLT|NCLAT"
    r"|Income Tax Appellate Tribunal|ITAT"
    r"|Consumer Disputes Redressal Commission))\b",
)

_NOISE = {"the", "and", "of", "in", "to", "it", "that", "this", "section",
          "article", "act", "code", "court", "state", "union", "no", "shall",
          "advs", "adv", "advocate", "advocates", "through", "mr", "ms", "sh",
          "respondent", "petitioner", "appellant", "applicant", "versus",
          "order", "judgment", "reserved", "pronounced", "coram", "bench",
          "date", "dated", "crl", "writ", "appeal", "bail", "appln", "ors",
          "anr", "others", "another"}

# Cause titles carry docket furniture: "....Respondent", "Through: Mr X, Advs",
# and a trailing " on 30 May, 2025" from the header or filename. All of that
# has to come off, or it becomes entity names.
_ROLE_TAIL = re.compile(
    r"[\s.]*\.{2,}\s*(?:Respondent|Petitioner|Appellant|Applicant)s?\b.*$",
    re.IGNORECASE)
_DATE_TAIL = re.compile(
    r"\s+on\s+\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+,?\s*\d{4}\s*$",
    re.IGNORECASE)
_TRAILING_ON = re.compile(r"\s+on\s*$", re.IGNORECASE)
# "NA" is pandas/CSV null-marker residue that Tika renders inline, so it ends
# up glued to the front of party names ("NA Solomon Ogbe"). NaN/None likewise.
_LEAD_ROLE = re.compile(
    r"^(?:Hon'?ble|the|Mr|Ms|Mrs|Sh|Shri|Smt|Dr|NA|NaN|None|Unnamed)\b\.?\s+")
_STRIP_CHARS = " .,;:-\u2013\u2014"


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "")
    s = _ROLE_TAIL.sub("", s)
    s = _DATE_TAIL.sub("", s)
    s = _TRAILING_ON.sub("", s)
    s = s.strip(_STRIP_CHARS)
    s = _LEAD_ROLE.sub("", s)
    # Drop a leading sentence/location fragment, without splitting initials.
    s = re.sub(r"^.*?(?<![A-Z])\.\s+(?=[A-Z])", "", s)
    return s.strip(_STRIP_CHARS)


def _ok(s: str) -> bool:
    if not s or len(s) < 3:
        return False
    if s.lower() in _NOISE:
        return False
    if not re.search(r"[A-Za-z]{3}", s):
        return False
    words = [w.strip(".") for w in re.split(r"\s+", s.lower()) if w]
    return not all(w in _NOISE for w in words)


def _canonical_act(raw: str | None, context: str) -> str | None:
    """Resolve an act mention (or the surrounding text) to a canonical name."""
    for probe in (raw, context):
        if not probe:
            continue
        for canonical, rx in _ACT_RE.items():
            if rx.search(probe):
                return canonical
    return None


# A Q&A file about one enactment cites "section 412" with no act beside it.
# The sentence cannot resolve that, but the file it came from can.
SOURCE_ACT_HINTS = {
    "ipc_qa": "Indian Penal Code",
    "crpc_qa": "Code of Criminal Procedure",
    "constitution_qa": "Constitution of India",
}

_BARE_SECTION = re.compile(r"(?:[Ss]ection|[Ss]ec\.?|u/s)\s*(\d+[A-Z]{0,2})")


def act_hint_for(source: str | None) -> str | None:
    """Which enactment a source file is about, if its name says so."""
    if not source:
        return None
    low = source.lower()
    return next((act for key, act in SOURCE_ACT_HINTS.items() if key in low), None)


def extract_triples(text: str, source: str | None = None) -> List[Dict[str, str]]:
    """Extract legal triples from one chunk. Deterministic, no LLM.

    `source` supplies the enactment for bare section references, which is the
    only way a Q&A record saying "under section 412" can be placed.
    """
    triples: List[Dict[str, str]] = []
    seen: set[tuple] = set()

    def add(head, head_type, relation, tail, tail_type):
        head, tail = _clean(head), _clean(tail)
        if not (_ok(head) and _ok(tail)) or head.lower() == tail.lower():
            return
        key = (head.lower(), relation, tail.lower())
        if key in seen:
            return
        seen.add(key)
        triples.append({"head": head, "head_type": head_type,
                        "relation": relation, "tail": tail,
                        "tail_type": tail_type})

    # --- courts
    courts = [_clean(m.group(1)) for m in _COURT.finditer(text)]

    # --- cases, and their parties
    cases: List[str] = []
    for m in _CASE.finditer(text):
        a, b = _clean(m.group(1)), _clean(m.group(2))
        if not (_ok(a) and _ok(b)) or len(a) > 60 or len(b) > 60:
            continue
        case = f"{a} vs {b}"
        cases.append(case)
        add(a, "Party", "party_to", case, "Case")
        add(b, "Party", "party_to", case, "Case")
        for court in courts[:1]:
            add(case, "Case", "decided_by", court, "Court")

    # --- statute sections -> acts
    for m in _SECTION_OF_ACT.finditer(text):
        num = _clean(m.group(1))
        if not num:
            continue
        window = text[max(0, m.start() - 60): m.end() + 80]
        act = _canonical_act(m.group(2), window)
        if not act:
            continue
        section = f"Section {num} {act}"
        add(section, "Statute", "part_of", act, "Act")
        for case in cases[:1]:
            add(case, "Case", "invokes", section, "Statute")
        for court in courts[:1]:
            add(court, "Court", "interpreted", section, "Statute")

    # --- bare sections, resolved by the source file's subject
    hint = act_hint_for(source)
    if hint:
        for m in _BARE_SECTION.finditer(text):
            section = f"Section {m.group(1)} {hint}"
            add(section, "Statute", "part_of", hint, "Act")
            for case in cases[:1]:
                add(case, "Case", "invokes", section, "Statute")

    # --- articles. Which instrument an article belongs to must come from
    # context, not assumption: "Article 1969 of the Civil Code" is a real
    # Portuguese Civil Code provision, not a Constitution article (the
    # Constitution of India stops at 395). Guessing here silently mislabels
    # every Goa civil matter.
    for m in _ARTICLE.finditer(text):
        num_raw = m.group(1)
        art_num = re.match(r"\d+", num_raw)
        window = text[m.end(): m.end() + 70]
        parent = _canonical_act(None, window)
        if parent is None:
            # No instrument named nearby: default to the Constitution only if
            # the number could plausibly be one of its articles.
            n = int(art_num.group()) if art_num else 0
            parent = "Constitution of India" if 1 <= n <= 395 else None
        art = f"Article {num_raw}" + (f" {parent}" if parent and
                                      parent != "Constitution of India" else "")
        if parent:
            add(art, "Statute", "part_of", parent, "Act")
        for case in cases[:1]:
            add(case, "Case", "invokes", art, "Statute")

    # Upstream-style "everything -> one Concept hub" edges are deliberately
    # NOT emitted. A single node every act points at makes all acts two hops
    # apart, so BFS traversal from any seed drags in the whole graph and
    # crowds out the relations that actually answer the question. Acts already
    # enter the graph through their sections and the cases that invoke them.

    return triples


def extract_many(chunks: Iterable[str], source: str | None = None
                 ) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in chunks:
        out.extend(extract_triples(c, source=source))
    return out
