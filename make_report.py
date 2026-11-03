"""Generate the project explainer PDF (docs/Legal_AI_Project_Report.pdf).

Reproducible: reads live counts from the running index and graph so the
numbers in the document are never stale.

    .venv/bin/python make_report.py
"""
from __future__ import annotations

import datetime as dt
import os
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, KeepTogether,
                                ListFlowable, ListItem, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

OUT = "docs/Sanhita_Project_Report.pdf"

OXBLOOD = colors.HexColor("#6E1423")
INK = colors.HexColor("#17181C")
SLATE = colors.HexColor("#5A6472")
PAPER = colors.HexColor("#F5F6F4")
EDGE = colors.HexColor("#DCDDD7")
GOLD = colors.HexColor("#A8812B")
OK = colors.HexColor("#2F6B4F")

# --------------------------------------------------------------- live stats --
def live_stats() -> dict:
    s = {"chunks": "n/a", "nodes": "n/a", "edges": "n/a", "docs": "n/a",
         "types": {}}
    try:
        from retriever import Retriever
        s["chunks"] = f"{Retriever().count():,}"
    except Exception:
        pass
    try:
        from graph_store import GraphStore
        g = GraphStore()
        s["nodes"] = f"{g.graph.number_of_nodes():,}"
        s["edges"] = f"{g.graph.number_of_edges():,}"
        s["docs"] = f"{len(g.documents):,}"
        s["types"] = g.get_stats()["type_counts"]
    except Exception:
        pass
    return s


# -------------------------------------------------------------- flowables ---
class Rule(Flowable):
    """A hairline with the accent colour, used to open a section."""
    def __init__(self, width=None, thickness=2, color=OXBLOOD):
        super().__init__()
        self.width_ = width
        self.thickness = thickness
        self.color = color

    def wrap(self, aw, ah):
        self.w = self.width_ or aw
        return (self.w, self.thickness + 2)

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 1, self.w, 1)


class Architecture(Flowable):
    """Hand-drawn architecture diagram: ingestion, storage, query."""
    H = 182 * mm

    def wrap(self, aw, ah):
        self.w = aw
        return (aw, self.H)

    def _box(self, x, y, w, h, label, sub="", fill=colors.white,
             stroke=EDGE, bold=True, lw=0.8, textcolor=INK):
        c = self.canv
        c.setFillColor(fill); c.setStrokeColor(stroke); c.setLineWidth(lw)
        c.roundRect(x, y, w, h, 2.2 * mm, stroke=1, fill=1)
        c.setFillColor(textcolor)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 7.6)
        c.drawCentredString(x + w / 2, y + h - 4.6 * mm, label)
        if sub:
            c.setFont("Helvetica", 6.4)
            c.setFillColor(SLATE)
            for i, ln in enumerate(sub.split("\n")):
                c.drawCentredString(x + w / 2, y + h - 8.0 * mm - i * 3.0 * mm, ln)

    def _arrow(self, x1, y1, x2, y2, label="", dash=False):
        c = self.canv
        c.setStrokeColor(SLATE); c.setLineWidth(0.8)
        c.setDash(2, 2) if dash else c.setDash()
        c.line(x1, y1, x2, y2)
        c.setDash()
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        for d in (2.6, -2.6):
            c.line(x2, y2,
                   x2 - 2.6 * mm * math.cos(ang - d * 0.1),
                   y2 - 2.6 * mm * math.sin(ang - d * 0.1))
        if label:
            c.setFillColor(SLATE); c.setFont("Helvetica", 6.2)
            c.drawCentredString((x1 + x2) / 2, (y1 + y2) / 2 + 1.4 * mm, label)

    def _band(self, y, h, title):
        c = self.canv
        c.setFillColor(PAPER); c.setStrokeColor(EDGE); c.setLineWidth(0.6)
        c.roundRect(0, y, self.w, h, 2 * mm, stroke=1, fill=1)
        c.setFillColor(OXBLOOD); c.setFont("Helvetica-Bold", 7.2)
        c.drawString(3 * mm, y + h - 5.2 * mm, title)

    def draw(self):
        W = self.w
        colw = (W - 8 * mm) / 4.0

        # ---------------- A: ingestion (offline) ----------------
        self._band(132 * mm, 48 * mm, "A.  INGESTION  (offline, run once)")
        y = 138 * mm
        for i, (lab, sub) in enumerate([
            ("413 source files", "397 CSV - clauses\n4 JSON - 39,150 Q&A\n12 PDF - judgments"),
            ("Apache Tika", "detects type from\nbytes, extracts text\nfrom any format"),
            ("Chunker", "sentence-aware\n500 tokens,\n100 overlap"),
            ("121,112 chunks", "deterministic ids\n-> resumable,\nidempotent"),
        ]):
            x = 2 * mm + i * colw
            self._box(x, y, colw - 4 * mm, 26 * mm, lab, sub)
            if i < 3:
                self._arrow(x + colw - 4 * mm, y + 13 * mm, x + colw, y + 13 * mm)

        # ---------------- B: storage ----------------
        self._band(84 * mm, 44 * mm, "B.  TWO STORES BUILT FROM THE SAME CHUNKS")
        self._box(2 * mm, 90 * mm, (W - 8 * mm) / 2, 30 * mm,
                  "VECTOR STORE  -  Chroma  (+ BM25)", 
                  "nomic-embed-text turns each chunk into a 768-number\n"
                  "vector for meaning; a BM25 index scores word overlap.\n"
                  "Both are searched and their rankings are fused.",
                  stroke=OXBLOOD, lw=1.1)
        self._box(W / 2 + 2 * mm, 90 * mm, (W - 8 * mm) / 2, 30 * mm,
                  "KNOWLEDGE GRAPH  -  NetworkX",
                  "Pattern extraction pulls legal triples from the text:\n"
                  "Case -invokes-> Statute,  Case -decided_by-> Court.\n"
                  "Stores how entities CONNECT, which vectors cannot.",
                  stroke=GOLD, lw=1.1)
        self._arrow(W / 4, 132 * mm, W / 4, 120 * mm, "embed")
        self._arrow(3 * W / 4, 132 * mm, 3 * W / 4, 120 * mm, "extract")

        # ---------------- C: query (online) ----------------
        # Single left-to-right row so no arrow crosses a box. The earlier
        # layout stacked the question above the retrievers and every arrow
        # ran through it.
        self._band(14 * mm, 64 * mm, "C.  ANSWERING A QUESTION  (online, per request)")
        bw = (W - 10 * mm) / 4.0
        by, bh = 48 * mm, 19 * mm
        steps = [
            ("1. Question", '"Does Article 21\napply to NDPS bail?"', INK, 1.0),
            ("2a. Vector search", "embed it, take the\n5 nearest passages", OXBLOOD, 1.0),
            ("2b. Graph traversal", "find entities, walk\n2 hops, read edges", GOLD, 1.0),
            ("3. Build prompt", "passages +\nrelations together", EDGE, 0.8),
        ]
        centres = []
        for i, (lab, sub, stroke, lw) in enumerate(steps):
            x = 2 * mm + i * bw
            self._box(x, by, bw - 4 * mm, bh, lab, sub, stroke=stroke, lw=lw)
            centres.append(x + (bw - 4 * mm) / 2)
            if i < 3:
                self._arrow(x + bw - 4 * mm, by + bh / 2, x + bw, by + bh / 2)

        self._box(W * 0.20, 24 * mm, W * 0.60, 13 * mm,
                  "4.  Qwen 2.5 (Ollama) writes the answer from that evidence",
                  "", stroke=OXBLOOD, lw=1.2)
        self._arrow(centres[3], by, W * 0.70, 37 * mm)

        # dashed: each store feeds the retriever directly below it
        self._arrow(W / 4, 90 * mm, centres[1], by + bh, "reads", dash=True)
        self._arrow(3 * W / 4, 90 * mm, centres[2], by + bh, "reads", dash=True)


# ----------------------------------------------------------------- styles ---
def build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                                fontSize=27, leading=31, textColor=INK,
                                spaceAfter=4)
    s["sub"] = ParagraphStyle("sb", parent=ss["Normal"], fontName="Times-Italic",
                              fontSize=12.5, leading=17, textColor=SLATE,
                              alignment=TA_CENTER, spaceAfter=16)
    s["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                             fontSize=14, leading=18, textColor=OXBLOOD,
                             spaceBefore=4, spaceAfter=7)
    s["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=10.6, leading=14, textColor=INK,
                             spaceBefore=10, spaceAfter=4)
    s["body"] = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman",
                               fontSize=10.4, leading=15.2, textColor=INK,
                               alignment=TA_JUSTIFY, spaceAfter=7)
    s["small"] = ParagraphStyle("sm", parent=s["body"], fontSize=9.1, leading=13,
                                textColor=SLATE)
    s["code"] = ParagraphStyle("c", parent=ss["Code"], fontName="Courier",
                               fontSize=8.1, leading=11, textColor=INK,
                               backColor=PAPER, borderColor=EDGE, borderWidth=0.6,
                               borderPadding=5, spaceBefore=3, spaceAfter=8)
    s["quote"] = ParagraphStyle("q", parent=s["body"], fontName="Times-Italic",
                                leftIndent=10, textColor=SLATE, spaceAfter=8)
    s["cap"] = ParagraphStyle("cp", parent=s["small"], alignment=TA_CENTER,
                              spaceBefore=4)
    return s


def tbl(data, widths, head=True, font=8.3):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, EDGE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    if head:
        cmds += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("TEXTCOLOR", (0, 0), (-1, 0), OXBLOOD),
                 ("LINEBELOW", (0, 0), (-1, 0), 1.0, OXBLOOD)]
    t.setStyle(TableStyle(cmds))
    return t


def bullets(items, st, color=OXBLOOD):
    return ListFlowable([ListItem(Paragraph(i, st), leftIndent=12) for i in items],
                        bulletType="bullet", start="square", bulletColor=color,
                        bulletFontSize=5.5, leftIndent=13)


# -------------------------------------------------------------------- doc ---
def main():
    st = build_styles()
    S = live_stats()
    today = dt.date.today().strftime("%d %B %Y")

    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=18 * mm, bottomMargin=18 * mm,
                          title="Sanhita - Graph RAG Project Report",
                          author="AYON ARYAN")
    fw = doc.width

    def deco(canv, d, first=False):
        canv.saveState()
        canv.setFillColor(OXBLOOD)
        canv.rect(0, A4[1] - 6 * mm, A4[0], 6 * mm, stroke=0, fill=1)
        if not first:
            canv.setFillColor(SLATE); canv.setFont("Helvetica", 7.4)
            canv.drawString(20 * mm, 11 * mm,
                            "Sanhita - Retrieval-Augmented Generation with a Knowledge Graph")
            canv.drawRightString(A4[0] - 20 * mm, 11 * mm, "Page %d" % d.page)
            canv.setStrokeColor(EDGE); canv.setLineWidth(0.5)
            canv.line(20 * mm, 14.5 * mm, A4[0] - 20 * mm, 14.5 * mm)
        canv.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[frame],
                     onPage=lambda c, d: deco(c, d, True)),
        PageTemplate(id="rest", frames=[frame], onPage=deco),
    ])

    E = []
    A = E.append

    # ============================== cover ==============================
    A(Spacer(1, 26 * mm))
    A(Paragraph("Sanhita", st["title"]))
    A(Paragraph("संहिता &mdash; the code, searchable<br/>"
                "A fully local question-answering system for Indian law, built with "
                "Retrieval-Augmented Generation and a legal knowledge graph",
                st["sub"]))
    A(Rule(width=fw, thickness=2.4))
    A(Spacer(1, 7 * mm))
    A(tbl([
        ["What it does", "Answers questions about Indian statutes, judgments and "
                         "contract clauses, and shows the evidence for every answer"],
        ["How it answers", "Retrieval-Augmented Generation (RAG) combined with "
                           "knowledge-graph traversal - a hybrid of two retrievers"],
        ["Where it runs", "Entirely on one laptop. No cloud API, no API key, "
                          "no data leaves the machine"],
        ["Language model", "Qwen 2.5 (3B instruct) served locally by Ollama"],
        ["Embeddings", "nomic-embed-text (768 dimensions), also local via Ollama"],
        ["Text extraction", "Apache Tika 3.3.3"],
        ["Stores", "Chroma (vectors) and NetworkX (knowledge graph)"],
        ["Corpus", "413 files - %s text passages, %s graph entities" % (S["chunks"], S["nodes"])],
        ["Report generated", today],
    ], [34 * mm, fw - 34 * mm], head=False, font=8.6))
    A(Spacer(1, 9 * mm))
    A(Paragraph(
        "<b>How to read this report.</b> Section 1 explains the problem and why "
        "plain language models are not enough. Section 2 explains RAG from first "
        "principles. Section 3 is the architecture diagram. Sections 4 to 6 walk "
        "through the three stages of the system. Section 7 follows one real "
        "question end to end. Section 8 records the engineering decisions and the "
        "measurements behind them, and Section 9 the defects found during testing "
        "and how each was fixed. Section 10 covers how to run it, and Section 11 "
        "the honest limitations.", st["body"]))
    A(NextPageTemplate("rest"))
    A(PageBreak())

    # ======================== 1. the problem ===========================
    A(Paragraph("1.  The problem", st["h1"]))
    A(Paragraph(
        "A language model such as Qwen or GPT has read a great deal of text, but it "
        "stores what it read as statistical patterns in its weights, not as "
        "retrievable documents. This produces three problems that matter a great "
        "deal in law.", st["body"]))
    A(bullets([
        "<b>It invents citations.</b> Asked for the punishment for theft under the "
        "Indian Penal Code, the 3B code-tuned model we first tried answered "
        "\"up to 7 years\". The correct answer is up to three years under Section 379. "
        "A confident wrong citation is worse than no answer.",
        "<b>It cannot cite a source.</b> Even when correct, the model cannot say "
        "which document the answer came from, so a lawyer cannot verify it.",
        "<b>It does not know your documents.</b> It has never seen the contract on "
        "your desk or a judgment delivered last month.",
    ], st["body"]))
    A(Paragraph(
        "Retraining the model on every new judgment is impossibly expensive. The "
        "practical answer is to stop asking the model to remember, and instead give "
        "it the relevant text at the moment the question is asked. That is "
        "Retrieval-Augmented Generation.", st["body"]))

    # ======================== 2. what RAG is ===========================
    A(Paragraph("2.  What Retrieval-Augmented Generation is", st["h1"]))
    A(Paragraph(
        "RAG splits the job in two. A <b>retriever</b> finds text likely to contain "
        "the answer. A <b>generator</b> (the language model) reads only that text and "
        "writes the answer. The model contributes language skill; the corpus "
        "contributes facts.", st["body"]))
    A(Paragraph("2.1  Why embeddings, not keyword search", st["h2"]))
    A(Paragraph(
        "Keyword search fails when the question and the document use different "
        "words for the same idea. A question about \"a cheque bouncing\" should find "
        "a passage about \"dishonour of a negotiable instrument\", which shares no "
        "keyword with it.", st["body"]))
    A(Paragraph(
        "An <b>embedding model</b> converts text into a list of numbers - here 768 of "
        "them - positioned so that passages with similar meaning sit close together. "
        "Finding relevant text becomes measuring distance. We compare using cosine "
        "similarity, the angle between two vectors, which ignores length and "
        "measures direction - that is, meaning rather than verbosity.", st["body"]))
    A(Paragraph(
        "One subtlety this project had to get right: nomic-embed-text is an "
        "<i>asymmetric</i> model. It expects the prefix \"search_document: \" on text "
        "being stored and \"search_query: \" on a question. Ollama does not add these "
        "automatically, and omitting them measurably weakens retrieval. The system "
        "adds them, and a test asserts that a document vector and a query vector of "
        "identical text differ - proving the prefixes are applied.", st["body"]))

    A(Paragraph("2.2  Why a knowledge graph as well", st["h2"]))
    A(Paragraph(
        "Vector search retrieves passages that <i>resemble</i> the question. It cannot "
        "answer questions about <i>connections</i>: which cases cite Article 21, which "
        "court decided a matter, which statute a section belongs to. Those are "
        "relationships, and relationships belong in a graph.", st["body"]))
    A(Paragraph(
        "So this project builds a second index: a directed graph whose nodes are "
        "legal entities (cases, statutes, acts, courts, parties) and whose edges are "
        "relations between them. Asking about Article 21 can then follow edges to "
        "every case that invoked it - something no amount of passage similarity "
        "would produce. Combining both retrievers is called <b>Graph RAG</b>.",
        st["body"]))
    A(Paragraph(
        "The graph currently holds %s entities and %s relations extracted from %s "
        "documents." % (S["nodes"], S["edges"], S["docs"]), st["small"]))
    if S["types"]:
        A(Spacer(1, 2))
        A(tbl([["Entity type", "Count", "Meaning"]] + [
            [k, f"{v:,}", {
                "Party": "a person or organisation named in a case",
                "Case": "a reported matter, e.g. \"X vs Y\"",
                "Statute": "a specific section or article",
                "Act": "a whole enactment, e.g. the NDPS Act",
                "Court": "the forum that decided a matter",
            }.get(k, "")] for k, v in S["types"].items()],
            [30 * mm, 22 * mm, fw - 52 * mm]))

    # ======================== 3. architecture ==========================
    A(Paragraph("3.  Architecture", st["h1"]))
    A(Paragraph(
        "The system has three stages. Stage A runs once and prepares the corpus. "
        "Stage B holds what was prepared. Stage C runs for every question.",
        st["body"]))
    A(Architecture())
    A(Paragraph(
        "Figure 1. Dashed arrows show the two stores being read at query time. "
        "Both retrievers run on every question and their results are merged into a "
        "single prompt.", st["cap"]))
    A(PageBreak())

    # ======================== 4. ingestion =============================
    A(Paragraph("4.  Stage A - preparing the corpus", st["h1"]))
    A(Paragraph("4.1  Reading any file format with Apache Tika", st["h2"]))
    A(Paragraph(
        "The corpus mixes CSV clause banks, JSON question-answer datasets and PDF "
        "judgments. Rather than one reader per format, the system runs an "
        "<b>Apache Tika</b> server and sends it every file. Tika identifies the type "
        "from the file's actual bytes and returns plain text, handling PDF, DOCX, "
        "XLSX, PPTX, HTML, CSV, RTF, ODT, EPUB and more.", st["body"]))
    A(Paragraph(
        "This choice fixed a real defect. The previous per-format code tested "
        "<font face='Courier'>filename.endswith(\".pdf\")</font>, which is "
        "case-sensitive, and ten of the twelve judgment files are named "
        "<font face='Courier'>.PDF</font> in capitals. The entire case-law "
        "collection was being skipped in silence. Because Tika detects type from "
        "content, there is no longer a filename to get wrong.", st["body"]))

    A(Paragraph("4.2  Chunking", st["h2"]))
    A(Paragraph(
        "A whole judgment is too long to put in a prompt, so text is split into "
        "<b>chunks</b> of about 500 tokens with 100 tokens of overlap. Splitting "
        "happens at sentence boundaries, because a chunk cut mid-sentence embeds "
        "poorly. The overlap means a sentence near a boundary still appears whole in "
        "one of the two neighbouring chunks.", st["body"]))

    A(Paragraph("4.3  Embedding, and why ordering turned out to matter", st["h2"]))
    A(Paragraph(
        "Each chunk is sent to Ollama and stored in Chroma with its vector, its text "
        "and its source filename. Chunk identifiers are derived by hashing the source "
        "path plus the record and chunk index, which makes the build "
        "<b>idempotent</b> - re-running it updates rather than duplicates - and "
        "<b>resumable</b>, which matters because the full build takes around three "
        "hours on this hardware.", st["body"]))
    A(Paragraph(
        "The first version walked the data directory alphabetically. That put "
        "<font face='Courier'>data/csvs/</font> - 397 files of contract boilerplate "
        "- ahead of the statute question-answer sets and the judgments. For the "
        "whole first hours of the build, every legal question retrieved debenture "
        "acceleration clauses, and the system looked broken although it was working "
        "exactly as written. Ingestion now orders the most answer-bearing material "
        "first. It is a one-line change and it was the single largest improvement to "
        "answer quality in the project.", st["body"]))

    # ======================== 5. the graph =============================
    A(Paragraph("5.  Stage B - building the knowledge graph", st["h1"]))
    A(Paragraph(
        "A <b>triple</b> is a fact in three parts: a subject, a relation and an "
        "object. \"Solomon Ogbe vs Narcotics Control Bureau\" - \"invokes\" - "
        "\"Article 21\". Collect enough triples and you have a graph.", st["body"]))
    A(Paragraph("5.1  Two extractors, and the measurement that chose between them", st["h2"]))
    A(Paragraph(
        "The conventional method is to ask a language model to read each chunk and "
        "return triples as JSON. It is accurate and finds subtle relations - it "
        "correctly extracted that a party was <i>represented by</i> particular "
        "counsel. We measured its cost on this machine before committing to it.",
        st["body"]))
    A(tbl([
        ["Method", "Speed", "Whole corpus (121,112 chunks)"],
        ["Language model (Qwen 2.5 3B)", "about 101 s per chunk", "about 141 days"],
        ["Deterministic legal patterns", "about 1,700 chunks/s", "about 1.4 minutes"],
    ], [52 * mm, 36 * mm, fw - 88 * mm]))
    A(Spacer(1, 3))
    A(Paragraph(
        "141 days is not an option, so the project adds a second extractor. Legal "
        "writing cites its entities in highly regular forms - \"Section 138 of the "
        "Negotiable Instruments Act\", \"Article 21\", \"X vs Y\", \"High Court of "
        "Delhi\". Those forms can be recognised by pattern with no model at all. The "
        "pattern extractor is the default and covers the whole corpus; the "
        "model-based extractor is kept and can be switched on for a subset where its "
        "richer output is worth the wait. Both produce the same triple shape, so they "
        "fill one graph.", st["body"]))
    A(Paragraph(
        "This is worth stating plainly as a lesson: the published method was "
        "unusable at our scale, and measuring it before adopting it is what revealed "
        "that. The replacement is not more sophisticated - it is a set of regular "
        "expressions - but it is 30,000 times faster and, for this domain, accurate "
        "enough to be useful.", st["quote"]))

    A(KeepTogether([
        Paragraph("5.2  Three problems that only appear at scale", st["h2"]),
        Paragraph(
            "A small demonstration graph behaves well. A 31,000-node graph does not, "
            "and three distinct faults had to be corrected.", st["body"])]))
    A(Paragraph("<b>(a) Not every \"Article\" is constitutional.</b> The extractor "
                "attributed every \"Article N\" to the Constitution of India. Then "
                "\"Article 1969\" and \"Article 2011\" appeared, which look like "
                "years. They are in fact genuine provisions of the <i>Portuguese "
                "Civil Code</i>: Goa inherited Portuguese civil law, and a judgment "
                "in the corpus is a Goan inheritance matter. Rejecting numbers above "
                "395 - the Constitution's real range - would have deleted correct "
                "data. The parent instrument is now read from the surrounding text.",
                st["body"]))
    A(Paragraph("<b>(b) Artificial hubs destroy traversal.</b> The reference "
                "implementation links every statute to a single shared concept node. "
                "That node makes every statute two steps from every other, so walking "
                "out from one starting point reaches the entire graph. Those edges "
                "are not created.", st["body"]))
    A(Paragraph("<b>(c) Real hubs need blocking.</b> \"Constitution of India\" "
                "legitimately has about 575 edges. Walking <i>through</i> it from "
                "\"Article 21\" returned forty relations that were all of the form "
                "\"Article N belongs to the Constitution\" - true, useless, and they "
                "crowded out the case law. Nodes above a degree threshold are now "
                "included as destinations but never walked through, and results are "
                "ordered by distance so truncation drops the furthest. After the fix "
                "the same question surfaces the NDPS bail cases that actually cite "
                "Article 21.", st["body"]))

    # ======================== 6. query =================================
    A(Paragraph("6.  Stage C - answering a question", st["h1"]))
    A(Paragraph("Four steps run for every question.", st["body"]))
    A(tbl([
        ["Step", "What happens"],
        ["1. Vector search",
         "The question is embedded with the query prefix and the five nearest "
         "passages are retrieved from Chroma, each with its similarity score and "
         "source filename."],
        ["2. Entity extraction",
         "Legal entities are identified in the question. If the question is plain "
         "English and names no statute, entities are taken instead from the "
         "passages just retrieved, which are topically relevant by construction."],
        ["3. Graph traversal",
         "Starting from those entities, the graph is walked up to two steps, "
         "hub nodes excluded from expansion, and each edge is written as a line of "
         "text: \"A --[relation]--> B\"."],
        ["4. Generation",
         "Passages and relations go into one prompt with an instruction to use both "
         "and to say so plainly if the context does not contain the answer. Qwen "
         "streams the answer back token by token."],
    ], [30 * mm, fw - 30 * mm]))
    A(Spacer(1, 4))
    A(Paragraph(
        "The instruction to admit ignorance matters. The system answers from a "
        "partially built index, and when the passages genuinely do not contain the "
        "answer the model says so instead of filling the gap with invention. In "
        "testing it reported \"the provided passages do not contain information "
        "about this case\" rather than fabricating a holding - the correct behaviour "
        "for a legal tool.", st["body"]))

    A(Paragraph("6.1  Why retrieval is hybrid", st["h2"]))
    A(Paragraph(
        "Dense search alone failed in a way worth recording. Asked \"What is the "
        "punishment for theft under the Indian Penal Code?\", every passage that "
        "contained the phrase \"under the Indian Penal Code\" outranked the record "
        "that actually answers the question, which is worded \"What is the "
        "punishment for 379 Theft?\". The correct chunk did not appear in the top "
        "hundred. Rephrasing the question with \"379\" in it put that same chunk "
        "first with a similarity of 0.895.", st["body"]))
    A(Paragraph(
        "A BM25 index was added alongside, and the two rankings are combined with "
        "Reciprocal Rank Fusion, which merges orderings without requiring their "
        "scores to be comparable. This makes retrieval less sensitive to how a "
        "question happens to be phrased. It does not rescue this particular "
        "example - after stop-word removal the target shares only two of the five "
        "query terms, while its competitors share all five, so both retrievers "
        "rank it low for defensible reasons. Fixing that case needs a "
        "cross-encoder reranker, which is listed as further work.", st["body"]))

    A(Paragraph("6.2  One request carries both answer and evidence", st["h2"]))
    A(Paragraph(
        "The interface needs the answer as it is generated <i>and</i> the evidence "
        "behind it. Making two requests would run the language model twice. Instead "
        "a single endpoint streams newline-delimited JSON: the evidence arrives "
        "first as one object, then the answer arrives as a sequence of tokens.",
        st["body"]))
    A(Paragraph(
        '{"type":"meta","vector_results":[...],"graph_relations":[...]}<br/>'
        '{"type":"token","t":"According"}<br/>'
        '{"type":"token","t":" to"}<br/>'
        '{"type":"done"}', st["code"]))
    A(PageBreak())

    # ======================== 7. worked example ========================
    A(Paragraph("7.  One question, end to end", st["h1"]))
    A(Paragraph("The question:", st["body"]))
    A(Paragraph('"Does Article 21 apply to prolonged incarceration under the '
                'NDPS Act?"', st["quote"]))
    A(Paragraph("<b>Step 1.</b> The question is embedded and Chroma returns the five "
                "closest passages with their sources and similarity scores.",
                st["body"]))
    A(Paragraph("<b>Step 2.</b> Pattern extraction reads two entities directly out "
                "of the question: <font face='Courier'>Article 21</font> and "
                "<font face='Courier'>Narcotic Drugs and Psychotropic Substances "
                "Act</font>.", st["body"]))
    A(Paragraph("<b>Step 3.</b> Traversal from those two nodes returns relations "
                "including:", st["body"]))
    A(Paragraph(
        "Kashif vs NCB --[invokes]--> Article 21<br/>"
        "Jitendra Jain vs NCB --[invokes]--> Article 21<br/>"
        "Solomon Ogbe vs Narcotics Control Bureau --[invokes]--> Article 21<br/>"
        "Solomon Ogbe vs Narcotics Control Bureau --[decided_by]--> Supreme Court<br/>"
        "Solomon Ogbe vs Narcotics Control Bureau --[invokes]--> Section 37(1) NDPS Act",
        st["code"]))
    A(Paragraph(
        "This is the point of the whole exercise. Those are the three narcotics bail "
        "cases in the corpus that actually rely on Article 21. No passage similarity "
        "search would group them, because they share a <i>relationship</i>, not "
        "wording. The graph found them by following edges.", st["body"]))
    A(Paragraph("<b>Step 4.</b> Both sets of context go to Qwen, which writes the "
                "answer. The interface shows the answer with its citations in the "
                "margin, the retrieved passages, the relations used, and the matched "
                "part of the graph lit up.", st["body"]))

    # ======================== 8. decisions =============================
    A(Paragraph("8.  Engineering decisions, with the measurements behind them", st["h1"]))
    A(tbl([
        ["Decision", "Reason"],
        ["Run everything locally via Ollama",
         "Legal documents are confidential. A local model means no document, "
         "question or client name ever leaves the machine. It also costs nothing "
         "per query and works offline."],
        ["Qwen 2.5 3B rather than 7B",
         "The machine is an Apple M2 with 8 GB of memory, about 5.3 GB usable by the "
         "GPU. The 7B model is 4.68 GB and the embedding model 0.37 GB; together with "
         "working memory they exceed the budget, so the two would be unloaded and "
         "reloaded on every question. At 3B (1.93 GB) both stay resident."],
        ["Chroma, not FAISS",
         "Chroma stores the chunk text and its metadata alongside the vector. FAISS "
         "stores vectors only, so the text has to be kept in a second structure and "
         "kept in step with it."],
        ["One embedding path for writing and reading",
         "The earlier version wrote vectors with one library and read them with "
         "another. The two were configured separately, so changing one model name "
         "would have silently degraded retrieval to noise without any error. Both "
         "now call the same function."],
        ["Verify the vector dimension before writing",
         "Changing embedding model changes the vector space and invalidates the "
         "index. The builder checks the dimension against configuration first and "
         "refuses to write on a mismatch, rather than corrupting the collection."],
        ["Pattern extraction for the graph",
         "Measured: 101 s per chunk by language model against 1,700 chunks per "
         "second by pattern. The model-based path remains available for subsets."],
        ["Produce document sections in code, not by prompting",
         "Asked to write one headed section per uploaded document, the 3B model "
         "merged them, even when told to emit exactly N verbatim headings. The "
         "system now generates once per document and writes the headings itself, so "
         "the structure is guaranteed rather than hoped for."],
    ], [48 * mm, fw - 48 * mm]))
    A(PageBreak())

    # ======================== 9. defects ===============================
    A(Paragraph("9.  Defects found during development", st["h1"]))
    A(Paragraph(
        "These are recorded because each was found by testing rather than by "
        "reading the code, and several were invisible in normal use.", st["body"]))
    A(tbl([
        ["Defect", "Effect", "Resolution"],
        ["Case-sensitive .pdf test",
         "Ten of twelve judgments silently skipped; the whole case-law set absent "
         "from the index",
         "Tika detects type from file content"],
        ["An unused file-count limit",
         "297 of 397 CSV files never indexed; code and index had drifted apart",
         "Limit removed; file coverage is now reported"],
        ["A JSON array named .jsonl",
         "The largest data file, 24,607 records, skipped by a filename test",
         "Structured files are detected by content"],
        ["\"LEGAL\" in \"NON-LEGAL\"",
         "The off-topic filter passed everything, because the rejection label "
         "contains the acceptance label as a substring",
         "Labels renamed so one cannot contain the other"],
        ["The filter then blocked real questions",
         "A question about Article 1 of the Constitution was classified as "
         "non-legal; uploads were rejected",
         "Filter disabled by default and documented"],
        ["A hardcoded address in the page",
         "The page requested 127.0.0.1 while being served from localhost - a "
         "different origin - so every request failed silently",
         "All addresses are now relative; a test forbids a hardcoded one"],
        ["Shared scratch collection name",
         "The second and later document uploads failed, because the library "
         "returns the same client each time",
         "Unique name per request, removed afterwards"],
        ["Highlighting expanded hub nodes",
         "A question mentioning the Constitution highlighted 798 nodes - "
         "effectively the whole visible graph",
         "Hub-aware and capped"],
        ["Bare substring matching in the graph",
         "The word \"grounds\" in a question matched \"Recreation Grounds Co Ltd "
         "vs Taylor\"",
         "Matches anchored to word boundaries"],
        ["Dense retrieval alone",
         "A passage phrased differently from the question was unreachable even "
         "at rank 100",
         "BM25 added; rankings fused with RRF"],
        ["Alphabetical ingestion order",
         "Hours spent indexing contract boilerplate while the statute Q&A and "
         "judgments waited; every answer drew on clause text",
         "Answer-bearing sources indexed first"],
    ], [40 * mm, 60 * mm, fw - 100 * mm], font=7.7))
    A(Spacer(1, 4))
    A(Paragraph(
        "The system is covered by 65 automated checks - 15 on the Python layer "
        "(<font face='Courier'>smoke_test.py</font>) and 50 on the HTTP interface "
        "(<font face='Courier'>test_http.sh</font>). Several were written "
        "specifically to stop a defect in this table from returning.", st["small"]))

    # ======================== 10. running ==============================
    A(Paragraph("10.  Running the system", st["h1"]))
    A(Paragraph(
        "./start.sh<br/>"
        "    # starts Ollama, starts the Tika server, reports the index, serves the app<br/><br/>"
        ".venv/bin/python embed_and_index.py     # build or resume the vector index<br/>"
        ".venv/bin/python build_graph.py         # build the knowledge graph<br/>"
        ".venv/bin/python smoke_test.py          # 15 checks on the Python layer<br/>"
        "./test_http.sh                          # 50 checks on the HTTP interface<br/><br/>"
        "then open  http://localhost:5002", st["code"]))
    A(PageBreak())
    A(Paragraph("Principal source files", st["h2"]))
    A(tbl([
        ["File", "Responsibility"],
        ["config.py", "Every tunable value, overridable by environment variable"],
        ["ollama_client.py", "Generation and embeddings, with retries and preflight checks"],
        ["tika_extract.py", "Tika server lifecycle and text extraction"],
        ["loaders.py", "Corpus discovery and extraction, with full skip accounting"],
        ["chunking.py", "Sentence-aware chunking"],
        ["embed_and_index.py", "Vector index builder - resumable and idempotent"],
        ["legal_entities.py", "Pattern-based legal triple extraction"],
        ["graph_store.py", "Persistent knowledge graph and hub-aware traversal"],
        ["graph_rag.py", "Hybrid retrieval and the model-based extractors"],
        ["build_graph.py", "Knowledge graph builder"],
        ["app.py", "HTTP interface"],
        ["templates/index.html", "The single user interface"],
    ], [42 * mm, fw - 42 * mm], font=8.0))

    # ======================== 11. limitations ==========================
    A(Paragraph("11.  Limitations, stated honestly", st["h1"]))
    A(bullets([
        "<b>A 3B model is a small model.</b> It is chosen to fit 8 GB of memory, not "
        "for maximum quality. It follows complex formatting instructions poorly, "
        "which is why document sections are assembled in code. A larger model on "
        "larger hardware would answer better with no other change - the model name "
        "is one line of configuration.",
        "<b>Pattern extraction has a ceiling.</b> It recognises citation forms it was "
        "written for. A relation phrased unusually is missed. It has no understanding, "
        "only patterns - and its output still contains some noise, such as docket "
        "fragments caught inside party names.",
        "<b>There is no reranking.</b> The five nearest passages go to the model as "
        "retrieved. A cross-encoder reranking stage would improve precision.",
        "<b>Retrieval is dense only.</b> There is no keyword or BM25 component, so an "
        "exact rare term - a precise section number - is matched by meaning rather "
        "than literally. A hybrid of the two is the standard improvement.",
        "<b>There is no evaluation set.</b> Correctness has been checked by reading "
        "answers, not measured against a labelled benchmark. Without one, claims "
        "about accuracy are impressions.",
        "<b>Answers are not legal advice.</b> The system retrieves and summarises "
        "text from a fixed corpus. It does not know whether a judgment has since been "
        "overruled or a section amended.",
    ], st["body"]))

    A(Paragraph("11.1  Corpus quality limits answer quality", st["h1"]))
    A(Paragraph(
        "Some wrong answers are not retrieval or generation faults but faithful "
        "reporting of poor source records. The question-answer datasets contain "
        "entries whose question names no particular offence while the answer states "
        "an absolute penalty, for example:", st["body"]))
    A(Paragraph(
        "Q: \"What are the possible punishments for a crime?\"<br/>"
        "A: \"Imprisonment for life, Imprisonment for 10 years, fine, or both...\"",
        st["code"]))
    A(Paragraph(
        "Fifteen such records exist in one file. They are semantically close to any "
        "question of the form \"what is the punishment for X\", so they are "
        "retrieved often, and a small model quoting them produces a confident wrong "
        "answer. No amount of prompt engineering repairs this: the system is "
        "reporting the corpus accurately. The corpus needs filtering, which is a "
        "data task rather than a retrieval one.", st["body"]))

    A(Paragraph("11.2  What we would do next", st["h1"]))
    A(bullets([
        "Build a labelled set of questions with known answers and measure retrieval "
        "precision and recall, so changes can be judged rather than guessed at.",
        "Add BM25 alongside the vector search and combine the rankings.",
        "Add a reranking stage over the retrieved passages.",
        "Run the model-based triple extractor over the judgments as a background "
        "job, to enrich the part of the graph where relations are most valuable.",
        "Carry the source of each graph relation through to the interface, so a "
        "relation can be traced to the passage that produced it.",
    ], st["body"]))

    A(Spacer(1, 7 * mm))
    A(Rule(width=fw, thickness=1, color=EDGE))
    A(Spacer(1, 3 * mm))
    A(Paragraph(
        "<b>Summary.</b> The system answers questions about Indian law without "
        "sending anything to a cloud service. It does so by retrieving relevant "
        "passages by meaning and relevant relations from a legal knowledge graph, "
        "then asking a locally-run Qwen model to answer from that evidence and show "
        "it. The engineering content of the project is less in assembling those "
        "parts than in the measurements that decided between approaches, and in the "
        "ten defects found by testing - most of which produced no error message and "
        "would have gone unnoticed.", st["body"]))
    A(Paragraph("Report generated %s from live system state." % today, st["small"]))

    doc.build(E)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT}  ({size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
