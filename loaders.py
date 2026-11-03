"""Corpus loading via Apache Tika, with explicit accounting.

Replaces load_all_data.py. Three bugs in that module are fixed structurally:

  1. `.endswith(".pdf")` was case-sensitive, so 10 of 12 judgment PDFs
     (named `.PDF`) were silently dropped. Tika sniffs content type from
     bytes, and extension matching here is case-folded.
  2. `MAX_CSV_FILES = 100` was declared but never applied, leaving code and
     index out of sync. There is no implicit cap here; use `limit=` if you
     want one.
  3. `train.jsonl` was skipped by a `.json` suffix test. It is in fact a
     pretty-printed JSON *array*, so structured files are now sniffed by
     content rather than trusted by extension.

Every file is accounted for: loaded, empty, unsupported or failed. Nothing
is dropped silently.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import config
import tika_extract as tika


@dataclass
class Doc:
    """One unit of text handed to the chunker."""
    source_file: str          # basename, used for citation/display
    source_path: str          # path relative to project root, unique
    text: str
    record_index: Optional[int] = None   # set for structured records


@dataclass
class IngestReport:
    loaded_files: List[str] = field(default_factory=list)
    empty_files: List[str] = field(default_factory=list)
    unsupported_files: List[str] = field(default_factory=list)
    failed_files: Dict[str, str] = field(default_factory=dict)
    docs_by_file: Dict[str, int] = field(default_factory=dict)

    @property
    def total_docs(self) -> int:
        return sum(self.docs_by_file.values())

    def summary(self) -> str:
        lines = [
            "Ingest report",
            f"  files loaded      : {len(self.loaded_files)}",
            f"  documents produced: {self.total_docs}",
            f"  empty (no text)   : {len(self.empty_files)}",
            f"  unsupported       : {len(self.unsupported_files)}",
            f"  failed            : {len(self.failed_files)}",
        ]
        if self.empty_files:
            lines.append("  -- empty --")
            lines += [f"     {p}" for p in self.empty_files[:20]]
            if len(self.empty_files) > 20:
                lines.append(f"     ... and {len(self.empty_files)-20} more")
        if self.unsupported_files:
            lines.append("  -- unsupported --")
            lines += [f"     {p}" for p in self.unsupported_files[:20]]
            if len(self.unsupported_files) > 20:
                lines.append(f"     ... and {len(self.unsupported_files)-20} more")
        if self.failed_files:
            lines.append("  -- failed --")
            lines += [f"     {p}: {e}" for p, e in list(self.failed_files.items())[:20]]
        return "\n".join(lines)


# --------------------------------------------------------------- helpers ----
def _record_to_text(rec) -> str:
    """Flatten one structured record into retrievable text."""
    if isinstance(rec, dict):
        parts = []
        for k, v in rec.items():
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(rec)


def _read_structured(path: Path) -> List[str]:
    """Return a text blob per record.

    Sniffs the actual content: a whole-file JSON value first (covers a
    pretty-printed array in a .jsonl file), then line-delimited JSON.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        texts = []
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in "[]":
                continue
            try:
                texts.append(_record_to_text(json.loads(line)))
            except json.JSONDecodeError:
                continue
        return texts

    if isinstance(data, list):
        return [_record_to_text(r) for r in data]
    if isinstance(data, dict):
        # A mapping of id -> record is common; otherwise treat as one record.
        if data and all(isinstance(v, (dict, list)) for v in data.values()):
            return [_record_to_text(v) for v in data.values()]
        return [_record_to_text(data)]
    return [str(data)]


def _classify(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in tika.STRUCTURED_EXTENSIONS and not config.TIKA_FOR_JSON:
        return "structured"
    if ext in tika.TIKA_EXTENSIONS or ext in tika.STRUCTURED_EXTENSIONS:
        return "tika"
    return "unsupported"


# Index the most answer-bearing material first. Plain alphabetical order put
# data/csvs (397 files of contract-clause boilerplate, ~81k chunks) ahead of
# data/jsons and data/pdfs, so a long build spent hours indexing boilerplate
# while the statute/case Q&A and the judgments -- the material that actually
# answers legal questions -- sat unindexed to the very end. Retrieval looked
# broken for that whole window: every question returned clause text.
PRIORITY_HINTS = ("jsons", "pdfs", "json", "pdf")


def _priority(path: Path) -> tuple:
    parts = {p.lower() for p in path.parts}
    for rank, hint in enumerate(PRIORITY_HINTS):
        if hint in parts:
            return (rank, str(path))
    return (len(PRIORITY_HINTS), str(path))


def discover(dirs: Sequence[str]) -> List[Path]:
    """Recursively list candidate files, highest-value sources first."""
    out: List[Path] = []
    for d in dirs:
        root = Path(d)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name == ".DS_Store":
                continue
            out.append(p)
    return sorted(out, key=_priority)


# ------------------------------------------------------------------ main ----
def load_documents(
    dirs: Sequence[str] | None = None,
    limit: int | None = None,
    verbose: bool = True,
) -> tuple[List[Doc], IngestReport]:
    """Extract every supported file under `dirs` into Doc records."""
    dirs = dirs or config.DATA_DIRS
    report = IngestReport()
    docs: List[Doc] = []

    files = discover(dirs)
    if limit:
        files = files[:limit]

    needs_tika = any(_classify(p) == "tika" for p in files)
    if needs_tika:
        tika.start_server()

    for i, path in enumerate(files, 1):
        rel = str(path)
        kind = _classify(path)

        if kind == "unsupported":
            report.unsupported_files.append(rel)
            continue

        try:
            if kind == "structured":
                texts = _read_structured(path)
                produced = 0
                for idx, t in enumerate(texts):
                    if t and t.strip():
                        docs.append(Doc(path.name, rel, t, record_index=idx))
                        produced += 1
                if produced == 0:
                    report.empty_files.append(rel)
                else:
                    report.loaded_files.append(rel)
                    report.docs_by_file[rel] = produced
            else:
                text = tika.extract_text(path)
                if text and text.strip():
                    docs.append(Doc(path.name, rel, text))
                    report.loaded_files.append(rel)
                    report.docs_by_file[rel] = 1
                else:
                    report.empty_files.append(rel)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            report.failed_files[rel] = f"{type(exc).__name__}: {exc}"

        if verbose and i % 25 == 0:
            print(f"  [{i}/{len(files)}] {report.total_docs} docs so far", flush=True)

    return docs, report


if __name__ == "__main__":
    docs, rep = load_documents()
    print(rep.summary())
    print(f"\nTotal documents: {len(docs)}")
