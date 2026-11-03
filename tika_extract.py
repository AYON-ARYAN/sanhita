"""Apache Tika document extraction.

Tika replaces the previous per-format readers (pdfplumber / PyMuPDF /
python-docx / pandas). One extractor now handles PDF, DOCX, DOC, RTF, ODT,
XLSX, XLS, CSV, TSV, PPTX, HTML, XML, EPUB, TXT and more, which also removes
the case-sensitive `.pdf` / `.PDF` extension bug in the old loaders: Tika
sniffs content type from the bytes, not the filename.

The server is run as a subprocess from the vendored fat jar and reused across
calls.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

import config

_server_proc: Optional[subprocess.Popen] = None

# Formats Tika handles that are worth feeding it. Anything not listed is
# skipped by the loader rather than guessed at.
TIKA_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".rtf", ".odt", ".txt", ".text", ".md",
    ".xlsx", ".xls", ".xlsm", ".ods", ".csv", ".tsv",
    ".pptx", ".ppt", ".odp",
    ".html", ".htm", ".xml", ".xhtml", ".epub",
    ".msg", ".eml",
}

STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".ndjson"}


def is_up() -> bool:
    try:
        r = requests.get(f"{config.TIKA_URL}/tika", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def start_server(wait_seconds: int = 90) -> None:
    """Start the Tika server if it is not already answering. Idempotent."""
    global _server_proc
    if is_up():
        return

    jar = Path(config.TIKA_JAR)
    if not jar.is_file():
        raise FileNotFoundError(
            f"Tika jar not found at {jar}\n"
            "Download it with:\n"
            "  mkdir -p vendor/tika && curl -sL -o vendor/tika/tika-server-standard-3.3.3.jar \\\n"
            "    https://repo1.maven.org/maven2/org/apache/tika/tika-server-standard/3.3.3/"
            "tika-server-standard-3.3.3.jar"
        )

    port = config.TIKA_URL.rsplit(":", 1)[-1]
    cmd = [
        "java", "-Xmx2g", "-Djava.awt.headless=true", "-jar", str(jar),
        "--port", port,
        "--host", "localhost",
    ]
    # Tika 3.x takes per-task timeouts from a config file, not the CLI.
    cfg = Path(config.TIKA_CONFIG)
    if cfg.is_file():
        cmd += ["--config", str(cfg)]
    log_path = Path(os.getenv("TIKA_LOG", "/tmp/tika-server.log"))
    log = open(log_path, "ab", buffering=0)
    _server_proc = subprocess.Popen(
        cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL
    )

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if is_up():
            return
        if _server_proc.poll() is not None:
            raise RuntimeError(
                f"Tika server exited with code {_server_proc.returncode}. "
                f"See {log_path}"
            )
        time.sleep(1)
    raise TimeoutError(
        f"Tika server did not come up within {wait_seconds}s. See {log_path}"
    )


def stop_server() -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
    _server_proc = None


def extract_text(path: str | Path) -> str:
    """Return plain text for any Tika-supported document.

    Raises on transport failure; returns "" when Tika parses the file but
    finds no text (e.g. an image-only PDF with no OCR configured).
    """
    path = Path(path)
    with path.open("rb") as fh:
        resp = requests.put(
            f"{config.TIKA_URL}/tika",
            data=fh,
            headers={
                "Accept": "text/plain",
                "Content-Disposition": f'attachment; filename="{path.name}"',
            },
            timeout=config.TIKA_TIMEOUT,
        )
    if resp.status_code == 422:
        # Tika could not parse (encrypted / corrupt). Treat as empty, not fatal.
        return ""
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text or ""


def extract_text_from_bytes(data: bytes, filename: str = "upload") -> str:
    """Same as extract_text but for in-memory uploads."""
    resp = requests.put(
        f"{config.TIKA_URL}/tika",
        data=data,
        headers={
            "Accept": "text/plain",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
        timeout=config.TIKA_TIMEOUT,
    )
    if resp.status_code == 422:
        return ""
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text or ""


def detect_type(path: str | Path) -> str:
    """Tika's content-type detection for a file (bytes-based, not extension)."""
    path = Path(path)
    with path.open("rb") as fh:
        resp = requests.put(
            f"{config.TIKA_URL}/detect/stream",
            data=fh,
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.text.strip()
