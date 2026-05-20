"""
Logging, path helpers, and shared utilities for AMPL MCP Server.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timezone


# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"
TEMP_DIR = ROOT_DIR / "temp"

for _d in (LOGS_DIR, RESULTS_DIR, TEMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─── Logger ──────────────────────────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("ampl_mcp_server")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(str(LOGS_DIR / "server.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


logger = _build_logger()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def timestamp_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_path(file_path: str) -> Path:
    """Resolve a user-supplied path relative to ROOT_DIR if not absolute."""
    p = Path(file_path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p.resolve()


def truncate_result(rows: list[dict], max_preview: int = 10) -> tuple[list[dict], int]:
    """Return (preview_slice, total_count)."""
    return rows[:max_preview], len(rows)
