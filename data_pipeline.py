"""
Data Pipeline — Excel / CSV ingestion with strict string preservation.

Key design decisions:
  - Read everything as string FIRST, then selectively parse numerics.
  - Leading-zero codes (000, 001, 01A, BUS001, 34E) are NEVER auto-cast.
  - All indexing is 1-based for AMPL compatibility.
  - Primary and secondary files share the same dtype rules and validation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from utils import logger, safe_path

# ─── Regex patterns for numeric detection ────────────────────────────────────

_PURE_INTEGER = re.compile(r"^[+-]?\d+$")
_PURE_FLOAT = re.compile(r"^[+-]?\d*\.\d+([eE][+-]?\d+)?$")
_LEADING_ZERO_NUMERIC = re.compile(r"^0\d+$")  # e.g. 000, 001, 0123 — keep as string
_ALPHANUMERIC_CODE = re.compile(r"^[A-Za-z0-9]+$")  # mixed codes like 01A, BUS001


def _is_definitely_numeric(series: pd.Series) -> bool:
    """
    Return True only if every non-null value in *series* unambiguously
    represents a numeric quantity (no leading-zero IDs, no mixed codes).
    """
    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return False

    for val in non_null:
        val = val.strip()
        if not val:
            continue
        if _LEADING_ZERO_NUMERIC.match(val):
            return False  # "001" is an ID, not a number
        if _PURE_FLOAT.match(val):
            continue
        if _PURE_INTEGER.match(val):
            continue
        return False  # contains non-numeric characters

    return True


def _is_integer_column(series: pd.Series) -> bool:
    """Check if a series that passed _is_definitely_numeric contains only integers."""
    non_null = series.dropna().astype(str).str.strip()
    for val in non_null:
        val = val.strip()
        if not val:
            continue
        if _PURE_FLOAT.match(val):
            return False
    return True


# ─── File type normalisation ─────────────────────────────────────────────────

COLUMN_NAME_NORMALIZE_RE = re.compile(r"[^a-zA-Z0-9_]")


def _normalize_column_name(name: str) -> str:
    """Replace non-alphanumeric characters with underscore for AMPL compatibility."""
    normalized = COLUMN_NAME_NORMALIZE_RE.sub("_", str(name))
    if normalized and normalized[0].isdigit():
        normalized = "_" + normalized
    if not normalized:
        normalized = "_col"
    return normalized


# ─── Main Pipeline ──────────────────────────────────────────────────────────

class DataPipeline:
    """
    Shared pipeline for all data files (primary and secondary).

    All files pass through the same:
      - file-type detection
      - string-safe reading
      - dtype inference
      - AMPL transformation
    """

    def __init__(self) -> None:
        self._dtype_rules: dict[str, str] = {}  # column → dtype
        self._normalized_columns: dict[str, str] = {}  # original → normalized
        self._string_columns: list[str] = []

    # ── File reading ──────────────────────────────────────────────────────

    def read_file(self, file_path: str) -> dict[str, pd.DataFrame]:
        """
        Read an Excel or CSV file, returning a dict of {sheet_name: DataFrame}.

        - .xlsx / .xls: one DataFrame per sheet.
        - .csv: single DataFrame keyed "data".
        - All columns are read as *object* (string) initially.
        """
        path = safe_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        tables: dict[str, pd.DataFrame] = {}

        if ext in (".xlsx", ".xls"):
            tables = self._read_excel(path)
        elif ext == ".csv":
            tables = self._read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .xlsx, .xls, .csv")

        logger.info(
            "File read | path=%s | sheets=%s | total_cells=%d",
            file_path,
            list(tables.keys()),
            sum(df.size for df in tables.values()),
        )
        return tables

    def _read_excel(self, path: Path) -> dict[str, pd.DataFrame]:
        """Read all sheets as string to prevent auto-numeric conversion."""
        xl = pd.ExcelFile(path, engine="openpyxl" if path.suffix == ".xlsx" else "xlrd")
        tables: dict[str, pd.DataFrame] = {}
        for sheet in xl.sheet_names:
            df = pd.read_excel(
                xl,
                sheet_name=sheet,
                dtype=str,  # force all columns to string
                header=0,
            )
            # Drop fully empty rows/columns
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                tables[sheet] = df
        return tables

    def _read_csv(self, path: Path) -> dict[str, pd.DataFrame]:
        """Read CSV with encoding detection, all columns as string."""
        encoding = self._detect_encoding(path)
        try:
            df = pd.read_csv(path, dtype=str, encoding=encoding, header=0)
        except UnicodeDecodeError:
            df = pd.read_csv(path, dtype=str, encoding="latin-1", header=0)

        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            return {}
        return {"data": df}

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """Lightweight encoding detection using chardet or a fallback."""
        try:
            import chardet
            with open(path, "rb") as f:
                raw = f.read(10000)
            result = chardet.detect(raw)
            return result["encoding"] or "utf-8"
        except ImportError:
            # chardet not available — try utf-8, fall back to latin-1
            try:
                with open(path, encoding="utf-8") as f:
                    f.read(1000)
                return "utf-8"
            except UnicodeDecodeError:
                return "latin-1"

    # ── Dtype inference (string-safe) ─────────────────────────────────────

    def infer_and_apply_dtypes(self, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        For each table, detect which columns are truly numeric and convert
        them while preserving string columns that look ambiguous.

        Returns the same tables dict with dtypes applied.
        """
        self._string_columns = []

        for sheet_name, df in tables.items():
            normalized: dict[str, str] = {}
            for col in df.columns:
                norm = _normalize_column_name(col)
                normalized[str(col)] = norm

            self._normalized_columns.update(normalized)
            df = df.rename(columns=normalized)
            tables[sheet_name] = self._apply_dtypes_to_dataframe(df)
            logger.debug("Sheet '%s' dtypes applied | columns=%d", sheet_name, len(df.columns))

        return tables

    def _apply_dtypes_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dtype inference on a single DataFrame."""
        for col in df.columns:
            series = df[col].copy()

            # Replace common null markers
            series = series.replace(["", "NA", "N/A", "null", "NULL", "NaN", "nan", "None"], None)

            if _is_definitely_numeric(series):
                if _is_integer_column(series):
                    df[col] = pd.to_numeric(series, errors="coerce")
                    self._dtype_rules[col] = "int"
                else:
                    df[col] = pd.to_numeric(series, errors="coerce")
                    self._dtype_rules[col] = "float"
            else:
                self._string_columns.append(col)
                self._dtype_rules[col] = "str"
                df[col] = series.astype(str)

        return df

    # ── AMPL transformation ───────────────────────────────────────────────

    def to_ampl_data(self, tables: dict[str, pd.DataFrame]) -> str:
        """
        Convert cleaned DataFrames to an AMPL .dat format string.

        Strategy:
          - Each sheet becomes a set + parameter group.
          - The first string column is treated as the set index.
          - Remaining string columns become indexed string parameters.
          - Numeric columns become indexed numeric parameters.
          - All indexing is 1-based.
        """
        lines: list[str] = []
        lines.append("# Auto-generated AMPL data — " + pd.Timestamp.now().isoformat())
        lines.append("")

        for sheet_name, df in tables.items():
            if df.empty:
                continue

            base_name = _normalize_column_name(sheet_name)
            n_rows = len(df)

            # ── Set definition (1-based) ──
            set_name = f"{base_name}_SET"
            set_members = [str(i + 1) for i in range(n_rows)]
            lines.append(f"set {set_name} := {' '.join(set_members)};")
            lines.append("")

            # ── String columns → indexed string params ──
            for col in df.columns:
                col_name = _normalize_column_name(col)
                if col_name in self._string_columns:
                    param_name = f"{base_name}_{col_name}"
                    lines.append(f"param {param_name} {{i in {set_name}}} symbolic;")
                    lines.append(f"param {param_name} :=")
                    for idx, val in enumerate(df[col], start=1):
                        safe_val = str(val) if pd.notna(val) else "''"
                        # Escape single quotes inside values
                        safe_val = safe_val.replace("'", "''")
                        lines.append(f"  {idx} '{safe_val}'")
                    lines.append(";")
                    lines.append("")

            # ── Numeric columns → indexed numeric params ──
            for col in df.columns:
                col_name = _normalize_column_name(col)
                if col_name not in self._string_columns:
                    param_name = f"{base_name}_{col_name}"
                    lines.append(f"param {param_name} {{i in {set_name}}};")
                    lines.append(f"param {param_name} :=")
                    for idx, val in enumerate(df[col], start=1):
                        if pd.isna(val):
                            lines.append(f"  {idx} .")
                        else:
                            lines.append(f"  {idx} {val}")
                    lines.append(";")
                    lines.append("")

        return "\n".join(lines)

    def get_loading_summary(self, tables: dict[str, pd.DataFrame]) -> dict:
        """Build the summary returned to the LLM after data injection."""
        table_info = []
        for name, df in tables.items():
            string_cols = [c for c in df.columns if c in self._string_columns]
            numeric_cols = [c for c in df.columns if c not in self._string_columns]
            table_info.append({
                "name": name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "string_columns": string_cols,
                "numeric_columns": numeric_cols,
            })

        return {
            "tables_loaded": table_info,
            "primary_keys_detected": self._string_columns.copy(),
        }

    # ── State reset ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear internal dtype rules and column cache."""
        self._dtype_rules.clear()
        self._normalized_columns.clear()
        self._string_columns.clear()
        logger.debug("DataPipeline state reset")
