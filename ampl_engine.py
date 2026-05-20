"""
AMPL Engine — Stateful singleton runtime for the AMPL MCP Server.

Manages the full AMPL lifecycle:
  - initialisation, model/data injection, solve, variable extraction,
  - infeasible diagnostics (IIS + slack analysis + relaxation suggestions),
  - solver switching, and state caching.

The module-level ``_engine`` singleton ensures exactly one AMPL process
across the lifetime of the MCP server.

Environment variables:
  AMPL_PATH   — path to the directory containing the ``ampl`` binary
                (if not on PATH; also checked via AMPLPATH for legacy compat)
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Optional

import amplpy
import pandas as pd

from schemas import (
    InfeasibleConstraint,
    InfeasibleDiagnostics,
)
from utils import logger, timestamp_iso, safe_path

# ─── AMPL installation detection ──────────────────────────────────────────────

def _find_ampl_directory() -> Optional[str]:
    """
    Locate the AMPL installation directory (the folder containing ``ampl`` / ``ampl.exe``).

    Checks (in order):
      1. ``AMPL_PATH`` environment variable
      2. ``AMPLPATH`` environment variable (legacy amplpy convention)
      3. ``PATH`` — search for ``ampl`` / ``ampl.exe``
      4. Glob scan of common install roots (``D:\\ampl*``, ``C:\\ampl*``, etc.)
      5. Standard install locations by platform
    """
    binary_name = "ampl.exe" if sys.platform == "win32" else "ampl"

    # 1 & 2 — environment variables
    for env_var in ("AMPL_PATH", "AMPLPATH"):
        val = os.environ.get(env_var)
        if val:
            p = Path(val)
            if p.is_file():
                # User pointed directly at the binary
                logger.debug("AMPL binary from %s: %s", env_var, p)
                return str(p.parent)
            if p.is_dir():
                logger.debug("AMPL directory from %s: %s", env_var, p)
                return str(p)

    # 3 — PATH
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        directory = directory.strip()
        if not directory:
            continue
        candidate = Path(directory) / binary_name
        if candidate.is_file():
            logger.debug("AMPL found on PATH: %s", candidate)
            return str(directory)

    # 4 — glob scan under common drive roots (Windows) or filesystem roots
    search_roots: list[Path] = []
    if sys.platform == "win32":
        for drive_letter in ("D", "C", "E", "F"):
            root = Path(f"{drive_letter}:\\")
            if root.exists():
                search_roots.append(root)
    else:
        search_roots = [Path("/opt"), Path("/usr/local"), Path.home()]

    for root in search_roots:
        try:
            for pattern in ("ampl*", "AMPL*", "Ampl*"):
                for candidate_dir in root.glob(pattern):
                    if candidate_dir.is_dir() and (candidate_dir / binary_name).is_file():
                        logger.debug("AMPL found via glob: %s", candidate_dir)
                        return str(candidate_dir)
        except (PermissionError, OSError):
            continue

    # 5 — standard install locations (fallback)
    standard: list[Path] = []
    if sys.platform == "win32":
        standard = [
            Path(r"D:\ampl_mswin64"),
            Path(r"D:\ampl"),
            Path(r"C:\AMPL\ampl"),
            Path(r"C:\AMPL"),
            Path(r"C:\Program Files\AMPL"),
            Path(r"C:\Program Files (x86)\AMPL"),
            Path.home() / "AMPL",
        ]
    elif sys.platform == "darwin":
        standard = [
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
            Path.home() / "AMPL",
        ]
    else:
        standard = [
            Path("/opt/ampl"),
            Path("/usr/local/ampl"),
            Path.home() / "AMPL",
        ]

    for d in standard:
        if (d / binary_name).is_file():
            logger.debug("AMPL found at standard location: %s", d)
            return str(d)
        if d.is_dir() and any(d.glob(binary_name)):
            logger.debug("AMPL found at standard location: %s", d)
            return str(d)

    logger.debug("No AMPL installation directory found — relying on amplpy defaults")
    return None

# ─── Module-level singleton ──────────────────────────────────────────────────

_engine: Optional["AMPLEngine"] = None
_ampl_dir: Optional[str] = _find_ampl_directory()


def get_engine() -> "AMPLEngine":
    """Return the module-global AMPLEngine singleton, creating it if needed."""
    global _engine
    if _engine is None:
        _engine = AMPLEngine(ampl_dir=_ampl_dir)
        _engine.initialize()
    return _engine


def get_ampl_directory() -> Optional[str]:
    """Return the detected AMPL installation directory (or None)."""
    return _ampl_dir


# ─── Engine ──────────────────────────────────────────────────────────────────

class AMPLEngine:
    """
    Stateful AMPL runtime wrapper.

    Maintains a single persistent ``amplpy.AMPL`` instance so that the LLM
    can upload a model, inject data, solve, modify parameters, and re-solve
    without losing the session.
    """

    def __init__(self, ampl_dir: Optional[str] = None) -> None:
        self.ampl_dir: Optional[str] = ampl_dir
        self.ampl: Optional[amplpy.AMPL] = None
        self.loaded_models: list[str] = []
        self.loaded_data_files: list[str] = []
        self.last_solve_result: Optional[str] = None
        self.last_solver_output: str = ""
        self.last_objective: Optional[float] = None
        self.last_solve_time: float = 0.0
        self.current_solver: str = "highs"
        self.available_solvers: list[str] = []
        self._initialized: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Create the AMPL instance, configure solver, and probe available solvers."""
        if self._initialized and self.ampl is not None:
            logger.info("AMPL already initialised — skipping")
            return

        # Register the AMPL directory with amplpy BEFORE creating the AMPL object.
        # This is the canonical approach per amplpy's error message.
        if self.ampl_dir:
            logger.info("Registering AMPL directory: %s", self.ampl_dir)
            amplpy.add_to_path(str(self.ampl_dir))

        try:
            self.ampl = amplpy.AMPL()
            logger.info("AMPL instance created | ampl_dir=%s", self.ampl_dir or "(default)")
        except Exception as exc:
            logger.critical("Failed to create AMPL instance: %s", exc)
            raise RuntimeError(
                f"Cannot initialise AMPL.  Is AMPL installed and configured?\n"
                f"  AMPL_PATH={os.environ.get('AMPL_PATH', 'not set')}\n"
                f"  ampl_dir={self.ampl_dir or '(none detected)'}\n"
                f"  Underlying error: {exc}\n"
                f"\n"
                f"To fix: set AMPL_PATH to the directory containing ampl.exe, e.g.:\n"
                f"  Windows:  $env:AMPL_PATH = 'D:\\\\ampl_mswin64'\n"
                f"  Linux:    export AMPL_PATH=/opt/ampl\n"
            ) from exc

        # Probe available solvers
        self._probe_solvers()

        # Set default solver if available; downgrade if missing
        if self.current_solver not in self.available_solvers and self.available_solvers:
            fallback = self.available_solvers[0]
            logger.warning(
                "Default solver '%s' not available — falling back to '%s'",
                self.current_solver, fallback,
            )
            self.current_solver = fallback

        try:
            self.ampl.set_option("solver", self.current_solver)
        except Exception:
            logger.warning("Could not set default solver '%s'", self.current_solver)

        self._initialized = True
        logger.info(
            "AMPL engine initialised | solver=%s | available=%s",
            self.current_solver, self.available_solvers,
        )

    def _probe_solvers(self) -> None:
        """Detect which solvers are available via the current AMPL installation."""
        if self.ampl is None:
            return
        known = ["highs", "gurobi", "cplex", "cbc", "scip", "xpress", "baron", "knitro", "ipopt", "bonmin", "copt"]
        self.available_solvers = []
        for s in known:
            try:
                self.ampl.set_option("solver", s)
                self.available_solvers.append(s)
            except Exception:
                pass
        logger.debug("Probed solvers: %s", self.available_solvers)

    def reset(self) -> None:
        """
        Full workspace reset: close and re-create the AMPL instance,
        clearing all models, data, variables, and cached state.
        """
        if self.ampl is not None:
            try:
                self.ampl.close()
            except Exception as exc:
                logger.warning("Error closing AMPL during reset: %s", exc)

        if self.ampl_dir:
            amplpy.add_to_path(str(self.ampl_dir))
        self.ampl = amplpy.AMPL()

        self.loaded_models.clear()
        self.loaded_data_files.clear()
        self.last_solve_result = None
        self.last_solver_output = ""
        self.last_objective = None
        self.last_solve_time = 0.0
        self.current_solver = self.available_solvers[0] if self.available_solvers else "highs"
        try:
            self.ampl.set_option("solver", self.current_solver)
        except Exception:
            pass
        self._initialized = True
        logger.info("AMPL workspace fully reset | solver=%s", self.current_solver)

    def ensure_runtime(self) -> amplpy.AMPL:
        """Return the active AMPL instance, initialising if necessary."""
        if self.ampl is None or not self._initialized:
            self.initialize()
        assert self.ampl is not None
        return self.ampl

    # ── Model injection ───────────────────────────────────────────────────

    def load_model(self, model_code: str) -> dict:
        """
        Inject a complete .mod file (as a string) into the AMPL session.

        Errors are caught with full AMPL line-number information so the LLM
        can self-correct.
        """
        ampl = self.ensure_runtime()
        captured = io.StringIO()

        try:
            with redirect_stdout(captured):
                ampl.eval(model_code)
            ampl_log = captured.getvalue()
            self.loaded_models.append(f"<inline model {len(self.loaded_models) + 1}>")
            logger.info(
                "Model loaded | chars=%d | lines=%d",
                len(model_code),
                model_code.count("\n") + 1,
            )
            return {
                "status": "success",
                "message": "Model loaded and validated successfully",
                "model_size_bytes": len(model_code.encode("utf-8")),
                "line_count": model_code.count("\n") + 1,
                "ampl_log": ampl_log,
            }
        except Exception as exc:
            captured.seek(0)
            ampl_output = captured.read()
            full_message = f"{exc}\n\nAMPL output:\n{ampl_output}"
            logger.error("Model load failed: %s", full_message)
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": full_message,
                "traceback": traceback.format_exc(),
            }

    # ── Data injection ────────────────────────────────────────────────────

    def inject_data(self, ampl_data_str: str, source_label: str) -> dict:
        """
        Inject a preprocessed data string (AMPL .dat format) into the session.

        ``source_label`` is used for logging and state tracking.
        """
        ampl = self.ensure_runtime()
        captured = io.StringIO()

        try:
            with redirect_stdout(captured):
                ampl.eval(ampl_data_str)
            ampl_log = captured.getvalue()
            self.loaded_data_files.append(source_label)
            logger.info("Data injected | source=%s | chars=%d", source_label, len(ampl_data_str))
            return {
                "status": "success",
                "message": f"Data from '{source_label}' injected successfully",
                "ampl_log": ampl_log,
            }
        except Exception as exc:
            captured.seek(0)
            ampl_output = captured.read()
            full_message = f"{exc}\n\nAMPL output:\n{ampl_output}"
            logger.error("Data injection failed: %s", full_message)
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": full_message,
                "traceback": traceback.format_exc(),
            }

    # ── Read data file directly ───────────────────────────────────────────

    def read_data_file(self, file_path: str) -> dict:
        """Use ampl.read_data() for files that are already valid AMPL .dat format."""
        ampl = self.ensure_runtime()
        captured = io.StringIO()

        try:
            with redirect_stdout(captured):
                ampl.read_data(file_path)
            ampl_log = captured.getvalue()
            self.loaded_data_files.append(file_path)
            logger.info("Data file read | path=%s", file_path)
            return {
                "status": "success",
                "message": f"Data file '{file_path}' read successfully",
                "ampl_log": ampl_log,
            }
        except Exception as exc:
            captured.seek(0)
            ampl_output = captured.read()
            full_message = f"{exc}\n\nAMPL output:\n{ampl_output}"
            logger.error("Data file read failed: %s", full_message)
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": full_message,
                "traceback": traceback.format_exc(),
            }

    # ── Solve ─────────────────────────────────────────────────────────────

    def solve(self, solver_name: str = "highs") -> dict:
        """
        Execute the current model with *solver_name*, capturing timing,
        solve_result, objective, and solver stdout.

        If the problem is infeasible, automatically run diagnostics.

        Returns a plain dict — server.py wraps it in a Pydantic response.
        """
        ampl = self.ensure_runtime()

        # Switch solver if requested
        if solver_name != self.current_solver:
            try:
                ampl.set_option("solver", solver_name)
                self.current_solver = solver_name
                logger.info("Solver switched to %s", solver_name)
            except Exception as exc:
                logger.warning("Failed to set solver '%s': %s", solver_name, exc)

        captured = io.StringIO()
        t0 = time.perf_counter()

        try:
            with redirect_stdout(captured):
                ampl.solve()
        except Exception as exc:
            captured.seek(0)
            raw_output = captured.read()
            self.last_solver_output = raw_output
            self.last_solve_result = "error"
            self.last_solve_time = time.perf_counter() - t0
            logger.error("Solve exception: %s", exc)
            return {
                "status": "error",
                "solve_result": "error",
                "objective_value": None,
                "solver_output": raw_output,
                "runtime_seconds": round(self.last_solve_time, 4),
                "solver_name": self.current_solver,
                "diagnostics": None,
                "variable_summary": {},
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_traceback": traceback.format_exc(),
            }

        elapsed = time.perf_counter() - t0
        captured.seek(0)
        raw_output = captured.read()

        self.last_solver_output = raw_output
        self.last_solve_time = elapsed

        solve_result = str(ampl.solve_result)
        self.last_solve_result = solve_result

        objective_value: Optional[float] = None
        var_summary: dict[str, Any] = {}

        if solve_result == "solved":
            objective_value = self._extract_objective(ampl)
            self.last_objective = objective_value
            var_summary = self._collect_variable_summary()

        diagnostics: Optional[InfeasibleDiagnostics] = None
        diagnostics_dict = None
        if solve_result == "infeasible":
            logger.warning("Solve returned INFEASIBLE — running diagnostics")
            diagnostics = self._diagnose_infeasibility()
            diagnostics_dict = diagnostics.model_dump() if diagnostics else None

        logger.info(
            "Solve complete | result=%s | solver=%s | time=%.3fs",
            solve_result,
            self.current_solver,
            elapsed,
        )

        return {
            "status": "success",
            "solve_result": solve_result,
            "objective_value": objective_value,
            "solver_output": raw_output,
            "runtime_seconds": round(elapsed, 4),
            "solver_name": self.current_solver,
            "diagnostics": diagnostics_dict,
            "variable_summary": var_summary,
        }

    # ── Variable extraction ───────────────────────────────────────────────

    def extract_variable(self, variable_name: str, export_dir: str = "./results") -> dict:
        """
        Extract a variable from the last solve as a list of dicts.

        Supports scalar, indexed (1D), and multi-dimensional variables.
        Returns a structured dict suitable for serialisation.
        """
        ampl = self.ensure_runtime()

        try:
            var = ampl.get_variable(variable_name)
        except Exception as exc:
            logger.error("Variable '%s' not found: %s", variable_name, exc)
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": f"Variable '{variable_name}' not found in model. {exc}",
                "traceback": traceback.format_exc(),
            }

        try:
            df: pd.DataFrame = var.get_values().to_pandas()
        except Exception as exc:
            logger.error("Failed to extract values for '%s': %s", variable_name, exc)
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "message": f"Failed to fetch values for '{variable_name}': {exc}",
                "traceback": traceback.format_exc(),
            }

        if df.empty:
            return {
                "status": "success",
                "variable_name": variable_name,
                "export_dir": export_dir,
                "row_count": 0,
                "file_path": None,
                "preview_rows": 0,
                "preview": [],
                "message": f"Variable '{variable_name}' has no assigned values.",
            }

        rows: list[dict] = df.to_dict(orient="records")
        total_rows = len(rows)

        if total_rows > 50:
            import os

            os.makedirs(export_dir, exist_ok=True)
            safe_name = variable_name.replace("[", "_").replace("]", "_").replace(",", "_")
            csv_path = f"{export_dir}/{safe_name}_{timestamp_iso().replace(':', '-')}.csv"
            df.to_csv(csv_path, index=False)
            preview = rows[:10]
            logger.info("Variable '%s' exported | rows=%d | path=%s", variable_name, total_rows, csv_path)
            return {
                "status": "success",
                "variable_name": variable_name,
                "export_dir": export_dir,
                "row_count": total_rows,
                "file_path": csv_path,
                "preview_rows": len(preview),
                "preview": preview,
                "message": f"Result too large ({total_rows} rows). Exported to CSV.",
            }

        return {
            "status": "success",
            "variable_name": variable_name,
            "export_dir": export_dir,
            "row_count": total_rows,
            "file_path": None,
            "preview_rows": total_rows,
            "preview": rows,
            "message": f"Extracted {total_rows} rows for variable '{variable_name}'.",
        }

    # ── Session state ─────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """Snapshot current engine state for the session-info tool."""
        ampl = self.ensure_runtime()
        var_count = 0
        con_count = 0
        try:
            var_count = ampl.get_variables().num_instances()
        except Exception:
            pass
        try:
            con_count = ampl.get_constraints().num_instances()
        except Exception:
            pass

        return {
            "status": "success",
            "models_loaded": list(self.loaded_models),
            "data_files_loaded": list(self.loaded_data_files),
            "last_solve_result": self.last_solve_result,
            "last_objective": self.last_objective,
            "variable_count": var_count,
            "constraint_count": con_count,
            "current_solver": self.current_solver,
            "available_solvers": list(self.available_solvers),
            "ampl_directory": self.ampl_dir,
        }

    # ── Solver options ────────────────────────────────────────────────────

    def set_solver_options(self, options: dict[str, Any]) -> dict:
        """Set solver-specific options on the AMPL instance."""
        ampl = self.ensure_runtime()
        set_opts: dict[str, Any] = {}
        errors: list[str] = []

        for key, value in options.items():
            try:
                ampl.set_option(key, str(value))
                set_opts[key] = value
                logger.info("Option set | %s = %s", key, value)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
                logger.error("Failed to set option '%s': %s", key, exc)

        if errors:
            return {
                "status": "error",
                "solver": self.current_solver,
                "options_set": set_opts,
                "message": "Some options failed: " + "; ".join(errors),
            }

        return {
            "status": "success",
            "solver": self.current_solver,
            "options_set": set_opts,
            "message": f"Set {len(set_opts)} solver option(s) on {self.current_solver}",
        }

    # ── .run script execution ─────────────────────────────────────────────

    def run_script(self, script_code: str, save_path: Optional[str] = None) -> dict:
        """
        Execute an AMPL .run script (command file).

        .run files can contain loops, conditionals, multiple solve statements,
        display commands, and parameter updates — far more powerful than
        injecting model code alone.

        If *save_path* is given, the script is also written to disk so the
        user can re-run or inspect it.
        """
        ampl = self.ensure_runtime()
        captured = io.StringIO()
        errors: list[str] = []
        solve_results: list[dict] = []
        t0 = time.perf_counter()

        # Optionally save to disk
        disk_path: Optional[Path] = None
        if save_path:
            disk_path = safe_path(save_path)
            try:
                Path(disk_path).parent.mkdir(parents=True, exist_ok=True)
                Path(disk_path).write_text(script_code, encoding="utf-8")
                logger.info("Script saved to %s", disk_path)
            except Exception as exc:
                errors.append(f"Failed to save script: {exc}")

        # Execute via ampl.eval() — AMPL processes .run commands the same way
        try:
            with redirect_stdout(captured):
                # Preprocess: intercept 'solve;' calls to track per-solve results
                statements = self._split_ampl_statements(script_code)
                solve_count = 0
                for stmt in statements:
                    stmt_stripped = stmt.strip()
                    if not stmt_stripped:
                        continue

                    t_stmt = time.perf_counter()
                    try:
                        ampl.eval(stmt)
                    except Exception as eval_exc:
                        errors.append(f"Statement error: {eval_exc}\n  near: {stmt_stripped[:200]}")
                        logger.warning("Script statement failed: %s", eval_exc)

                    # If this statement was a solve, capture the result
                    if stmt_stripped.startswith("solve") or stmt_stripped == "solve;":
                        solve_count += 1
                        try:
                            sr = str(ampl.solve_result)
                            obj = self._extract_objective(ampl)
                            solve_results.append({
                                "solve_index": solve_count,
                                "label": "",
                                "solve_result": sr,
                                "objective_value": obj,
                                "runtime_seconds": round(time.perf_counter() - t_stmt, 4),
                            })
                            self.last_solve_result = sr
                            self.last_objective = obj
                        except Exception:
                            solve_results.append({
                                "solve_index": solve_count,
                                "label": "",
                                "solve_result": "unknown",
                                "objective_value": None,
                                "runtime_seconds": round(time.perf_counter() - t_stmt, 4),
                            })
        except Exception as exc:
            errors.append(f"Script execution failed: {exc}")
            logger.error("run_script exception: %s", exc)

        captured.seek(0)
        stdout = captured.read()
        total_elapsed = time.perf_counter() - t0

        self.loaded_models.append(f"<run script (solves={len(solve_results)})>")
        logger.info(
            "Script executed | solves=%d | errors=%d | time=%.3fs",
            len(solve_results), len(errors), total_elapsed,
        )

        return {
            "status": "error" if errors else "success",
            "message": f"Script executed: {len(solve_results)} solve(s), {len(errors)} error(s) in {total_elapsed:.3f}s",
            "script_path": str(disk_path) if disk_path else None,
            "total_solves": len(solve_results),
            "solve_results": solve_results,
            "ampl_stdout": stdout,
            "errors": errors,
        }

    @staticmethod
    def _split_ampl_statements(code: str) -> list[str]:
        """
        Split AMPL code into individual statements, respecting:
          - Semicolons that end statements
          - Compound statements (for { }, repeat { }, if { })
          - Multi-line quoted strings
        Returns a list of complete AMPL statements.
        """
        statements: list[str] = []
        depth = 0          # brace depth
        current: list[str] = []
        in_single_quote = False
        in_double_quote = False
        i = 0

        while i < len(code):
            ch = code[i]

            # Quote toggling (skip escaped quotes)
            if ch == "'" and not in_double_quote:
                if i == 0 or code[i - 1] != "\\":
                    in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                if i == 0 or code[i - 1] != "\\":
                    in_double_quote = not in_double_quote

            if not in_single_quote and not in_double_quote:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)
                elif ch == ";" and depth == 0:
                    current.append(ch)
                    statements.append("".join(current))
                    current = []
                    i += 1
                    continue

            current.append(ch)
            i += 1

        # Flush trailing content
        remainder = "".join(current).strip()
        if remainder:
            statements.append(remainder)

        return statements

    # ── Gurobi configuration ──────────────────────────────────────────────

    def configure_gurobi(
        self,
        params: Optional[dict[str, Any]] = None,
        preset: Optional[str] = None,
    ) -> dict:
        """
        Configure Gurobi solver parameters with built-in domain knowledge.

        *preset* can be one of:
          - "default"   — reset to Gurobi defaults
          - "tune"      — enable automatic parameter tuning
          - "fast"      — emphasise speed over accuracy
          - "precise"   — emphasise accuracy (tight MIP gaps, no shortcuts)
          - "heuristic" — prioritise finding feasible solutions quickly
          - "balanced"  — moderate settings for general use
          - "barrier"   — use barrier algorithm for LP/QP

        *params* is a dict of explicit Gurobi parameter key-value pairs
        (e.g., {"MIPGap": 0.01, "TimeLimit": 300, "Threads": 4}).
        """
        ampl = self.ensure_runtime()

        # Ensure Gurobi is the active solver
        if "gurobi" not in self.available_solvers:
            return {
                "status": "error",
                "solver": self.current_solver,
                "params_set": [],
                "params_failed": [],
                "ampl_option_string": "",
                "message": "Gurobi solver is not available on this system. Detected solvers: "
                           + ", ".join(self.available_solvers),
            }

        try:
            ampl.set_option("solver", "gurobi")
            self.current_solver = "gurobi"
        except Exception as exc:
            logger.warning("Could not switch to Gurobi: %s", exc)

        merged_params: dict[str, Any] = {}

        # Apply preset
        preset_str = (preset or "").lower().strip()
        if preset_str and preset_str in GUROBI_PRESETS:
            merged_params.update(GUROBI_PRESETS[preset_str])
        elif preset_str and preset_str not in GUROBI_PRESETS:
            return {
                "status": "error",
                "solver": "gurobi",
                "params_set": [],
                "params_failed": [],
                "ampl_option_string": "",
                "message": f"Unknown preset '{preset}'. Available: {list(GUROBI_PRESETS.keys())}",
            }

        # Overlay explicit params
        if params:
            merged_params.update(params)

        # Build the gurobi_options string
        option_parts = []
        for key, value in merged_params.items():
            option_parts.append(f"{key}={value}")
        option_string = " ".join(option_parts)

        # Apply via AMPL option
        params_set: list[dict] = []
        params_failed: list[str] = []

        try:
            ampl.set_option("gurobi_options", option_string)
            logger.info("Gurobi options set: %s", option_string)

            for key, value in merged_params.items():
                info = GUROBI_PARAM_DB.get(key, {})
                params_set.append({
                    "name": key,
                    "value": value,
                    "description": info.get("desc", ""),
                    "category": info.get("category", ""),
                })
        except Exception as exc:
            params_failed.append(str(exc))
            logger.error("Failed to set Gurobi options: %s", exc)

        return {
            "status": "success" if not params_failed else "error",
            "solver": "gurobi",
            "params_set": params_set,
            "params_failed": params_failed,
            "ampl_option_string": f"option gurobi_options '{option_string}';",
            "message": f"Set {len(params_set)} Gurobi parameter(s)"
                       + (f" (preset={preset_str})" if preset_str else ""),
        }

    # ── Infeasible diagnostics ────────────────────────────────────────────

    def _diagnose_infeasibility(self) -> InfeasibleDiagnostics:
        """
        Run automated diagnostics when solve_result == 'infeasible'.

        Strategy (tried in order):
          1. IIS / Conflict refinement (Gurobi / CPLEX).
          2. Slack analysis — scan every constraint for violations.
          3. Human-readable pattern matching on the solver output.
        """
        ampl = self.ensure_runtime()
        infeasible_constraints: list[InfeasibleConstraint] = []
        possible_causes: list[str] = []
        relaxation_suggestions: list[str] = []
        iis_available = False

        # ── 1) IIS attempt (Gurobi / CPLEX) ───────────────────────────────
        iis_available = self._try_iis(ampl, infeasible_constraints)

        # ── 2) Slack / violation scan (always) ────────────────────────────
        self._slack_analysis(ampl, infeasible_constraints, possible_causes)

        # ── 3) Solver output keyword analysis ─────────────────────────────
        self._parse_solver_messages(possible_causes, relaxation_suggestions)

        # ── 4) Deduplicate ────────────────────────────────────────────────
        seen = set()
        unique_constraints: list[InfeasibleConstraint] = []
        for c in infeasible_constraints:
            if c.constraint_name not in seen:
                seen.add(c.constraint_name)
                unique_constraints.append(c)

        return InfeasibleDiagnostics(
            infeasible_constraints=unique_constraints,
            possible_causes=possible_causes if possible_causes else [
                "No constraints found with significant slack violations — the infeasibility may involve variable bounds or structural issues.",
            ],
            relaxation_suggestions=relaxation_suggestions if relaxation_suggestions else [
                "Try relaxing the most binding constraints by 1-5% and re-solve.",
                "Check for contradicting fixed variable bounds (var >= ub + epsilon).",
                "Verify input data consistency — e.g. demand exceeding capacity with zero inventory.",
            ],
            iis_available=iis_available,
        )

    def _try_iis(
        self,
        ampl: amplpy.AMPL,
        constraints_list: list[InfeasibleConstraint],
    ) -> bool:
        """Attempt IIS computation for Gurobi or CPLEX. Returns True if IIS ran."""
        solver_lower = self.current_solver.lower()
        if solver_lower not in ("gurobi", "cplex"):
            return False

        try:
            if solver_lower == "gurobi":
                ampl.set_option("gurobi_options", "iis=1")
            elif solver_lower == "cplex":
                ampl.set_option("cplex_options", "iis=1")

            # Re-solve with IIS enabled, ignoring the inevitable infeasibility
            captured = io.StringIO()
            with redirect_stdout(captured):
                try:
                    ampl.solve()
                except Exception:
                    pass

            # Collect constraints flagged in the IIS
            for c in ampl.get_constraints():
                try:
                    iis_val = c.get("iis")
                    if iis_val is not None and str(iis_val).lower() not in ("none", "", "0", "false"):
                        constraints_list.append(InfeasibleConstraint(
                            constraint_name=str(c.name()),
                            violation=float("inf"),
                        ))
                except Exception:
                    pass

            logger.info("IIS analysis completed with %s", self.current_solver)
            return True
        except Exception as exc:
            logger.warning("IIS attempt failed: %s", exc)
            return False

    def _slack_analysis(
        self,
        ampl: amplpy.AMPL,
        constraints_list: list[InfeasibleConstraint],
        possible_causes: list[str],
    ) -> None:
        """
        Scan every constraint and compute violation via slack / dual / lb / ub.
        Flag constraints with infeasible slack.
        """
        try:
            all_cons = ampl.get_constraints()
        except Exception:
            return

        violations: list[tuple[float, str, float, float | None, float | None, float | None]] = []

        for c in all_cons:
            try:
                body = float(c.body())
                lb = float(c.lb()) if c.lb() is not None and c.lb() != float("-inf") else None
                ub = float(c.ub()) if c.ub() is not None and c.ub() != float("inf") else None
                slack_val = float(c.slack())

                violation = 0.0
                if lb is not None and body < lb - 1e-6:
                    violation = max(violation, lb - body)
                if ub is not None and body > ub + 1e-6:
                    violation = max(violation, body - ub)

                if violation > 1e-6:
                    violations.append((violation, str(c.name()), body, lb, ub, slack_val))
            except Exception:
                continue

        # Sort by violation (largest first), keep top 20
        violations.sort(key=lambda x: x[0], reverse=True)
        for v in violations[:20]:
            violation, name, body, lb, ub, slack_val = v
            constraints_list.append(InfeasibleConstraint(
                constraint_name=name,
                violation=round(violation, 6),
                body=round(body, 6),
                lbound=round(lb, 6) if lb is not None else None,
                ubound=round(ub, 6) if ub is not None else None,
                slack=round(slack_val, 6) if slack_val is not None else None,
            ))
            possible_causes.append(
                f"Constraint '{name}' violated by {violation:.4f} "
                f"(body={body:.4f}, lb={lb}, ub={ub}, slack={slack_val})"
            )

        if violations:
            logger.info("Slack analysis found %d violated constraints", len(violations))

    def _parse_solver_messages(
        self,
        possible_causes: list[str],
        relaxation_suggestions: list[str],
    ) -> None:
        """Extract plain-English hints from the solver output."""
        output = self.last_solver_output.lower()

        keywords = {
            "bound infeasibility": "The solver detected a bound infeasibility — check variable lower/upper bounds.",
            "presolve": "Infeasibility detected during presolve — likely a structural issue with constraints.",
            "primal infeasible": "The primal problem is proven infeasible. Consider relaxing constraints.",
            "dual infeasible": "The dual is infeasible — the primal may be unbounded or the model ill-posed.",
            "infeasible": "Solver reports general infeasibility.",
        }

        for keyword, suggestion in keywords.items():
            if keyword in output:
                possible_causes.append(suggestion)

        if "infeasible" in output:
            relaxation_suggestions.append(
                "Use the infeasible constraints list above to identify binding constraints "
                "— try relaxing their RHS by 1-5%."
            )
            relaxation_suggestions.append(
                "If the problem is a MIP/MILP, check that integrality constraints are not "
                "causing conflicting requirements."
            )

    @staticmethod
    def _extract_objective(ampl: amplpy.AMPL) -> Optional[float]:
        """Extract the current objective value from the AMPL session."""
        try:
            objectives = list(ampl.get_objectives())
            if objectives:
                return float(objectives[0].value())
        except Exception:
            pass
        try:
            return float(ampl.get_value("_obj"))
        except Exception:
            pass
        return None

    def _collect_variable_summary(self) -> dict[str, Any]:
        """Build a lightweight summary of variables post-solve.

        Only fetches values for scalar variables to avoid drowning
        the LLM in large indexed-variable DataFrames.
        """
        ampl = self.ensure_runtime()
        summary: dict[str, Any] = {
            "total_variables": 0,
            "variables": {},
        }
        try:
            vars_ = ampl.get_variables()
            summary["total_variables"] = vars_.num_instances()
            for v in vars_:
                try:
                    vals = v.get_values().to_pandas()
                    # Only include scalar values (single-row results)
                    if len(vals) == 1 and len(vals.columns) == 1:
                        summary["variables"][str(v.name())] = float(vals.iloc[0, 0])
                except Exception:
                    pass
        except Exception:
            pass
        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Gurobi Parameter Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

GUROBI_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "tune": {
        "TuneTimeLimit": 60,
        "TuneOutput": 1,
    },
    "fast": {
        "MIPGap": 0.01,
        "MIPFocus": 1,
        "Presolve": 2,
        "Cuts": 1,
        "Heuristics": 0.3,
        "Threads": 4,
        "Method": 2,
    },
    "precise": {
        "MIPGap": 0.0001,
        "MIPFocus": 3,
        "NumericFocus": 1,
        "Presolve": 2,
        "Cuts": 3,
        "IntegralityFocus": 1,
        "Symmetry": 2,
    },
    "heuristic": {
        "MIPFocus": 1,
        "Heuristics": 0.8,
        "Cuts": 0,
        "Presolve": 1,
        "NoRelHeurTime": 30,
        "ZeroObjNodes": 10000,
    },
    "balanced": {
        "MIPGap": 0.001,
        "MIPFocus": 0,
        "Presolve": 2,
        "Cuts": 2,
        "Threads": 4,
    },
    "barrier": {
        "Method": 2,
        "Crossover": 0,
        "BarHomogeneous": 1,
        "BarConvTol": 1e-12,
    },
}

GUROBI_PARAM_DB: dict[str, dict[str, str]] = {
    # ── Termination ──
    "TimeLimit":     {"desc": "Solver time limit in seconds", "category": "Termination"},
    "MIPGap":        {"desc": "Relative MIP optimality gap tolerance", "category": "Termination"},
    "MIPGapAbs":     {"desc": "Absolute MIP optimality gap tolerance", "category": "Termination"},
    "NodeLimit":     {"desc": "Maximum MIP nodes to explore", "category": "Termination"},
    "IterationLimit":{"desc": "Maximum simplex iterations", "category": "Termination"},
    "SolutionLimit": {"desc": "Stop after finding this many feasible solutions", "category": "Termination"},
    "BestBdStop":    {"desc": "Stop when best bound reaches this value", "category": "Termination"},
    "BestObjStop":   {"desc": "Stop when best objective reaches this value", "category": "Termination"},
    "Cutoff":        {"desc": "Discard solutions worse than this value", "category": "Termination"},

    # ── MIP Strategy ──
    "MIPFocus":      {"desc": "MIP solver focus: 0=balanced, 1=feasible, 2=optimal, 3=bound", "category": "MIP"},
    "Heuristics":    {"desc": "Time spent on heuristics (0-1)", "category": "MIP"},
    "NoRelHeurTime": {"desc": "Heuristic time before root relaxation (seconds)", "category": "MIP"},
    "ZeroObjNodes":  {"desc": "Number of zero-objective nodes to explore", "category": "MIP"},
    "BranchDir":     {"desc": "Branch direction preference (-1=down, 0=auto, 1=up)", "category": "MIP"},
    "VarBranch":     {"desc": "Variable branching rule (-1=auto, 0=pseudo, 1=strong)", "category": "MIP"},
    "PumpPasses":    {"desc": "Number of feasibility pump passes", "category": "MIP"},
    "RINS":          {"desc": "RINS heuristic frequency (-1=auto, 0=off)", "category": "MIP"},
    "ZeroHalfCuts":  {"desc": "Zero-half cut aggressiveness (0-2)", "category": "MIP"},
    "SubMIPNodes":   {"desc": "Nodes explored in sub-MIP heuristics", "category": "MIP"},

    # ── Cuts ──
    "Cuts":          {"desc": "Global cut aggressiveness (-1=auto, 0=none, 1=moderate, 2=aggressive, 3=very aggressive)", "category": "Cuts"},
    "CliqueCuts":    {"desc": "Clique cut aggressiveness", "category": "Cuts"},
    "CoverCuts":     {"desc": "Cover cut aggressiveness", "category": "Cuts"},
    "FlowCoverCuts": {"desc": "Flow cover cut aggressiveness", "category": "Cuts"},
    "GomoryPasses":  {"desc": "Gomory cut passes", "category": "Cuts"},
    "MIRCuts":       {"desc": "MIR cut aggressiveness", "category": "Cuts"},
    "NetworkCuts":   {"desc": "Network cut aggressiveness", "category": "Cuts"},
    "ModKCuts":      {"desc": "Mod-K cut aggressiveness", "category": "Cuts"},

    # ── Presolve ──
    "Presolve":      {"desc": "Presolve aggressiveness (-1=auto, 0=none, 1=conservative, 2=aggressive)", "category": "Presolve"},
    "PreDual":       {"desc": "Presolve dual reduction (-1=auto, 0=off, 1=on)", "category": "Presolve"},
    "PrePasses":     {"desc": "Presolve passes limit (-1=auto)", "category": "Presolve"},
    "Symmetry":      {"desc": "Symmetry detection (0=none, 1=moderate, 2=aggressive)", "category": "Presolve"},
    "Aggregate":     {"desc": "Aggregate variables in presolve (0=off, 1=on)", "category": "Presolve"},

    # ── Algorithm ──
    "Method":        {"desc": "LP solver method (-1=auto, 0=primal simplex, 1=dual simplex, 2=barrier, 3=concurrent)", "category": "Algorithm"},
    "Crossover":     {"desc": "Barrier crossover (0=off, 1=on)", "category": "Algorithm"},
    "BarHomogeneous":{"desc": "Barrier homogeneous algorithm (0=off, 1=on)", "category": "Algorithm"},
    "BarConvTol":    {"desc": "Barrier convergence tolerance", "category": "Algorithm"},
    "BarOrder":      {"desc": "Barrier ordering algorithm (-1=auto, 0=approx minimum degree, 1=nested dissection)", "category": "Algorithm"},
    "ConcurrentMIP": {"desc": "Number of concurrent MIP solves", "category": "Algorithm"},
    "ConcurrentJobs": {"desc": "Distributed concurrent jobs", "category": "Algorithm"},

    # ── Numerics ──
    "NumericFocus":  {"desc": "Numerical precision focus (0=auto, 1=moderate, 2=most precise)", "category": "Numerics"},
    "FeasibilityTol":{"desc": "Feasibility tolerance", "category": "Numerics"},
    "OptimalityTol": {"desc": "Optimality tolerance (reduced cost)", "category": "Numerics"},
    "IntFeasTol":    {"desc": "Integer feasibility tolerance", "category": "Numerics"},
    "MarkowitzTol":  {"desc": "Markowitz tolerance for simplex", "category": "Numerics"},
    "ScaleFlag":     {"desc": "Model scaling (0=none, 1=row, 2=column, 3=both)", "category": "Numerics"},

    # ── Performance ──
    "Threads":       {"desc": "Number of threads / CPU cores", "category": "Performance"},
    "MemLimit":      {"desc": "Memory limit in GB", "category": "Performance"},
    "NodefileStart": {"desc": "Nodefile start threshold (GB of memory)", "category": "Performance"},
    "NodefileDir":   {"desc": "Directory for nodefile storage", "category": "Performance"},

    # ── Tuning ──
    "TuneTimeLimit": {"desc": "Time limit for parameter tuning (seconds)", "category": "Tuning"},
    "TuneOutput":    {"desc": "Tuning output verbosity (0-3)", "category": "Tuning"},
    "TuneTrials":    {"desc": "Number of tuning trials per parameter set", "category": "Tuning"},
    "TuneResults":   {"desc": "Number of tuning results to return", "category": "Tuning"},

    # ── Output ──
    "OutputFlag":    {"desc": "Enable (1) or disable (0) solver output", "category": "Output"},
    "LogFile":       {"desc": "Write solver log to file", "category": "Output"},
    "DisplayInterval": {"desc": "Log line interval (seconds)", "category": "Output"},

    # ── IIS ──
    "IISMethod":     {"desc": "IIS computation method (0=fast heuristic, 1=more thorough)", "category": "IIS"},

    # ── Distributed ──
    "DistributedMIPJobs": {"desc": "Number of distributed MIP workers", "category": "Distributed"},
    "WorkerPassword":      {"desc": "Distributed worker password", "category": "Distributed"},
    "WorkerPool":          {"desc": "Cluster worker pool specification", "category": "Distributed"},
}
