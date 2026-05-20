#!/usr/bin/env python3
"""
AMPL MCP Server — Academic Optimization Research Edition.

A production-grade, stateful MCP server that lets LLMs (Claude Desktop,
VS Code Agent, Cursor) build AMPL models, inject experimental data,
execute optimisation, diagnose infeasibility, and extract results —
all within a persistent AMPL session.

Start:
    python server.py

Transport: stdio (default for MCP).
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    print("FATAL: fastmcp package not installed. Run: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

from ampl_engine import get_engine, get_ampl_directory
from data_pipeline import DataPipeline
from schemas import (
    ResetWorkspaceResponse,
    SetModelResponse,
    InjectDataResponse,
    SolveResultResponse,
    ExtractSolutionResponse,
    SessionStateResponse,
    SetSolverOptionsResponse,
    ErrorResponse,
    ErrorDetail,
)
from utils import logger, timestamp_iso, safe_path

# ─── FastMCP application ─────────────────────────────────────────────────────

mcp = FastMCP("AMPL Research MCP Server")

# Module-level pipeline (dtype rules persist across calls for multi-file experiments)
_pipeline = DataPipeline()


# ─── Helper: structured error ────────────────────────────────────────────────

def _error_response(exc: Exception) -> ErrorResponse:
    """Build a uniform error envelope from an exception."""
    return ErrorResponse(
        status="error",
        error=ErrorDetail(
            error_type=type(exc).__name__,
            message=str(exc)[:2000],
            traceback=traceback.format_exc()[-3000:],
        ),
    )


# ─── Tool 1: reset_workspace ─────────────────────────────────────────────────

@mcp.tool()
def reset_workspace() -> dict:
    """
    Completely reset the AMPL workspace.

    Clears all loaded models, data, variables, solver state,
    cached results, and the data pipeline dtype rules.
    Use this before starting a brand-new experiment.
    """
    logger.info("Tool: reset_workspace")
    try:
        engine = get_engine()
        engine.reset()
        _pipeline.reset()
        return ResetWorkspaceResponse(
            status="success",
            message="AMPL workspace reset successfully. All models, data, and solver state cleared.",
        ).model_dump()
    except Exception as exc:
        logger.error("reset_workspace failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Tool 2: set_ampl_model ──────────────────────────────────────────────────

@mcp.tool()
def set_ampl_model(model_code: str) -> dict:
    """
    Inject a complete AMPL .mod file (as a string) into the persistent session.

    The model_code should contain valid AMPL syntax: sets, parameters, variables,
    objective, and constraints.  Full syntax errors are returned with line numbers
    so the LLM can self-correct.

    Args:
        model_code: Complete AMPL model source code as a string.
    """
    logger.info("Tool: set_ampl_model | chars=%d", len(model_code))
    try:
        engine = get_engine()
        result = engine.load_model(model_code)
        if result.get("status") == "error":
            return ErrorResponse(
                status="error",
                error=ErrorDetail(
                    error_type=result.get("error_type", "AMPLSyntaxError"),
                    message=result.get("message", ""),
                    traceback=result.get("traceback", ""),
                ),
            ).model_dump()
        return SetModelResponse(
            status="success",
            message=result["message"],
            model_size_bytes=result["model_size_bytes"],
            line_count=result["line_count"],
            ampl_log=result.get("ampl_log", ""),
        ).model_dump()
    except Exception as exc:
        logger.error("set_ampl_model failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Tool 3: inject_experiment_data ──────────────────────────────────────────

@mcp.tool()
def inject_experiment_data(file_path: str, is_secondary_file: bool = False) -> dict:
    """
    Load experimental data from an Excel (.xlsx/.xls) or CSV file into the
    active AMPL session.

    The pipeline:
      1. Reads all columns as strings to prevent silent numeric coercion.
      2. Detects genuinely numeric columns and converts them safely.
      3. Preserves ID-like strings (001, 01A, BUS001) as strings.
      4. Transforms the DataFrame into AMPL .dat format with 1-based indexing.
      5. Injects the data into the persistent AMPL session.

    Args:
        file_path:  Absolute or relative path to the data file.
        is_secondary_file: If True, the file shares dtype rules with a
                           previously loaded primary file (multi-file
                           experiment support).
    """
    logger.info(
        "Tool: inject_experiment_data | path=%s | secondary=%s",
        file_path,
        is_secondary_file,
    )
    try:
        if not is_secondary_file:
            _pipeline.reset()

        # Step 1: Read
        tables = _pipeline.read_file(file_path)

        # Step 2: Infer & apply dtypes
        tables = _pipeline.infer_and_apply_dtypes(tables)

        # Step 3: Transform to AMPL .dat
        ampl_data_str = _pipeline.to_ampl_data(tables)

        # Step 4: Inject into AMPL session
        engine = get_engine()
        result = engine.inject_data(ampl_data_str, source_label=file_path)

        if result.get("status") == "error":
            return ErrorResponse(
                status="error",
                error=ErrorDetail(
                    error_type=result.get("error_type", "AMPLDataError"),
                    message=result.get("message", ""),
                    traceback=result.get("traceback", ""),
                ),
            ).model_dump()

        # Step 5: Build summary
        summary = _pipeline.get_loading_summary(tables)

        return InjectDataResponse(
            status="success",
            message=f"Data from '{file_path}' loaded ({len(summary['tables_loaded'])} table(s)).",
            tables_loaded=summary["tables_loaded"],
            primary_keys_detected=summary["primary_keys_detected"],
        ).model_dump()

    except FileNotFoundError as exc:
        return ErrorResponse(
            status="error",
            error=ErrorDetail(
                error_type="FileNotFoundError",
                message=str(exc),
                traceback=traceback.format_exc(),
            ),
        ).model_dump()
    except Exception as exc:
        logger.error("inject_experiment_data failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Tool 4: run_optimization ────────────────────────────────────────────────

@mcp.tool()
def run_optimization(solver_name: str = "highs") -> dict:
    """
    Execute the loaded model with the specified solver.

    Supports dynamic solver switching.  If the solve result is 'infeasible',
    automatic diagnostics are run: IIS/conflict analysis (Gurobi/CPLEX),
    slack-based violation scanning, and human-readable relaxation suggestions.

    Args:
        solver_name: Solver to use. Supported: highs, gurobi, cplex, cbc, scip, etc.
                     Must be installed and licensed separately.
    """
    logger.info("Tool: run_optimization | solver=%s", solver_name)
    try:
        engine = get_engine()
        result = engine.solve(solver_name=solver_name)

        if result.get("status") == "error":
            error_detail = ErrorDetail(
                error_type=result.get("error_type", "SolveError"),
                message=result.get("error_message", result.get("solver_output", "Unknown solve error")),
                traceback=result.get("error_traceback", ""),
            )
            return ErrorResponse(status="error", error=error_detail).model_dump()

        return SolveResultResponse(**result).model_dump()
    except Exception as exc:
        logger.error("run_optimization failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Tool 5: extract_solution ────────────────────────────────────────────────

@mcp.tool()
def extract_solution(variable_name: str, export_dir: str = "./results") -> dict:
    """
    Extract the value of a named variable from the last solve.

    Supports scalar, indexed (1D), and multi-dimensional variables.
    If the result exceeds 50 rows, it is automatically exported to CSV in
    *export_dir* and only a 10-row preview is returned to prevent token overflow.

    Args:
        variable_name: AMPL variable name (e.g. 'x', 'flow', 'cost[i,j]').
        export_dir:   Directory for CSV export when results exceed 50 rows.
    """
    logger.info("Tool: extract_solution | var=%s | dir=%s", variable_name, export_dir)
    try:
        import os
        os.makedirs(export_dir, exist_ok=True)

        engine = get_engine()
        result = engine.extract_variable(variable_name, export_dir=export_dir)

        if result.get("status") == "error":
            return ErrorResponse(
                status="error",
                error=ErrorDetail(
                    error_type=result.get("error_type", "VariableError"),
                    message=result.get("message", ""),
                    traceback=result.get("traceback", ""),
                ),
            ).model_dump()

        return ExtractSolutionResponse(
            status="success",
            variable_name=variable_name,
            export_dir=export_dir,
            row_count=result["row_count"],
            file_path=result.get("file_path"),
            preview_rows=result["preview_rows"],
            preview=result["preview"],
            message=result["message"],
        ).model_dump()

    except Exception as exc:
        logger.error("extract_solution failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Tool 6: get_session_state ───────────────────────────────────────────────

@mcp.tool()
def get_session_state() -> dict:
    """
    Return the current state of the AMPL session.

    Reports loaded models, data files, last solve result, objective value,
    variable/constraint counts, and the active solver.
    """
    logger.info("Tool: get_session_state")
    try:
        engine = get_engine()
        state = engine.get_state()
        return SessionStateResponse(**state).model_dump()
    except Exception as exc:
        logger.error("get_session_state failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Tool 7: set_solver_options ──────────────────────────────────────────────

@mcp.tool()
def set_solver_options(options_json: str) -> dict:
    """
    Set solver-specific options on the active AMPL session.

    Args:
        options_json: A JSON string of key-value pairs, e.g.
                      '{"outlev": 1, "timelim": 60, "threads": 4}'.
    """
    logger.info("Tool: set_solver_options | options=%s", options_json)
    try:
        import json

        options: dict[str, Any] = json.loads(options_json)
        if not isinstance(options, dict):
            return ErrorResponse(
                status="error",
                error=ErrorDetail(
                    error_type="ValueError",
                    message="options_json must deserialize to a JSON object (key-value pairs).",
                    traceback="",
                ),
            ).model_dump()

        engine = get_engine()
        result = engine.set_solver_options(options)

        if result.get("status") == "error":
            return ErrorResponse(
                status="error",
                error=ErrorDetail(
                    error_type="SolverOptionError",
                    message=result.get("message", ""),
                    traceback="",
                ),
            ).model_dump()

        return SetSolverOptionsResponse(**result).model_dump()

    except json.JSONDecodeError as exc:
        return ErrorResponse(
            status="error",
            error=ErrorDetail(
                error_type="JSONDecodeError",
                message=f"Invalid JSON: {exc}",
                traceback="",
            ),
        ).model_dump()
    except Exception as exc:
        logger.error("set_solver_options failed: %s", exc)
        return _error_response(exc).model_dump()


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Start the AMPL MCP Server with a health check."""
    print("=" * 60, file=sys.stderr)
    print("  AMPL MCP Server — Academic Optimization Research Edition", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    ampl_dir = get_ampl_directory()
    print(f"  AMPL directory: {ampl_dir or '(not detected — relying on amplpy defaults)'}", file=sys.stderr)

    try:
        engine = get_engine()
        state = engine.get_state()
        print(f"  Solver:         {state['current_solver']}", file=sys.stderr)
        print(f"  Available:      {', '.join(state['available_solvers']) or '(none)'}", file=sys.stderr)
        print(f"  Models loaded:  {len(state['models_loaded'])}", file=sys.stderr)
        print(f"  Data files:     {len(state['data_files_loaded'])}", file=sys.stderr)
    except Exception as exc:
        print(f"  WARNING: AMPL engine health check failed: {exc}", file=sys.stderr)
        logger.warning("Startup health check failed: %s", exc)

    print("=" * 60, file=sys.stderr)
    print("  Starting stdio transport...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    try:
        mcp.run(transport="stdio")
    except Exception as _startup_err:
        msg = f"FATAL: server crashed on startup: {_startup_err}\n{traceback.format_exc()}"
        print(msg, file=sys.stderr)
        logger.critical(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
