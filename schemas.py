"""
Pydantic schemas for AMPL MCP Server — Academic Optimization Research Edition.

Defines all Tool Request/Response models, Error types, and Solve Result structures.
No bare dicts cross module boundaries.
"""

from __future__ import annotations

from typing import Any, Optional, Union
from pydantic import BaseModel, Field


# ─── Base / Shared ───────────────────────────────────────────────────────────

class BaseResponse(BaseModel):
    """Every tool response inherits from this."""
    status: str = Field(default="success", description="Execution status: success | error")


class ErrorDetail(BaseModel):
    """Structured error payload attached to error responses."""
    error_type: str = Field(default="", description="Exception class or error category")
    message: str = Field(default="", description="Human-readable error description")
    traceback: str = Field(default="", description="Python traceback string")


class ErrorResponse(BaseResponse):
    """Unified error envelope returned by all tools on failure."""
    status: str = Field(default="error")
    error: ErrorDetail = Field(default_factory=ErrorDetail)


# ─── Tool 1: reset_workspace ─────────────────────────────────────────────────

class ResetWorkspaceResponse(BaseResponse):
    message: str = Field(default="AMPL workspace reset successfully")


# ─── Tool 2: set_ampl_model ──────────────────────────────────────────────────

class SetModelResponse(BaseResponse):
    message: str = Field(default="")
    model_size_bytes: int = Field(default=0)
    line_count: int = Field(default=0)
    ampl_log: str = Field(default="", description="Raw AMPL interpreter output")


# ─── Tool 3: inject_experiment_data ───────────────────────────────────────────

class TableLoadInfo(BaseModel):
    """Summary for one loaded table / sheet."""
    name: str
    row_count: int
    column_count: int
    string_columns: list[str] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)


class InjectDataResponse(BaseResponse):
    message: str = Field(default="")
    tables_loaded: list[TableLoadInfo] = Field(default_factory=list)
    primary_keys_detected: list[str] = Field(default_factory=list)


# ─── Tool 4: run_optimization ────────────────────────────────────────────────

class InfeasibleConstraint(BaseModel):
    constraint_name: str
    violation: float
    body: Optional[float] = None
    lbound: Optional[float] = None
    ubound: Optional[float] = None
    slack: Optional[float] = None


class InfeasibleDiagnostics(BaseModel):
    """Populated when solve_result == 'infeasible'."""
    infeasible_constraints: list[InfeasibleConstraint] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    relaxation_suggestions: list[str] = Field(default_factory=list)
    iis_available: bool = Field(default=False)


class SolveResultResponse(BaseResponse):
    solve_result: str = Field(default="", description="AMPL solve_result string e.g. 'solved', 'infeasible'")
    objective_value: Optional[float] = Field(default=None)
    solver_output: str = Field(default="", description="Raw solver stdout")
    runtime_seconds: float = Field(default=0.0)
    solver_name: str = Field(default="highs")
    diagnostics: Optional[InfeasibleDiagnostics] = Field(default=None)
    variable_summary: dict[str, Any] = Field(default_factory=dict)


# ─── Tool 5: extract_solution ────────────────────────────────────────────────

class ExtractSolutionResponse(BaseResponse):
    variable_name: str = Field(default="")
    export_dir: str = Field(default="./results")
    row_count: int = Field(default=0, description="Total result rows (may exceed preview)")
    file_path: Optional[str] = Field(default=None, description="CSV path if exported (>50 rows)")
    preview_rows: int = Field(default=0)
    preview: list[dict[str, Any]] = Field(default_factory=list)
    message: str = Field(default="")


# ─── Tool 6: get_session_state ───────────────────────────────────────────────

class SessionStateResponse(BaseResponse):
    models_loaded: list[str] = Field(default_factory=list)
    data_files_loaded: list[str] = Field(default_factory=list)
    last_solve_result: Optional[str] = Field(default=None)
    last_objective: Optional[float] = Field(default=None)
    variable_count: int = Field(default=0)
    constraint_count: int = Field(default=0)
    current_solver: str = Field(default="highs")
    available_solvers: list[str] = Field(default_factory=list)
    ampl_directory: Optional[str] = Field(default=None)


# ─── Tool 7: set_solver_options ──────────────────────────────────────────────

class SetSolverOptionsResponse(BaseResponse):
    solver: str = Field(default="")
    options_set: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="")


# ─── Union type for tool dispatch ────────────────────────────────────────────

ToolResponse = Union[
    ResetWorkspaceResponse,
    SetModelResponse,
    InjectDataResponse,
    SolveResultResponse,
    ExtractSolutionResponse,
    SessionStateResponse,
    SetSolverOptionsResponse,
    ErrorResponse,
]
