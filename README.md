# AMPL MCP Server — Academic Optimization Research Edition

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![uv](https://img.shields.io/badge/uv-ready-5A4FE4?logo=astral)](https://docs.astral.sh/uv/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0%2B-FF6B00)](https://github.com/fastmcp-me/fastmcp-python)
[![AMPL](https://img.shields.io/badge/AMPL-ready-005282)](https://ampl.com)
[![MCP](https://img.shields.io/badge/Model%20Context%20Protocol-1.0-404040)](https://modelcontextprotocol.io)

**Give LLMs a persistent AMPL optimisation session.**

</div>

A **production-grade, stateful** [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
server that enables LLMs — Claude Desktop, VS Code Agent, Cursor — to build AMPL models,
inject experimental data, solve, diagnose infeasibility, extract results, and iterate
across multiple rounds of an optimisation study, all within a single persistent session.

Designed for **academic operations research** and **mathematical programming research**
workflows where multi-round experimentation is the norm, not a one-off.

---

## Table of Contents

- [AMPL MCP Server — Academic Optimization Research Edition](#ampl-mcp-server--academic-optimization-research-edition)
  - [Table of Contents](#table-of-contents)
  - [Why This Exists](#why-this-exists)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Steps](#setup-steps)
    - [AMPL Detection](#ampl-detection)
    - [Solver Setup](#solver-setup)
  - [MCP Tools](#mcp-tools)
    - [1. `reset_workspace`](#1-reset_workspace)
    - [2. `set_ampl_model`](#2-set_ampl_model)
    - [3. `inject_experiment_data`](#3-inject_experiment_data)
    - [4. `run_optimization`](#4-run_optimization)
    - [5. `extract_solution`](#5-extract_solution)
    - [6. `get_session_state`](#6-get_session_state)
    - [7. `set_solver_options`](#7-set_solver_options)
    - [8. `run_ampl_script`](#8-run_ampl_script)
    - [9. `configure_gurobi`](#9-configure_gurobi)
  - [Example Workflow](#example-workflow)
  - [LLM Client Configuration](#llm-client-configuration)
    - [Claude Desktop](#claude-desktop)
    - [VS Code / Cursor](#vs-code--cursor)
  - [Project Structure](#project-structure)
  - [Key Design Principles](#key-design-principles)
    - [1. Stateful Session Architecture](#1-stateful-session-architecture)
    - [2. String-Preserving Data Pipeline](#2-string-preserving-data-pipeline)
    - [3. 1-Based Indexing](#3-1-based-indexing)
    - [4. Production Error Handling](#4-production-error-handling)
    - [5. Token Overflow Prevention](#5-token-overflow-prevention)
  - [Infeasible Diagnostics](#infeasible-diagnostics)
    - [Layer 1: IIS / Conflict Analysis](#layer-1-iis--conflict-analysis)
    - [Layer 2: Slack-Based Violation Scanning](#layer-2-slack-based-violation-scanning)
    - [Layer 3: Solver Message Parsing](#layer-3-solver-message-parsing)
    - [Response Shape](#response-shape)
  - [Error Handling](#error-handling)
  - [Logging](#logging)
  - [Development](#development)
    - [Requirements](#requirements)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [Related Projects](#related-projects)
  - [License](#license)

---

## Why This Exists

LLMs are increasingly used as research assistants in optimization modeling. However,
existing MCP servers for AMPL follow a **"one-shot script"** pattern — each tool call
spawns a new subprocess, losing all state between interactions. That makes iterative
model refinement impossible.

This server adopts a **stateful session architecture**: the AMPL runtime is a singleton.
You can:

```
Upload model → Load data → Solve → Adjust parameters → Re-solve → Extract results
```

...all in the same session, exactly as you would work interactively at the AMPL prompt.

This follows the design philosophy of the [MATLAB MCP Core Server](https://github.com/MathWorks-Teaching-Resources/MATLAB-mcp-core-server),
where tools are *state operations*, not ephemeral scripts.

---

## Architecture

```
┌──────────────────────┐     stdio      ┌──────────────────────────────┐
│  Claude Desktop /    │ ◄────────────► │  AMPL MCP Server             │
│  VS Code Agent /     │   JSON-RPC     │  (FastMCP 2.x + stdio)       │
│  Cursor               │                │                              │
└──────────────────────┘                │  ┌────────────────────────┐  │
                                        │  │  ampl_engine.py        │  │
                                        │  │  · Singleton AMPL      │  │
                                        │  │  · Solver management   │  │
                                        │  │  · IIS diagnostics     │  │
                                        │  └───────────┬────────────┘  │
                                        │              │               │
                                        │  ┌───────────▼────────────┐  │
                                        │  │  data_pipeline.py      │  │
                                        │  │  · String-safe read    │  │
                                        │  │  · 1-based indexing    │  │
                                        │  │  · Excel/CSV → AMPL    │  │
                                        │  └────────────────────────┘  │
                                        │                              │
                                        │  ┌────────────────────────┐  │
                                        │  │  schemas.py             │  │
                                        │  │  · Pydantic models      │  │
                                        │  │  · Structured errors    │  │
                                        │  └────────────────────────┘  │
                                        └──────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jysw980/ampl-mcp-server.git
cd ampl-mcp-server

# 2. Install dependencies (uv creates the venv automatically)
uv sync

# 3. If AMPL is not on PATH, set the environment variable
#    Windows (PowerShell):
$env:AMPL_PATH = "path_to_your_ampl.exe"
#    macOS / Linux:
export AMPL_PATH="/opt/ampl"

# 4. Start the server
uv run python server.py
```

You should see a startup banner:
```
============================================================
  AMPL MCP Server — Academic Optimization Research Edition
============================================================
  AMPL directory: path_to_your_ampl.exe
  Solver:         highs
  Available:      highs, gurobi, cplex
  Models loaded:  0
  Data files:     0
============================================================
  Starting stdio transport...
============================================================
```

---

## Installation

### Prerequisites

- **Python 3.10+** (tested on 3.11)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package and project manager
- **AMPL** with at least one solver (see [Solver Setup](#solver-setup))

### Setup Steps

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and sync
git clone https://github.com/jysw980/ampl-mcp-server.git
cd ampl-mcp-server
uv sync
```

### AMPL Detection

The server detects your AMPL installation automatically, in this order:

| Priority | Method | Example |
|----------|--------|---------|
| 1 | `AMPL_PATH` env var | `export AMPL_PATH=/opt/ampl` |
| 2 | `AMPLPATH` env var | (legacy amplpy convention) |
| 3 | `PATH` search | Finds `ampl.exe` on your PATH |
| 4 | Drive-root glob scan | Windows: scans `D:\`, `C:\` for `ampl*` |
| 5 | Standard install paths | `C:\AMPL`, `/opt/ampl`, `~/AMPL` |

If detection fails, set `AMPL_PATH` explicitly to the directory containing `ampl`
(or `ampl.exe` on Windows).

### Solver Setup

The server automatically probes available solvers at startup and selects the first
working one if the default (`highs`) is unavailable.

| Solver | License | Installation |
|--------|---------|-------------|
| [HiGHS](https://highs.dev/) | Open-source | `uv pip install highspy` or bundled with AMPL |
| [Gurobi](https://www.gurobi.com/) | Academic / Commercial | `uv pip install gurobipy` + license file |
| [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) | Academic / Commercial | `uv pip install cplex` + license |
| [CBC](https://github.com/coin-or/Cbc) | Open-source | Bundled with AMPL Community Edition |
| [SCIP](https://www.scipopt.org/) | Academic | `uv pip install pyscipopt` |
| [IPOPT](https://coin-or.github.io/Ipopt/) | Open-source | Bundled with AMPL |

> **No commercial solvers?** Install [HiGHS](https://highs.dev/) — it's open-source
> and handles LP, MIP, and QP problems.

---

## MCP Tools

All nine tools are documented below with their purpose, signature, and return shape.

### 1. `reset_workspace`

Purge the entire AMPL session — models, data, variables, solver state, and cache.

```
reset_workspace() → { status: "success", message: "..." }
```

Use this when starting a completely new experiment.

### 2. `set_ampl_model`

Inject a full `.mod` file as a string into the persistent AMPL session. Syntax errors
are returned **with line numbers** so the LLM can self-correct.

```
set_ampl_model(model_code: str) → {
  status: "success" | "error",
  message: str,
  model_size_bytes: int,
  line_count: int,
  ampl_log: str
}
```

Errors are *never* swallowed — the full AMPL error stack is surfaced.

### 3. `inject_experiment_data`

Load `.xlsx`, `.xls`, or `.csv` files into the AMPL session. The pipeline:

1. Reads **all columns as strings** to prevent silent numeric coercion
2. Detects unambiguously numeric columns and converts them safely
3. Preserves ID-like strings (`001`, `01A`, `BUS001`, `34E`) as-is
4. Transforms the DataFrame to AMPL `.dat` format with 1-based indexing

```
inject_experiment_data(
  file_path: str,
  is_secondary_file: bool = false
) → {
  status: str,
  message: str,
  tables_loaded: [{ name, row_count, column_count, string_columns, numeric_columns }],
  primary_keys_detected: [str]
}
```

Set `is_secondary_file: true` to share dtype rules with a previously loaded file
(multi-file experiment support).

### 4. `run_optimization`

Solve the loaded model. Supports dynamic solver switching. If the result is
`infeasible`, automatic diagnostics kick in (IIS, slack analysis, human-readable
suggestions).

```
run_optimization(solver_name: str = "highs") → {
  status: str,
  solve_result: "solved" | "infeasible" | "unbounded" | "error" | ...,
  objective_value: float | null,
  solver_output: str,
  runtime_seconds: float,
  solver_name: str,
  diagnostics: InfeasibleDiagnostics | null,
  variable_summary: { total_variables, variables: { name: value } }
}
```

See [Infeasible Diagnostics](#infeasible-diagnostics) for details on what
`diagnostics` contains.

### 5. `extract_solution`

Extract a variable's values from the last solve. Handles scalar, indexed (1D),
and multi-dimensional variables. If the result exceeds **50 rows**, it is
automatically exported to CSV and only a 10-row preview is returned — preventing
token overflow.

```
extract_solution(
  variable_name: str,
  export_dir: str = "./results"
) → {
  status: str,
  variable_name: str,
  row_count: int,
  file_path: str | null,    // CSV path if >50 rows
  preview_rows: int,
  preview: [dict],           // max 10 rows
  message: str
}
```

### 6. `get_session_state`

Snapshot the current AMPL session — loaded models, data files, last solve result,
objective value, variable/constraint counts, active solver, and available solvers.

```
get_session_state() → {
  models_loaded: [str],
  data_files_loaded: [str],
  last_solve_result: str | null,
  last_objective: float | null,
  variable_count: int,
  constraint_count: int,
  current_solver: str,
  available_solvers: [str],
  ampl_directory: str | null
}
```

### 7. `set_solver_options`

Set solver-specific parameters as a JSON key-value object.

```
set_solver_options(options_json: str) → {
  solver: str,
  options_set: dict,
  message: str
}
```

Example: `'{"outlev": 1, "timelim": 60, "threads": 4}'`

### 8. `run_ampl_script`

Execute an AMPL `.run` script (command file) in the persistent session.

`.run` files are AMPL's scripting mechanism — they support `for` / `repeat` loops,
`if-then-else` conditionals, multiple `solve` statements with parameter sweeps,
`let` for in-place parameter changes, and `display` / `printf` for output.

This is the preferred tool for **iterative optimisation workflows**: sensitivity
analysis, parameter sweeps, decomposition algorithms, and any workflow requiring
more than a single solve. The tool captures the result of **every** solve within
the script.

```
run_ampl_script(
  script_code: str,
  save_path: str = ""
) → {
  status: str,
  message: str,
  script_path: str | null,     // path if saved to disk
  total_solves: int,
  solve_results: [{
    solve_index: int,           // 1-based solve index
    label: str,
    solve_result: str,
    objective_value: float | null,
    runtime_seconds: float
  }],
  ampl_stdout: str,             // full AMPL interpreter output
  errors: [str]                 // per-statement errors (non-fatal)
}
```

### 9. `configure_gurobi`

Configure Gurobi solver parameters with a built-in knowledge base of 50+ parameters
and 7 named presets.

**Presets**: `default`, `tune` (auto parameter tuning), `fast` (speed over accuracy),
`precise` (tight tolerances), `heuristic` (feasibility first), `balanced` (general use),
`barrier` (interior-point LP/QP).

**Known parameters** (50+): TimeLimit, MIPGap, MIPFocus, Heuristics, Cuts, Presolve,
Method, Threads, NumericFocus, Symmetry, BarConvTol, and many more — each with
human-readable descriptions.

```
configure_gurobi(
  params_json: str = "{}",      // e.g. '{"MIPGap": 0.005, "TimeLimit": 120}'
  preset: str = ""              // named preset (see above)
) → {
  solver: str,
  params_set: [{
    name: str,
    value: any,
    description: str,
    category: str               // e.g. MIP, Termination, Cuts, Algorithm
  }],
  params_failed: [str],
  ampl_option_string: str,      // assembled 'option gurobi_options ...' string
  message: str
}
```

---

## Example Workflow

Here's how an LLM interacts with the server across a typical session:

```
# Session 1: Model building
LLM → set_ampl_model("""
    set PRODUCTS;
    param profit{PRODUCTS};
    param hours{PRODUCTS};
    param capacity;
    var x{p in PRODUCTS} >= 0;
    maximize total_profit: sum{p in PRODUCTS} profit[p] * x[p];
    s.t. labor: sum{p in PRODUCTS} hours[p] * x[p] <= capacity;
""")

# Session 2: Data loading
LLM → inject_experiment_data("./data/products.xlsx")

# Session 3: First solve
LLM → run_optimization("highs")
→ { solve_result: "solved", objective_value: 14250.0, ... }

# Session 4: Extract results
LLM → extract_solution("x")
→ { row_count: 45, preview: [{...}, ...], ... }

# Session 5: Adjust & re-solve (no state loss!)
LLM → set_ampl_model("... capacity := capacity * 1.1; ...")
LLM → run_optimization("highs")

# Session 6: If infeasible
→ { solve_result: "infeasible",
    diagnostics: { infeasible_constraints: [...], relaxation_suggestions: [...] } }

# Session 7: Parameter sweep with .run script
LLM → configure_gurobi(preset="fast", params_json='{"TimeLimit": 60}')
LLM → run_ampl_script("""
    for {c in 0..100 by 10} {
        let capacity := capacity * (1 + c/100.0);
        solve;
        display capacity, total_profit;
    }
""")
→ { total_solves: 11,
    solve_results: [
      { solve_index: 1, solve_result: "solved", objective_value: 14250.0 },
      { solve_index: 2, solve_result: "solved", objective_value: 15675.0 },
      ...
    ] }
```
---

## Real Example(from [AMPL: A Modeling Language for Mathematical Programming Chapter 2: Diet and Other Input Models: Minimizing Costs](https://ampl.com/wp-content/uploads/Chapter-2-Diet-and-Other-Input-Models-Minimizing-Costs-AMPL-Book.pdf))

### Prompt:
Consider the problem of choosing prepared foods to meet certain nutritional requirements. Suppose that precooked dinners of the following kinds are available for the following prices per package: BEEF beef $3.19 CHK chicken 2.59 FISH fish 2.29 HAM ham 2.89 MCH macaroni & cheese 1.89 MTL meat loaf 1.99 SPG spaghetti 1.99 TUR turkey 2.49 These dinners provide the following percentages, per package, of the minimum daily requirements for vitamins A, C, B1 and B2: A C B1 B2 BEEF 60% 20% 10% 15% CHK 8 0 20 20 FISH 8 10 15 10 HAM 40 40 35 10 MCH 15 35 15 15 MTL 70 30 15 15 SPG 25 50 25 15 TUR 60 20 15 10 The problem is to find the cheapest combination of packages that will meet a week’s requirements — that is, at least 700% of the daily requirement for each nutrient. 
Write an AMPL model to optimize this problem, and solve by gurobi.

### Response:

The results are evident from the initial extraction. Below is the summary:

#### AMPL Optimization Problem – Minimum-Cost Diet Plan
##### Model Description
**Decision Variable**: `buy[f]` – integer number of packages purchased for each food item
**Objective**: Minimize total cost
**Constraints**: The intake of each vitamin (A, C, B1, B2) shall meet or exceed the weekly required amount (≥700%)

##### Optimal Solution (Solved by Gurobi)
| Food | Unit Price | Purchase Quantity | Cost |
| ---- | ---- | ---- | ---- |
| BEEF | $3.19 | 0 | $0.00 |
| CHK (Chicken) | $2.59 | 2 | $5.18 |
| FISH | $2.29 | 0 | $0.00 |
| HAM | $2.89 | 0 | $0.00 |
| MCH (Macaroni) | $1.89 | 43 | $81.27 |
| MTL (Meatloaf) | $1.99 | 1 | $1.99 |
| SPG | $1.99 | 0 | $0.00 |
| TUR (Turkey) | $2.49 | 0 | $0.00 |

**Total Cost**: $88.44

##### Nutrient Compliance Status
| Nutrient | Actual Intake | Required Amount | Compliance |
| ---- | ---- | ---- | ---- |
| Vitamin A | 731% | 700% | ✓ |
| Vitamin C | 1535% | 700% | ✓ |
| Vitamin B1 | 700% | 700% | ✓ (Exactly met) |
| Vitamin B2 | 700% | 700% | ✓ (Exactly met) |

##### Core Solution Logic
Macaroni (MCH) features the highest cost-performance ratio, serving as the staple food to supply most Vitamin C and B vitamins. Meatloaf (MTL) supplements Vitamin A, while chicken (CHK) precisely fills the deficit of Vitamin B1 and B2.

---

## LLM Client Configuration

### Claude Desktop

Add to your `claude_desktop_config.json` (Windows — adjust paths):

```json
{
  "mcpServers": {
    "ampl-research": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "path_to_ampl-mcp-server",
        "python", "server.py"
      ],
      "env": {
        "AMPL_PATH": "path_to_your_ampl.exe"
      }
    }
  }
}
```

macOS / Linux:

```json
{
  "mcpServers": {
    "ampl-research": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/Users/you/ampl-mcp-server",
        "python", "server.py"
      ],
      "env": {
        "AMPL_PATH": "/opt/ampl"
      }
    }
  }
}
```

If AMPL is already on your system `PATH`, omit the `env` block entirely.

### VS Code / Cursor

Use the same configuration in your editor's MCP settings panel. The JSON format
is identical.

---

## Project Structure

```
ampl-mcp-server/
├── server.py             # FastMCP entry point + 7 tool definitions
├── ampl_engine.py        # Singleton AMPL runtime (solve, diagnostics, solver mgmt)
├── data_pipeline.py      # Excel/CSV ingestion → AMPL .dat transformation
├── schemas.py            # Pydantic request/response models (no bare dicts)
├── utils.py              # Logging infrastructure, path helpers
├── pyproject.toml        # uv dependencies + project metadata
├── .python-version       # Pinned Python version
├── LICENSE               # MIT license
├── README.md             # This file
├── logs/                 # Server logs (server.log)
├── results/              # CSV exports from extract_solution (>50 rows)
└── temp/                 # Temporary files
```

---

## Key Design Principles

### 1. Stateful Session Architecture

The AMPL runtime is a **module-level singleton**. Every tool call operates on the
same in-memory AMPL instance. No subprocess spawning, no temporary files, no
state serialization. The LLM experiences a continuous interactive session.

### 2. String-Preserving Data Pipeline

Academic datasets frequently use string IDs that look numeric: `001`, `01A`,
`BUS001`, `34E`. The pipeline reads **all columns as object dtype first**, then
selectively converts only columns that are unambiguously numeric. Leading-zero
identifiers are never silently turned into integers.

### 3. 1-Based Indexing

All AMPL set members and parameter indices start at 1, matching AMPL's native
conventions. Zero-based indexing — a common source of off-by-one bugs when
bridging Python and AMPL — is explicitly avoided.

### 4. Production Error Handling

Every tool catches exceptions and returns a structured error response:
```json
{
  "status": "error",
  "error": {
    "error_type": "AMPLSyntaxError",
    "message": "syntax error\ncontext:  >>> var x >= 0 <<<\n...",
    "traceback": "..."
  }
}
```

AMPL errors are returned **in full** — line numbers, context, and solver messages —
so the LLM can self-correct without human intervention.

### 5. Token Overflow Prevention

Variable extractions with >50 rows are automatically exported to CSV in
`./results/` and only a 10-row preview is returned to the LLM. No raw
megabyte dumps through the MCP transport.

---

## Infeasible Diagnostics

When `solve_result == "infeasible"`, the server runs a three-layer diagnostic:

### Layer 1: IIS / Conflict Analysis

For Gurobi and CPLEX, the server enables Irreducible Inconsistent Subsystem
(IIS) computation to identify the minimal set of conflicting constraints.

### Layer 2: Slack-Based Violation Scanning

Every constraint is inspected. The server computes:
- `body` — evaluated constraint body
- `lb` / `ub` — lower / upper bounds
- `slack` — solver-reported slack
- `violation` — distance from feasibility

The top 20 most violated constraints are returned, sorted by violation magnitude.

### Layer 3: Solver Message Parsing

The solver output is scanned for keyword patterns (`bound infeasibility`,
`primal infeasible`, `dual infeasible`) and matched to plain-English
explanations and relaxation suggestions.

### Response Shape

```json
{
  "diagnostics": {
    "infeasible_constraints": [
      {
        "constraint_name": "labor",
        "violation": 15.3,
        "body": 115.3,
        "lbound": null,
        "ubound": 100.0,
        "slack": -15.3
      }
    ],
    "possible_causes": [
      "Constraint 'labor' violated by 15.3 (body=115.3, lb=None, ub=100.0, slack=-15.3)"
    ],
    "relaxation_suggestions": [
      "Try relaxing the most binding constraints by 1-5% and re-solve.",
      "If the problem is a MIP/MILP, check that integrality constraints are not causing conflicting requirements."
    ],
    "iis_available": true
  }
}
```

---

## Error Handling

All tool responses follow a strict contract. Success and error paths use the
same top-level envelope:

| Field | Success | Error |
|-------|---------|-------|
| `status` | `"success"` | `"error"` |
| `error` | absent | `{ error_type, message, traceback }` |

AMPL semantic errors (syntax, data, solver) are never converted to generic
messages. The raw AMPL output is preserved so the LLM has full debugging
context.

---

## Logging

All tool invocations, solver executions, infeasible analyses, and exceptions
are written to `./logs/server.log` with timestamps:

```
2026-05-20 09:30:00 | INFO     | ampl_mcp_server | Tool: set_ampl_model | chars=847
2026-05-20 09:30:01 | INFO     | ampl_mcp_server | Model loaded | chars=847 | lines=12
2026-05-20 09:30:05 | INFO     | ampl_mcp_server | Tool: run_optimization | solver=highs
2026-05-20 09:30:07 | INFO     | ampl_mcp_server | Solve complete | result=solved | solver=highs | time=2.134s
```

Log level to stderr is `WARNING`; the file receives `DEBUG` level.

---

## Development

```bash
uv sync              # Install all deps (including pytest, ruff)
uv run ruff check .  # Lint
uv run pytest        # Run tests

# Run the server with hot-reload on code changes
uv run watchfiles --filter python "python server.py"
```

### Requirements

- `fastmcp>=2.0.0` — standalone FastMCP framework (not the Anthropic SDK fork)
- `amplpy>=0.13.0` — AMPL Python API
- `pandas>=2.0.0` — DataFrame engine for the data pipeline
- `pydantic>=2.0.0` — schema validation
- `openpyxl>=3.1.0` — Excel (.xlsx) support
- `xlrd>=2.0.1` — legacy Excel (.xls) support

---

## Contributing

Contributions are welcome — especially from the OR/MS community.

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes
4. Run `uv run ruff check .` to lint
5. Run `uv run pytest` to verify
6. Open a pull request

Areas where contributions are especially valuable:

- **Additional solvers** — BARON, KNITRO, COPT, XPRESS support
- **NL file export** — for use with external solvers
- **AMPL script replay** — record and replay interactive sessions
- **Test suite** — integration tests with real AMPL models
- **Docker image** — containerized deployment

---

## Citation

If you use this server in academic research, please cite:

```bibtex
@software{ampl-mcp-server,
  title     = {{AMPL MCP Server — Academic Optimization Research Edition}},
  url       = {https://github.com/jysw980/ampl-mcp-server},
  note      = {Stateful MCP server bridging LLMs and AMPL for multi-round optimisation research},
  year      = {2025},
}
```

---

## Related Projects

- [MATLAB MCP Core Server](https://github.com/MathWorks-Teaching-Resources/MATLAB-mcp-core-server) — same stateful MCP architecture, different ecosystem
- [AMPL Python API (amplpy)](https://dev.ampl.com/ampl/python/getting-started.html) — the underlying AMPL ↔ Python bridge
- [FastMCP](https://github.com/fastmcp-me/fastmcp-python) — standalone MCP framework
- [Model Context Protocol](https://modelcontextprotocol.io/) — MCP specification

---

## License

MIT — see [LICENSE](LICENSE) for full text.
