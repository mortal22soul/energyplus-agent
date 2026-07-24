# How to Run Eco-Loop

## 1. Setup

```powershell
uv sync
uv run python -m pytest tests/ -v
```

All 80 tests should pass.

## 2. Run the full baseline vs AI comparison (offline, no EnergyPlus needed)

```powershell
uv run ecoloop demo --mode compare
```

This generates synthetic time-series data from the EPW weather file, runs both the
baseline (fixed-schedule) and AI (multi-agent LLM) controllers, and prints
energy/peak/comfort comparison metrics. Output lands under `data/output/`.

Modes: `baseline`, `ai`, `compare`.

## 3. LLM Configuration

The controller uses a **multi-agent LangGraph architecture** with a **fallback chain**.

### Provider Chain

```
Multi-Agent (4 Azure calls) --> Single-Agent LLM --> Ollama --> Deterministic
```

Set `ECOLOOP_LLM_MODE` to change the chain:

| `ECOLOOP_LLM_MODE` | Chain |
|---|---|
| `azure-first` *(default)* | Azure OpenAI -> Ollama -> deterministic |
| `hybrid` | Same as `azure-first` |
| `azure` | Azure OpenAI -> deterministic (no Ollama) |
| `ollama` | Ollama (primary + fallback model) -> deterministic |
| `deterministic` | No LLM, rule-based only |

### Required: Set your Azure credentials in `.env`

The `.env` file must contain your Azure OpenAI Foundry credentials:

```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4-mini
ECOLOOP_LLM_MODE=azure-first
```

Note: The endpoint can include or exclude the `/openai/v1` suffix; the system
handles both formats automatically.

### Optional: Ollama (local fallback)

1. Install Ollama and pull the desired models:

   ```powershell
   ollama pull llama3.1:8b-instruct
   ```

2. Ensure Ollama is running (default: `http://localhost:11434`).

With `azure-first` mode, Ollama is used automatically if Azure is unavailable.

### Fallback behavior

The controller **never crashes** on LLM unavailability. The chain is:

```
Multi-Agent Graph --> Single-Agent Azure --> Ollama --> Deterministic hold
```

Each provider failure is logged with the source (`primary`, `fallback`, `deterministic`)
and written to the audit log for inspection.

## 4. Multi-Agent Architecture

The AI mode uses a **hierarchical multi-agent system** via LangGraph:

- **EnergyAgent**: Minimizes HVAC energy consumption
- **ComfortAgent**: Maintains thermal comfort (PMV in [-0.5, 0.5])
- **ForecastAgent**: Optimizes based on weather predictions
- **SupervisorAgent**: Merges proposals with priority Safety > Comfort > Energy

Each agent calls Azure OpenAI independently. If the multi-agent graph fails,
the system falls back to the single-agent LLM client.

## 5. MPC Optimizer

A **3R-2C thermal network model** provides predictive control:
- Predicts zone temperatures over a 4-hour horizon
- Optimizes setpoints using brute-force enumeration
- Results included as evaluator signal in the LLM prompt

The MPC runs automatically as part of the evaluator pipeline.

## 6. MCP Server

```powershell
# Stdio (for Claude Desktop, etc.)
uv run ecoloop mcp

# HTTP (for remote clients)
uv run ecoloop mcp --transport streamable-http --port 8000
```

Available tools: `read_zone_telemetry`, `read_zone_history`, `read_safety_limits`,
`check_action_safety`, `propose_setpoints`, `run_simulation_comparison`,
`read_audit_log`, `list_output_directories`.

## 7. Streamlit Dashboard

```powershell
uv run ecoloop dashboard
```

Opens the dashboard at `http://localhost:8501`. Requires at least one run output
directory under `data/output/` (produced by `demo --mode compare`).

Tabs:
- **Control Loop**: Step-by-step replay with zone temp + power charts
- **Energy Comparison**: Baseline vs AI with cumulative energy curves
- **Comfort Analysis**: PMV tracking with comfort band visualization
- **Audit Trail**: Searchable, filterable event log with CSV download
- **Architecture**: Multi-agent system diagram and safety rules

## 8. EnergyPlus Runtime Spike

```powershell
uv run ecoloop spike --idf "D:\EnergyPlus\ExampleFiles\5ZoneAirCooled.idf" --epw "D:\EnergyPlus\WeatherData\USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
```

This proves live callback actuation works with the installed EnergyPlus.
Requires EnergyPlus to be installed at `D:\EnergyPlus` (or set `ECOLOOP_ENERGYPLUS_HOME`).

## 9. Running Tests

```powershell
# All tests
uv run python -m pytest tests/ -v

# Specific test file
uv run python -m pytest tests/test_multi_agent.py -v
uv run python -m pytest tests/test_mpc.py -v
```
