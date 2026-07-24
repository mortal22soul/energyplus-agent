# Eco-Loop Building Agents

> **Eco-Loop** is a safety-governed autonomous building operator: a multi-agent LLM system
> (Azure OpenAI / Ollama) reasons over live EnergyPlus digital-twin telemetry, proposes
> structured HVAC actions through specialized Energy/Comfort/Forecast agents, and a
> deterministic safety layer applies safe commands -- proving measurable energy savings
> while maintaining thermal comfort.

[View the Architecture Report](docs/architecture.md) | [How to Run](HOW_TO_RUN.md)

## What It Does

1. **Reads** zone temperatures, humidity, occupancy, and HVAC power from EnergyPlus simulation
2. **Evaluates** comfort (PMV/PPD), energy demand, weather forecast, and MPC predictions
3. **Decides** via multi-agent LangGraph (Energy + Comfort + Forecast + Supervisor agents)
4. **Validates** all proposals through a strict SafetyEngine (clamping, rate limits, deadband)
5. **Logs** every decision and outcome to a JSONL audit trail
6. **Compares** baseline (fixed-schedule) vs AI-controlled operation
7. **Visualizes** results in an interactive Streamlit dashboard with Plotly charts

## Quick Start

```powershell
# Install dependencies
uv sync

# Run all tests (80 tests)
uv run python -m pytest tests/ -v

# Run a full baseline vs AI comparison (generates synthetic time-series data)
uv run ecoloop demo --mode compare

# Or run just the baseline
uv run ecoloop demo --mode baseline

# Or run the AI mode (uses Azure OpenAI, falls back to Ollama/deterministic)
uv run ecoloop demo --mode ai

# Launch the Streamlit dashboard
uv run ecoloop dashboard

# Start the MCP server (for Claude Desktop / external LLM clients)
uv run ecoloop mcp

# Run EnergyPlus runtime callback spike (proves live actuation works)
uv run ecoloop spike
```

## LLM Configuration

Set your Azure OpenAI credentials in `.env`:

```
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4-mini
ECOLOOP_LLM_MODE=azure-first
```

The fallback chain is: **Azure OpenAI -> Ollama -> Deterministic**. The system never
crashes on LLM unavailability.

## Project Structure

```
energyplus-agent/
+-- src/ecoloop/
|   +-- agent.py           # Azure OpenAI / Ollama LLM client with fallback chain
|   +-- multi_agent.py     # LangGraph multi-agent (Energy/Comfort/Forecast/Supervisor)
|   +-- mpc.py             # 3R-2C thermal model + MPC optimizer (4h horizon)
|   +-- baseline.py        # Fixed-schedule baseline controller
|   +-- cli.py             # CLI entry point: spike, demo, run, dashboard, mcp
|   +-- comfort.py         # ASHRAE-55 PMV/PPD thermal comfort calculator
|   +-- config.py          # Paths, safety limits, LLM config, zone definitions
|   +-- controller.py      # Closed-loop orchestration (observe->decide->validate->actuate)
|   +-- dashboard.py       # Streamlit dashboard with Plotly charts
|   +-- metrics.py         # Energy/peak/comfort comparison metrics
|   +-- runtime.py         # EnergyPlus runtime callback adapter
|   +-- safety.py          # Deterministic safety engine (clamp, deadband, rate-limit)
|   +-- schemas.py         # Typed data classes: BuildingState, ControlAction, ZoneSetpoints
|   +-- state.py           # State reader, PMV/PPD evaluator, forecast, MPC evaluator
|   +-- synthetic.py       # Synthetic time-series generator from EPW weather data
|   +-- mcp_server.py      # 8 MCP tools via FastMCP (stdio/HTTP)
|   +-- audit.py           # Append-only JSONL event logger
+-- tests/                 # Unit tests (80 tests: comfort, safety, agent, metrics, MCP, multi-agent, MPC)
+-- prompts/               # LLM system prompt + action examples
+-- data/                  # EPW weather, IDF model, output directories
+-- docs/                  # Architecture report and diagrams
+-- models/                # EnergyPlus IDF models
```

## Architecture

```
                    Building State (from EnergyPlus/Synthetic)
                                    |
                            StateReader + Evaluators
                    (comfort, energy, forecast, MPC, oscillation)
                                    |
                    +---------------+---------------+
                    |               |               |
               EnergyAgent    ComfortAgent    ForecastAgent
               (minimize kWh) (maintain PMV)  (pre-condition)
                    |               |               |
                    +-------+-------+-------+-------+
                            |
                    SupervisorAgent
                    (merge: Safety > Comfort > Energy)
                            |
                       SafetyEngine
                   (clamp, deadband, rate-limit)
                            |
                    Approved Setpoints
                            |
                    Audit Log (JSONL)
                            |
                   Dashboard + Metrics
```

## Key Design Decisions

- **Multi-agent LLM**: Specialized agents for energy, comfort, and forecast with a supervisor
  that merges proposals using priority rules (Safety > Comfort > Energy)
- **MPC-informed**: A 3R-2C thermal model provides predictive recommendations over a 4-hour
  horizon, guiding the LLM agents with model-based optimization
- **Safety-first design**: The LLM never writes to EnergyPlus directly. SafetyEngine validates
  all proposals before any actuation
- **Graceful degradation**: Multi-agent -> single-agent -> Ollama -> deterministic fallback
- **Azure OpenAI primary**: Cloud-hosted gpt-5.4-mini for reliable structured output
- **MCP-native**: 8 tools via FastMCP for any MCP-compatible client
- **Offline-first testing**: Controller runs over parsed EnergyPlus output CSV for fast iteration

## Evaluation

The system is evaluated against:
- **Energy savings**: kWh reduction of AI mode vs baseline (fixed-schedule)
- **Comfort compliance**: % of occupied intervals where zone temps fall within comfort band
- **Safety clamping**: Number of actions that required safety intervention

Run the comparison and see results in the dashboard.

## Building Model

- **Model**: 5ZoneAirCooled.idf (EnergyPlus example)
- **Zones**: SPACE1-1 through SPACE5-1
- **Weather**: San Francisco TMY3 EPW
- **Simulation**: 7 days, 15-minute timestep

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Simulation | EnergyPlus 26.1 |
| LLM Primary | Azure OpenAI (gpt-5.4-mini) |
| LLM Fallback | Ollama (Llama 3.1 8B) |
| Multi-Agent | LangGraph |
| MPC | Custom 3R-2C thermal model |
| Comfort | ASHRAE-55 PMV/PPD |
| Dashboard | Streamlit + Plotly |
| MCP | FastMCP |
| Data | JSONL + CSV |

## License

MIT
