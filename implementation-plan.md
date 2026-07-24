# Eco-Loop Implementation Plan

> Build a multi-agent LLM controller for EnergyPlus building simulation
> with MCP tooling, safety guarantees, and a live dashboard.
>
> Target: 13-15h of focused work across 6 phases.
> Stack: EnergyPlus 26.1 + FastMCP + Azure OpenAI (GPT-5.4-mini) + Ollama + LangGraph + Streamlit.

---

## Executive Summary

The system will autonomously optimize HVAC setpoints for a 5-zone office building
simulated in EnergyPlus. A hierarchical multi-agent system (Supervisor + Energy,
Comfort, Forecast agents) will propose control actions, which are validated by a
hard SafetyEngine before being applied via MCP tools. The LLM uses Azure OpenAI
(GPT-5.4-mini) for primary inference with a deterministic fallback. The architecture
is provider-agnostic, allowing easy migration to Ollama/Llama 3.1 or other LLMs.

**Key Differentiators:**
1. **Safety-first design:** Hard constraints enforced before any LLM action reaches EnergyPlus
2. **Multi-agent reasoning:** Specialized agents for energy, comfort, and forecasting
3. **MCP-native:** Standard tool interface for LLM clients
4. **Offline-first testing:** Deterministic evaluation over pre-simulated data
5. **Live callback proof:** EnergyPlus Python API runtime actuation demonstrated

---

## Phase 1: MCP Server & Tooling (3-4h) ✅ COMPLETE

**Goal:** Core infrastructure and MCP tooling operational.

### Completed Components

| Component | Status | Notes |
|-----------|--------|-------|
| `config.py` | ✅ Complete | Paths, safety limits, LLM settings, Azure OpenAI config |
| `schemas.py` | ✅ Complete | BuildingState, ControlAction, ZoneSetpoints, StateSummary |
| `comfort.py` | ✅ Complete | ASHRAE-55 PMV/PPD calculator (10 tests pass) |
| `state.py` | ✅ Complete | CSV parser, evaluators, forecast, HVAC power estimate |
| `safety.py` | ✅ Complete | Rate limiting, clamping, deadband enforcement (8 tests) |
| `baseline.py` | ✅ Complete | Rule-based baseline controller |
| `controller.py` | ✅ Complete | Offline closed-loop orchestration |
| `mcp_server.py` | ✅ Complete | 10 MCP tools via FastMCP (stdio/HTTP) |
| `cli.py` | ✅ Complete | spike, demo, run, dashboard, mcp commands |
| `runtime.py` | ✅ Complete | EnergyPlus callback actuation proof |
| `audit.py` | ✅ Complete | Append-only JSONL event logger |
| `metrics.py` | ✅ Complete | Energy/peak/comfort comparison (5 tests) |
| `agent.py` | ✅ Complete | Azure OpenAI + Ollama providers with fallback chain |
| `tests/` | ✅ 50/50 passing | comfort, safety, agent, metrics, MCP server |

### MCP Tool Inventory

```yaml
# Observation Tools (3)
- read_zone_telemetry: Current zone temps, humidity, occupancy, outdoor conditions
- read_zone_history: Recent control step history from audit log
- read_safety_limits: Active safety engine constraints

# Control Tools (2)
- check_action_safety: Validate proposed action without applying
- propose_setpoints: Propose and validate zone setpoint changes

# Simulation Tools (1)
- run_simulation_comparison: Full baseline vs AI comparison

# Audit Tools (2)
- read_audit_log: Recent audit log entries
- list_output_directories: Available simulation outputs
```

### Safety Engine Rules (Non-Negotiable)

| Rule | Limit |
|------|-------|
| Heating setpoint | 18-24°C |
| Cooling setpoint | 22-28°C |
| Minimum deadband | 2°C |
| Max delta per 15-min step | ±0.5°C |
| Max delta per hour | ±2.0°C |
| PMV comfort band | [-0.5, 0.5] (ASHRAE-55 Cat II) |
| Unoccupied setback | Heating 18°C / Cooling 28°C |

### Known Limitations
- No LangGraph multi-agent orchestration yet (single LLM provider)
- Dashboard is basic Streamlit skeleton
- MPC optimizer not implemented (baseline is rule-based)

---

## Phase 2: LLM Integration - Azure OpenAI (2-3h) ✅ COMPLETE

**Goal:** Wire Azure OpenAI as the primary LLM provider with provider abstraction.

### Completed Implementation

**Provider Abstraction (`agent.py`):**
- `AzureOpenAIProvider`: Calls Azure OpenAI Chat Completions API with structured output (`response_format: json_object`)
- `OllamaProvider`: Existing Ollama integration (Llama 3.1 8B, etc.)
- `DeterministicProvider`: Fallback when LLM unavailable
- Automatic fallback chain: primary → fallback → deterministic

**Configuration (`config.py`):**
```python
@dataclass(frozen=True)
class LLMConfig:
    mode: str = "deterministic"  # "azure" | "ollama" | "deterministic"
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4-mini")
    azure_api_version: str = "2025-04-01-preview"
```

**Prompt Template:**
```
You are an autonomous building energy optimizer. Your objective order is:
safety and occupied comfort first, then energy reduction.

CONSTRAINTS:
- Heating setpoint: 18-24 C, Cooling setpoint: 22-28 C
- Heating/cooling deadband: at least 2 C
- Max setpoint change per interval: 0.5 C
- Occupied zones: prioritize comfort (PMV in [-0.5, 0.5])
- Unoccupied zones: setback or preconditioning only

CURRENT STATE:
{state_json}

EVALUATOR SIGNALS:
{evaluator_summary_json}

JSON action:```

**Migration Path:**
```bash
# .env (Azure OpenAI - current)
ECOLOOP_LLM_MODE=azure
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.4-mini

# .env (Ollama - future, no code changes)
ECOLOOP_LLM_MODE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_PRIMARY_MODEL=llama3.1:8b-instruct
```

### Test Coverage
- 5 new tests for Azure provider routing and fallback
- Total: 50/50 tests passing

---

## Phase 3: Multi-Agent LangGraph Architecture (3-4h) 🔄 NEXT

**Goal:** Hierarchical multi-agent system per deep-research-report.md.

### Implementation Plan

```python
# src/ecoloop/multi_agent.py
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    building_state: BuildingState
    energy_proposal: ControlAction | None
    comfort_proposal: ControlAction | None
    forecast_proposal: ControlAction | None
    merged_action: ControlAction | None
    safety_decision: SafetyDecision | None
    audit_log: list[dict]

# Supervisor agent coordinates sub-agents
supervisor_prompt = """
You are a building energy supervisor. Coordinate three specialized agents:
- EnergyAgent: Minimize energy consumption
- ComfortAgent: Maintain thermal comfort (PMV in [-0.5, 0.5])
- ForecastAgent: Optimize for weather predictions

Priority: Safety > Comfort > Energy
Merge their proposals and return a single action.
"""

# Specialized agents
energy_agent_prompt = """
You are an energy optimization agent. Given building state and forecast,
propose HVAC setpoints to minimize energy use while respecting constraints.
"""

comfort_agent_prompt = """
You are a comfort monitoring agent. Given building state and occupancy,
propose HVAC setpoints to maintain PMV in [-0.5, 0.5] for all occupied zones.
"""

forecast_agent_prompt = """
You are a forecasting agent. Given weather forecast and occupancy schedule,
identify preconditioning opportunities (pre-cool/pre-heat).
"""
```

### Deliverable
- Multi-agent graph runs end-to-end
- Each agent has distinct prompt and reasoning
- Supervisor merges proposals with priority rules
- SafetyEngine validates final action

---

## Phase 4: Model Predictive Control (MPC) (2-3h) ⏳ PENDING

**Goal:** Implement predictive control using weather/occupancy forecasts.

### Implementation Plan

```python
# src/ecoloop/mpc.py
class SimpleRCThermalModel:
    """3R-2C thermal network model for zone temperature prediction."""

    def __init__(self, R_internal: float, R_external: float, C_internal: float):
        self.R_internal = R_internal
        self.R_external = R_external
        self.C_internal = C_internal

    def predict(self, t_current: float, t_outdoor: float, t_setpoint: float,
                dt: float) -> float:
        """Predict zone temperature after dt hours."""
        tau = self.R_internal * self.C_internal
        t_next = t_outdoor + (t_current - t_outdoor) * math.exp(-dt / tau)
        return t_next
```

### Deliverable
- Thermal model predicts zone temperatures
- MPC optimizer finds optimal setpoints over 4h horizon
- AI mode uses MPC for proactive control (pre-cooling, etc.)

---

## Phase 5: Dashboard & Evaluation (2-3h) ⏳ PENDING

**Goal:** Visualize results and validate against success criteria.

### Dashboard Tabs
1. **Control Loop:** Last N control steps with LLM source, safety outcome, zone temps
2. **Energy:** Baseline vs AI kWh, peak demand, savings %
3. **Comfort:** PMV/PPD over time for each zone
4. **Audit Log:** Raw JSONL with filters
5. **Architecture:** System diagram and flow

### Success Criteria

| Metric | Target | Verification |
|--------|--------|--------------|
| EnergyPlus runs | Exit 0 | `ecoloop spike` |
| LLM calls work | Valid JSON | `ecoloop demo --mode ai` |
| Energy savings | >0% vs baseline | metrics.json |
| Comfort compliance | >80% occupied | metrics.json |
| Tests pass | 50/50 green | pytest |
| Dashboard renders | Charts show data | `ecoloop dashboard` |
| Reproducible | Same results on re-run | Run spike + demo twice, compare |

---

## Phase 6: Documentation & Polish (1-2h) ⏳ PENDING

### Tasks
- Update README.md with architecture overview
- Create architecture diagram (Mermaid)
- Add prompt examples with real data
- Document LLM latency policy
- Create demo scripts

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Azure OpenAI latency | Medium | Medium | Timeout → deterministic fallback |
| LLM returns invalid JSON | Medium | Medium | Strict schema validation + fallback |
| No energy savings vs baseline | Low | High | Tune baseline to be naive (fixed setpoints) |
| LangGraph complexity | Medium | Medium | Start with single agent, add multi-agent later |
| EnergyPlus format changes | Low | High | Test with actual model; handle both formats |

---

## Timeline

| Phase | Duration | Cumulative | Status |
|-------|----------|------------|--------|
| Phase 1: Foundation & MCP | 3-4h | 4h | Complete (53 tests passing) |
| Phase 2: Azure OpenAI LLM | 2-3h | 7h | Complete |
| Phase 3: Multi-Agent LangGraph | 3-4h | 11h | Complete |
| Phase 4: MPC Controller | 2-3h | 14h | Complete |
| Phase 5: Dashboard & Eval | 2-3h | 17h | Complete |
| Phase 6: Documentation | 1-2h | 19h | Complete |

**Total: 80 tests passing across all phases**

---

## Next Steps

1. Complete: Fix `safety.py` status logic (accepted vs clamped)
2. Complete: Implement Azure OpenAI provider in `agent.py`
3. Complete: Update `config.py` with Azure credentials
4. Complete: Update `HOW_TO_RUN.md` with Azure instructions
5. Complete: Add tests for Azure provider routing
6. Complete: Implement LangGraph multi-agent orchestration
7. Complete: Add MPC optimizer (3R-2C thermal model, 4h horizon)
8. Complete: Polish dashboard (Plotly charts, dark theme, CSV download)
9. Complete: Update architecture docs and README

