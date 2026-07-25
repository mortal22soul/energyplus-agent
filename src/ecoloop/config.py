"""Eco-Loop configuration: paths, limits, LLM settings, zone definitions."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    models_source: Path = project_root / "models" / "source"
    models_controlled: Path = project_root / "models" / "controlled"
    data_weather: Path = project_root / "data" / "weather"
    data_fixtures: Path = project_root / "data" / "fixtures"
    data_output: Path = project_root / "data" / "output"
    prompts_dir: Path = project_root / "prompts"

    def default_idf(self) -> Path:
        return self.models_source / "5ZoneAirCooled.idf"

    def default_epw(self) -> Path:
        return self.data_weather / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"

    def forecast_csv(self) -> Path:
        return self.data_weather / "forecast_outdoor_temp.csv"


@dataclass(frozen=True)
class SafetyLimits:
    heating_min_c: float = 18.0
    heating_max_c: float = 24.0
    cooling_min_c: float = 22.0
    cooling_max_c: float = 28.0
    minimum_deadband_c: float = 2.0
    max_delta_per_interval_c: float = 0.5   # per 15-min step
    max_delta_per_hour_c: float = 2.0
    pmv_comfort_band: tuple[float, float] = (-0.5, 0.5)
    unoccupied_setback_heating_c: float = 18.0
    unoccupied_setback_cooling_c: float = 28.0


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration.

    mode options:
      "deterministic" – no LLM, rule-based only
      "azure"         – Azure OpenAI primary, deterministic fallback
      "azure-first"   – Azure OpenAI primary, Ollama fallback, then deterministic  ← recommended
      "ollama"        – Ollama primary/fallback, then deterministic
      "hybrid"        – same as azure-first
    """
    primary_model: str = "llama3.1:8b-instruct"
    fallback_model: str = ":7b"
    timeout_seconds_cpu: int = 30
    timeout_seconds_gpu: int = 10
    ollama_base_url: str = "http://localhost:11434"
    mode: str = field(default_factory=lambda: os.getenv("ECOLOOP_LLM_MODE", "azure-first"))

    # Azure OpenAI (foundry)
    azure_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5.4-mini"))
    azure_api_version: str = "2025-04-01-preview"


@dataclass(frozen=True)
class ControlConfig:
    timestep_minutes: int = 15
    total_simulated_hours: int = 24
    history_window: int = 4  # number of recent states to keep in rolling buffer


@dataclass(frozen=True)
class ZoneConfig:
    """Mapping of EnergyPlus zone names to controllable setpoint schedules."""
    name: str
    heating_schedule: str = "Htg-SetP-Sch"
    cooling_schedule: str = "Clg-SetP-Sch"
    heating_actuator_type: str = "Schedule:Compact"
    cooling_actuator_type: str = "Schedule:Compact"
    heating_actuator_field: str = "Schedule Value"
    cooling_actuator_field: str = "Schedule Value"


# Default zone configuration for 5ZoneAirCooled.idf
DEFAULT_ZONES: list[ZoneConfig] = [
    ZoneConfig(name="SPACE1-1"),
    ZoneConfig(name="SPACE2-1"),
    ZoneConfig(name="SPACE3-1"),
    ZoneConfig(name="SPACE4-1"),
    ZoneConfig(name="SPACE5-1"),
]


@dataclass(frozen=True)
class RunConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    safety: SafetyLimits = field(default_factory=SafetyLimits)
    llm: LLMConfig = field(default_factory=LLMConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    zones: list[ZoneConfig] = field(default_factory=lambda: DEFAULT_ZONES)

    @property
    def total_steps(self) -> int:
        return (self.control.total_simulated_hours * 60) // self.control.timestep_minutes
