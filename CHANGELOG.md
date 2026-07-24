# Changelog

All notable project changes are recorded here.

## [0.1.0] - 2026-07-26

### Added

- `uv`-managed Python project foundation.
- Typed building-state and control-action models.
- Deterministic safety validation, baseline policy, PMV/PPD calculation, metrics, and JSONL audit logging.
- EnergyPlus runtime adapter boundary with a clear availability check.
- Offline-first Azure/OpenAI-compatible configuration and Ollama run instructions; no model was installed.
- EnergyPlus 26.1 runtime spike using `5ZoneAirCooled.idf`: live `SPACE1-1` temperature reads and `Htg-SetP-Sch` schedule actuation through PyEnergyPlus callbacks.

### Known limitations

- EnergyPlus is not installed in this workspace, so callback actuation has not yet been validated against a real IDF/EPW model.
