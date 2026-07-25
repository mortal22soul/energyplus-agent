from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ComparisonMetrics:
    baseline_kwh: float; ai_kwh: float; baseline_peak_kw: float; ai_peak_kw: float
    energy_savings_pct: float; peak_reduction_pct: float

def compare(baseline_kwh: float, ai_kwh: float, baseline_peak_kw: float, ai_peak_kw: float) -> ComparisonMetrics:
    if baseline_kwh <= 0 or baseline_peak_kw <= 0: raise ValueError("baseline metrics must be positive")
    return ComparisonMetrics(baseline_kwh, ai_kwh, baseline_peak_kw, ai_peak_kw, 100*(baseline_kwh-ai_kwh)/baseline_kwh, 100*(baseline_peak_kw-ai_peak_kw)/baseline_peak_kw)
