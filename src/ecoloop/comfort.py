from dataclasses import dataclass
from math import exp, sqrt

@dataclass(frozen=True)
class ComfortResult:
    pmv: float
    ppd_pct: float

def calculate_pmv_ppd(air_temp_c: float, radiant_temp_c: float, relative_humidity_pct: float, air_speed_m_s: float = 0.1, metabolic_rate_met: float = 1.2, clothing_clo: float = 0.5) -> ComfortResult:
    """Fanger PMV/PPD (ISO 7730 iterative formulation)."""
    if not 0 <= relative_humidity_pct <= 100:
        raise ValueError("relative humidity must be between 0 and 100")
    pa = relative_humidity_pct * 10 * exp(16.6536 - 4030.183 / (air_temp_c + 235))
    icl, m = 0.155 * clothing_clo, metabolic_rate_met * 58.15
    fcl = 1 + 1.29 * icl if icl <= .078 else 1.05 + .645 * icl
    taa, tra, hcf = air_temp_c + 273, radiant_temp_c + 273, 12.1 * sqrt(air_speed_m_s)
    p1 = icl * fcl; p2 = p1 * 3.96; p3 = p1 * 100; p4 = p1 * taa
    p5 = 308.7 - .028 * m + p2 * (tra / 100) ** 4
    xn, xf = (taa + (35.5 - air_temp_c) / (3.5 * icl + .1)) / 100, (taa + (35.5 - air_temp_c) / (3.5 * icl + .1)) / 50
    for _ in range(150):
        xf = (xf + xn) / 2; hc = max(hcf, 2.38 * abs(100 * xf - taa) ** .25)
        nxt = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        if abs(nxt - xn) <= .00015: xn = nxt; break
        xn = nxt
    tcl = 100 * xn - 273
    losses = 3.05e-3 * (5733 - 6.99 * m - pa) + (0.42 * (m - 58.15) if m > 58.15 else 0) + 1.7e-5 * m * (5867 - pa) + .0014 * m * (34 - air_temp_c) + 3.96 * fcl * (xn ** 4 - (tra / 100) ** 4) + fcl * hc * (tcl - air_temp_c)
    pmv = (.303 * exp(-.036 * m) + .028) * (m - losses)
    return ComfortResult(pmv, 100 - 95 * exp(-.03353 * pmv ** 4 - .2179 * pmv ** 2))
