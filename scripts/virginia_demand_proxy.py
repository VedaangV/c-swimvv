"""
Population-based demand proxy (order-of-magnitude only) -- converts a
substation's Voronoi-cell population into a rough estimated average/peak
MW demand, for sanity-checking matched generation capacity against local
demand before relying on it as a self-sufficiency anchor.

Extracted out of scripts/deprecated_sci_islanding/check_generation_sci_
overlap.py when the SCI-driven greedy islanding strategy was deprecated
(see that folder's README.md) -- this proxy itself has nothing to do
with SCI, it's a general "does this much generation plausibly cover this
much population's demand" check, needed by scripts/compute_island_
benefit_cost.py regardless of how an island's members were chosen.

The pipeline has no real per-substation electrical load (MW) data
anywhere -- population/GDP density (scripts/virginia_risk_data.py's
load_economic_exposure()) is an economic-impact proxy, not electrical
demand, and there is no load-flow model here. This is explicitly NOT a
real load estimate -- it ignores the substation's actual customer mix (a
residential feeder and an industrial substation with the same
population would have very different real demand), any diurnal/seasonal
timing, and any transmission losses. It exists only to catch an
order-of-magnitude mismatch (a 20 MW plant matched to a substation whose
Voronoi cell implies ~500 MW of demand is obviously not self-sufficient;
one matched to a cell implying ~5 MW might be). A real substation-level
load estimate (e.g. from EIA-861 utility sales data or actual PJM/
utility load data, if available) would be a strictly better input than
this if available -- this proxy is a stand-in for that, not a permanent
replacement.

US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR: commonly cited order-of-magnitude
figure for combined residential+commercial+industrial U.S. electricity
use, NOT a Virginia-specific or substation-specific measurement.
US_TYPICAL_ANNUAL_LOAD_FACTOR: average/peak ratio -- storms don't
necessarily hit at off-peak load, so sizing against peak, not average,
is the conservative direction.

Usage: import as a library, e.g.
    from scripts.virginia_demand_proxy import estimate_demand_mw
"""

import pandas as pd

from scripts.virginia_risk_data import load_economic_exposure

US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR = 12000.0
US_TYPICAL_ANNUAL_LOAD_FACTOR = 0.6


def load_population_lookup() -> dict:
    """sub_id -> population, reusing virginia_risk_data.py's econ-exposure
    loader rather than re-deriving it."""
    exposure = load_economic_exposure()
    return dict(zip(exposure["sub_id"], exposure["population"]))


def estimate_demand_mw(population: pd.Series) -> tuple:
    """Rough order-of-magnitude average and peak MW estimates from
    population alone, not a real load estimate -- see module docstring.
    Returns (avg_demand_mw, peak_demand_mw) as parallel Series."""
    avg_demand_mw = population * US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR / 8760 / 1000
    peak_demand_mw = avg_demand_mw / US_TYPICAL_ANNUAL_LOAD_FACTOR
    return avg_demand_mw, peak_demand_mw
