"""
Population-based demand proxy (order-of-magnitude only) -- converts a
substation's Voronoi-cell population into a rough average/peak MW demand
estimate, for sanity-checking matched generation capacity against local
demand. Not a real load estimate: ignores customer mix, diurnal/seasonal
timing, and transmission losses. Only good for catching order-of-magnitude
mismatches (e.g. a 20 MW plant against a cell implying ~500 MW of demand).

US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR is a generic national figure, not
Virginia-specific. US_TYPICAL_ANNUAL_LOAD_FACTOR converts average to peak;
sizing against peak is the conservative choice since storms don't
necessarily hit during off-peak hours.

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
