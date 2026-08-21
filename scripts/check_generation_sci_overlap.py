"""
Diagnostic for generation-anchored island selection (Phase 6 extension):
checks whether any substation with real nearby EIA-860 generation also
clears a meaningful SCI risk threshold, before spending an overnight
Monte Carlo run on an island anchored there. Motivated directly by Island
F's real result -- its generation-eligible seed pool topped out at SCI
0.1257, ~7x below Island A/B's seed SCI (~0.85-0.9), raising the
question of whether Virginia has any substation that's both high-risk
(worth protecting) and generation-backed (the way NZ's Manapouri anchors
Scenario 5), or whether those are structurally disjoint sets here. If
they're disjoint, that's a legitimate negative result for generation-
anchored islanding in Virginia specifically, not a bug to work around.

Reports, without running any island-selection or heavy pipeline stage:
- how many eligible substations have matched EIA-860 generation at all
- the SCI percentile of the single best generation-backed substation
- for a few SCI percentile cutoffs (p50/p75/p90/p95), how many of the
  substations above that cutoff also have generation nearby

DEMAND PROXY (population-based, order-of-magnitude only)
----------------------------------------------------------
The pipeline has no per-substation electrical load (MW) data anywhere --
econ_exposure_i (compute_virginia_criticality.py) uses population/GDP
*density* as an economic-impact proxy, not electrical demand. There is
no load-flow model here, so "does this substation's nearby generation
actually cover its own demand" can't be answered with real data already
on hand. This adds a rough proxy instead of leaving the question
unanswered: each substation's Voronoi-cell population (already computed
for econ_exposure_i, reused via compute_virginia_criticality.load_
economic_exposure()) is converted to an estimated average continuous MW
demand using a single national per-capita consumption constant
(US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR -- commonly cited order-of-
magnitude figure for combined residential+commercial+industrial U.S.
electricity use, NOT a Virginia-specific or substation-specific
measurement), then inflated to a rough peak-demand estimate via
US_TYPICAL_ANNUAL_LOAD_FACTOR (average/peak ratio -- storms don't
necessarily hit at off-peak load, so sizing against peak, not average,
is the conservative direction).

This is explicitly NOT a real load estimate -- it ignores the
substation's actual customer mix (a residential feeder and an
industrial substation with the same population would have very
different real demand), any diurnal/seasonal timing, and any
transmission losses. It exists only to catch an order-of-magnitude
mismatch (a 20 MW plant matched to a substation whose Voronoi cell
implies ~500 MW of demand is obviously not self-sufficient; one matched
to a cell implying ~5 MW might be) before spending an overnight run on
a candidate where the generation clearly couldn't cover the load at
all. A real substation-level load estimate (e.g. from EIA-861 utility
sales data or actual PJM/utility load data, if available) would be a
strictly better input than this if the user has access to it -- this
proxy is a stand-in for that, not a permanent replacement.

Usage:
    python scripts/check_generation_sci_overlap.py
    python scripts/check_generation_sci_overlap.py --min-generation-mw 50
"""

import argparse

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR
import scripts.check_island_generation as gen_check
import scripts.compute_virginia_criticality as crit

logger = setup_logger(log_file="logs/check_generation_sci_overlap.log")

# See DEMAND PROXY in the module docstring -- rough order-of-magnitude
# constants only, not a real load estimate.
US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR = 12000.0
US_TYPICAL_ANNUAL_LOAD_FACTOR = 0.6


def load_sci_table() -> pd.DataFrame:
    path = VA_ECON_DIR / "va_substation_criticality.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/compute_virginia_criticality.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df


def load_coords() -> pd.DataFrame:
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df


def load_population_lookup() -> dict:
    """sub_id -> population, reusing compute_virginia_criticality.py's
    econ-exposure loader (same source econ_exposure_i already draws
    population from) rather than re-deriving it."""
    exposure = crit.load_economic_exposure()
    return dict(zip(exposure["sub_id"], exposure["population"]))


def estimate_demand_mw(population: pd.Series) -> tuple:
    """See module docstring's DEMAND PROXY -- rough order-of-magnitude
    average and peak MW estimates from population alone, not a real load
    estimate. Returns (avg_demand_mw, peak_demand_mw) as parallel Series."""
    avg_demand_mw = population * US_AVG_ELECTRICITY_KWH_PER_CAPITA_YEAR / 8760 / 1000
    peak_demand_mw = avg_demand_mw / US_TYPICAL_ANNUAL_LOAD_FACTOR
    return avg_demand_mw, peak_demand_mw


def main(min_generation_mw: float = 0.0, percentiles=(0.5, 0.75, 0.9, 0.95)):
    sci_df = load_sci_table()
    topo = load_coords()
    sub_coords = topo[topo["sub_id"].isin(sci_df["sub_id"])][["sub_id", "latitude", "longitude"]]

    generation_lookup = gen_check.compute_generation_by_substation(sub_coords)
    if not generation_lookup:
        raise ValueError(
            "No EIA-860 generation data available -- download it first, see "
            "scripts/check_island_generation.py's module docstring."
        )
    population_lookup = load_population_lookup()

    sci_df = sci_df.copy()
    sci_df["generation_mw"] = sci_df["sub_id"].map(generation_lookup).fillna(0.0)
    # sub_id must actually be a key in generation_lookup (a real EIA-860
    # match), not just pass the >= threshold after an unmatched substation's
    # missing value gets filled to 0.0 -- otherwise min_generation_mw=0.0
    # (the default) would flag every substation as "has_generation".
    sci_df["has_generation"] = sci_df["sub_id"].isin(generation_lookup) & (
        sci_df["generation_mw"] >= min_generation_mw
    )
    sci_df["population"] = sci_df["sub_id"].map(population_lookup).fillna(0.0)
    sci_df["avg_demand_mw_est"], sci_df["peak_demand_mw_est"] = estimate_demand_mw(sci_df["population"])
    # inf, not a large finite number, when peak_demand_mw_est is 0 (no
    # population in this substation's service area) -- genuinely undefined
    # coverage, not "infinite coverage" in any meaningful sense, but inf
    # sorts naturally to the top when ranking by coverage below.
    sci_df["generation_covers_peak_demand_ratio"] = (
        sci_df["generation_mw"] / sci_df["peak_demand_mw_est"].replace(0.0, float("nan"))
    )

    n_total = len(sci_df)
    n_gen = int(sci_df["has_generation"].sum())
    logger.info(
        f"{n_gen}/{n_total} eligible substations have >= {min_generation_mw} MW of nearby "
        "matched EIA-860 generation."
    )

    gen_subs = sci_df[sci_df["has_generation"]].sort_values("SCI", ascending=False)
    if not gen_subs.empty:
        logger.info(
            "Top 10 generation-backed substations by SCI (demand estimates are a rough "
            "population-based proxy -- see module docstring's DEMAND PROXY, not a real load model):\n"
            + gen_subs[
                ["sub_id", "SCI", "generation_mw", "population", "avg_demand_mw_est",
                 "peak_demand_mw_est", "generation_covers_peak_demand_ratio"]
            ].head(10).to_string(index=False)
        )
        n_covers_peak = int((gen_subs["generation_covers_peak_demand_ratio"] >= 1.0).sum())
        logger.info(
            f"{n_covers_peak}/{len(gen_subs)} generation-backed substations have matched "
            "generation >= their estimated peak demand proxy (coverage ratio >= 1)."
        )

    for p in percentiles:
        threshold = sci_df["SCI"].quantile(p)
        n_total_above = int((sci_df["SCI"] >= threshold).sum())
        n_gen_above = int(((sci_df["SCI"] >= threshold) & sci_df["has_generation"]).sum())
        logger.info(
            f"SCI >= p{int(p*100)} ({threshold:.4f}): {n_total_above} substations total, "
            f"{n_gen_above} of them also have >= {min_generation_mw} MW generation."
        )

    max_gen_sci = float(gen_subs["SCI"].max()) if not gen_subs.empty else None
    if max_gen_sci is not None:
        overall_percentile = float((sci_df["SCI"] < max_gen_sci).mean())
        logger.info(
            f"Best generation-backed substation's SCI ({max_gen_sci:.4f}) sits at the "
            f"{overall_percentile*100:.1f}th percentile of all {n_total} eligible substations."
        )
    else:
        logger.info(f"No substation clears the >= {min_generation_mw} MW generation threshold at all.")

    return {"generation_lookup": generation_lookup, "sci_df": sci_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check overlap between SCI risk ranking and EIA-860 generation before "
                     "committing to a generation-anchored island selection run."
    )
    parser.add_argument("--min-generation-mw", type=float, default=0.0)
    args = parser.parse_args()
    main(min_generation_mw=args.min_generation_mw)
