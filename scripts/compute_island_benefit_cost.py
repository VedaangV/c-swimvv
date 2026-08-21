"""
Phase 5, part 2: implements the benefit-cost accounting
ISLANDING_MITIGATION_PLAN.md's Phase 5 section designed but that was
never actually coded. scripts/compare_island_econ.py's diff is purely
the fragility-driven $/day change from redistributed GIC -- it never
included a switching-labor cost or the island's own temporary-outage
cost, and it never credited generation for anything (nothing in the
GIC/fragility physics models supply continuity at all -- see the
conversation that led here: an island's own $/day loss and gain come
entirely from its transformers' own GIC-driven failure probability
changing when the boundary is cut, not from any accounting of whether
the island keeps serving its own load during the disconnection).

This adds the two missing cost terms, with generation entering ONLY
here -- discounting the island's own-outage cost by how much of its own
estimated demand its matched EIA-860 generation covers.

ASSUMPTIONS (all explicitly flagged order-of-magnitude choices, none are
real utility data -- same spirit as the NZ paper's own ~$250k switching-
cost assumption and this project's earlier DEMAND PROXY):

- SWITCHING_COST_PER_LINE_USD: labor cost per boundary line switched,
  scaled by the island's boundary_line_count (already computed by
  scripts/select_virginia_island.py, stored in va_island_{id}.json).
  Set an order of magnitude below the NZ paper's total ~$250k figure --
  that figure covered a much larger multi-line national switching plan,
  not one island's boundary.
- ASSUMED_DISCONNECTION_HOURS: how long the island is assumed offline
  during a storm's peak-risk window -- sizes the OWN-OUTAGE COST term
  (island members' annual GDP x hours/year-hours).
- REPAIR_DURATION_DAYS_VARIANTS: how many days the fragility-driven
  $/day diff should be integrated over to become a one-time $ benefit,
  representing how long a damaged-but-unprotected transformer would
  keep costing money before repair/replacement -- the plan's own stated
  "12-24+ months to replace a damaged large power transformer" reasoning
  for why avoiding transformer damage matters at all. Reported as a
  low/mid/high range (180/365/730 days) rather than one arbitrary
  number, since this is the single biggest uncertainty in the whole
  calculation -- collapsing it to a point estimate would hide that.

GENERATION CREDIT
-------------------
own_outage_cost is discounted by min(1, island_generation_mw /
island_peak_demand_mw_est) -- both computed exactly as in scripts/
check_generation_sci_overlap.py's DEMAND PROXY, just summed over the
island's members instead of reported per-substation. An island whose
matched EIA-860 generation fully covers its own demand-proxy has its
own-outage cost zeroed (assumed continuously served through the
disconnection); partial coverage scales the discount linearly; zero
generation (Islands A-E) leaves the term at full value -- i.e. exactly
what an implicit generation-blind calculation would have given them,
so A-E's numbers from this script are directly comparable to what's
already in the plan doc, just with the cost side finally filled in.

Usage:
    python scripts/compute_island_benefit_cost.py --island f
    python scripts/compute_island_benefit_cost.py --island a
"""

import argparse
import json

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR
import scripts.check_island_generation as gen_check
import scripts.check_generation_sci_overlap as demand_proxy
import scripts.compare_island_econ as econ_cmp
import scripts.compute_virginia_criticality as crit

logger = setup_logger(log_file="logs/compute_island_benefit_cost.log")

SWITCHING_COST_PER_LINE_USD = 25_000.0
ASSUMED_DISCONNECTION_HOURS = 6.0
REPAIR_DURATION_DAYS_VARIANTS = {"low_180d": 180, "mid_365d": 365, "high_730d": 730}


def load_island(island_id: str) -> dict:
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/select_virginia_island.py first.")
    with open(path) as f:
        return json.load(f)


def load_topo_coords() -> pd.DataFrame:
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df


def compute_island_generation_and_demand(members: list) -> dict:
    """Island-wide generation_mw, peak_demand_mw_est, and their ratio --
    same mechanics as check_generation_sci_overlap.py's per-substation
    DEMAND PROXY, summed across the island's members instead."""
    topo = load_topo_coords()
    sub_coords = topo[topo["sub_id"].isin(members)][["sub_id", "latitude", "longitude"]]

    generation_lookup = gen_check.compute_generation_by_substation(sub_coords)
    island_generation_mw = sum(generation_lookup.get(m, 0.0) for m in members)

    exposure = crit.load_economic_exposure()
    island_exposure = exposure[exposure["sub_id"].isin(members)]
    island_population = float(island_exposure["population"].sum())
    island_gdp_total_musd = float(island_exposure["gdp_total"].sum())

    _, peak_mw = demand_proxy.estimate_demand_mw(pd.Series([island_population]))
    island_peak_demand_mw_est = float(peak_mw.iloc[0])

    if island_peak_demand_mw_est > 0:
        coverage_ratio = island_generation_mw / island_peak_demand_mw_est
    else:
        coverage_ratio = float("inf") if island_generation_mw > 0 else 0.0

    return {
        "island_generation_mw": island_generation_mw,
        "island_population": island_population,
        "island_gdp_total_musd": island_gdp_total_musd,
        "island_peak_demand_mw_est": island_peak_demand_mw_est,
        "coverage_ratio": coverage_ratio,
    }


def compute_own_outage_cost_musd(
    island_gdp_total_musd: float, coverage_ratio: float, disconnection_hours: float = ASSUMED_DISCONNECTION_HOURS
) -> dict:
    """Raw cost: island's annual GDP x (disconnection hours / hours-in-a-year)
    -- the fraction of a year's economic output assumed lost while the
    island is disconnected. Discounted by generation coverage: an island
    whose matched generation fully covers its demand-proxy has this cost
    zeroed; zero generation leaves it at full value."""
    raw_musd = island_gdp_total_musd * (disconnection_hours / (365 * 24))
    discount = max(0.0, 1.0 - min(coverage_ratio, 1.0))
    return {"raw_musd": raw_musd, "discount": discount, "effective_musd": raw_musd * discount}


def compute_switching_cost_musd(boundary_line_count: int, per_line_usd: float = SWITCHING_COST_PER_LINE_USD) -> float:
    return boundary_line_count * per_line_usd / 1e6


def main(
    island_id: str = "a",
    disconnection_hours: float = ASSUMED_DISCONNECTION_HOURS,
    per_line_usd: float = SWITCHING_COST_PER_LINE_USD,
):
    island = load_island(island_id)
    members = island["members"]
    boundary_line_count = island["boundary_line_count"]

    gen_demand = compute_island_generation_and_demand(members)
    outage_cost = compute_own_outage_cost_musd(
        gen_demand["island_gdp_total_musd"], gen_demand["coverage_ratio"], disconnection_hours
    )
    switching_cost_musd = compute_switching_cost_musd(boundary_line_count, per_line_usd)
    total_cost_musd = outage_cost["effective_musd"] + switching_cost_musd

    logger.info(
        f"Island {island_id}: {len(members)} members, {boundary_line_count} boundary lines. "
        f"Generation {gen_demand['island_generation_mw']:.1f} MW vs. estimated peak demand "
        f"{gen_demand['island_peak_demand_mw_est']:.1f} MW (coverage ratio "
        f"{gen_demand['coverage_ratio']:.2f}). Own-outage cost: ${outage_cost['raw_musd']:.4f}M raw, "
        f"discounted {outage_cost['discount']*100:.1f}% by generation -> ${outage_cost['effective_musd']:.4f}M "
        f"effective. Switching cost: ${switching_cost_musd:.4f}M. Total one-time cost: ${total_cost_musd:.4f}M."
    )

    econ_result = econ_cmp.main(island_id=island_id, confidence="mean")
    by_scenario = econ_result["by_scenario"]  # $M/day diff, positive = benefit

    rows = []
    for variant_name, days in REPAIR_DURATION_DAYS_VARIANTS.items():
        for scenario, diff_per_day in by_scenario.items():
            total_benefit_musd = diff_per_day * days
            net_musd = total_benefit_musd - total_cost_musd
            if total_cost_musd > 0:
                ratio = total_benefit_musd / total_cost_musd
            else:
                ratio = float("inf") if total_benefit_musd > 0 else float("-inf") if total_benefit_musd < 0 else None
            rows.append({
                "island_id": island_id, "repair_duration_variant": variant_name, "repair_duration_days": days,
                "scenario": scenario, "diff_per_day_musd": diff_per_day,
                "total_benefit_musd": total_benefit_musd, "total_cost_musd": total_cost_musd,
                "net_musd": net_musd, "benefit_cost_ratio": ratio,
            })

    df = pd.DataFrame(rows)
    out_path = VA_ECON_DIR / f"va_island_{island_id}_benefit_cost.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(df)} rows)")

    mid = df[df["repair_duration_variant"] == "mid_365d"].set_index("scenario")
    logger.info(
        f"Island {island_id}, mid_365d repair-duration variant, net $M by scenario:\n"
        + mid[["diff_per_day_musd", "total_benefit_musd", "net_musd", "benefit_cost_ratio"]].to_string()
    )
    n_net_positive = int((mid["net_musd"] > 0).sum())
    logger.info(
        f"{n_net_positive}/{len(mid)} scenarios net-positive after switching + own-outage cost "
        f"(mid_365d variant, {disconnection_hours}h disconnection assumption)."
    )

    return {"gen_demand": gen_demand, "outage_cost": outage_cost, "switching_cost_musd": switching_cost_musd, "df": df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute an island's full benefit-cost accounting, including generation-discounted own-outage cost.")
    parser.add_argument("--island", type=str, default="a")
    parser.add_argument("--disconnection-hours", type=float, default=ASSUMED_DISCONNECTION_HOURS)
    parser.add_argument("--per-line-usd", type=float, default=SWITCHING_COST_PER_LINE_USD)
    args = parser.parse_args()
    main(island_id=args.island, disconnection_hours=args.disconnection_hours, per_line_usd=args.per_line_usd)
