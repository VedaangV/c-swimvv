"""
Phase 5, part 2: the benefit-cost accounting ISLANDING_MITIGATION_PLAN.md's
Phase 5 section designed but never coded. compare_island_econ.py's diff is
just the fragility-driven $/day change from redistributed GIC -- it doesn't
account for switching-labor cost or the island's own outage while
disconnected. This adds both, unconditionally (every island is assumed to
go dark for ASSUMED_DISCONNECTION_HOURS).

Stranded-substation cost: cutting an island's boundary can split off
non-member substations too, if the island happened to sit on their only
path to the rest of PJM (seen for real on Island K: 51 components after the
cut). check_island_connectivity() in run_virginia_islanded_gic.py persists
which substations end up stranded (va_island_{id}_stranded.json); this
script charges them the same outage-cost formula as members, reported
separately as stranded_outage_cost_musd. Islands run before this file
existed get zero stranded cost and a warning -- their total_cost_musd is an
optimistic lower bound until re-run.

Generation credit removed (not just unused): an earlier version discounted
own-outage cost by matched EIA-860 generation coverage. Dropped because
Island G, selected with no generation requirement, beat Island F (9/10 vs
6/10 net-positive scenarios) -- see
scripts/deprecated_sci_islanding/README.md for the reasoning; the retired
code lives there too.

Assumptions below are order-of-magnitude guesses, not real utility data
(same spirit as the NZ paper's ~$250k switching-cost assumption):
- SWITCHING_COST_PER_LINE_USD: an order of magnitude below the NZ paper's
  total figure, since that covered a national multi-line plan, not one
  island's boundary.
- SWITCHING_EVENTS_PER_STORM = 2: lines get opened before the storm and
  reclosed after -- two real labor events, not one.
- ASSUMED_DISCONNECTION_HOURS: how long the island is offline during a
  storm's peak-risk window; sizes the own-outage cost term.
- REPAIR_DURATION_DAYS_VARIANTS: how many days to integrate the $/day diff
  over to get a one-time benefit (stands in for how long a damaged
  transformer stays broken -- 12-24+ months per the plan). Reported as
  low/mid/high (180/365/730d) since this is the biggest uncertainty here.

Usage:
    python scripts/compute_island_benefit_cost.py --island g
    python scripts/compute_island_benefit_cost.py --island a
"""

import argparse
import json

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR
import scripts.compare_island_econ as econ_cmp
import scripts.virginia_risk_data as crit

logger = setup_logger(log_file="logs/compute_island_benefit_cost.log")

SWITCHING_COST_PER_LINE_USD = 25_000.0
SWITCHING_EVENTS_PER_STORM = 2  # open before the storm's peak-risk window, reclose after
ASSUMED_DISCONNECTION_HOURS = 6.0
REPAIR_DURATION_DAYS_VARIANTS = {"low_180d": 180, "mid_365d": 365, "high_730d": 730}


def load_island(island_id: str) -> dict:
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run an island-selection script first (e.g. "
            "scripts/select_island_by_redistribution_sensitivity.py)."
        )
    with open(path) as f:
        return json.load(f)


def compute_island_gdp(members: list) -> dict:
    """Island-wide population and annual GDP, summed across members. Sizes
    the own-outage cost term below."""
    exposure = crit.load_economic_exposure()
    island_exposure = exposure[exposure["sub_id"].isin(members)]
    island_population = float(island_exposure["population"].sum())
    island_gdp_total_musd = float(island_exposure["gdp_total"].sum())

    return {
        "island_population": island_population,
        "island_gdp_total_musd": island_gdp_total_musd,
    }


def load_stranded_substations(island_id: str) -> list:
    """Reads va_island_{id}_stranded.json (written by check_island_
    connectivity() in run_virginia_islanded_gic.py) -- substations outside
    the formal island that got stranded in their own component when the
    boundary was cut. Missing file means the island predates this
    tracking; logs a warning and returns [] rather than failing."""
    path = VA_NETWORK_DIR / f"va_island_{island_id}_stranded.json"
    if not path.exists():
        logger.warning(
            f"No {path} -- Island {island_id} was solved before stranded-substation cost "
            "tracking was added (or run_virginia_islanded_gic.py hasn't been re-run since). "
            "Proceeding with zero stranded cost; total_cost_musd below may be an optimistic "
            f"lower bound. Re-run `python scripts/run_virginia_islanded_gic.py --island "
            f"{island_id}` to regenerate this file and get the corrected total."
        )
        return []
    with open(path) as f:
        data = json.load(f)
    stranded = data.get("stranded_substations", [])
    if stranded:
        logger.info(f"Island {island_id}: {len(stranded)} substation(s) stranded outside the island by the cut.")
    return stranded


def compute_stranded_gdp(stranded_substations: list) -> dict:
    """Same population/GDP lookup as compute_island_gdp(), for substations
    stranded by the cut rather than formal members. Kept separate so the
    two never get conflated in logs/output."""
    if not stranded_substations:
        return {"stranded_population": 0.0, "stranded_gdp_total_musd": 0.0}
    exposure = crit.load_economic_exposure()
    stranded_exposure = exposure[exposure["sub_id"].isin(stranded_substations)]
    return {
        "stranded_population": float(stranded_exposure["population"].sum()),
        "stranded_gdp_total_musd": float(stranded_exposure["gdp_total"].sum()),
    }


def compute_own_outage_cost_musd(
    island_gdp_total_musd: float, disconnection_hours: float = ASSUMED_DISCONNECTION_HOURS
) -> dict:
    """Island's annual GDP x (disconnection hours / hours in a year) --
    the fraction of a year's output assumed lost while disconnected.
    Unconditional; see module docstring on why generation credit was
    dropped."""
    musd = island_gdp_total_musd * (disconnection_hours / (365 * 24))
    return {"musd": musd}


def compute_switching_cost_musd(
    boundary_line_count: int,
    per_line_usd: float = SWITCHING_COST_PER_LINE_USD,
    events_per_storm: int = SWITCHING_EVENTS_PER_STORM,
) -> float:
    return boundary_line_count * per_line_usd * events_per_storm / 1e6


def main(
    island_id: str = "g",
    disconnection_hours: float = ASSUMED_DISCONNECTION_HOURS,
    per_line_usd: float = SWITCHING_COST_PER_LINE_USD,
    events_per_storm: int = SWITCHING_EVENTS_PER_STORM,
):
    island = load_island(island_id)
    members = island["members"]
    boundary_line_count = island["boundary_line_count"]

    island_gdp = compute_island_gdp(members)
    outage_cost = compute_own_outage_cost_musd(island_gdp["island_gdp_total_musd"], disconnection_hours)
    switching_cost_musd = compute_switching_cost_musd(boundary_line_count, per_line_usd, events_per_storm)

    stranded_substations = load_stranded_substations(island_id)
    stranded_gdp = compute_stranded_gdp(stranded_substations)
    stranded_outage_cost = compute_own_outage_cost_musd(stranded_gdp["stranded_gdp_total_musd"], disconnection_hours)

    total_cost_musd = outage_cost["musd"] + switching_cost_musd + stranded_outage_cost["musd"]

    logger.info(
        f"Island {island_id}: {len(members)} members, {boundary_line_count} boundary lines, "
        f"{len(stranded_substations)} stranded by the cut. "
        f"Own-outage cost: ${outage_cost['musd']:.4f}M ({disconnection_hours}h assumed disconnection). "
        f"Stranded-substation outage cost: ${stranded_outage_cost['musd']:.4f}M. "
        f"Switching cost: ${switching_cost_musd:.4f}M ({boundary_line_count} lines x "
        f"{events_per_storm} switching events). Total one-time cost: ${total_cost_musd:.4f}M."
    )

    econ_result = econ_cmp.main(island_id=island_id, confidence="mean")
    by_scenario = econ_result["by_scenario"]  # $M/day diff; positive = benefit

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
                "own_outage_cost_musd": outage_cost["musd"], "switching_cost_musd": switching_cost_musd,
                "stranded_outage_cost_musd": stranded_outage_cost["musd"],
                "n_stranded_substations": len(stranded_substations),
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

    return {
        "island_gdp": island_gdp, "outage_cost": outage_cost, "switching_cost_musd": switching_cost_musd,
        "stranded_substations": stranded_substations, "stranded_gdp": stranded_gdp,
        "stranded_outage_cost": stranded_outage_cost, "df": df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute an island's full benefit-cost accounting (switching cost + unconditional own-outage cost).")
    parser.add_argument("--island", type=str, default="g")
    parser.add_argument("--disconnection-hours", type=float, default=ASSUMED_DISCONNECTION_HOURS)
    parser.add_argument("--per-line-usd", type=float, default=SWITCHING_COST_PER_LINE_USD)
    parser.add_argument("--events-per-storm", type=int, default=SWITCHING_EVENTS_PER_STORM)
    args = parser.parse_args()
    main(
        island_id=args.island, disconnection_hours=args.disconnection_hours,
        per_line_usd=args.per_line_usd, events_per_storm=args.events_per_storm,
    )
