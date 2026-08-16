"""
Widens scripts/compare_island_gic.py's baseline-vs-islanded GIC comparison
from Island A's 3 members to every Virginia substation, to answer the
question that comparison alone couldn't: the aggregate whole-Virginia
Ghosh economic result already confirmed islanding is net-worse (see
ISLANDING_MITIGATION_PLAN.md), and compare_island_gic.py confirmed the
seed substation (708152212) really was protected while two other island
members got worse -- but neither says WHERE the rest of the extra damage,
beyond those 3 substations, actually landed.

Two possible pictures, both actionable:
- Concentrated: a handful of specific substations account for most of the
  increase -- worth checking whether a different boundary (excluding
  whatever's near those substations) could plausibly avoid them.
- Diffuse: increases spread thinly across most of Virginia -- suggests the
  redistribution effect is closer to fundamental for this cut, not a
  boundary-drawing mistake.

Reuses scripts/compare_island_gic.py's load_member_gic()/summarize()/
build_comparison() unmodified -- same computation, just given Virginia's
full substation set instead of Island A's 3 members. Reuses scripts/
extract_virginia_results.py's _load_va_substation_lookup() for the
Virginia-membership check (va_boundary.contains(Point(lon, lat))) rather
than re-deriving it.

WARNING: this loads and filters all ~3000 baseline + ~3000 islanded
ground_gic*.csv files down to ~2000 Virginia substations instead of 3 --
same number of files as scripts/compare_island_gic.py (so similar wall-
clock time, dominated by file I/O either way), but the resulting combined
DataFrame is much larger. Expect a few GB of memory use, not an issue on
a normal machine but worth knowing if it runs somewhere constrained.

Run after both scripts/run_virginia_gic.py and
scripts/run_virginia_islanded_gic.py have completed.

Usage:
    python scripts/compare_virginia_gic.py [--top N]
"""

import argparse
import json

import pandas as pd

from configs import setup_logger
from configs.virginia_config import (
    VA_ADMITTANCE_DIR, VA_GND_GIC_DIR, VA_ISLAND_A_GND_GIC_DIR, VA_NETWORK_DIR, VA_ECON_DIR,
    get_virginia_boundary,
)
from scripts.compare_island_gic import load_member_gic, summarize, build_comparison
from scripts.extract_virginia_results import _load_va_substation_lookup

logger = setup_logger(log_file="logs/compare_virginia_gic.log")


def load_virginia_members() -> set:
    va_boundary = get_virginia_boundary()
    sub_lookup = _load_va_substation_lookup(va_boundary)
    va_members = set(sub_lookup.loc[sub_lookup["in_virginia"], "name"])
    logger.info(f"{len(va_members)} Virginia substations (of {len(sub_lookup)} total in VA+PJM).")
    return va_members


def load_island_a_members() -> set:
    path = VA_NETWORK_DIR / "va_island_a.json"
    if not path.exists():
        return set()
    with open(path) as f:
        return {str(m) for m in json.load(f)["members"]}


def main(top_n: int = 25):
    va_members = load_virginia_members()
    island_a_members = load_island_a_members()

    base_df = load_member_gic(VA_GND_GIC_DIR, va_members, "baseline")
    isl_df = load_member_gic(VA_ISLAND_A_GND_GIC_DIR, va_members, "island_a")

    base_summary = summarize(base_df, "baseline")
    isl_summary = summarize(isl_df, "island_a")
    combined = base_summary.join(isl_summary, how="outer")

    gic_cols = sorted({c.split("GIC_", 1)[1] for c in base_df.columns if c.startswith("GIC_")})
    result = build_comparison(combined, gic_cols)
    result["in_island_a"] = result["sub_id"].isin(island_a_members)

    out_path = VA_ECON_DIR / "va_virginia_gic_comparison.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(result)} rows, {result['sub_id'].nunique()} substations)")

    mean_rows = result[result["stat"] == "mean"]
    n_reduced = (mean_rows["diff"] < 0).sum()
    n_increased = (mean_rows["diff"] > 0).sum()
    n_total = len(mean_rows)
    logger.info(
        f"Across all Virginia substations x scenarios (mean stat): GIC reduced in "
        f"{n_reduced}/{n_total} rows, increased in {n_increased}/{n_total} rows."
    )

    for scenario_label, period in [("real historical storm", "gannon"), ("100-year return period", "100")]:
        scen_rows = mean_rows[mean_rows["period"] == period]
        if scen_rows.empty:
            continue
        non_island = scen_rows[~scen_rows["in_island_a"]]

        print(f"\n=== {scenario_label} (GIC_{period}, mean) -- top {top_n} substations by INCREASE, excluding Island A itself ===")
        top_increase = non_island.sort_values("diff", ascending=False).head(top_n)
        print(top_increase[["sub_id", "baseline_gic", "island_a_gic", "diff", "pct_change"]].to_string(index=False))

        total_diff_all = scen_rows["diff"].sum()
        total_diff_island = scen_rows[scen_rows["in_island_a"]]["diff"].sum()
        total_diff_rest = scen_rows[~scen_rows["in_island_a"]]["diff"].sum()
        top_n_diff = top_increase["diff"].sum()
        print(
            f"\nTotal diff across all VA substations: {total_diff_all:.2f} A "
            f"(Island A's 3 members: {total_diff_island:.2f} A, rest of Virginia: {total_diff_rest:.2f} A). "
            f"Top {top_n} non-island substations alone account for {top_n_diff:.2f} A of that "
            f"({100 * top_n_diff / total_diff_rest if total_diff_rest else float('nan'):.1f}% of the non-island total)."
        )

    print(f"\nOverall (mean stat, all periods): GIC reduced in {n_reduced}/{n_total} rows, increased in {n_increased}/{n_total} rows.")
    print(f"Full detail: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GIC across all Virginia substations, baseline vs islanded.")
    parser.add_argument("--top", type=int, default=25, help="How many top-increase substations to print per scenario.")
    args = parser.parse_args()
    main(top_n=args.top)
