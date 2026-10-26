"""
Widens compare_island_gic.py's baseline-vs-islanded GIC comparison from an
island's members to every Virginia substation, to see whether the worse
whole-Virginia econ result (see ISLANDING_MITIGATION_PLAN.md) comes from a
concentrated redistribution onto a few substations (fixable by redrawing
the boundary) or a diffuse one (more fundamental to this cut).

Reuses compare_island_gic.py's load_member_gic()/summarize()/
build_comparison() as-is, just given Virginia's full substation set.

WARNING: loads/filters the same ~3000+3000 ground_gic*.csv files as
compare_island_gic.py but keeps ~2000 substations instead of a handful, so
expect a few GB of memory use.

Run after both run_virginia_gic.py and run_virginia_islanded_gic.py (same
island_id) have completed.

Usage:
    python scripts/compare_virginia_gic.py [--island b] [--top N]
"""

import argparse
import json

import pandas as pd

from configs import setup_logger
from configs.virginia_config import (
    VA_GND_GIC_DIR, VA_NETWORK_DIR, VA_ECON_DIR, get_island_dirs, get_virginia_boundary,
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


def load_island_members(island_id: str = "a") -> set:
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        return set()
    with open(path) as f:
        return {str(m) for m in json.load(f)["members"]}


def main(top_n: int = 25, island_id: str = "a"):
    island_label = f"island_{island_id}"
    va_members = load_virginia_members()
    island_members = load_island_members(island_id)

    gnd_gic_dir = get_island_dirs(island_id)["gnd_gic_dir"]
    base_df = load_member_gic(VA_GND_GIC_DIR, va_members, "baseline")
    isl_df = load_member_gic(gnd_gic_dir, va_members, island_label)

    base_summary = summarize(base_df, "baseline")
    isl_summary = summarize(isl_df, island_label)
    combined = base_summary.join(isl_summary, how="outer")

    gic_cols = sorted({c.split("GIC_", 1)[1] for c in base_df.columns if c.startswith("GIC_")})
    result = build_comparison(combined, gic_cols, island_label)
    in_island_col = f"in_island_{island_id}"
    result[in_island_col] = result["sub_id"].isin(island_members)

    out_path = VA_ECON_DIR / f"va_virginia_gic_comparison_island_{island_id}.csv"
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
        non_island = scen_rows[~scen_rows[in_island_col]]

        print(f"\n=== {scenario_label} (GIC_{period}, mean) -- top {top_n} substations by INCREASE, excluding Island {island_id} itself ===")
        top_increase = non_island.sort_values("diff", ascending=False).head(top_n)
        print(top_increase[["sub_id", "baseline_gic", "island_gic", "diff", "pct_change"]].to_string(index=False))

        total_diff_all = scen_rows["diff"].sum()
        total_diff_island = scen_rows[scen_rows[in_island_col]]["diff"].sum()
        total_diff_rest = scen_rows[~scen_rows[in_island_col]]["diff"].sum()
        top_n_diff = top_increase["diff"].sum()
        print(
            f"\nTotal diff across all VA substations: {total_diff_all:.2f} A "
            f"(Island {island_id}'s members: {total_diff_island:.2f} A, rest of Virginia: {total_diff_rest:.2f} A). "
            f"Top {top_n} non-island substations alone account for {top_n_diff:.2f} A of that "
            f"({100 * top_n_diff / total_diff_rest if total_diff_rest else float('nan'):.1f}% of the non-island total)."
        )

    print(f"\nOverall (mean stat, all periods): GIC reduced in {n_reduced}/{n_total} rows, increased in {n_increased}/{n_total} rows.")
    print(f"Full detail: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GIC across all Virginia substations, baseline vs islanded.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'b'.")
    parser.add_argument("--top", type=int, default=25, help="How many top-increase substations to print per scenario.")
    args = parser.parse_args()
    main(top_n=args.top, island_id=args.island)
