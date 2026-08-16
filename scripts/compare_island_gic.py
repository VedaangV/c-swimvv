"""
Diagnostic for Phase 5's surprising first result: the islanded-topology
Ghosh economic-impact output shows *worse* whole-Virginia total impact than
baseline in every scenario/confidence combination, not better (see
ISLANDING_MITIGATION_PLAN.md). Before treating that as a real "islanding
backfires" finding rather than a bug, this checks the more basic question:
did cutting an island's boundary lines even reduce GIC *at that island's
own members* -- the one thing the topology edit should unambiguously do,
regardless of what happens to the rest of the network.

Compares mean/p95 |ground GIC| at an island's members only, baseline
(VA_GND_GIC_DIR) vs islanded (that island's gnd_gic_dir), per storm
scenario. Same "take the magnitude, not the signed value" reasoning as
scripts/extract_virginia_results.py's summarize_ground_gic() -- ground
GIC's sign depends on storm field direction, not on how large/damaging the
current actually is, so averaging signed values could understate real
magnitude.

Two possible outcomes, both informative:
- The island's own GIC drops in the islanded run (local protection
  worked), but the aggregate econ result is still worse -> supports the
  network-wide redistribution explanation (cutting the boundary rerouted
  more current onto other paths elsewhere in Virginia than it removed
  locally) -- a real, reportable finding, not a bug.
- The island's own GIC does NOT drop (flat or worse even locally) ->
  points at something wrong upstream (the cut, the connectivity check, or
  the Monte Carlo itself), worth debugging before trusting either result.

Takes an island_id ("a", "b", ...; default "a") -- reads va_island_{id}.json
and configs.virginia_config.get_island_dirs(island_id)'s gnd_gic_dir, same
generalization as scripts/run_virginia_islanded_gic.py and friends.

Run after both scripts/run_virginia_gic.py and scripts/run_virginia_
islanded_gic.py (same island_id) have completed.

Usage:
    python scripts/compare_island_gic.py [--island b]
"""

import argparse
import json
from glob import glob

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_GND_GIC_DIR, VA_NETWORK_DIR, VA_ECON_DIR, get_island_dirs

logger = setup_logger(log_file="logs/compare_island_gic.log")


def load_island_members(island_id: str = "a") -> list:
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/select_virginia_island.py (island_id='{island_id}') first."
        )
    with open(path) as f:
        island = json.load(f)
    return [str(m) for m in island["members"]]


def load_member_gic(gnd_gic_dir, members: set, label: str) -> pd.DataFrame:
    files = sorted(glob(str(gnd_gic_dir / "ground_gic*.csv")))
    if not files:
        raise FileNotFoundError(f"No ground_gic*.csv under {gnd_gic_dir}.")

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        df["Substation"] = df["Substation"].astype(str)
        df = df[df["Substation"].isin(members)]
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(
            f"None of the {len(members)} members appear in any ground_gic*.csv under "
            f"{gnd_gic_dir} -- check sub_id matching between the island json and this run's output."
        )

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"{label}: {len(combined)} rows across {len(files)} files for {combined['Substation'].nunique()} members.")
    return combined


def summarize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    gic_cols = [c for c in df.columns if c.startswith("GIC_")]
    if not gic_cols:
        raise ValueError(f"No GIC_* columns found ({label}); got {list(df.columns)}.")

    df_abs = df.copy()
    df_abs[gic_cols] = df_abs[gic_cols].abs()

    grouped = df_abs.groupby("Substation")[gic_cols]
    summary = grouped.mean().add_suffix("_mean").join(grouped.quantile(0.95).add_suffix("_p95"))
    summary.columns = [f"{label}_{c}" for c in summary.columns]
    return summary


def build_comparison(combined: pd.DataFrame, gic_cols: list, island_label: str) -> pd.DataFrame:
    """Reshape the wide baseline_GIC_{period}_{stat} / {island_label}_GIC_{period}_{stat}
    summary into a long sub_id x period x stat table with diff/pct_change --
    shared by this script and scripts/compare_virginia_gic.py (same summarize()
    output shape, different substation scope)."""
    rows = []
    for period in gic_cols:
        for stat in ("mean", "p95"):
            base_col = f"baseline_GIC_{period}_{stat}"
            isl_col = f"{island_label}_GIC_{period}_{stat}"
            if base_col not in combined.columns or isl_col not in combined.columns:
                continue
            for sub_id, row in combined.iterrows():
                base_val = row.get(base_col)
                isl_val = row.get(isl_col)
                if pd.isna(base_val) or pd.isna(isl_val):
                    continue
                diff = isl_val - base_val
                pct = 100 * diff / base_val if base_val != 0 else float("nan")
                rows.append({
                    "sub_id": sub_id, "period": period, "stat": stat,
                    "baseline_gic": base_val, "island_gic": isl_val,
                    "diff": diff, "pct_change": pct,
                })
    return pd.DataFrame(rows).sort_values(["sub_id", "period", "stat"])


def main(island_id: str = "a"):
    island_label = f"island_{island_id}"
    members = set(load_island_members(island_id))
    logger.info(f"Comparing GIC at Island {island_id}'s {len(members)} members: {sorted(members)}")

    gnd_gic_dir = get_island_dirs(island_id)["gnd_gic_dir"]
    base_df = load_member_gic(VA_GND_GIC_DIR, members, "baseline")
    isl_df = load_member_gic(gnd_gic_dir, members, island_label)

    base_summary = summarize(base_df, "baseline")
    isl_summary = summarize(isl_df, island_label)

    combined = base_summary.join(isl_summary, how="outer")

    gic_cols = sorted({c.split("GIC_", 1)[1] for c in base_df.columns if c.startswith("GIC_")})
    result = build_comparison(combined, gic_cols, island_label)
    out_path = VA_ECON_DIR / f"va_island_{island_id}_gic_comparison.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(result)} rows)")

    mean_rows = result[result["stat"] == "mean"]
    n_reduced = (mean_rows["diff"] < 0).sum()
    n_increased = (mean_rows["diff"] > 0).sum()
    n_total = len(mean_rows)
    logger.info(
        f"Across Island {island_id}'s own members x scenarios (mean stat): GIC reduced in "
        f"{n_reduced}/{n_total} rows, increased in {n_increased}/{n_total} rows."
    )

    print(f"\n=== GIC at Island {island_id}'s own substations: baseline vs islanded (mean stat) ===")
    print(mean_rows[["sub_id", "period", "baseline_gic", "island_gic", "diff", "pct_change"]].to_string(index=False))

    print(f"\nSummary: reduced in {n_reduced}/{n_total}, increased in {n_increased}/{n_total} (mean stat).")
    if n_reduced == n_total:
        print(
            "Local protection worked cleanly everywhere checked -- if the aggregate econ result is "
            "still worse, that points at network-wide GIC redistribution onto other substations, "
            "not a bug here."
        )
    elif n_increased == n_total:
        print(
            "Local GIC did NOT drop anywhere checked -- worth investigating the cut/connectivity "
            "check before trusting either result. Check logs/run_virginia_islanded_gic.log's "
            "'Connectivity after cut' line."
        )
    else:
        print(f"Mixed result across Island {island_id}'s own members/scenarios -- inspect {out_path.name} directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GIC at an island's own members, baseline vs islanded.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'b'.")
    args = parser.parse_args()
    main(island_id=args.island)
