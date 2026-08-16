"""
Diagnostic for Phase 5's surprising first result: the islanded-topology
Ghosh economic-impact output shows *worse* whole-Virginia total impact than
baseline in every scenario/confidence combination, not better (see
ISLANDING_MITIGATION_PLAN.md). Before treating that as a real "islanding
backfires" finding rather than a bug, this checks the more basic question:
did cutting Island A's boundary lines even reduce GIC *at Island A's own
3 substations* -- the one thing the topology edit should unambiguously do,
regardless of what happens to the rest of the network.

Compares mean/p95 |ground GIC| at Island A's members only, baseline
(VA_GND_GIC_DIR) vs islanded (VA_ISLAND_A_GND_GIC_DIR), per storm scenario.
Same "take the magnitude, not the signed value" reasoning as scripts/
extract_virginia_results.py's summarize_ground_gic() -- ground GIC's sign
depends on storm field direction, not on how large/damaging the current
actually is, so averaging signed values could understate real magnitude.

Two possible outcomes, both informative:
- Island A's own GIC drops in the islanded run (local protection worked),
  but the aggregate econ result is still worse -> supports the network-wide
  redistribution explanation (cutting the boundary rerouted more current
  onto other paths elsewhere in Virginia than it removed locally) -- a
  real, reportable finding, not a bug.
- Island A's own GIC does NOT drop (flat or worse even locally) -> points
  at something wrong upstream (the cut, the connectivity check, or the
  Monte Carlo itself), worth debugging before trusting either result.

Run after both scripts/run_virginia_gic.py and
scripts/run_virginia_islanded_gic.py have completed.

Usage:
    python scripts/compare_island_gic.py
"""

import json
from glob import glob

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_GND_GIC_DIR, VA_ISLAND_A_GND_GIC_DIR, VA_NETWORK_DIR, VA_ECON_DIR

logger = setup_logger(log_file="logs/compare_island_gic.log")


def load_island_members() -> list:
    path = VA_NETWORK_DIR / "va_island_a.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/select_virginia_island.py first.")
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
            f"None of Island A's {len(members)} members appear in any ground_gic*.csv under "
            f"{gnd_gic_dir} -- check sub_id matching between va_island_a.json and this run's output."
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


def build_comparison(combined: pd.DataFrame, gic_cols: list) -> pd.DataFrame:
    """Reshape the wide baseline_GIC_{period}_{stat} / island_a_GIC_{period}_{stat}
    summary into a long sub_id x period x stat table with diff/pct_change --
    shared by this script and scripts/compare_virginia_gic.py (same summarize()
    output shape, different substation scope)."""
    rows = []
    for period in gic_cols:
        for stat in ("mean", "p95"):
            base_col = f"baseline_GIC_{period}_{stat}"
            isl_col = f"island_a_GIC_{period}_{stat}"
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
                    "baseline_gic": base_val, "island_a_gic": isl_val,
                    "diff": diff, "pct_change": pct,
                })
    return pd.DataFrame(rows).sort_values(["sub_id", "period", "stat"])


def main():
    members = set(load_island_members())
    logger.info(f"Comparing GIC at Island A's {len(members)} members: {sorted(members)}")

    base_df = load_member_gic(VA_GND_GIC_DIR, members, "baseline")
    isl_df = load_member_gic(VA_ISLAND_A_GND_GIC_DIR, members, "island_a")

    base_summary = summarize(base_df, "baseline")
    isl_summary = summarize(isl_df, "island_a")

    combined = base_summary.join(isl_summary, how="outer")

    gic_cols = sorted({c.split("GIC_", 1)[1] for c in base_df.columns if c.startswith("GIC_")})
    result = build_comparison(combined, gic_cols)
    out_path = VA_ECON_DIR / "va_island_a_gic_comparison.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(result)} rows)")

    mean_rows = result[result["stat"] == "mean"]
    n_reduced = (mean_rows["diff"] < 0).sum()
    n_increased = (mean_rows["diff"] > 0).sum()
    n_total = len(mean_rows)
    logger.info(
        f"Across Island A's own members x scenarios (mean stat): GIC reduced in "
        f"{n_reduced}/{n_total} rows, increased in {n_increased}/{n_total} rows."
    )

    print("\n=== GIC at Island A's own substations: baseline vs islanded (mean stat) ===")
    print(mean_rows[["sub_id", "period", "baseline_gic", "island_a_gic", "diff", "pct_change"]].to_string(index=False))

    print(f"\nSummary: reduced in {n_reduced}/{n_total}, increased in {n_increased}/{n_total} (mean stat).")
    if n_reduced == n_total:
        print(
            "Local protection worked cleanly everywhere checked -- the aggregate econ result being "
            "worse points at network-wide GIC redistribution onto other substations, not a bug here."
        )
    elif n_increased == n_total:
        print(
            "Local GIC did NOT drop anywhere checked -- worth investigating the cut/connectivity "
            "check before trusting either result. Check logs/run_virginia_islanded_gic.log's "
            "'Connectivity after cut' line."
        )
    else:
        print("Mixed result across Island A's own members/scenarios -- inspect va_island_a_gic_comparison.csv directly.")


if __name__ == "__main__":
    main()
