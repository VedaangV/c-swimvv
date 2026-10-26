"""
Phase 5/6 diagnostic: diffs an island's va_ghosh_results.csv against the
baseline's, giving the whole-Virginia $/day impact difference per
scenario/confidence/sector. Was being done by eye for every island run (A,
B, AB, C, D, E, F); now a script.

Sign convention: total_impact is always <= 0. diff = island - baseline, so
diff > 0 means the loss shrank (islanding helped); diff < 0 means it grew
(islanding hurt).

Usage:
    python scripts/compare_island_econ.py --island f
    python scripts/compare_island_econ.py --island f --confidence p95
"""

import argparse

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, get_island_dirs

logger = setup_logger(log_file="logs/compare_island_econ.log")


def load_ghosh(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/run_virginia_econ_analysis.py (baseline) or "
            "scripts/run_virginia_islanded_econ_analysis.py (island) first."
        )
    return pd.read_csv(path)


def main(island_id: str = "a", confidence: str = "mean"):
    baseline_path = VA_ECON_DIR / "va_ghosh_results.csv"
    island_path = get_island_dirs(island_id)["econ_dir"] / "va_ghosh_results.csv"

    baseline = load_ghosh(baseline_path).rename(columns={"total_impact": "baseline_total_impact"})
    island = load_ghosh(island_path).rename(columns={"total_impact": "island_total_impact"})

    merged = baseline.merge(
        island[["scenario", "confidence", "sector", "island_total_impact"]],
        on=["scenario", "confidence", "sector"], how="inner",
    )
    if len(merged) != len(baseline):
        logger.warning(
            f"{len(baseline) - len(merged)} baseline rows had no matching island row "
            "(scenario/confidence/sector mismatch) -- check both files cover the same scenarios."
        )

    merged["diff"] = merged["island_total_impact"] - merged["baseline_total_impact"]

    out_path = VA_ECON_DIR / f"va_island_{island_id}_econ_comparison.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(merged)} rows)")

    conf_df = merged[merged["confidence"] == confidence]
    by_scenario = conf_df.groupby("scenario")["diff"].sum().sort_index()
    logger.info(
        f"Whole-Virginia total_impact diff by scenario ({confidence} confidence, summed across "
        "sectors -- positive means islanding reduced the loss):\n"
        + by_scenario.to_string()
    )
    n_positive = int((by_scenario > 0).sum())
    logger.info(
        f"{n_positive}/{len(by_scenario)} scenarios net-positive (islanding reduced whole-VA "
        f"loss) at {confidence} confidence. Overall sum across scenarios: {by_scenario.sum():.4f}."
    )

    return {"merged": merged, "by_scenario": by_scenario}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diff an island's Ghosh econ results against baseline.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'f'.")
    parser.add_argument("--confidence", type=str, default="mean", choices=["mean", "p5", "p95"])
    args = parser.parse_args()
    main(island_id=args.island, confidence=args.confidence)
