"""
Pulls real numbers for a paper table (e.g. "Estimated daily economic and
human impact on Virginia by storm return period") from this project's saved
econ-pipeline output instead of hand-copying/recomputing them.

Reads two files run_virginia_econ_analysis.py already produces:
- va_leontief_consumption_production_results.csv (or ghosh/leontief_cons
  via --model) for total/direct loss, summing |total_impact|/|direct_shock|
  across sectors per scenario x confidence.
- va_confidence_intervals.csv for POP_AFFECTED/EST_TOTAL (population /
  establishments at risk).

Caveat on "mean +/- ...": the saved output only has mean/p5/p95, never a
real sample std. What this reports is HALF THE WIDTH OF THE EMPIRICAL 90%
INTERVAL, (p95-p5)/2 -- matching plot_virginia_econ_impact.py's convention.
An earlier revision divided by 2*1.645 instead (treating p5/p95 as exactly
mean +/- 1.645*sigma), which gives a ~1.645x smaller number and is why two
versions of this disagreed before. Stuck with the plain half-CI-width since
it doesn't assume normality, which these fragility-driven, often
right-skewed quantities don't satisfy anyway. Report it as "half the 90% CI
width" in the paper, not as a standard deviation.

Usage:
    python scripts/extract_econ_table_values.py --model leontief_cp
    python scripts/extract_econ_table_values.py --model ghosh --periods 100 150 200 250 gannon
"""

import argparse

import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR

logger = setup_logger(log_file="logs/extract_econ_table_values.log")

MODEL_FILES = {
    "ghosh": "va_ghosh_results.csv",
    "leontief_cons": "va_leontief_consumption_results.csv",
    "leontief_cp": "va_leontief_consumption_production_results.csv",
}

def _match_scenario(all_scenarios, period_key: str) -> str:
    """Find the one scenario string containing `period_key` (e.g. '100'
    matches '100-year'). Matches by substring since scenario naming differs
    between the IO-model output (cleaned) and va_confidence_intervals.csv
    (raw), so we can't assume one exact string."""
    needle = f"{period_key}-year"
    matches = [s for s in all_scenarios if needle in str(s)]
    if not matches:
        raise ValueError(f"No scenario matching '{needle}' in {list(all_scenarios)}")
    if len(matches) > 1:
        logger.warning(f"Multiple scenarios matched '{needle}': {matches}; using the first.")
    return matches[0]


def econ_loss_row(io_df: pd.DataFrame, period_key: str) -> dict:
    """Total/direct loss (mean +/- half the 90% CI width, $M/day), summed
    over sectors, for one scenario. See module docstring for why half-CI-
    width rather than a std."""
    scenario = _match_scenario(io_df["scenario"].unique(), period_key)
    scen_df = io_df[io_df["scenario"] == scenario]

    out = {}
    for label, col in [("total", "total_impact"), ("direct", "direct_shock")]:
        by_conf = scen_df.groupby("confidence")[col].apply(lambda s: s.abs().sum())
        mean_val = by_conf.get("mean", float("nan"))
        p5_val = by_conf.get("p5", float("nan"))
        p95_val = by_conf.get("p95", float("nan"))
        half_ci_width = abs(p95_val - p5_val) / 2
        out[f"{label}_mean"] = mean_val
        out[f"{label}_half_ci"] = half_ci_width
    return out


def risk_counts_row(ci_df: pd.DataFrame, period_key: str) -> dict:
    """Population/establishments at risk (thousands, mean only -- the
    table just shows plain approximate values here, not mean+/-std)."""
    scenario = _match_scenario(ci_df["scenario"].unique(), period_key)
    scen_df = ci_df[ci_df["scenario"] == scenario]

    def _mean_for(variable):
        row = scen_df[scen_df["variable"] == variable]
        return float(row["mean"].iloc[0]) if not row.empty else float("nan")

    return {
        "pop_thousands": _mean_for("POP_AFFECTED") / 1000.0,
        "est_thousands": _mean_for("EST_TOTAL") / 1000.0,
    }


def main(model: str, periods: list):
    io_path = VA_ECON_DIR / MODEL_FILES[model]
    ci_path = VA_ECON_DIR / "va_confidence_intervals.csv"
    if not io_path.exists():
        raise FileNotFoundError(f"Missing {io_path}. Run scripts/run_virginia_econ_analysis.py first.")
    if not ci_path.exists():
        raise FileNotFoundError(f"Missing {ci_path}. Run scripts/run_virginia_econ_analysis.py first.")

    io_df = pd.read_csv(io_path)
    ci_df = pd.read_csv(ci_path)

    print(f"\n{'Return period':<15}{'Total loss ($M/day)':<24}{'Direct loss ($M/day)':<24}"
          f"{'Pop at risk (k)':<18}{'Establishments at risk (k)':<26}")
    print("-" * 107)
    for period_key in periods:
        loss = econ_loss_row(io_df, period_key)
        risk = risk_counts_row(ci_df, period_key)
        label = "Gannon (2024)" if period_key == "gannon" else f"{period_key}-year"
        total_str = f"{loss['total_mean']:.1f} ± {loss['total_half_ci']:.1f}"
        direct_str = f"{loss['direct_mean']:.1f} ± {loss['direct_half_ci']:.1f}"
        pop_str = f"{risk['pop_thousands']:.0f}"
        est_str = f"{risk['est_thousands']:.1f}"
        print(f"{label:<15}{total_str:<24}{direct_str:<24}{pop_str:<18}{est_str}")

    print("\nLaTeX rows (Leontief consumption+production style):")
    for period_key in periods:
        loss = econ_loss_row(io_df, period_key)
        risk = risk_counts_row(ci_df, period_key)
        label = "Gannon (2024)" if period_key == "gannon" else f"{period_key}-year"
        print(
            f"{label} & ${loss['total_mean']:.1f} \\pm {loss['total_half_ci']:.1f}$ & "
            f"${loss['direct_mean']:.1f} \\pm {loss['direct_half_ci']:.1f}$ & "
            f"{risk['pop_thousands']:.0f} & {risk['est_thousands']:.1f} \\\\"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull real econ-table values from saved Virginia econ-pipeline output.")
    parser.add_argument("--model", type=str, default="leontief_cp", choices=list(MODEL_FILES.keys()))
    parser.add_argument("--periods", type=str, nargs="*", default=["100", "150", "200", "250", "gannon"])
    args = parser.parse_args()
    main(model=args.model, periods=args.periods)
