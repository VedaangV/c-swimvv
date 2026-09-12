"""
Pull real numbers for a paper table (e.g. "Estimated daily economic and
human impact on Virginia by storm return period") straight from this
project's own saved econ-pipeline output, instead of hand-copying/
recomputing them.

Reads two files scripts/run_virginia_econ_analysis.py already produces:
- va_leontief_consumption_production_results.csv (or the ghosh/leontief_
  cons equivalent, via --model) -- per scenario x confidence(mean/p5/p95)
  x sector rows of direct_shock/total_impact. "Total loss" sums
  |total_impact| across sectors; "Direct loss" sums |direct_shock| --
  same aggregation viz/plot_utils.py's export_econ_comparison_table()
  already uses for its own comparison table, just not scaled to $B and not
  restricted to one model.
- va_confidence_intervals.csv (get_confidence_intervals_unrounded()'s
  output) -- already has POP_AFFECTED/EST_TOTAL rows with mean/p5/p95 per
  scenario; this is where "population/establishments at risk" come from,
  not the IO model output.

HONEST LIMITATION ON "MEAN +/- STANDARD DEVIATION": this pipeline's saved
output only ever has mean/p5/p95, never a directly-computed sample
standard deviation (get_confidence_intervals_unrounded() computes
np.percentile, not np.std, and nothing downstream retains per-realization
total_impact after the IO transform). What this script reports as "std"
is the standard normal-distribution APPROXIMATION from a 90% CI:
std ~= (p95 - p5) / (2 * 1.645) ~= (p95 - p5) / 3.29 -- exact only if the
underlying distribution is genuinely normal, which a fragility-driven,
right-skewed quantity generally is NOT (see viz/plot_virginia_winding_gic_
ranking.py's docstring for a documented real case of that skew in this
same pipeline). Report it as an approximation in the paper, not a
directly-measured standard deviation, unless a true bootstrap/realization-
level std is computed separately.

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

# z-score for a 90% CI (p5-p95); std ~= (p95-p5) / (2*Z) under a normal
# approximation -- see module docstring's HONEST LIMITATION section.
Z_90 = 1.645


def _match_scenario(all_scenarios, period_key: str) -> str:
    """Find the one scenario string containing `period_key` (e.g. '100'
    matches '100-year', 'gannon' matches 'gannon-year' or the raw,
    uncleaned 'e_gannon-year-hazard A/ph' -- scenario naming differs
    between the IO-model output (cleaned by _clean_scenario_name()) and
    va_confidence_intervals.csv (raw ds.scenario.values), so this matches
    by substring against whichever form is present rather than assuming
    one exact string)."""
    needle = f"{period_key}-year"
    matches = [s for s in all_scenarios if needle in str(s)]
    if not matches:
        raise ValueError(f"No scenario matching '{needle}' in {list(all_scenarios)}")
    if len(matches) > 1:
        logger.warning(f"Multiple scenarios matched '{needle}': {matches}; using the first.")
    return matches[0]


def econ_loss_row(io_df: pd.DataFrame, period_key: str) -> dict:
    """Total/direct loss (mean +/- approx std, $M/day), summed over
    sectors, for one scenario."""
    scenario = _match_scenario(io_df["scenario"].unique(), period_key)
    scen_df = io_df[io_df["scenario"] == scenario]

    out = {}
    for label, col in [("total", "total_impact"), ("direct", "direct_shock")]:
        by_conf = scen_df.groupby("confidence")[col].apply(lambda s: s.abs().sum())
        mean_val = by_conf.get("mean", float("nan"))
        p5_val = by_conf.get("p5", float("nan"))
        p95_val = by_conf.get("p95", float("nan"))
        approx_std = abs(p95_val - p5_val) / (2 * Z_90)
        out[f"{label}_mean"] = mean_val
        out[f"{label}_approx_std"] = approx_std
    return out


def risk_counts_row(ci_df: pd.DataFrame, period_key: str) -> dict:
    """Population/establishments at risk (thousands, mean only -- the
    table template shows these as plain approximate values, not
    mean+/-std)."""
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
        total_str = f"{loss['total_mean']:.1f} ± {loss['total_approx_std']:.1f}"
        direct_str = f"{loss['direct_mean']:.1f} ± {loss['direct_approx_std']:.1f}"
        pop_str = f"{risk['pop_thousands']:.0f}"
        est_str = f"{risk['est_thousands']:.1f}"
        print(f"{label:<15}{total_str:<24}{direct_str:<24}{pop_str:<18}{est_str}")

    print("\nLaTeX rows (Leontief consumption+production style):")
    for period_key in periods:
        loss = econ_loss_row(io_df, period_key)
        risk = risk_counts_row(ci_df, period_key)
        label = "Gannon (2024)" if period_key == "gannon" else f"{period_key}-year"
        print(
            f"{label} & ${loss['total_mean']:.1f} \\pm {loss['total_approx_std']:.1f}$ & "
            f"${loss['direct_mean']:.1f} \\pm {loss['direct_approx_std']:.1f}$ & "
            f"{risk['pop_thousands']:.0f} & {risk['est_thousands']:.1f} \\\\"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull real econ-table values from saved Virginia econ-pipeline output.")
    parser.add_argument("--model", type=str, default="leontief_cp", choices=list(MODEL_FILES.keys()))
    parser.add_argument("--periods", type=str, nargs="*", default=["100", "150", "200", "250", "gannon"])
    args = parser.parse_args()
    main(model=args.model, periods=args.periods)
