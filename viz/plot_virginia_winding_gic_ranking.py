"""
Top-N most vulnerable Virginia substations by winding GIC (A/ph) for a
given storm period, replacing an ad hoc chart that had two real problems:
unexplained/duplicate-looking coordinate labels instead of real substation
names, and a linear-axis mean+p95-whisker bar chart where the whisker
often dwarfed the bar itself.

Reads scripts/extract_virginia_results.py's summarize_and_rank() output
(va_gic_summary.csv) -- WINDING GIC (A/ph, per-transformer-winding), not
the ground-GIC ranking viz/plot_virginia_gic_risk.py already covers (see
that function's own docstring for why the two are ranked separately:
winding GIC is signed and its sign depends on storm direction and which
winding -- HV/LV/Series/Common -- is measured, not on magnitude, so
ranking uses |GIC| there already).

NAMES: reuses viz/plot_virginia_gic_risk.py's _load_substation_labels()
completely unmodified -- the same tiered real-name lookup (external
va_substation_names.csv -> OSM SS_NAME -> operator+voltage+coordinates ->
"Unnamed" + coordinates) that script already built and uses for the
ground-GIC chart. Not duplicated here.

WHY THE P95 WHISKER IS OFTEN SEVERAL TIMES THE MEAN, AND WHY THAT'S REAL:
summarize_and_rank()'s own docstring explains winding GIC's sign flips
with storm direction and which winding is measured, so even after ranking
on |GIC| the underlying per-Monte-Carlo-realization distribution can be
heavily right-skewed -- usually modest, occasionally spiking hard on a
particular random transformer-composition draw. A p95 several times the
mean is a real property of that skew, observed directly in this project's
own real data, not an error. What WAS a real problem is rendering that on
a LINEAR x-axis with a symmetric-looking error bar: a whisker several
times the bar's own length visually dominates and reads as broken, and a
symmetric xerr (as viz/plot_virginia_gic_risk.py's ground-GIC chart uses)
can imply a leftward extent that goes negative and gets silently clipped.
This script uses (a) an ASYMMETRIC error bar (lower=0, upper=p95-mean --
there's no meaningful "below the mean" whisker for a magnitude quantity)
and (b) a LOG-scale x-axis, which compresses the real dynamic range into
something legible while showing the exact same underlying numbers -- no
data is altered or hidden, only how the same two numbers are drawn.

Usage:
    python viz/plot_virginia_winding_gic_ranking.py --period 100
    python viz/plot_virginia_winding_gic_ranking.py --period gannon --top-n 20
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import VA_ECON_DIR, VA_FIGURES_DIR
from viz.plot_virginia_gic_risk import _load_substation_labels

logger = setup_logger(log_file="logs/plot_virginia_winding_gic_ranking.log")

TOP_N = 15


def _find_hazard_columns(columns, period: str) -> tuple:
    """Match this period's _mean/_p95 columns by substring rather than an
    exact hardcoded prefix, since the hazard-column naming observed across
    this project varies (e.g. '100-year-hazard A/ph' vs
    'e_100-year-hazard A/ph') -- 'in' matches either. Works for both
    synthetic return periods ('100', '250') and real storms ('gannon',
    'halloween', 'st_patricks'), which all share the same '<name>-year-
    hazard A/ph' suffix convention in this pipeline."""
    needle = f"{period}-year-hazard A/ph"
    mean_col = next((c for c in columns if needle in c and c.endswith("_mean")), None)
    p95_col = next((c for c in columns if needle in c and c.endswith("_p95")), None)
    if mean_col is None or p95_col is None:
        raise ValueError(
            f"No '_mean'/'_p95' column matching '{needle}' found in va_gic_summary.csv. "
            f"Available columns: {list(columns)}"
        )
    return mean_col, p95_col


def main(period: str = "100", top_n: int = TOP_N):
    setup_matplotlib()

    summary_path = VA_ECON_DIR / "va_gic_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run scripts/extract_virginia_results.py first.")

    summary = pd.read_csv(summary_path)
    summary["sub_id"] = summary["sub_id"].astype(str)

    mean_col, p95_col = _find_hazard_columns(summary.columns, period)
    logger.info(f"Using columns: mean={mean_col!r}, p95={p95_col!r}")

    labels = _load_substation_labels()
    summary["display_name"] = summary["sub_id"].map(labels).fillna(summary["sub_id"])

    top = summary.sort_values(mean_col, ascending=False).head(top_n)
    order = top.iloc[::-1]  # largest at the top of a horizontal bar chart

    mean_vals = order[mean_col].to_numpy(dtype=float)
    p95_vals = order[p95_col].to_numpy(dtype=float)
    upper_err = np.clip(p95_vals - mean_vals, a_min=0, a_max=None)
    lower_err = np.zeros_like(upper_err)

    y = range(len(order))
    fig, ax = plt.subplots(figsize=(11, 8))

    ax.barh(
        y, mean_vals, xerr=[lower_err, upper_err], color="#d73027", ecolor="0.2",
        capsize=3, zorder=2,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    for yi, (mean_v, p95_v) in enumerate(zip(mean_vals, p95_vals)):
        ax.text(max(mean_v, p95_v) * 1.08, yi, f"{mean_v:.1f} (p95 {p95_v:.1f})", va="center", fontsize=8)

    ax.set_yticks(list(y))
    ax.set_yticklabels(order["display_name"], fontsize=8)
    ax.set_xlabel(f"Winding GIC, |A/ph| ({period}-year storm) — log scale, whisker = p95")
    ax.set_title(f"Top {len(top)} most vulnerable Virginia substations ({period}-year return period storm)")

    fig.tight_layout()
    out_path = VA_FIGURES_DIR / f"va_winding_gic_ranking_{period}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    if len(top):
        logger.info(f"Top substation: {top.iloc[0]['display_name']} ({top.iloc[0][mean_col]:.2f} A/ph mean)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Top-N Virginia substations by winding GIC for one storm period, real names, log-scale whisker."
    )
    parser.add_argument("--period", type=str, default="100", help="Return period, e.g. '100', '250', 'gannon'.")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()
    main(period=args.period, top_n=args.top_n)
