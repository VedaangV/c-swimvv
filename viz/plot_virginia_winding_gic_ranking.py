"""
Top-N most vulnerable Virginia substations by winding GIC (A/ph) for a
given storm period, with real substation names and a one-sided p95
whisker bar chart.

Reads va_gic_summary.csv -- WINDING GIC (A/ph, per-transformer-winding),
not the ground-GIC ranking viz/plot_virginia_gic_risk.py covers (ranked
separately since winding GIC's sign depends on storm direction/winding,
so ranking uses |GIC|). Reuses that module's _load_substation_labels()
for names.

The p95 whisker is one-sided since p95 of a magnitude is always >= the
mean, so there's nothing meaningful behind the bar. It can legitimately
be several times the mean (a real right-skew from the Monte Carlo, not a
bug) -- pass --scale log when that ratio gets large for a given period.

Usage:
    python viz/plot_virginia_winding_gic_ranking.py --period 100
    python viz/plot_virginia_winding_gic_ranking.py --period gannon --top-n 20 --scale log
"""

import argparse
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import VA_ECON_DIR, VA_FIGURES_DIR
from viz.plot_virginia_gic_risk import _load_substation_labels
from viz.plot_utils import savefig_with_retry

logger = setup_logger(log_file="logs/plot_virginia_winding_gic_ranking.log")

TOP_N = 15
LABEL_WRAP_WIDTH = 36

TITLE_FONTSIZE = 22
AXIS_LABEL_FONTSIZE = 17
TICK_LABEL_FONTSIZE = 16
VALUE_LABEL_FONTSIZE = 14


def _find_hazard_columns(columns, period: str) -> tuple:
    """Match by substring rather than exact prefix, since the hazard-column
    naming varies (e.g. '100-year-hazard A/ph' vs 'e_100-year-hazard A/ph').
    Works for both return periods ('100') and named storms ('gannon')."""
    needle = f"{period}-year-hazard A/ph"
    mean_col = next((c for c in columns if needle in c and c.endswith("_mean")), None)
    p95_col = next((c for c in columns if needle in c and c.endswith("_p95")), None)
    if mean_col is None or p95_col is None:
        raise ValueError(
            f"No '_mean'/'_p95' column matching '{needle}' found in va_gic_summary.csv. "
            f"Available columns: {list(columns)}"
        )
    return mean_col, p95_col


def main(period: str = "100", top_n: int = TOP_N, scale: str = "linear"):
    setup_matplotlib()

    summary_path = VA_ECON_DIR / "va_gic_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run scripts/extract_virginia_results.py first.")

    summary = pd.read_csv(summary_path)
    summary["sub_id"] = summary["sub_id"].astype(str)

    mean_col, p95_col = _find_hazard_columns(summary.columns, period)
    logger.info(f"Using columns: mean={mean_col!r}, p95={p95_col!r}")

    # COORD_NAME_OVERRIDES is applied inside _load_substation_labels() itself
    labels = _load_substation_labels(include_coords=True)
    summary["display_name"] = summary["sub_id"].map(labels).fillna(summary["sub_id"])

    top = summary.sort_values(mean_col, ascending=False).head(top_n)
    order = top.iloc[::-1]  # largest at the top of a horizontal bar chart

    mean_vals = order[mean_col].to_numpy(dtype=float)
    p95_vals = order[p95_col].to_numpy(dtype=float)
    upper_err = np.clip(p95_vals - mean_vals, a_min=0, a_max=None)
    lower_err = np.zeros_like(upper_err)

    y = range(len(order))
    # height scales with bar count so wrapped two-line labels don't collide
    fig, ax = plt.subplots(figsize=(15, max(8.0, 0.95 * top_n + 2.0)))

    ax.barh(
        y, mean_vals, xerr=[lower_err, upper_err], color="#d73027", ecolor="0.2",
        capsize=3, zorder=3,
    )

    max_p95 = float(np.max(p95_vals)) if len(p95_vals) else 1.0
    if scale == "log":
        ax.set_xscale("log")
        # without minor ticks, a dataset inside one decade shows only one labeled tick
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.set_xlim(right=max_p95 * 1.6)
    else:
        # headroom for value-label text at the longest whisker
        ax.set_xlim(left=0, right=max_p95 * 1.35)
        ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)

    for yi, (mean_v, p95_v) in enumerate(zip(mean_vals, p95_vals)):
        ax.text(
            max(mean_v, p95_v) * 1.03, yi, f"{mean_v:.1f} (p95 {p95_v:.1f})",
            va="center", fontsize=VALUE_LABEL_FONTSIZE, fontweight="bold",
        )

    wrapped_labels = [
        "\n".join(textwrap.wrap(name, LABEL_WRAP_WIDTH)) for name in order["display_name"]
    ]
    ax.set_yticks(list(y))
    ax.set_yticklabels(wrapped_labels, fontsize=TICK_LABEL_FONTSIZE, fontweight="bold")
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")
    scale_note = "log scale" if scale == "log" else "linear scale"
    ax.set_xlabel(
        f"Winding GIC, |A/ph| ({period}-year storm) — {scale_note}, whisker = p95",
        fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold",
    )
    ax.set_title(
        f"Top {len(top)} most vulnerable Virginia substations ({period}-year return period storm)",
        fontsize=TITLE_FONTSIZE, fontweight="bold",
    )

    fig.tight_layout()
    out_path = VA_FIGURES_DIR / f"va_winding_gic_ranking_{period}.png"
    savefig_with_retry(fig, out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    if len(top):
        logger.info(f"Top substation: {top.iloc[0]['display_name']} ({top.iloc[0][mean_col]:.2f} A/ph mean)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Top-N Virginia substations by winding GIC for one storm period, real names, one-sided p95 whisker."
    )
    parser.add_argument("--period", type=str, default="100", help="Return period, e.g. '100', '250', 'gannon'.")
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument(
        "--scale", type=str, default="linear", choices=["linear", "log"],
        help="X-axis scale. 'linear' (default) is fine for mild mean-to-p95 ratios; "
             "use 'log' when that ratio is much larger for a given period/top-N.",
    )
    args = parser.parse_args()
    main(period=args.period, top_n=args.top_n, scale=args.scale)
