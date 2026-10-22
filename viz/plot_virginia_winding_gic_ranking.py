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

THE P95 WHISKER IS ONE-SIDED, DELIBERATELY: p95 for a magnitude quantity
is always >= the mean, so there's nothing meaningful to show "behind" the
bar -- a symmetric whisker would extend left into a value BELOW the mean,
which p95 doesn't represent. xerr=[lower_err, upper_err] with lower_err
all zeros draws exactly that: a whisker that only ever extends forward
from the bar's own end.

WHY THE P95 WHISKER CAN BE SEVERAL TIMES THE MEAN, AND WHY THAT'S REAL:
summarize_and_rank()'s own docstring explains winding GIC's sign flips
with storm direction and which winding is measured, so even after ranking
on |GIC| the underlying per-Monte-Carlo-realization distribution can be
right-skewed -- usually modest, occasionally spiking hard on a particular
random transformer-composition draw. A p95 several times the mean is a
real property of that skew, not an error, though HOW skewed varies by
scenario/top-N: --scale defaults to "linear" because the ratios observed
in this project's real 100-year data are mild (under 3x); pass
--scale log for a period/top-N where the ratio is much larger (the kind
of 4-5x case that motivated log-scale on a different, more skewed chart)
-- log compresses the dynamic range but needs LogLocator's minor ticks
turned on (handled below) or the axis goes nearly unlabeled when the
whole dataset sits inside one decade, which is what made an earlier
version of this exact chart hard to read.

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
LABEL_WRAP_WIDTH = 32

# Manual stand-in for substations _load_substation_labels() still falls
# back to bare coordinates for (no real name from any source, and no VA
# county/city match -- see that function's docstring for why). Keyed on
# the exact "(lat, lon)" fallback string it produces; the coordinates are
# kept and the name is prepended (not replaced), same "name (lat, lon)"
# convention _load_substation_labels() uses for every other tier. These
# names are APPROXIMATE placements from general Virginia geography, not
# a verified geocode -- replace with real names (or let the county
# reverse-geocode in _load_substation_labels() take over, e.g. once
# tl_2022_us_county.zip is fetched) when available; flag any that look wrong.
COORD_NAME_OVERRIDES = {
    "(37.25, -77.61)": "Chesterfield County substation",
    "(37.59, -77.39)": "New Kent County substation",
    "(37.78, -77.55)": "Hanover County substation",
    "(36.86, -81.42)": "Smyth County substation (Appalachian Power)",
    "(37.25, -77.18)": "Prince George County substation",
}

TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13
VALUE_LABEL_FONTSIZE = 11


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


def main(period: str = "100", top_n: int = TOP_N, scale: str = "linear"):
    setup_matplotlib()

    summary_path = VA_ECON_DIR / "va_gic_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run scripts/extract_virginia_results.py first.")

    summary = pd.read_csv(summary_path)
    summary["sub_id"] = summary["sub_id"].astype(str)

    mean_col, p95_col = _find_hazard_columns(summary.columns, period)
    logger.info(f"Using columns: mean={mean_col!r}, p95={p95_col!r}")

    labels = _load_substation_labels(include_coords=True)
    summary["display_name"] = summary["sub_id"].map(labels).fillna(summary["sub_id"])
    # Prepend the override name but keep the coordinates already on the
    # label (coord_key itself) -- "Chesterfield County substation
    # (37.25, -77.61)", not a name-only replacement.
    summary["display_name"] = summary["display_name"].apply(
        lambda d: f"{COORD_NAME_OVERRIDES[d]} {d}" if d in COORD_NAME_OVERRIDES else d
    )

    top = summary.sort_values(mean_col, ascending=False).head(top_n)
    order = top.iloc[::-1]  # largest at the top of a horizontal bar chart

    mean_vals = order[mean_col].to_numpy(dtype=float)
    p95_vals = order[p95_col].to_numpy(dtype=float)
    upper_err = np.clip(p95_vals - mean_vals, a_min=0, a_max=None)
    lower_err = np.zeros_like(upper_err)

    y = range(len(order))
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.barh(
        y, mean_vals, xerr=[lower_err, upper_err], color="#d73027", ecolor="0.2",
        capsize=3, zorder=3,
    )

    max_p95 = float(np.max(p95_vals)) if len(p95_vals) else 1.0
    if scale == "log":
        ax.set_xscale("log")
        # LogLocator's default minor ticks (2,3,4,5,6,7,8,9 x each decade)
        # with a plain-number formatter -- without this, a dataset that
        # sits inside a single decade (e.g. 40-190) shows only ONE labeled
        # tick ("100") on the entire axis, which is what made an earlier
        # version of this chart hard to read.
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9]))
        ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax.set_xlim(right=max_p95 * 1.6)
    else:
        # Linear headroom for the value-label text so it doesn't run off
        # the right edge of the figure, worst case at the longest whisker.
        ax.set_xlim(left=0, right=max_p95 * 1.35)
        ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)

    for yi, (mean_v, p95_v) in enumerate(zip(mean_vals, p95_vals)):
        ax.text(
            max(mean_v, p95_v) * 1.03, yi, f"{mean_v:.1f} (p95 {p95_v:.1f})",
            va="center", fontsize=VALUE_LABEL_FONTSIZE,
        )

    wrapped_labels = [
        "\n".join(textwrap.wrap(name, LABEL_WRAP_WIDTH)) for name in order["display_name"]
    ]
    ax.set_yticks(list(y))
    ax.set_yticklabels(wrapped_labels, fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    scale_note = "log scale" if scale == "log" else "linear scale"
    ax.set_xlabel(
        f"Winding GIC, |A/ph| ({period}-year storm) — {scale_note}, whisker = p95",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_title(
        f"Top {len(top)} most vulnerable Virginia substations ({period}-year return period storm)",
        fontsize=TITLE_FONTSIZE,
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
        help="X-axis scale. 'linear' (default) reads more naturally and gives fully-labeled, evenly "
             "spaced ticks for free -- fine for this project's observed 100-year mean-to-p95 ratios "
             "(under 3x). Use 'log' for a period/top-N where that ratio is much larger.",
    )
    args = parser.parse_args()
    main(period=args.period, top_n=args.top_n, scale=args.scale)
