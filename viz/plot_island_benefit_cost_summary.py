"""
Combined results figure replacing the three per-island benefit-cost
tables (Tables 5/6/7) with one chart, per review feedback asking for
graphs over tables. Raw numbers stay in the Supporting Information.

Reads scripts/compute_island_benefit_cost.py's output CSVs, filtered to
the mid_365d variant (same as the main-text tables). One horizontal bar
per (scenario, island), colored by a fixed island order (A/B/C) so color
stays consistent regardless of which islands are passed on the CLI.

Plots net benefit ($M) rather than benefit-cost ratio, since the ratio
is dominated by each island's very different switching-cost denominator
and wouldn't be comparable at a glance. Cost is shown as a text
annotation rather than a bar segment, since it's 1-2 orders of magnitude
smaller than the benefits and would be invisible as a waterfall segment.

Usage:
    python viz/plot_island_benefit_cost_summary.py
    python viz/plot_island_benefit_cost_summary.py --islands a g o --variant mid_365d
"""

import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import VA_ECON_DIR, VA_FIGURES_DIR
from viz.plot_island_maps import PAPER_ISLAND_LABELS
from viz.plot_utils import savefig_with_retry

logger = setup_logger(log_file="logs/plot_island_benefit_cost_summary.log")

# fixed per island identity, not re-cycled if an island is dropped from CLI args
ISLAND_COLORS = {"A": "#E41A1C", "B": "#FF8C00", "C": "#FFD700"}

TITLE_FONTSIZE = 20
SUBTITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 15
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13

SCENARIO_ORDER = [
    "50-year", "75-year", "100-year", "125-year", "150-year", "175-year",
    "200-year", "225-year", "250-year", "gannon-year", "halloween-year",
    "st_patricks-year",
]
SCENARIO_DISPLAY = {
    "gannon-year": "Gannon (2024)",
    "halloween-year": "Halloween (2003)",
    "st_patricks-year": "St. Patrick's Day (2015)",
}


def _format_sig_figs(value: float, sig: int = 3) -> str:
    """Format to sig figs with commas, no scientific notation (`.3g` and
    fixed-decimal formats both misbehave on this project's cost range)."""
    if value == 0:
        return "0"
    digits = max(sig - int(math.floor(math.log10(abs(value)))) - 1, 0)
    return f"{round(value, digits):,.{digits}f}"


def load_island_benefit_cost(island_id: str, variant: str) -> pd.DataFrame:
    path = VA_ECON_DIR / f"va_island_{island_id}_benefit_cost.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/compute_island_benefit_cost.py --island {island_id} first."
        )
    df = pd.read_csv(path)
    df = df[df["repair_duration_variant"] == variant].copy()
    df["paper_label"] = PAPER_ISLAND_LABELS.get(island_id, island_id.upper())
    return df


def main(island_ids=("a", "g", "o"), variant: str = "mid_365d"):
    setup_matplotlib()

    all_df = pd.concat([load_island_benefit_cost(i, variant) for i in island_ids], ignore_index=True)
    labels_present = [PAPER_ISLAND_LABELS.get(i, i.upper()) for i in island_ids]

    present_scenarios = [s for s in SCENARIO_ORDER if s in all_df["scenario"].unique()]
    n_scenarios = len(present_scenarios)
    n_islands = len(labels_present)

    fig, ax = plt.subplots(figsize=(11, 0.65 * n_scenarios + 1.8))

    bar_height = 0.8 / n_islands
    y_base = np.arange(n_scenarios)

    cost_by_label = {}
    for j, label in enumerate(labels_present):
        sub = all_df[all_df["paper_label"] == label].set_index("scenario")
        y = y_base - 0.4 + bar_height * (j + 0.5)
        values = [sub.loc[s, "net_musd"] if s in sub.index else np.nan for s in present_scenarios]
        ax.barh(
            y, values, height=bar_height * 0.9, color=ISLAND_COLORS.get(label, "0.5"),
            edgecolor="black", linewidth=1.0, label=f"Island {label}", zorder=3,
        )
        # cost is fixed per island across scenarios, so one value is enough
        cost_by_label[label] = sub["total_cost_musd"].iloc[0]

    ax.axvline(0, color="0.3", linewidth=0.8, zorder=2)
    ax.set_yticks(y_base)
    ax.set_yticklabels([SCENARIO_DISPLAY.get(s, s) for s in present_scenarios], fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Net benefit ($M, 365-day repair-duration variant)",
        fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold",
    )
    # suptitle + ax title (not one title with two lines) so the cost
    # annotation doesn't overlap the main title
    fig.suptitle("Island benefit–cost by scenario", fontsize=TITLE_FONTSIZE, fontweight="bold", y=0.995)
    # "both switching events" = open + reclose around the storm window, not a single action
    cost_line = "Cost per storm (both switching events):  " + "   ".join(
        f"Island {l} - ${_format_sig_figs(cost_by_label[l])}M" for l in labels_present
    )
    ax.set_title(cost_line, fontsize=SUBTITLE_FONTSIZE, color="black", pad=10)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, framealpha=0.9)
    ax.grid(axis="x", color="0.85", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = VA_FIGURES_DIR / "va_island_benefit_cost_summary.png"
    savefig_with_retry(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined net-benefit chart for all islands, replacing per-island tables.")
    parser.add_argument("--islands", type=str, nargs="+", default=["a", "g", "o"])
    parser.add_argument("--variant", type=str, default="mid_365d", choices=["low_180d", "mid_365d", "high_730d"])
    args = parser.parse_args()
    main(island_ids=args.islands, variant=args.variant)
