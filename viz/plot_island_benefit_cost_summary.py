"""
Single results figure replacing the three per-island benefit-cost tables
(Tables 5/6/7 in the paper) with one chart -- per Ed's review: "Graphs are
better though as they show magnitudes without needing to read each number."
The raw per-scenario numbers stay available in the paper's Supporting
Information as the original tables; this is the main-text figure.

Reads scripts/compute_island_benefit_cost.py's own output CSVs directly
(va_island_{id}_benefit_cost.csv), filtered to the mid_365d repair-duration
variant -- the same variant the paper's main-text tables report. One
horizontal diverging bar per (scenario, island), grouped by scenario,
colored by island using a fixed categorical order (A/B/C) so color always
means the same island regardless of which islands are passed on the CLI.

Plots net benefit ($M), not the benefit-cost ratio -- net benefit is in a
single consistent dollar unit across all three islands and scenarios, so
magnitudes are directly comparable at a glance; benefit-cost ratio is
dominated by each island's tiny, very different switching-cost denominator
(2 vs 2 vs 4 boundary lines), so its magnitude alone would mostly reflect
cost-side differences rather than the economic story this figure exists to
tell. Exact ratios remain in the Supporting Information tables.

Bars are net_musd, as before. Cost is reported too, but as an explicit
numeric annotation rather than encoded in bar geometry -- an earlier
version of this figure drew each bar as a two-segment waterfall (a gray
cost segment, then a colored benefit segment landing on the net), which
works only when cost and benefit are comparable in magnitude. Against
the real data they aren't: each island's one-time switching/outage cost
(total_cost_musd) is one to two orders of magnitude smaller than the
per-scenario benefits, so the cost segment was too thin to see at any
reasonable bar height -- a real scale mismatch, not a rendering bug.
Since total_cost_musd is fixed per island (the same for every scenario,
since it doesn't depend on the storm), it only needs to be stated once
per island, not redrawn on every bar -- so it's printed as a one-line
annotation instead.

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

# Okabe-Ito colorblind-safe triad, assigned by fixed island identity (A/B/C)
# -- never re-cycled if an island is dropped from the CLI args, so color
# always means the same island across any version of this figure.
ISLAND_COLORS = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}

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
    """Format to a fixed number of significant figures with commas, never
    scientific notation -- plain f"{v:.3g}" switches to "1.23e+03" above
    ~1000 and f"{v:.0f}"/f"{v:.2f}" round anything under $1M straight to
    "$0M", both of which came up across this project's real island costs
    (some well under $1M, one in the thousands)."""
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

    fig, ax = plt.subplots(figsize=(9, 0.55 * n_scenarios + 1.5))

    bar_height = 0.8 / n_islands
    y_base = np.arange(n_scenarios)

    cost_by_label = {}
    for j, label in enumerate(labels_present):
        sub = all_df[all_df["paper_label"] == label].set_index("scenario")
        y = y_base - 0.4 + bar_height * (j + 0.5)
        values = [sub.loc[s, "net_musd"] if s in sub.index else np.nan for s in present_scenarios]
        ax.barh(
            y, values, height=bar_height * 0.9, color=ISLAND_COLORS.get(label, "0.5"),
            label=f"Island {label}", zorder=3,
        )
        # total_cost_musd is fixed per island (same for every scenario),
        # so one representative value is enough -- see module docstring.
        cost_by_label[label] = sub["total_cost_musd"].iloc[0]

    ax.axvline(0, color="0.3", linewidth=0.8, zorder=2)
    ax.set_yticks(y_base)
    ax.set_yticklabels([SCENARIO_DISPLAY.get(s, s) for s in present_scenarios])
    ax.invert_yaxis()
    ax.set_xlabel("Net benefit ($M, 365-day repair-duration variant)")
    # fig.suptitle sits above the whole figure, ax.set_title right above
    # the axes -- using both (rather than one title with a second line)
    # keeps the cost annotation from overlapping the main title text.
    fig.suptitle("Island benefit–cost by scenario", fontsize=14, y=0.995)
    cost_line = "One-time cost:  " + "   ".join(
        f"Island {l} - ${_format_sig_figs(cost_by_label[l])}M" for l in labels_present
    )
    ax.set_title(cost_line, fontsize=9, color="black", pad=8)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
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
