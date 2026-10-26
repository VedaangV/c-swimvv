"""
Sector-level economic-impact figures for Virginia: direct + indirect
impact per NAICS sector, stacked horizontal bars, one panel per storm
severity. Mirrors viz/plots.py's plot_econo_naics() but plots in $M/day
instead of Bn $/day, since Virginia's totals are too small for the
national-scale units to read well.

Reads scripts/run_virginia_econ_analysis.py's output CSVs and produces
one PNG per model under VA_FIGURES_DIR.

Usage:
    python viz/plot_virginia_econ_impact.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from configs import setup_logger, setup_matplotlib
from configs.virginia_config import VA_ECON_DIR, VA_FIGURES_DIR

logger = setup_logger(log_file="logs/plot_virginia_econ_impact.log")

SCALE = 1.0
LABEL = "Impact ($M/day)"
SCENARIOS = ["100-year", "150-year", "200-year", "250-year"]

# same palette as plot_utils.ECON_PALETTES, inlined to avoid pulling in
# plot_utils' cartopy dependency just for two color dicts
ECON_PALETTES = {
    "ghosh": {"direct": "darkred", "indirect": "lightcoral", "label": "Ghosh (supply-driven)"},
    "leontief_cons": {"direct": "#0072B2", "indirect": "#56B4E9", "label": "Leontief consumption"},
    "leontief_cp": {"direct": "#E69F00", "indirect": "#FFCC80", "label": "Leontief consumption + production"},
}

RESULT_FILES = {
    "ghosh": "va_ghosh_results.csv",
    "leontief_cons": "va_leontief_consumption_results.csv",
    "leontief_cp": "va_leontief_consumption_production_results.csv",
}

SECTOR_LABELS = {
    "AGR": "Agriculture &\nForestry",
    "MINING": "Mining &\nOil/Gas Extraction",
    "UTIL_CONST": "Utilities &\nConstruction",
    "MANUF": "Manufacturing",
    "TRADE_TRANSP": "Trade &\nTransportation",
    "INFO": "Information",
    "FIRE": "Finance &\nReal Estate",
    "PROF_OTHER": "Professional &\nOther Services",
    "EDUC_ENT": "Education &\nEntertainment",
    "G": "Government",
}


def plot_econo_naics_va(econ_results: pd.DataFrame, model_type: str) -> None:
    palette = ECON_PALETTES.get(model_type, ECON_PALETTES["ghosh"])
    color_direct = palette["direct"]
    color_indirect = palette["indirect"]

    scen_last = SCENARIOS[-1]
    scenario_last_mean = econ_results[
        (econ_results.scenario == scen_last) & (econ_results.confidence == "mean")
    ].set_index("sector")
    sector_order = scenario_last_mean.total_impact.abs().sort_values(ascending=False).index
    short_labels = [SECTOR_LABELS[s] for s in sector_order]

    scenario_last_p95 = econ_results[
        (econ_results.scenario == scen_last) & (econ_results.confidence == "p95")
    ].set_index("sector")
    max_x = scenario_last_p95.total_impact.abs().max() / SCALE * 1.1
    baseline_gap = 0.02 * max_x

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    for i, scen in enumerate(SCENARIOS):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        df = econ_results[econ_results.scenario == scen]
        mean = df[df.confidence == "mean"].set_index("sector").reindex(sector_order)
        p5 = df[df.confidence == "p5"].set_index("sector").reindex(sector_order)
        p95 = df[df.confidence == "p95"].set_index("sector").reindex(sector_order)

        direct = mean.direct_shock.abs() / SCALE
        multiplier = mean.multiplier_effect.abs() / SCALE

        y = np.arange(len(sector_order))[::-1]

        ax.barh(y, direct, left=baseline_gap, color=color_direct, alpha=0.8, height=0.8, zorder=3)
        ax.barh(y, multiplier, left=direct + baseline_gap, color=color_indirect, alpha=0.8, height=0.8, zorder=3)

        ax.set_xlim(0, max_x)
        ax.set_ylim(-0.5, len(sector_order) - 0.5)
        ax.grid(True, axis="x", linewidth=0.25, color="0.9", zorder=0)
        ax.spines["left"].set_color("0.3")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["left"].set_zorder(5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        direct_mean = direct.sum()
        direct_low = p5.direct_shock.abs().sum() / SCALE
        direct_high = p95.direct_shock.abs().sum() / SCALE
        direct_err = (direct_high - direct_low) / 2

        indirect_mean = multiplier.sum()
        indirect_low = p5.multiplier_effect.abs().sum() / SCALE
        indirect_high = p95.multiplier_effect.abs().sum() / SCALE
        indirect_err = (indirect_high - indirect_low) / 2

        total_mean = mean.total_impact.abs().sum() / SCALE
        total_low = p5.total_impact.abs().sum() / SCALE
        total_high = p95.total_impact.abs().sum() / SCALE
        total_err = (total_high - total_low) / 2

        sx, sy, dy = 0.35, 0.5, 0.07
        ax.text(sx, sy, scen, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
        ax.text(sx, sy - dy, f"Direct: ${direct_mean:.1f} ± {direct_err:.1f} M", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.text(sx, sy - 2 * dy, f"Indirect: ${indirect_mean:.1f} ± {indirect_err:.1f} M", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.text(sx, sy - 3 * dy, f"Total: ${total_mean:.1f} ± {total_err:.1f} M", transform=ax.transAxes, ha="left", va="top", fontsize=9)

        ax.add_patch(Rectangle((sx, sy - 4.7 * dy), 0.04, 0.03, facecolor=color_direct, alpha=0.8, transform=ax.transAxes))
        ax.text(sx + 0.05, sy - 4.7 * dy + 0.015, "Direct Impact", transform=ax.transAxes, ha="left", va="center", fontsize=10)

        ax.add_patch(Rectangle((sx, sy - 5.7 * dy), 0.04, 0.03, facecolor=color_indirect, alpha=0.8, transform=ax.transAxes))
        ax.text(sx + 0.05, sy - 5.7 * dy + 0.015, "Indirect Impact", transform=ax.transAxes, ha="left", va="center", fontsize=10)

        ax.set_yticks(y)
        if col == 0:
            ax.set_yticklabels(short_labels, fontsize=10)
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel(LABEL, fontsize=11)
        else:
            ax.tick_params(labelbottom=False)
        ax.tick_params(labelsize=8)
        ax.set_facecolor("#F0F0F0")

    fig.patch.set_facecolor("#F0F0F0")
    fig.suptitle(f"Virginia economic impact by sector — {palette['label']}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    out_png = VA_FIGURES_DIR / f"va_econ_impact_{model_type}.png"
    out_pdf = VA_FIGURES_DIR / f"va_econ_impact_{model_type}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_png}")


def main():
    setup_matplotlib()

    for model_type, filename in RESULT_FILES.items():
        path = VA_ECON_DIR / filename
        if not path.exists():
            logger.warning(f"Missing {path}; skipping {model_type}. Run scripts/run_virginia_econ_analysis.py first.")
            continue
        econ_results = pd.read_csv(path)
        missing = [s for s in SCENARIOS if s not in econ_results["scenario"].unique()]
        if missing:
            logger.warning(f"{filename} is missing scenarios {missing}; those panels will be blank.")
        plot_econo_naics_va(econ_results, model_type)


if __name__ == "__main__":
    main()
