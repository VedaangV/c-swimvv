"""Visualization module for economic analysis results."""

import os
import gc
import pickle
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

import matplotlib.gridspec as gridspec
import xarray as xr

from configs import (
    USE_ALPHA_BETA_SCENARIO,
    FIGURES_DIR,
    setup_logger,
    PROCESS_GND_FILES,
    get_data_dir,
    DATA_DIR,
    setup_matplotlib,
)
from viz.plot_utils import (
    setup_map,
    linestring_to_array,
    process_substations,
    generate_grid_and_mask,
    extract_line_coordinates,
    add_ferc_regions,
    load_all_econ_results,
    export_econ_comparison_table,
    ECON_MODELS,
    ECON_PALETTES,
)
from econ.scripts.l_prepr_data import (
    read_pickle,
    load_gic_results,
    load_and_aggregate_tiles,
    process_vulnerability_chunks,
    load_network_data,
    find_vulnerable_substations,
    load_and_process_gic_data,
)

DATA_LOC = get_data_dir(econ=True)
logger = setup_logger("visualization module")

setup_matplotlib()

warnings.filterwarnings("ignore")


def plot_socio_economic_impact(
    results_data, confidence_df, model_type="io", file_suffix=""
):
    """Plot socio-economic impact showing establishments, population, and economic losses."""
    if USE_ALPHA_BETA_SCENARIO:
        scenario_map = {
            "gic_75yr": 75,
            "gic_100yr": 100,
            "gic_125yr": 125,
            "gic_150yr": 150,
            "gic_175yr": 175,
            "gic_200yr": 200,
            "gic_225yr": 225,
            "gic_250yr": 250,
        }
    else:
        scenario_map = {
            "50-year": 50,
            "75-year": 75,
            "100-year": 100,
            "125-year": 125,
            "150-year": 150,
            "175-year": 175,
            "200-year": 200,
            "225-year": 225,
            "250-year": 250,
        }

    econ_filtered = results_data[results_data["scenario"] != "gannon-year"]
    conf_filtered = confidence_df[~confidence_df["scenario"].str.contains("gannon")]

    est_data = conf_filtered[conf_filtered["variable"] == "EST_TOTAL"].set_index(
        "scenario"
    )
    est_x = [
        scenario_map[s.replace("e_", "").replace("-hazard A/ph", "")]
        for s in est_data.index
    ]
    est_mean = est_data["mean"].astype(int) / 1000
    est_err_low = est_mean - est_data["p5"].astype(int) / 1000
    est_err_high = est_data["p95"].astype(int) / 1000 - est_mean
    est_sorted = sorted(zip(est_x, est_mean, est_err_low, est_err_high))
    est_x_sorted, est_mean_sorted, est_err_low_sorted, est_err_high_sorted = zip(
        *est_sorted
    )

    pop_data = conf_filtered[conf_filtered["variable"] == "POP_AFFECTED"].set_index(
        "scenario"
    )
    pop_x = [
        scenario_map[s.replace("e_", "").replace("-hazard A/ph", "")]
        for s in pop_data.index
    ]
    pop_mean = pop_data["mean"] / 1e6
    pop_err_low = pop_mean - pop_data["p5"] / 1e6
    pop_err_high = pop_data["p95"] / 1e6 - pop_mean
    pop_sorted = sorted(zip(pop_x, pop_mean, pop_err_low, pop_err_high))
    pop_x_sorted, pop_mean_sorted, pop_err_low_sorted, pop_err_high_sorted = zip(
        *pop_sorted
    )

    direct_summary = (
        econ_filtered.groupby(["scenario", "confidence"])["direct_shock"]
        .sum()
        .abs()
        .unstack()
    )
    direct_x = [scenario_map[s] for s in direct_summary.index]
    direct_mean = direct_summary["mean"] / 1000
    direct_err_low = direct_mean - direct_summary["p5"] / 1000
    direct_err_high = direct_summary["p95"] / 1000 - direct_mean
    direct_sorted = sorted(zip(direct_x, direct_mean, direct_err_low, direct_err_high))
    (
        direct_x_sorted,
        direct_mean_sorted,
        direct_err_low_sorted,
        direct_err_high_sorted,
    ) = zip(*direct_sorted)

    total_summary = (
        econ_filtered.groupby(["scenario", "confidence"])["total_impact"]
        .sum()
        .abs()
        .unstack()
    )
    total_x = [scenario_map[s] for s in total_summary.index]
    total_mean = total_summary["mean"] / 1000
    total_err_low = total_mean - total_summary["p5"] / 1000
    total_err_high = total_summary["p95"] / 1000 - total_mean
    total_sorted = sorted(zip(total_x, total_mean, total_err_low, total_err_high))
    total_x_sorted, total_mean_sorted, total_err_low_sorted, total_err_high_sorted = (
        zip(*total_sorted)
    )

    max_est = max(np.array(est_mean_sorted) + np.array(est_err_high_sorted))
    est_ylim = max_est * 1.05

    max_pop = max(np.array(pop_mean_sorted) + np.array(pop_err_high_sorted))
    pop_ylim = max_pop * 1.05

    max_direct = max(np.array(direct_mean_sorted) + np.array(direct_err_high_sorted))
    max_total = max(np.array(total_mean_sorted) + np.array(total_err_high_sorted))
    econ_ylim = max(max_direct, max_total) * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].errorbar(
        est_x_sorted,
        est_mean_sorted,
        yerr=[est_err_low_sorted, est_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="green",
    )
    axes[0, 0].plot(est_x_sorted, est_mean_sorted, "-", color="green", alpha=0.7)
    axes[0, 0].set_ylabel(r"Establishments (Thousands)")
    axes[0, 0].set_title("(a) Business Impact", loc="left", fontsize=11)
    axes[0, 0].set_ylim(0, est_ylim)

    axes[0, 1].errorbar(
        pop_x_sorted,
        pop_mean_sorted,
        yerr=[pop_err_low_sorted, pop_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="red",
    )
    axes[0, 1].plot(pop_x_sorted, pop_mean_sorted, "-", color="red", alpha=0.7)
    axes[0, 1].set_ylabel(r"Population (Millions)")
    axes[0, 1].set_title("(b) Population Impact", loc="left", fontsize=11)
    axes[0, 1].set_ylim(0, pop_ylim)

    axes[1, 0].errorbar(
        direct_x_sorted,
        direct_mean_sorted,
        yerr=[direct_err_low_sorted, direct_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="purple",
    )
    axes[1, 0].plot(direct_x_sorted, direct_mean_sorted, "-", color="purple", alpha=0.7)
    axes[1, 0].set_ylabel(r"Direct Impact (\$Bn/day)")
    axes[1, 0].set_xlabel("Storm Return Period (years)")
    axes[1, 0].set_title("(c) Direct Economic Loss per Day", loc="left", fontsize=11)
    axes[1, 0].set_ylim(0, econ_ylim)

    axes[1, 1].errorbar(
        total_x_sorted,
        total_mean_sorted,
        yerr=[total_err_low_sorted, total_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="blue",
    )
    axes[1, 1].plot(total_x_sorted, total_mean_sorted, "-", color="blue", alpha=0.7)
    axes[1, 1].set_ylabel(r"Total Impact (\$Bn/day)")
    axes[1, 1].set_xlabel("Storm Return Period (years)")
    axes[1, 1].set_title("(d) Total Economic Loss per Day", loc="left", fontsize=11)
    axes[1, 1].set_ylim(0, econ_ylim)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F0F0F0")

    plt.tight_layout()

    fig.patch.set_facecolor("#F0F0F0")
    fig.savefig(
        FIGURES_DIR / f"economic_impact_{model_type}_{file_suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / f"economic_impact_{model_type}_{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_vuln_trafos(vuln_data, df_lines, file_suffix=""):
    """Plot substations with failure probability controlling size and color."""
    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)

    if USE_ALPHA_BETA_SCENARIO:
        scenarios_to_plot = [
            "gic_100yr_mean_prediction",
            "gic_150yr_mean_prediction",
            "gic_200yr_mean_prediction",
            "gic_250yr_mean_prediction",
        ]
    else:
        scenarios_to_plot = [
            "e_100-year-hazard A/ph",
            "e_150-year-hazard A/ph",
            "e_200-year-hazard A/ph",
            "e_250-year-hazard A/ph",
        ]

    scenario_display_names = {
        "gic_100yr_mean_prediction": "100-year",
        "gic_150yr_mean_prediction": "150-year",
        "gic_200yr_mean_prediction": "200-year",
        "gic_250yr_mean_prediction": "250-year",
        "e_100-year-hazard A/ph": "100-year",
        "e_150-year-hazard A/ph": "150-year",
        "e_200-year-hazard A/ph": "200-year",
        "e_250-year-hazard A/ph": "250-year",
    }

    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(2, 2, hspace=0.02, wspace=0.02, figure=fig)
    axes = [
        fig.add_subplot(gs[i, j], projection=projection)
        for i in range(2)
        for j in range(2)
    ]

    voltage_levels = [161, 230, 345, 500, 765]
    voltage_colors = ["purple", "blue", "green", "orange", "maroon"]
    voltage_widths = [0.2, 0.3, 0.4, 0.5, 0.6]
    voltage_color_map = dict(zip(voltage_levels, voltage_colors))
    voltage_width_map = dict(zip(voltage_levels, voltage_widths))

    p_bins = [0, 10, 20, 40, 50, 75, 101]
    p_labels = ["<10%", "10–20%", "20–40%", "40–50%", "50–75%", ">75%"]
    cmap = mpl.colormaps["YlOrRd"].resampled(5)

    p_colors = ["#9ca3af"] + [cmap(i) for i in range(5)]

    p_sizes = [5, 50, 65, 120, 160, 210]
    p_leg_ms = [5, 6, 7, 9, 11, 12]
    color_map = dict(zip(p_labels, p_colors))
    size_map = dict(zip(p_labels, p_sizes))
    leg_ms_map = dict(zip(p_labels, p_leg_ms))

    all_coords = vuln_data.groupby("sub_id")[["latitude", "longitude"]].first()

    for i, scenario in enumerate(scenarios_to_plot):
        ax = axes[i]

        sc_df = vuln_data[vuln_data["scenario"] == scenario]
        p_by_sub = sc_df.groupby("sub_id")["mean_failure_prob"].mean() * 100.0

        extent = [-120, -75, 25, 50]
        ax = setup_map(ax, extent)

        coords = all_coords.loc[all_coords.index.intersection(p_by_sub.index)].copy()
        coords["p_pct"] = p_by_sub.reindex(coords.index)

        coords["bin"] = pd.cut(
            coords["p_pct"],
            bins=p_bins,
            labels=p_labels,
            include_lowest=True,
            right=False,
        )
        coords.loc[coords["p_pct"] >= 75, "bin"] = ">75%"

        for lbl in p_labels:
            sel = coords[coords["bin"] == lbl]
            if sel.empty:
                continue
            ax.scatter(
                sel["longitude"],
                sel["latitude"],
                s=size_map[lbl],
                facecolors=color_map[lbl],
                edgecolors="black",
                linewidths=0.3,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        line_coords = [list(geom.coords) for geom in df_lines["geometry"]]
        line_colors = [voltage_color_map[v] for v in df_lines["V"]]
        line_widths = [voltage_width_map[v] for v in df_lines["V"]]
        lc = LineCollection(
            line_coords,
            linewidths=line_widths,
            alpha=0.6,
            colors=line_colors,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax.add_collection(lc)

        scenario_name = scenario_display_names[scenario]
        ax.set_title(f"({chr(97 + i)}) {scenario_name} Storm", fontsize=10, loc="left")

    prob_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color_map[lbl],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=leg_ms_map[lbl],
            label=lbl,
        )
        for lbl in p_labels
    ]
    leg_prob = fig.legend(
        handles=prob_handles,
        loc="lower center",
        bbox_to_anchor=(0.7, 0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
        title="Probability of failure",
        title_fontsize=9,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.add_artist(leg_prob)

    volt_handles = [
        Line2D(
            [0],
            [0],
            color=voltage_color_map[v],
            linewidth=voltage_width_map[v],
            alpha=0.6,
            label=f"{v} kV",
        )
        for v in voltage_levels
    ]
    leg_tl = fig.legend(
        handles=volt_handles,
        loc="lower center",
        bbox_to_anchor=(0.3, 0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
        title="Transmission line voltages",
        title_fontsize=9,
        handletextpad=0.6,
        columnspacing=0.9,
    )
    fig.add_artist(leg_tl)

    for ax in axes:
        ax.spines["geo"].set_visible(False)
        ax.set_facecolor("#F0F0F0")
    fig.patch.set_facecolor("#F0F0F0")

    plt.tight_layout(rect=(0, 0.14, 1, 1))

    fig.savefig(
        FIGURES_DIR / f"vulnerable_trafos_{file_suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / f"vulnerable_trafos_{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_econo_naics(econ_results, model_type="ghosh", file_suffix=""):
    """Plot economic impacts by NAICS sector as stacked bars (direct + indirect).

    Colors vary by model type so the three specifications are visually
    distinguishable in the paper appendix while keeping an identical layout.
    """
    SCALE = 1_000
    LABEL = "Impact (Bn $/day)"

    palette = ECON_PALETTES.get(model_type, ECON_PALETTES["ghosh"])
    color_direct = palette["direct"]
    color_indirect = palette["indirect"]

    if USE_ALPHA_BETA_SCENARIO:
        scenarios = ["gic_100yr", "gic_150yr", "gic_200yr", "gic_250yr"]
        scen_250 = "gic_250yr"
    else:
        scenarios = ["100-year", "150-year", "200-year", "250-year"]
        scen_250 = "250-year"

    scenario_250_mean = econ_results[
        (econ_results.scenario == scen_250) & (econ_results.confidence == "mean")
    ].set_index("sector")

    sector_order = (
        scenario_250_mean.total_impact.abs().sort_values(ascending=False).index
    )

    sector_labels_full = {
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
    short_labels = [sector_labels_full[s] for s in sector_order]

    scenario_250_p5 = econ_results[
        (econ_results.scenario == scen_250) & (econ_results.confidence == "p5")
    ].set_index("sector")
    scenario_250_p95 = econ_results[
        (econ_results.scenario == scen_250) & (econ_results.confidence == "p95")
    ].set_index("sector")

    max_x = scenario_250_p95.total_impact.abs().max() / SCALE * 1.1
    baseline_gap = 0.02 * max_x

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    for i, scen in enumerate(scenarios):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        df = econ_results[econ_results.scenario == scen]
        mean = df[df.confidence == "mean"].set_index("sector").reindex(sector_order)
        p5 = df[df.confidence == "p5"].set_index("sector").reindex(sector_order)
        p95 = df[df.confidence == "p95"].set_index("sector").reindex(sector_order)

        direct = mean.direct_shock.abs() / SCALE
        multiplier = mean.multiplier_effect.abs() / SCALE

        y = np.arange(len(sector_order))[::-1]

        ax.barh(
            y,
            direct,
            left=baseline_gap,
            color=color_direct,
            alpha=0.8,
            height=0.8,
            zorder=3,
        )
        ax.barh(
            y,
            multiplier,
            left=direct + baseline_gap,
            color=color_indirect,
            alpha=0.8,
            height=0.8,
            zorder=3,
        )

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

        sx, sy, dy = 0.6, 0.5, 0.07
        ax.text(
            sx,
            sy,
            scen,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            sx,
            sy - dy,
            f"Direct: ${direct_mean:.1f} ± {direct_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            sx,
            sy - 2 * dy,
            f"Indirect: ${indirect_mean:.1f} ± {indirect_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            sx,
            sy - 3 * dy,
            f"Total: ${total_mean:.1f} ± {total_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        ax.add_patch(
            Rectangle(
                (sx, sy - 4.7 * dy),
                0.04,
                0.03,
                facecolor=color_direct,
                alpha=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(
            sx + 0.05,
            sy - 4.7 * dy + 0.015,
            "Direct Impact",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )

        ax.add_patch(
            Rectangle(
                (sx, sy - 5.7 * dy),
                0.04,
                0.03,
                facecolor=color_indirect,
                alpha=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(
            sx + 0.05,
            sy - 5.7 * dy + 0.015,
            "Indirect Impact",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )

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
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    fig.savefig(
        FIGURES_DIR / f"indirect_impact_{model_type}_{file_suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / f"indirect_impact_{model_type}_{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_econo_naics_dodged(
    econ_results,
    model_type="ghosh",
    file_suffix="",
    confidence_df=None,
    us_pop_total=328_948_581,
):
    """Plot economic impacts by NAICS sector with dodged bars for direct and
    indirect impacts.  Colors vary by model type; when confidence_df is
    provided the per-scenario affected population is added to the annotation.
    """
    SCALE = 1_000
    LABEL = "Impact (Bn $/day)"

    palette = ECON_PALETTES.get(model_type, ECON_PALETTES["ghosh"])
    color_direct = palette["direct"]
    color_indirect = palette["indirect"]

    if USE_ALPHA_BETA_SCENARIO:
        scenarios = ["gic_100yr", "gic_150yr", "gic_200yr", "gic_250yr"]
        scen_250 = "gic_250yr"
    else:
        scenarios = ["100-year", "150-year", "200-year", "250-year"]
        scen_250 = "250-year"

    scenario_250_mean = econ_results[
        (econ_results.scenario == scen_250) & (econ_results.confidence == "mean")
    ].set_index("sector")
    sector_order = (
        scenario_250_mean.total_impact.abs().sort_values(ascending=False).index
    )

    sector_labels_full = {
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
    short_labels = [sector_labels_full[s] for s in sector_order]

    scenario_250_p95 = econ_results[
        (econ_results.scenario == scen_250) & (econ_results.confidence == "p95")
    ].set_index("sector")
    max_x = scenario_250_p95.total_impact.abs().max() / SCALE * 1.1
    baseline_gap = 0.02 * max_x

    max_direct = np.max(scenario_250_p95.direct_shock.abs().max() / SCALE * 1.1)
    max_indirect = max_x - max_direct
    x_lim = np.max([max_direct, max_indirect])

    # Helper: pull (mean, p5, p95) affected population for a scenario.
    def _pop_affected(scen_name):
        if confidence_df is None:
            return None
        # Scenario names in confidence_df carry the 'e_..-hazard A/ph' wrapper
        # while econ_results already has it stripped. Try both.
        candidates = [
            scen_name,
            f"e_{scen_name}-hazard A/ph",
            f"gic_{scen_name.replace('-year', 'yr')}",
        ]
        rows = confidence_df[
            (confidence_df["scenario"].isin(candidates))
            & (confidence_df["variable"] == "POP_AFFECTED")
        ]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return {
            "mean": float(r["mean"]),
            "p5": float(r["p5"]),
            "p95": float(r["p95"]),
        }

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    bar_width = 0.35
    y_positions = np.arange(len(sector_order))[::-1]
    y_direct = y_positions + bar_width / 2
    y_indirect = y_positions - bar_width / 2

    for i, scen in enumerate(scenarios):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        df = econ_results[econ_results.scenario == scen]
        mean = df[df.confidence == "mean"].set_index("sector").reindex(sector_order)
        p5 = df[df.confidence == "p5"].set_index("sector").reindex(sector_order)
        p95 = df[df.confidence == "p95"].set_index("sector").reindex(sector_order)

        direct_mean = mean.direct_shock.abs() / SCALE
        indirect_mean = mean.multiplier_effect.abs() / SCALE

        direct_p5 = p5.direct_shock.abs() / SCALE
        direct_p95 = p95.direct_shock.abs() / SCALE
        direct_lower_err = np.abs(direct_mean - np.minimum(direct_p5, direct_p95))
        direct_upper_err = np.abs(np.maximum(direct_p5, direct_p95) - direct_mean)

        indirect_p5 = p5.multiplier_effect.abs() / SCALE
        indirect_p95 = p95.multiplier_effect.abs() / SCALE
        indirect_lower_err = np.abs(
            indirect_mean - np.minimum(indirect_p5, indirect_p95)
        )
        indirect_upper_err = np.abs(
            np.maximum(indirect_p5, indirect_p95) - indirect_mean
        )

        ax.barh(
            y_direct,
            direct_mean,
            left=baseline_gap,
            color=color_direct,
            alpha=0.8,
            height=bar_width,
            zorder=3,
            label="Direct Impact" if i == 0 else "",
        )
        ax.barh(
            y_indirect,
            indirect_mean,
            left=baseline_gap,
            color=color_indirect,
            alpha=0.8,
            height=bar_width,
            zorder=3,
            label="Indirect Impact" if i == 0 else "",
        )

        ax.errorbar(
            direct_mean + baseline_gap,
            y_direct,
            xerr=[direct_lower_err, direct_upper_err],
            fmt="none",
            color="black",
            capsize=2,
            capthick=1,
            elinewidth=1,
            zorder=4,
        )
        ax.errorbar(
            indirect_mean + baseline_gap,
            y_indirect,
            xerr=[indirect_lower_err, indirect_upper_err],
            fmt="none",
            color="black",
            capsize=2,
            capthick=1,
            elinewidth=1,
            zorder=4,
        )

        ax.set_xlim(0, x_lim)
        ax.set_ylim(-0.5, len(sector_order) - 0.5)
        ax.grid(True, axis="x", linewidth=0.25, color="0.9", zorder=0)
        ax.spines["left"].set_color("0.3")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["left"].set_zorder(5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        direct_sum_mean = direct_mean.sum()
        direct_sum_low = direct_p5.sum()
        direct_sum_high = direct_p95.sum()
        direct_err = (direct_sum_high - direct_sum_low) / 2

        indirect_sum_mean = indirect_mean.sum()
        indirect_sum_low = indirect_p5.sum()
        indirect_sum_high = indirect_p95.sum()
        indirect_err = (indirect_sum_high - indirect_sum_low) / 2

        total_mean = direct_sum_mean + indirect_sum_mean
        total_low = direct_sum_low + indirect_sum_low
        total_high = direct_sum_high + indirect_sum_high
        total_err = (total_high - total_low) / 2

        # Shift annotation block up slightly to make room for the new pop line.
        sx, sy, dy = 0.5, 0.55, 0.07
        ax.text(
            sx,
            sy,
            scen,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            sx,
            sy - dy,
            f"Direct: ${direct_sum_mean:.1f} ± {direct_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            sx,
            sy - 2 * dy,
            f"Indirect: ${indirect_sum_mean:.1f} ± {indirect_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            sx,
            sy - 3 * dy,
            f"Total: ${total_mean:.1f} ± {total_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        # Affected population (millions and percent).
        pop = _pop_affected(scen)
        if pop is not None:
            pop_mean_m = pop["mean"] / 1e6
            pop_err_m = (pop["p95"] - pop["p5"]) / 2 / 1e6
            pop_pct = pop["mean"] / us_pop_total * 100
            ax.text(
                sx,
                sy - 4 * dy,
                f"Pop: {pop_mean_m:.2f} ± {pop_err_m:.2f} M ({pop_pct:.2f}%)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
            legend_base = sy - 5.7 * dy
        else:
            legend_base = sy - 4.7 * dy

        ax.add_patch(
            Rectangle(
                (sx, legend_base),
                0.04,
                0.03,
                facecolor=color_direct,
                alpha=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(
            sx + 0.05,
            legend_base + 0.015,
            "Direct Impact",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )

        ax.add_patch(
            Rectangle(
                (sx, legend_base - dy),
                0.04,
                0.03,
                facecolor=color_indirect,
                alpha=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(
            sx + 0.05,
            legend_base - dy + 0.015,
            "Indirect Impact",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )

        ax.set_yticks(y_positions)
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
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(
        FIGURES_DIR / f"indirect_impact_{model_type}_{file_suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES_DIR / f"indirect_impact_{model_type}_{file_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def create_tl_sub_visualization(gdf_sub, tl_df):
    """Create substation and transmission line visualization."""
    logger.info("Processing substation data...")
    _gdf, substations_gdf = process_substations(gdf_sub)

    PLOT_CONFIG = {
        "categories": [
            "transmission",
            "distribution",
            "generation",
            "switching",
            "unknown",
        ],
        "colors": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#999999"],
        "markers": ["s", "o", "^", "D", "x"],
        "sizes": [14, 15, 17, 16, 13],
        "spatial_extent": [-120, -75, 25, 50],
        "figsize": (10, 7),
    }

    if tl_df is None:
        return

    coord_arrays = tl_df["geometry"].apply(linestring_to_array)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    proj_data = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["figsize"], subplot_kw={"projection": projection}
    )
    ax = setup_map(ax)

    coll = mpl.collections.LineCollection(coord_arrays)
    coll.set_linewidth(0.7)
    linecolor = "tab:orange"
    coll.set_color(linecolor)
    coll.set_alpha(0.9)
    coll.set_zorder(20)
    coll.set_transform(proj_data)
    ax.add_collection(coll)
    ax.plot([], color=linecolor, linewidth=0.7, label="Transmission Lines")

    for category, color, marker, size in zip(
        PLOT_CONFIG["categories"],
        PLOT_CONFIG["colors"],
        PLOT_CONFIG["markers"],
        PLOT_CONFIG["sizes"],
    ):
        if category in ["transmission", "generation"]:
            mask = substations_gdf["SS_TYPE_CATEGORY"] == category
            if mask.sum() > 0:
                ax.scatter(
                    substations_gdf.loc[mask, "lon"],
                    substations_gdf.loc[mask, "lat"],
                    s=size,
                    c=color,
                    marker=marker,
                    label=f"{category.title()}",
                    zorder=10,
                    transform=proj_data,
                    alpha=0.85,
                    linewidths=0.7,
                    edgecolor="black",
                )

    ax.legend(
        loc="lower left",
        fontsize=10,
        framealpha=1,
        fancybox=True,
        edgecolor="k",
        title=None,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "ehv_grid.png", dpi=300)
    fig.savefig(FIGURES_DIR / "ehv_grid.pdf", dpi=300)

    plt.show()


def _trimmed_cmap(cmap_name="viridis", cut_low=0.1):
    """Create a trimmed colormap to avoid near-black colors at low end."""
    base = mpl.colormaps[cmap_name]
    return mpl.colors.LinearSegmentedColormap.from_list(
        "Ebar", base(np.linspace(cut_low, 1.0, 256))
    )


def plot_transmission_lines(
    ax,
    line_coordinates,
    values,
    min_value,
    max_value,
    cmap="magma",
    line_width=2.5,
    alpha=0.9,
):
    """Plot transmission lines with fixed unsaturated log range."""
    if not line_coordinates:
        logger.error("No valid line segments found.")
        return None

    vals = np.ma.masked_invalid(np.abs(values).astype(float))
    if vals.count() == 0:
        logger.warning("All TL values are NaN; skipping.")
        return None

    vmin_unsat = max(10.0, float(min_value), 0.0)
    vmax_unsat = max(1000.0, float(max_value))

    base = mpl.colormaps[cmap]
    cmap_trim = mpl.colors.LinearSegmentedColormap.from_list(
        f"{cmap}_trimmed", base(np.linspace(0.15, 1.0, 256))
    )

    norm = mpl.colors.LogNorm(vmin=vmin_unsat, vmax=vmax_unsat, clip=True)

    vals_clipped = np.clip(vals.filled(np.nan), vmin_unsat, vmax_unsat)

    coll = LineCollection(
        line_coordinates,
        cmap=cmap_trim,
        norm=norm,
        linewidths=1.0,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
    )
    coll.set_array(vals_clipped)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.log10(
            np.clip(vals.filled(0.0), max(vmin_unsat, 1e-12), vmax_unsat)
        )
        log_min = np.log10(vmin_unsat)
        log_max = np.log10(vmax_unsat)
        w = (log_vals - log_min) / max(log_max - log_min, 1e-12)
    coll.set_linewidths(0.6 + line_width * w)

    ax.add_collection(coll)
    return coll


def plot_mt_sites_e_fields_contour(
    ax,
    data,
    global_min,
    global_max,
    cmap="viridis",
):
    """Plot geoelectric field mesh with SymLogNorm."""
    grid_x, grid_y, grid_z, e_fields = data

    cmapE = _trimmed_cmap(cmap)
    normE = mpl.colors.SymLogNorm(
        linthresh=max(0.03 * float(global_max), 1e-6),
        linscale=0.1,
        vmin=float(global_min),
        vmax=float(global_max),
    )

    mesh = ax.pcolormesh(
        grid_x,
        grid_y,
        grid_z,
        cmap=cmapE,
        alpha=0.7,
        norm=normE,
        shading="gouraud",
        transform=ccrs.PlateCarree(),
    )

    current_min, current_max = np.nanmin(e_fields), np.nanmax(e_fields)
    del grid_x, grid_y, grid_z
    gc.collect()
    return mesh, current_min, current_max


def create_custom_colorbar_e_field(
    ax, obj, label, current_min, current_max, title, vmin, vmax, e_field=True
):
    """Create custom colorbar with annotated ticks."""
    bbox = ax.get_position()
    cax = ax.figure.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])

    if not e_field and isinstance(getattr(obj, "norm", None), mpl.colors.LogNorm):
        cb = plt.colorbar(obj, cax=cax, label=label, orientation="vertical")
        cb.ax.set_ylabel(title, rotation=90, labelpad=1, fontsize=8)

        ticks = [10, 100, 1000]
        labels = ["1–10", "100", r"$\geq 1000$"]
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
        cb.ax.minorticks_off()
        return cb

    cb = plt.colorbar(obj, cax=cax, label=label, orientation="vertical")
    cb.ax.set_ylabel(title, rotation=90, labelpad=1, fontsize=8)

    custom_ticks = [1, 10, vmax]
    if (
        np.isfinite(current_max)
        and current_max != vmax
        and current_max not in custom_ticks
    ):
        custom_ticks.append(current_max)
    custom_ticks = sorted(set(custom_ticks))
    cb.set_ticks(custom_ticks)

    def _fmt(x):
        return f"{x:.2e}" if x < 0.01 else f"{x:.0f}"

    labels = [_fmt(t) for t in custom_ticks]
    for i, t in enumerate(custom_ticks):
        if np.isfinite(current_max) and np.isclose(t, current_max):
            labels[i] += "*"
    cb.set_ticklabels(labels)
    for lbl, t in zip(cb.ax.get_yticklabels(), custom_ticks):
        if np.isfinite(current_max) and np.isclose(t, current_max):
            lbl.set_color("red")
    cb.ax.minorticks_off()

    return cb


def carto_e_field(
    ax,
    label_titles,
    spatial_extent=[-120, -75, 25, 50],
    add_grid_regions=True,
    df_tl=None,
    df_substations=None,
    cmap="viridis_r",
    value_column=None,
    show_legend=False,
    gic_global_vals=None,
    data_e=None,
    global_min=None,
    global_max=None,
):
    """Create cartographic visualization of geoelectric data and transmission lines."""
    ax = setup_map(ax, spatial_extent)

    if add_grid_regions:
        ax = add_ferc_regions(ax)

    if df_tl is not None and value_column is not None:
        if value_column not in df_tl.columns:
            logger.error(f"Column '{value_column}' not found in df_tl")
        else:
            line_collection = plot_transmission_lines(
                ax,
                line_coordinates,
                np.abs(df_tl[value_column].values),
                global_min,
                global_max,
                cmap=cmap,
            )
            label_title = label_titles.get(value_column, value_column)
            if line_collection is not None:
                create_custom_colorbar_e_field(
                    ax,
                    line_collection,
                    current_min=np.nanmin(np.abs(df_tl[value_column].values)),
                    current_max=np.nanmax(np.abs(df_tl[value_column].values)),
                    label=label_title,
                    title=label_title,
                    vmin=global_min,
                    vmax=global_max,
                    e_field=False,
                )
                del line_collection
            else:
                logger.error("Failed to create line collection.")
            logger.info("Line collection added to the axis.")

    if data_e is not None:
        mesh, current_min, current_max = plot_mt_sites_e_fields_contour(
            ax, data_e, global_min=global_min, global_max=global_max, cmap=cmap
        )
        create_custom_colorbar_e_field(
            ax,
            mesh,
            current_min=current_min,
            current_max=current_max,
            label="E Field (V/km)",
            title="E Field (V/km)",
            vmin=global_min,
            vmax=global_max,
            e_field=True,
        )
        del mesh, global_min, global_max

    return ax


def create_hazard_maps(e_fields, gannon_e, mt_coords, df_lines):
    """Create hazard maps for geoelectrics and transmission line voltages."""
    global line_coordinates, valid_indices

    viz_data_path = DATA_LOC / "viz_data"
    viz_data_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating grid and mask for plotting...")

    grid_e_75_path = viz_data_path / "grid_e_75.pkl"
    grid_e_100_path = viz_data_path / "grid_e_100.pkl"
    grid_e_150_path = viz_data_path / "grid_e_150.pkl"
    grid_e_200_path = viz_data_path / "grid_e_200.pkl"
    grid_e_250_path = viz_data_path / "grid_e_250.pkl"
    grid_e_gannon_path = viz_data_path / "grid_e_gannon.pkl"

    grid_file_paths = [
        grid_e_75_path,
        grid_e_100_path,
        grid_e_150_path,
        grid_e_200_path,
        grid_e_250_path,
        grid_e_gannon_path,
    ]

    e_field_75 = e_fields[75]
    e_field_100 = e_fields[100]
    e_field_150 = e_fields[150]
    e_field_200 = e_fields[200]
    e_field_250 = e_fields[250]
    e_field_gannon = gannon_e

    e_fields_period = [
        e_field_75,
        e_field_100,
        e_field_150,
        e_field_200,
        e_field_250,
        e_field_gannon,
    ]

    for grid_filename, e_field in zip(grid_file_paths, e_fields_period):
        generate_grid_and_mask(
            e_field,
            mt_coords,
            resolution=(500, 1000),
            filename=grid_filename,
        )

    logger.info("Grid and mask generated and saved.")

    line_coords_file = viz_data_path / "line_coords.pkl"
    if not os.path.exists(line_coords_file):
        line_coordinates, valid_indices = extract_line_coordinates(
            df_lines, filename=line_coords_file
        )
    else:
        with open(line_coords_file, "rb") as f:
            line_coordinates, valid_indices = pickle.load(f)

    figures_path = FIGURES_DIR
    figures_path.mkdir(exist_ok=True)

    _, _, _, e_fields_75 = read_pickle(grid_e_75_path)
    _, _, _, e_fields_100 = read_pickle(grid_e_100_path)
    _, _, _, e_fields_200 = read_pickle(grid_e_200_path)
    _, _, _, e_fields_250 = read_pickle(grid_e_250_path)
    _, _, _, e_fields_gannon = read_pickle(grid_e_gannon_path)

    stacked_vals_e = np.hstack(
        [e_fields_75, e_fields_100, e_fields_200, e_fields_250, e_fields_gannon]
    )
    stacked_vals_e = np.abs(stacked_vals_e)

    max_e_field = np.nanmax(stacked_vals_e)
    min_e_field = np.nanmin(stacked_vals_e)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.01)
    data_e_100 = read_pickle(grid_e_100_path)
    ax = fig.add_subplot(gs[0], projection=projection)
    ax = carto_e_field(
        ax,
        label_titles={},
        data_e=data_e_100,
        cmap="magma_r",
        global_max=max_e_field,
        global_min=min_e_field,
    )

    plt.tight_layout()
    plt.show()
    gc.collect()
    fig.savefig(figures_path / "hazard_map_100.png", dpi=300)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.01)
    ax = fig.add_subplot(gs[0], projection=projection)

    stacked_values = np.abs(
        np.hstack(df_lines[["V_100", "V_150", "V_250", "V_gannon"]].values)
    )

    global_min_v, global_max_v = np.nanmin(stacked_values), np.nanmax(stacked_values)
    logger.info(f"Global min and maxes {global_min_v}, {global_max_v}")
    offset = 1e-10
    global_min_v = max(global_min_v, offset)

    ax = carto_e_field(
        ax,
        label_titles={"V_100": "V"},
        df_tl=df_lines,
        cmap="magma_r",
        value_column="V_100",
        add_grid_regions=True,
        global_min=global_min_v,
        global_max=global_max_v,
    )

    plt.tight_layout()
    plt.show()
    gc.collect()

    years = [100, 150, 250]
    field_names = {
        "gannon": "gannon_e",
        100: "e_field_100",
        150: "e_field_150",
        250: "e_field_250",
    }

    storm_titles = {
        "gannon": {
            "e_field": "2024 Gannon Storm",
            "v_field": "2024 Gannon Storm",
            "cmap": "magma_r",
        },
        100: {"e_field": "1/100", "v_field": "1/100", "cmap": "magma_r"},
        150: {"e_field": "1/150", "v_field": "1/150", "cmap": "magma_r"},
        250: {"e_field": "1/250", "v_field": "1/250", "cmap": "magma_r"},
    }

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.2, hspace=0.2)

    num = 0
    for i, year in enumerate(field_names.keys()):
        ax_e = fig.add_subplot(gs[i, 0], projection=projection)
        title_v = storm_titles[year]["v_field"]
        title_e = storm_titles[year]["e_field"]
        cmap = storm_titles[year]["cmap"]
        data_e = read_pickle(viz_data_path / f"grid_e_{year}.pkl")

        ax_e = carto_e_field(
            ax_e,
            label_titles={field_names[year]: "Geoelectric Field (V/km)"},
            data_e=data_e,
            cmap=cmap,
            add_grid_regions=True,
            global_max=max_e_field,
            global_min=min_e_field,
        )

        ax_tl = fig.add_subplot(gs[i, 1], projection=projection)
        v_nodal_column = f"V_{year}"

        ax_tl = carto_e_field(
            ax_tl,
            label_titles={v_nodal_column: "Voltage (V)"},
            df_tl=df_lines,
            cmap=cmap,
            value_column=v_nodal_column,
            add_grid_regions=True,
            global_min=global_min_v,
            global_max=global_max_v,
        )

        if i == 0:
            ax_e.text(
                0.0,
                1.3,
                "Maximum Geoelectric Field Amplitudes",
                transform=ax_e.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=11,
            )
            ax_tl.text(
                0.0,
                1.3,
                " Maximum Transmission Lines Voltages",
                transform=ax_tl.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=11,
            )

        ax_e.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {title_v}",
            transform=ax_e.transAxes,
            fontsize=11,
        )
        num += 1
        ax_tl.text(
            0.0,
            1.1,
            f"({chr(97 + num)}) {title_e}",
            transform=ax_tl.transAxes,
            fontsize=11,
        )
        num += 1
        del v_nodal_column, title_e, title_v, ax_e, ax_tl

    plt.tight_layout()
    plt.show()
    fig.savefig(figures_path / "hazard_maps.png", dpi=300, bbox_inches="tight")
    # Save pdf version as well for high-quality print use.
    fig.savefig(figures_path / "hazard_maps.pdf", dpi=300, bbox_inches="tight")


def create_storm_hazard_maps(
    e_fields,
    gannon_e,
    halloween_e,
    st_patricks_e,
    hydro_quebec_e,
    mt_coords,
    df_lines,
    regen_grids=True,
):
    """Create hazard maps for historical storm events."""
    global line_coordinates, valid_indices

    viz_data_path = DATA_LOC / "viz_data"
    viz_data_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating event grids and masks...")

    grid_e_hq_path = viz_data_path / "grid_e_hydro_quebec.pkl"
    grid_e_halloween_path = viz_data_path / "grid_e_halloween.pkl"
    grid_e_stp_path = viz_data_path / "grid_e_st_patricks.pkl"
    grid_e_gannon_path = viz_data_path / "grid_e_gannon.pkl"

    grid_file_paths = [
        grid_e_hq_path,
        grid_e_halloween_path,
        grid_e_stp_path,
        grid_e_gannon_path,
    ]

    e_fields_events = [
        hydro_quebec_e,
        halloween_e,
        st_patricks_e,
        gannon_e,
    ]

    if regen_grids:
        for p in grid_file_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as _e:
                logger.warning(f"Could not remove {p}: {_e}")

    for grid_filename, e_field in zip(grid_file_paths, e_fields_events):
        if not os.path.exists(grid_filename):
            generate_grid_and_mask(
                e_field,
                mt_coords,
                resolution=(500, 1000),
                filename=grid_filename,
            )

    logger.info("Event grids ready.")

    line_coords_file = viz_data_path / "line_coords.pkl"
    if not os.path.exists(line_coords_file):
        line_coordinates, valid_indices = extract_line_coordinates(
            df_lines, filename=line_coords_file
        )
    else:
        with open(line_coords_file, "rb") as f:
            line_coordinates, valid_indices = pickle.load(f)

    figures_path = FIGURES_DIR
    figures_path.mkdir(exist_ok=True)

    _, _, grid_hq, e_hq = read_pickle(grid_e_hq_path)
    _, _, grid_hw, e_hw = read_pickle(grid_e_halloween_path)
    _, _, grid_stp, e_stp = read_pickle(grid_e_stp_path)
    _, _, grid_gannon, e_gan = read_pickle(grid_e_gannon_path)

    stacked_vals_e = np.hstack([e_hq, e_hw, e_stp, e_gan])
    stacked_vals_e = np.abs(stacked_vals_e)
    max_e_field = np.nanmax(stacked_vals_e)
    min_e_field = np.nanmin(stacked_vals_e)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)

    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.01)
    ax = fig.add_subplot(gs[0], projection=projection)
    data_e_gannon = read_pickle(grid_e_gannon_path)
    ax = carto_e_field(
        ax,
        label_titles={},
        data_e=data_e_gannon,
        cmap="magma_r",
        global_max=max_e_field,
        global_min=min_e_field,
    )
    plt.tight_layout()
    plt.show()
    gc.collect()
    fig.savefig(figures_path / "hazard_map_gannon.png", dpi=300)

    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.01)
    ax = fig.add_subplot(gs[0], projection=projection)

    v_cols = ["V_hydro_quebec", "V_halloween", "V_st_patricks", "V_gannon"]
    stacked_values = np.abs(np.hstack(df_lines[v_cols].values))
    global_min_v, global_max_v = np.nanmin(stacked_values), np.nanmax(stacked_values)
    logger.info(f"Global min/max V: {global_min_v}, {global_max_v}")
    offset = 1e-10
    global_min_v = max(global_min_v, offset)

    ax = carto_e_field(
        ax,
        label_titles={"V_gannon": "V"},
        df_tl=df_lines,
        cmap="magma_r",
        value_column="V_gannon",
        add_grid_regions=True,
        global_min=global_min_v,
        global_max=global_max_v,
    )
    plt.tight_layout()
    plt.show()
    gc.collect()
    fig.savefig(figures_path / "hazard_map_gannon_voltages.png", dpi=300)

    storm_keys = ["hydro_quebec", "halloween", "st_patricks", "gannon"]
    field_names = {
        "hydro_quebec": "hydro_quebec_e",
        "halloween": "halloween_e",
        "st_patricks": "st_patricks_e",
        "gannon": "gannon_e",
    }
    storm_titles = {
        "hydro_quebec": {
            "e_field": "1989 March (Hydro-Québec)",
            "v_field": "1989 March",
            "cmap": "magma_r",
        },
        "halloween": {
            "e_field": "2003 Halloween",
            "v_field": "2003 Halloween",
            "cmap": "magma_r",
        },
        "st_patricks": {
            "e_field": "2015 St. Patrick's",
            "v_field": "2015 St. Patrick's",
            "cmap": "magma_r",
        },
        "gannon": {
            "e_field": "2024 Gannon",
            "v_field": "2024 Gannon",
            "cmap": "magma_r",
        },
    }
    grid_map = {
        "hydro_quebec": grid_e_hq_path,
        "halloween": grid_e_halloween_path,
        "st_patricks": grid_e_stp_path,
        "gannon": grid_e_gannon_path,
    }
    v_col_map = {
        "hydro_quebec": "V_hydro_quebec",
        "halloween": "V_halloween",
        "st_patricks": "V_st_patricks",
        "gannon": "V_gannon",
    }

    fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.2, hspace=0.2)
    num = 0
    for i, key in enumerate(storm_keys):
        ax_e = fig.add_subplot(gs[i, 0], projection=projection)
        title_v = storm_titles[key]["v_field"]
        title_e = storm_titles[key]["e_field"]
        cmap = storm_titles[key]["cmap"]
        data_e = read_pickle(grid_map[key])

        ax_e = carto_e_field(
            ax_e,
            label_titles={field_names[key]: "Geoelectric (V/km)"},
            data_e=data_e,
            cmap=cmap,
            add_grid_regions=True,
            global_max=max_e_field,
            global_min=min_e_field,
        )

        ax_tl = fig.add_subplot(gs[i, 1], projection=projection)
        v_nodal_column = v_col_map[key]
        ax_tl = carto_e_field(
            ax_tl,
            label_titles={v_nodal_column: "Voltage (V)"},
            df_tl=df_lines,
            cmap=cmap,
            value_column=v_nodal_column,
            add_grid_regions=True,
            global_min=global_min_v,
            global_max=global_max_v,
        )

        if i == 0:
            ax_e.text(
                0.0,
                1.3,
                "Event Geoelectric Maps",
                transform=ax_e.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=11,
            )
            ax_tl.text(
                0.0,
                1.3,
                "Event-Derived Voltages",
                transform=ax_tl.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=11,
            )

        ax_e.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {title_e}",
            transform=ax_e.transAxes,
            fontsize=11,
        )
        num += 1
        ax_tl.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {title_v}",
            transform=ax_tl.transAxes,
            fontsize=11,
        )
        num += 1

        del v_nodal_column, title_e, title_v, ax_e, ax_tl

    plt.tight_layout()
    plt.show()
    fig.savefig(figures_path / "hazard_maps_events.png", dpi=300, bbox_inches="tight")
    # save pdf version as well
    fig.savefig(figures_path / "hazard_maps_events.pdf", dpi=300, bbox_inches="tight")


def create_B_E_maps(
    e_fields,
    b_fields,
    hydro_quebec_e,
    hydro_quebec_b,
    halloween_e,
    halloween_b,
    st_patricks_e,
    st_patricks_b,
    gannon_e,
    gannon_b,
    mt_coords,
    df_lines,
    mode="events",
    regen_grids=False,
):
    """Create side-by-side geomagnetic (B) and geoelectric (E) field maps."""
    viz_data_path = DATA_LOC / "viz_data"
    viz_data_path.mkdir(parents=True, exist_ok=True)
    figures_path = FIGURES_DIR
    figures_path.mkdir(exist_ok=True)

    if mode == "events":
        rows = [
            ("march89", "1989 March (Hydro-Québec)"),
            ("halloween", "2003 Halloween"),
            ("st_patricks", "2015 St. Patrick's"),
            ("gannon", "2024 Gannon"),
        ]
        E_arrays = {
            "march89": hydro_quebec_e,
            "halloween": halloween_e,
            "st_patricks": st_patricks_e,
            "gannon": gannon_e,
        }
        B_arrays = {
            "march89": hydro_quebec_b,
            "halloween": halloween_b,
            "st_patricks": st_patricks_b,
            "gannon": gannon_b,
        }
        figshape = (4, 2)
        out_png = "event_maps_BE.png"
    else:
        rows = [
            ("gannon", "2024 Gannon"),
            (100, "1/100"),
            (150, "1/150"),
            (250, "1/250"),
        ]
        E_arrays = {
            "gannon": gannon_e,
            100: e_fields[100],
            150: e_fields[150],
            250: e_fields[250],
        }
        B_arrays = {
            "gannon": gannon_b,
            100: b_fields[100],
            150: b_fields[150],
            250: b_fields[250],
        }
        figshape = (4, 2)
        out_png = "extreme_maps_BE.png"

    def _grid_path(kind, key):
        return viz_data_path / f"grid_{kind}_{key}.pkl"

    for key, _ in rows:
        ge = _grid_path("e", key)
        gb = _grid_path("b", key)
        if regen_grids or (not ge.exists()):
            generate_grid_and_mask(
                E_arrays[key], mt_coords, resolution=(500, 1000), filename=ge
            )
        if regen_grids or (not gb.exists()):
            generate_grid_and_mask(
                B_arrays[key], mt_coords, resolution=(500, 1000), filename=gb
            )

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
    gs = gridspec.GridSpec(
        figshape[0], figshape[1], figure=fig, wspace=0.2, hspace=0.25
    )

    def _vals(pth):
        _, _, z, _ = read_pickle(pth)
        a = np.asarray(z)
        if np.ma.isMaskedArray(z):
            a = z.compressed()
        else:
            a = a.ravel()
        return np.abs(a)

    E_stack = np.hstack([_vals(_grid_path("e", key)) for key, _ in rows])
    B_stack = np.hstack([_vals(_grid_path("b", key)) for key, _ in rows])

    E_min, E_max = float(np.nanmin(E_stack)), float(np.nanmax(E_stack))
    B_min, B_max = float(np.nanmin(B_stack)), float(np.nanmax(B_stack))

    if not np.isfinite(B_min) or not np.isfinite(B_max) or B_min >= B_max:
        B_min = float(np.nanpercentile(B_stack, 1))
        B_max = float(np.nanpercentile(B_stack, 99))

    for i, (key, title_str) in enumerate(rows):
        gx, gy, gz, _rawB = read_pickle(_grid_path("b", key))
        if np.ma.isMaskedArray(gz):
            gz_plot = np.ma.filled(gz, np.nan)
        else:
            gz_plot = gz
        bz = np.abs(gz_plot)

        valid_b = bz[np.isfinite(bz)]
        b_local_min = float(np.nanmin(valid_b)) if len(valid_b) > 0 else B_min
        b_local_max = float(np.nanmax(valid_b)) if len(valid_b) > 0 else B_max

        ax_l = fig.add_subplot(gs[i, 0], projection=projection)
        ax_l = setup_map(ax_l, [-120, -75, 25, 50])
        ax_l = add_ferc_regions(ax_l)

        meshB = ax_l.pcolormesh(
            gx,
            gy,
            bz,
            cmap="viridis_r",
            norm=mpl.colors.Normalize(vmin=B_min, vmax=B_max),
            shading="gouraud",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
        )

        ax_l.text(
            0.0,
            1.08,
            f"({chr(97+2*i)}) {title_str}",
            transform=ax_l.transAxes,
            fontsize=10,
        )

        bbox = ax_l.get_position()
        caxB = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
        cbB = plt.colorbar(meshB, cax=caxB, orientation="vertical")
        cbB.ax.set_ylabel(r"$\Delta B$ (nT)", rotation=90, labelpad=1, fontsize=8)

        ticksB = [B_min, B_max]
        if not any(np.isclose(b_local_max, t) for t in ticksB):
            ticksB.append(b_local_max)

        mid_ticks = np.linspace(B_min, B_max, 4)[1:-1]
        for mt in mid_ticks:
            if not any(np.isclose(mt, t) for t in ticksB):
                ticksB.append(mt)

        ticksB = sorted(set(ticksB))
        cbB.set_ticks(ticksB)

        labelsB = []
        for t in ticksB:
            if np.isclose(t, b_local_max):
                labelsB.append(f"{t:.0f}*")
            else:
                labelsB.append(f"{t:.0f}")

        cbB.set_ticklabels(labelsB)

        for lbl, t in zip(cbB.ax.get_yticklabels(), ticksB):
            if np.isclose(t, b_local_max):
                lbl.set_color("red")

        cbB.ax.minorticks_off()

        data_e = read_pickle(_grid_path("e", key))
        ax_r = fig.add_subplot(gs[i, 1], projection=projection)
        ax_r = carto_e_field(
            ax_r,
            label_titles={"E": r"$E$ (V/km)"},
            data_e=data_e,
            cmap="magma_r",
            add_grid_regions=True,
            global_min=E_min,
            global_max=E_max,
        )
        ax_r.text(
            0.0,
            1.08,
            f"({chr(97+2*i+1)}) {title_str}",
            transform=ax_r.transAxes,
            fontsize=10,
        )

        if i == 0:
            ax_l.text(
                0.0,
                1.28,
                "Geomagnetic Maps",
                transform=ax_l.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=11,
            )
            ax_r.text(
                0.0,
                1.28,
                "Geoelectric Maps",
                transform=ax_r.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                fontsize=11,
            )

    plt.tight_layout()
    fig.savefig(figures_path / out_png, dpi=300, bbox_inches="tight")
    # save as pdf
    fig.savefig(figures_path / "hazard_maps_ratios.pdf", dpi=300, bbox_inches="tight")
    plt.show()


def create_ratio_hazard_maps(
    e_fields, gannon_e, mt_coords, df_lines, regen_grids=False
):
    """Create hazard maps showing baseline and ratios relative to 1/100."""
    global line_coordinates, valid_indices

    viz_data_path = DATA_LOC / "viz_data"
    viz_data_path.mkdir(parents=True, exist_ok=True)
    figures_path = FIGURES_DIR
    figures_path.mkdir(exist_ok=True)

    logger.info("Generating ratio comparison grids...")

    scenarios = {
        100: {"e": e_fields[100], "title": "1/100 (Baseline)"},
        "gannon": {"e": gannon_e, "title": "2024 Gannon / 1/100"},
        150: {"e": e_fields[150], "title": "1/150 / 1/100"},
        250: {"e": e_fields[250], "title": "1/250 / 1/100"},
    }

    grid_paths = {}
    for key in scenarios.keys():
        grid_paths[key] = viz_data_path / f"grid_e_{key}.pkl"
        if regen_grids or (not grid_paths[key].exists()):
            generate_grid_and_mask(
                scenarios[key]["e"],
                mt_coords,
                resolution=(500, 1000),
                filename=grid_paths[key],
            )

    line_coords_file = viz_data_path / "line_coords.pkl"
    if not os.path.exists(line_coords_file):
        line_coordinates, valid_indices = extract_line_coordinates(
            df_lines, filename=line_coords_file
        )
    else:
        with open(line_coords_file, "rb") as f:
            line_coordinates, valid_indices = pickle.load(f)

    logger.info("Loading grid data...")

    gx_base, gy_base, gz_base, e_base = read_pickle(grid_paths[100])
    if np.ma.isMaskedArray(gz_base):
        gz_base_filled = np.ma.filled(gz_base, np.nan)
    else:
        gz_base_filled = gz_base

    e_stack = np.abs(e_base)
    e_min, e_max = float(np.nanmin(e_stack)), float(np.nanmax(e_stack))

    v_base = np.abs(df_lines["V_100"].values)
    v_min, v_max = float(np.nanmin(v_base)), float(np.nanmax(v_base))
    v_min = max(v_min, 1e-10)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.2, hspace=0.25)

    num = 0

    ax_e = fig.add_subplot(gs[0, 0], projection=projection)
    data_e_100 = read_pickle(grid_paths[100])
    ax_e = carto_e_field(
        ax_e,
        label_titles={"E": "Geoelectric (V/km)"},
        data_e=data_e_100,
        cmap="magma_r",
        add_grid_regions=True,
        global_min=e_min,
        global_max=e_max,
    )
    ax_e.text(
        0.0,
        1.1,
        f"({chr(97+num)}) {scenarios[100]['title']}",
        transform=ax_e.transAxes,
        fontsize=11,
    )
    num += 1

    ax_v = fig.add_subplot(gs[0, 1], projection=projection)
    ax_v = carto_e_field(
        ax_v,
        label_titles={"V_100": "Voltage (V)"},
        df_tl=df_lines,
        cmap="magma_r",
        value_column="V_100",
        add_grid_regions=True,
        global_min=v_min,
        global_max=v_max,
    )
    ax_v.text(
        0.0,
        1.1,
        f"({chr(97+num)}) {scenarios[100]['title']}",
        transform=ax_v.transAxes,
        fontsize=11,
    )
    num += 1

    if num == 2:
        ax_e.text(
            0.0,
            1.3,
            "Geoelectric Comparisons",
            transform=ax_e.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
        )
        ax_v.text(
            0.0,
            1.3,
            "Voltage Comparisons",
            transform=ax_v.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
        )

    ratio_keys = ["gannon", 150, 250]
    v_cols = {"gannon": "V_gannon", 150: "V_150", 250: "V_250"}

    for i, key in enumerate(ratio_keys, start=1):
        gx, gy, gz, e_vals = read_pickle(grid_paths[key])
        if np.ma.isMaskedArray(gz):
            gz_filled = np.ma.filled(gz, np.nan)
        else:
            gz_filled = gz

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_e = np.abs(gz_filled) / np.abs(gz_base_filled)
        ratio_e = np.where(np.isfinite(ratio_e), ratio_e, np.nan)

        ratio_vals = ratio_e[np.isfinite(ratio_e)]
        if len(ratio_vals) > 0:
            ratio_min, ratio_max = np.nanmin(ratio_vals), np.nanmax(ratio_vals)
            ratio_range = max(abs(ratio_max - 1.0), abs(1.0 - ratio_min))
            vmin_ratio = 1.0 - ratio_range
            vmax_ratio = 1.0 + ratio_range
        else:
            vmin_ratio, vmax_ratio = 0.5, 1.5

        ax_e = fig.add_subplot(gs[i, 0], projection=projection)
        ax_e = setup_map(ax_e, [-120, -75, 25, 50])
        ax_e = add_ferc_regions(ax_e)

        mesh_e = ax_e.pcolormesh(
            gx,
            gy,
            ratio_e,
            cmap="RdBu_r",
            norm=mpl.colors.TwoSlopeNorm(vmin=vmin_ratio, vcenter=1.0, vmax=vmax_ratio),
            shading="gouraud",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
        )

        bbox = ax_e.get_position()
        cax_e = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
        cb_e = plt.colorbar(mesh_e, cax=cax_e, orientation="vertical")
        cb_e.ax.set_ylabel("Ratio", rotation=90, labelpad=1, fontsize=8)
        cb_e.set_ticks([vmin_ratio, 1.0, vmax_ratio])
        cb_e.set_ticklabels([f"{vmin_ratio:.2f}", "1.00", f"{vmax_ratio:.2f}"])
        cb_e.ax.minorticks_off()

        ax_e.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {scenarios[key]['title']}",
            transform=ax_e.transAxes,
            fontsize=11,
        )
        num += 1

        v_col = v_cols[key]
        v_vals = np.abs(df_lines[v_col].values)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_v = v_vals / v_base
        ratio_v = np.where(np.isfinite(ratio_v), ratio_v, np.nan)

        ratio_v_vals = ratio_v[np.isfinite(ratio_v)]
        if len(ratio_v_vals) > 0:
            ratio_v_min, ratio_v_max = np.nanmin(ratio_v_vals), np.nanmax(ratio_v_vals)
            ratio_v_range = max(abs(ratio_v_max - 1.0), abs(1.0 - ratio_v_min))
            vmin_v_ratio = 1.0 - ratio_v_range
            vmax_v_ratio = 1.0 + ratio_v_range
        else:
            vmin_v_ratio, vmax_v_ratio = 0.5, 1.5

        ax_v = fig.add_subplot(gs[i, 1], projection=projection)
        ax_v = setup_map(ax_v, [-120, -75, 25, 50])
        ax_v = add_ferc_regions(ax_v)

        norm_v = mpl.colors.TwoSlopeNorm(
            vmin=vmin_v_ratio, vcenter=1.0, vmax=vmax_v_ratio
        )
        cmap_v = mpl.colormaps["RdBu_r"]

        line_segments = []
        line_colors = []
        for idx, (coords, val) in enumerate(zip(line_coordinates, ratio_v)):
            if np.isfinite(val) and len(coords) > 0:
                line_segments.append(coords)
                line_colors.append(val)

        lc = LineCollection(
            line_segments,
            cmap=cmap_v,
            norm=norm_v,
            linewidths=1.5,
            transform=ccrs.PlateCarree(),
        )
        lc.set_array(np.array(line_colors))
        ax_v.add_collection(lc)

        bbox = ax_v.get_position()
        cax_v = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
        cb_v = plt.colorbar(lc, cax=cax_v, orientation="vertical")
        cb_v.ax.set_ylabel("Ratio", rotation=90, labelpad=1, fontsize=8)
        cb_v.set_ticks([vmin_v_ratio, 1.0, vmax_v_ratio])
        cb_v.set_ticklabels([f"{vmin_v_ratio:.2f}", "1.00", f"{vmax_v_ratio:.2f}"])
        cb_v.ax.minorticks_off()

        ax_v.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {scenarios[key]['title']}",
            transform=ax_v.transAxes,
            fontsize=11,
        )
        num += 1

    plt.tight_layout()
    fig.savefig(figures_path / "hazard_maps_ratios.png", dpi=300, bbox_inches="tight")
    # pdf version
    fig.savefig(figures_path / "hazard_maps_ratios.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    logger.info("Ratio comparison maps created successfully!")


# 2
def create_event_ratio_maps(
    e_fields,
    gannon_e,
    halloween_e,
    st_patricks_e,
    hydro_quebec_e,
    mt_coords,
    df_lines,
    regen_grids=False,
):
    """Create hazard maps showing baseline and ratios of historical events relative to 1/100."""
    global line_coordinates, valid_indices

    viz_data_path = DATA_LOC / "viz_data"
    viz_data_path.mkdir(parents=True, exist_ok=True)
    figures_path = FIGURES_DIR
    figures_path.mkdir(exist_ok=True)

    logger.info("Generating event ratio comparison grids...")

    scenarios = {
        100: {"e": e_fields[100], "title": "1/100 (Baseline)"},
        "hydro_quebec": {"e": hydro_quebec_e, "title": "1989 March / 1/100"},
        "halloween": {"e": halloween_e, "title": "2003 Halloween / 1/100"},
        "st_patricks": {"e": st_patricks_e, "title": "2015 St. Patrick's / 1/100"},
        "gannon": {"e": gannon_e, "title": "2024 Gannon / 1/100"},
    }

    grid_paths = {}
    for key in scenarios.keys():
        grid_paths[key] = viz_data_path / f"grid_e_{key}.pkl"
        if regen_grids or (not grid_paths[key].exists()):
            generate_grid_and_mask(
                scenarios[key]["e"],
                mt_coords,
                resolution=(500, 1000),
                filename=grid_paths[key],
            )

    line_coords_file = viz_data_path / "line_coords.pkl"
    if not os.path.exists(line_coords_file):
        line_coordinates, valid_indices = extract_line_coordinates(
            df_lines, filename=line_coords_file
        )
    else:
        with open(line_coords_file, "rb") as f:
            line_coordinates, valid_indices = pickle.load(f)

    logger.info("Loading grid data...")

    gx_base, gy_base, gz_base, e_base = read_pickle(grid_paths[100])
    if np.ma.isMaskedArray(gz_base):
        gz_base_filled = np.ma.filled(gz_base, np.nan)
    else:
        gz_base_filled = gz_base

    e_stack = np.abs(e_base)
    e_min, e_max = float(np.nanmin(e_stack)), float(np.nanmax(e_stack))

    v_base = np.abs(df_lines["V_100"].values)
    v_min, v_max = float(np.nanmin(v_base)), float(np.nanmax(v_base))
    v_min = max(v_min, 1e-10)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    fig = plt.figure(figsize=(8.5, 11.5), dpi=300)
    gs = gridspec.GridSpec(5, 2, figure=fig, wspace=0.2, hspace=0.25)

    num = 0

    ax_e = fig.add_subplot(gs[0, 0], projection=projection)
    data_e_100 = read_pickle(grid_paths[100])
    ax_e = carto_e_field(
        ax_e,
        label_titles={"E": "Geoelectric (V/km)"},
        data_e=data_e_100,
        cmap="magma_r",
        add_grid_regions=True,
        global_min=e_min,
        global_max=e_max,
    )
    ax_e.text(
        0.0,
        1.1,
        f"({chr(97+num)}) {scenarios[100]['title']}",
        transform=ax_e.transAxes,
        fontsize=11,
    )
    num += 1

    ax_v = fig.add_subplot(gs[0, 1], projection=projection)
    ax_v = carto_e_field(
        ax_v,
        label_titles={"V_100": "Voltage (V)"},
        df_tl=df_lines,
        cmap="magma_r",
        value_column="V_100",
        add_grid_regions=True,
        global_min=v_min,
        global_max=v_max,
    )
    ax_v.text(
        0.0,
        1.1,
        f"({chr(97+num)}) {scenarios[100]['title']}",
        transform=ax_v.transAxes,
        fontsize=11,
    )
    num += 1

    if num == 2:
        ax_e.text(
            0.0,
            1.3,
            "Geoelectric Event Comparisons",
            transform=ax_e.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
        )
        ax_v.text(
            0.0,
            1.3,
            "Voltage Event Comparisons",
            transform=ax_v.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=11,
        )

    event_keys = ["hydro_quebec", "halloween", "st_patricks", "gannon"]
    v_cols = {
        "hydro_quebec": "V_hydro_quebec",
        "halloween": "V_halloween",
        "st_patricks": "V_st_patricks",
        "gannon": "V_gannon",
    }

    for i, key in enumerate(event_keys, start=1):
        gx, gy, gz, e_vals = read_pickle(grid_paths[key])
        if np.ma.isMaskedArray(gz):
            gz_filled = np.ma.filled(gz, np.nan)
        else:
            gz_filled = gz

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_e = np.abs(gz_filled) / np.abs(gz_base_filled)
        ratio_e = np.where(np.isfinite(ratio_e), ratio_e, np.nan)

        ratio_vals = ratio_e[np.isfinite(ratio_e)]
        if len(ratio_vals) > 0:
            vmin_ratio = float(np.nanpercentile(ratio_vals, 2))
            vmax_ratio = float(np.nanpercentile(ratio_vals, 98))
            if vmin_ratio > 1.0:
                vmin_ratio = 0.95
            if vmax_ratio < 1.0:
                vmax_ratio = 1.05
        else:
            vmin_ratio, vmax_ratio = 0.5, 1.5

        ax_e = fig.add_subplot(gs[i, 0], projection=projection)
        ax_e = setup_map(ax_e, [-120, -75, 25, 50])
        ax_e = add_ferc_regions(ax_e)

        mesh_e = ax_e.pcolormesh(
            gx,
            gy,
            ratio_e,
            cmap="RdBu_r",
            norm=mpl.colors.TwoSlopeNorm(vmin=vmin_ratio, vcenter=1.0, vmax=vmax_ratio),
            shading="gouraud",
            transform=ccrs.PlateCarree(),
            alpha=0.7,
        )

        bbox = ax_e.get_position()
        cax_e = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
        cb_e = plt.colorbar(mesh_e, cax=cax_e, orientation="vertical")
        cb_e.ax.set_ylabel("Ratio", rotation=90, labelpad=1, fontsize=8)

        median_ratio = float(np.nanmedian(ratio_vals)) if len(ratio_vals) > 0 else 1.0
        ticks_e = [vmin_ratio, 1.0, median_ratio, vmax_ratio]
        ticks_e = sorted(
            list(set([t for t in ticks_e if vmin_ratio <= t <= vmax_ratio]))
        )
        cb_e.set_ticks(ticks_e)
        cb_e.set_ticklabels([f"{t:.2f}" for t in ticks_e])
        cb_e.ax.minorticks_off()

        ax_e.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {scenarios[key]['title']}",
            transform=ax_e.transAxes,
            fontsize=11,
        )
        num += 1

        v_col = v_cols[key]
        v_vals = np.abs(df_lines[v_col].values)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_v = v_vals / v_base
        ratio_v = np.where(np.isfinite(ratio_v), ratio_v, np.nan)

        ratio_v_vals = ratio_v[np.isfinite(ratio_v)]
        if len(ratio_v_vals) > 0:
            vmin_v_ratio = float(np.nanpercentile(ratio_v_vals, 2))
            vmax_v_ratio = float(np.nanpercentile(ratio_v_vals, 98))
            if vmin_v_ratio > 1.0:
                vmin_v_ratio = 0.95
            if vmax_v_ratio < 1.0:
                vmax_v_ratio = 1.05
        else:
            vmin_v_ratio, vmax_v_ratio = 0.5, 1.5

        ax_v = fig.add_subplot(gs[i, 1], projection=projection)
        ax_v = setup_map(ax_v, [-120, -75, 25, 50])
        ax_v = add_ferc_regions(ax_v)

        norm_v = mpl.colors.TwoSlopeNorm(
            vmin=vmin_v_ratio, vcenter=1.0, vmax=vmax_v_ratio
        )
        cmap_v = mpl.colormaps["RdBu_r"]

        line_segments = []
        line_colors = []
        for idx, (coords, val) in enumerate(zip(line_coordinates, ratio_v)):
            if np.isfinite(val) and len(coords) > 0:
                line_segments.append(coords)
                line_colors.append(val)

        lc = LineCollection(
            line_segments,
            cmap=cmap_v,
            norm=norm_v,
            linewidths=1.5,
            transform=ccrs.PlateCarree(),
        )
        lc.set_array(np.array(line_colors))
        ax_v.add_collection(lc)

        bbox = ax_v.get_position()
        cax_v = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
        cb_v = plt.colorbar(lc, cax=cax_v, orientation="vertical")
        cb_v.ax.set_ylabel("Ratio", rotation=90, labelpad=1, fontsize=8)

        median_v_ratio = (
            float(np.nanmedian(ratio_v_vals)) if len(ratio_v_vals) > 0 else 1.0
        )
        ticks_v = [vmin_v_ratio, 1.0, median_v_ratio, vmax_v_ratio]
        ticks_v = sorted(
            list(set([t for t in ticks_v if vmin_v_ratio <= t <= vmax_v_ratio]))
        )
        cb_v.set_ticks(ticks_v)
        cb_v.set_ticklabels([f"{t:.2f}" for t in ticks_v])
        cb_v.ax.minorticks_off()

        ax_v.text(
            0.0,
            1.1,
            f"({chr(97+num)}) {scenarios[key]['title']}",
            transform=ax_v.transAxes,
            fontsize=11,
        )
        num += 1

    plt.tight_layout()
    fig.savefig(
        figures_path / "hazard_maps_event_ratios.png", dpi=300, bbox_inches="tight"
    )
    # pdf version
    fig.savefig(
        figures_path / "hazard_maps_event_ratios.pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()

    logger.info("Event ratio comparison maps created successfully!")


def plot_gnd_gic_panels(ds, df_substations, file_suffix=""):
    """Plot mean of the ground GIC."""
    df_substations = df_substations.rename(columns={"name": "sub_id"})

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    spatial_extent = [-120, -75, 25, 50]
    scenarios = ["GIC_gannon", "GIC_50", "GIC_100", "GIC_150", "GIC_200", "GIC_250"]
    titles = {
        "GIC_gannon": "Gannon",
        "GIC_50": "50-year",
        "GIC_100": "100-year",
        "GIC_150": "150-year",
        "GIC_200": "200-year",
        "GIC_250": "250-year",
    }
    cap_val = 400.0
    alpha_pts = 0.7

    bin_edges = np.array([0, 20, 40, 60, 80, 100, 200, 400], dtype=float)
    bin_labels = ["<20", "20–40", "40–60", "60–80", "80–100", "100–200", "200–400"]
    cmap_base = mpl.colormaps["YlOrRd"].resampled(6)
    ylorrd6 = [cmap_base(i) for i in range(6)]
    colors_list = ["#9ca3af"] + ylorrd6
    cmap_disc = colors.ListedColormap(colors_list)
    norm_disc = colors.BoundaryNorm(bin_edges, ncolors=cmap_disc.N, clip=True)

    size_map = {
        "<20": 5,
        "20–40": 10,
        "40–60": 25,
        "60–80": 40,
        "80–100": 60,
        "100–200": 80,
        "200–400": 100,
    }

    name_dim = "substatopn" if "substatopn" in ds.dims else "substation"
    ds_ids = ds.coords[name_dim].astype(str).values
    id_col = "substation" if "substation" in df_substations.columns else "sub_id"
    dfc = df_substations[[id_col, "latitude", "longitude"]].copy()
    dfc[id_col] = dfc[id_col].astype(str)
    dfc = dfc.set_index(id_col).reindex(ds_ids).dropna(subset=["latitude", "longitude"])
    lons, lats = dfc["longitude"].values, dfc["latitude"].values
    valid_xy = np.isfinite(lons) & np.isfinite(lats)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.05, hspace=0.25)
    row_axes = {0: [], 1: [], 2: []}

    panel_names = {
        "GIC_gannon": "2024 Gannon storm",
        "GIC_50": "50-year storm",
        "GIC_100": "100-year storm",
        "GIC_150": "150-year storm",
        "GIC_200": "200-year storm",
        "GIC_250": "250-year storm",
    }

    for i, scen in enumerate(scenarios):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c], projection=projection)
        row_axes[r].append(ax)
        ax = setup_map(ax, spatial_extent)

        vals = np.clip(
            np.abs(ds["gic_stat"].sel(stat="mean", scenario=scen).values), 0, cap_val
        )
        mask = valid_xy & np.isfinite(vals)

        bins = np.digitize(vals[mask], bin_edges, right=False) - 1
        bins = np.clip(bins, 0, len(bin_labels) - 1)
        sizes = np.vectorize(size_map.get)(np.array(bin_labels)[bins])

        ax.scatter(
            lons[mask],
            lats[mask],
            s=sizes,
            c=vals[mask],
            cmap=cmap_disc,
            norm=norm_disc,
            alpha=alpha_pts,
            edgecolors="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.set_title(f"({chr(97 + i)}) {panel_names[scen]}", fontsize=10, loc="left")
        ax.spines["geo"].set_visible(False)
        ax.set_facecolor("#F0F0F0")

    tick_locs = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    sm = plt.cm.ScalarMappable(cmap=cmap_disc, norm=norm_disc)
    sm.set_array([])

    pad, cax_w = 0.012, 0.015
    for r in (0, 1, 2):
        ax_l, ax_r = row_axes[r]
        bb_l, bb_r = ax_l.get_position(), ax_r.get_position()
        y0, y1 = min(bb_l.y0, bb_r.y0), max(bb_l.y1, bb_r.y1)
        x_right = max(bb_l.x1, bb_r.x1)

        cax = fig.add_axes([x_right + pad, y0, cax_w, y1 - y0])
        cb = fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_ticks(tick_locs)
        cb.set_ticklabels(bin_labels)
        cb.ax.tick_params(length=0)
        cb.set_label("|Mean GIC| (A/ph)", fontsize=8)

    fig.patch.set_facecolor("#F0F0F0")
    plt.tight_layout(rect=(0.05, 0.05, 0.90, 0.95))
    fig.savefig(
        FIGURES_DIR / f"gnd_gic_panels_{file_suffix}.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        FIGURES_DIR / f"gnd_gic_panels_{file_suffix}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def create_tl_sub_visualization(gdf_sub, tl_df):
    """Create transmission line and substation visualization."""
    import matplotlib.patheffects as pe

    logger.info("Processing substation data...")
    _gdf, substations_gdf = process_substations(gdf_sub)

    SCHEME = {
        "LINE": "#00204D",
        "TX_NODE": "#56B4E9",
        "GEN_NODE": "#D55E00",
    }

    PLOT_CONFIG = {
        "categories": ["transmission", "generation"],
        "markers": {"transmission": "s", "generation": "^"},
        "sizes": {"transmission": 3, "generation": 6},
        "spatial_extent": [-120, -75, 25, 50],
        "figsize": (10, 7),
    }

    if tl_df is None:
        return

    coord_arrays = tl_df["geometry"].apply(linestring_to_array)

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)
    proj_data = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=PLOT_CONFIG["figsize"], subplot_kw={"projection": projection}
    )
    ax = setup_map(ax)
    ax.set_extent(PLOT_CONFIG["spatial_extent"], crs=proj_data)

    coll = mpl.collections.LineCollection(coord_arrays, transform=proj_data)
    coll.set_linewidth(0.75)
    coll.set_color(SCHEME["LINE"])
    coll.set_alpha(0.9)
    coll.set_zorder(3)
    ax.add_collection(coll)
    ax.plot(
        [], color=SCHEME["LINE"], linewidth=0.75, label="Transmission lines (≥161 kV)"
    )

    for category in PLOT_CONFIG["categories"]:
        mask = substations_gdf["SS_TYPE_CATEGORY"] == category
        if mask.sum() == 0:
            continue
        node_color = (
            SCHEME["TX_NODE"] if category == "transmission" else SCHEME["GEN_NODE"]
        )
        sc = ax.scatter(
            substations_gdf.loc[mask, "lon"],
            substations_gdf.loc[mask, "lat"],
            s=PLOT_CONFIG["sizes"][category],
            c=node_color,
            marker=PLOT_CONFIG["markers"][category],
            label=f"{category.capitalize()} substations",
            zorder=10 if category == "generation" else 9,
            transform=proj_data,
            alpha=0.96,
            linewidths=0.4,
        )
        sc.set_path_effects([pe.withStroke(linewidth=0.8, foreground="#1a1a1a")])

    ax.legend(
        loc="lower left",
        fontsize=9,
        framealpha=0.95,
        fancybox=True,
        edgecolor="k",
        title=None,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "ehv_grid.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "ehv_grid.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    aggregate_gdf = load_and_aggregate_tiles()

    combined_ds, combined_vuln, vuln_table = load_gic_results()

    mean_vuln_all = process_vulnerability_chunks(
        combined_vuln, chunk_size=50, max_realizations=2000
    )

    df_lines, df_substations, ss_gdf_pkl = load_network_data()

    common_vulnerable, vulnerability_matrix, target_scenarios = (
        find_vulnerable_substations(mean_vuln_all)
    )

    (
        df_lines,
        mt_coords,
        mt_names,
        e_fields,
        b_fields,
        v_fields,
        gannon_e,
        v_cols,
        halloween_e,
        st_patricks_e,
        hydro_quebec_e,
        gannon_b,
        halloween_b,
        st_patricks_b,
        hydro_quebec_b,
    ) = load_and_process_gic_data(df_lines)

    ds = xr.open_dataset(DATA_DIR / "gnd_gic_processed" / "gnd_gic_aggregated.nc")

    # Econ results: load Ghosh + both Leontief specifications
    suffix = (
        "_alpha_beta"
        if USE_ALPHA_BETA_SCENARIO
        else "_gnd_gic" if PROCESS_GND_FILES else ""
    )
    filename_suffix = suffix.lstrip("_") or "eff_gic"

    econ_results = load_all_econ_results(suffix)
    try:
        confidence_df = pd.read_csv(FIGURES_DIR / f"confidence_intervals{suffix}.csv")
    except Exception as e:
        logger.error(f"Error loading confidence intervals: {e}")
        confidence_df = None

    logger.info("Generating Visualizations - Hazard Maps")
    create_hazard_maps(e_fields, gannon_e, mt_coords, df_lines)

    logger.info("Generating Visualizations - Hazard Event Maps")
    create_storm_hazard_maps(
        e_fields,
        gannon_e,
        halloween_e,
        st_patricks_e,
        hydro_quebec_e,
        mt_coords,
        df_lines,
        regen_grids=True,
    )

    logger.info("Generating Visualizations - B and E Field Maps")
    create_B_E_maps(
        e_fields,
        b_fields,
        hydro_quebec_e,
        hydro_quebec_b,
        halloween_e,
        halloween_b,
        st_patricks_e,
        st_patricks_b,
        gannon_e,
        gannon_b,
        mt_coords,
        df_lines,
        mode="events",
        regen_grids=True,
    )

    logger.info("Generating Visualizations - B and E Field Maps for Extreme Events")
    create_B_E_maps(
        e_fields,
        b_fields,
        hydro_quebec_e,
        hydro_quebec_b,
        halloween_e,
        halloween_b,
        st_patricks_e,
        st_patricks_b,
        gannon_e,
        gannon_b,
        mt_coords,
        df_lines,
        mode="extremes",
        regen_grids=True,
    )

    create_ratio_hazard_maps(e_fields, gannon_e, mt_coords, df_lines, regen_grids=False)
    create_event_ratio_maps(
        e_fields,
        gannon_e,
        halloween_e,
        st_patricks_e,
        hydro_quebec_e,
        mt_coords,
        df_lines,
        regen_grids=False,
    )

    logger.info("Generating Visualizations - TLs and Subs")
    create_tl_sub_visualization(ss_gdf_pkl, df_lines)

    plot_vuln_trafos(mean_vuln_all, df_lines, file_suffix=filename_suffix)

    plot_gnd_gic_panels(ds, df_substations, file_suffix=filename_suffix)

    # Econ impact figs
    for mid in ("ghosh", "leontief_cons", "leontief_cp"):
        if mid in econ_results and confidence_df is not None:
            plot_econo_naics_dodged(
                econ_results[mid],
                model_type=mid,
                file_suffix=filename_suffix,
                confidence_df=confidence_df,
            )

            plot_socio_economic_impact(
                econ_results[mid],
                confidence_df,
                model_type=mid,
                file_suffix=filename_suffix,
            )

    # Comparison table across all three models
    if econ_results:
        comparison_df = export_econ_comparison_table(
            econ_results, file_suffix=f"_{filename_suffix}"
        )
        print("\nComparison table preview:")
        print(comparison_df.head(10))

    # Substations data
    voronoi_gdf = gpd.read_file(
        DATA_LOC / "processed_voronoi" / "voronoi_polygons_clipped.geojson"
    )

    # Pivot vulnerability: one row per substation, columns = scenarios
    vuln_pivot = (
        mean_vuln_all.groupby(["sub_id", "scenario"])["mean_failure_prob"]
        .mean()
        .unstack("scenario")
        .reset_index()
    )

    # Ensure sub_id is string in both dataframes
    vuln_pivot["sub_id"] = vuln_pivot["sub_id"].astype(str)
    aggregate_gdf["sub_id"] = aggregate_gdf["sub_id"].astype(str)

    # Merge: substation with pop/econ + failure probabilities
    substation_risk = aggregate_gdf.merge(vuln_pivot, on="sub_id", how="left")

    # Add coordinates from mean_vuln_all
    sub_coords = (
        mean_vuln_all.groupby("sub_id")[["latitude", "longitude"]].first().reset_index()
    )
    sub_coords["sub_id"] = sub_coords["sub_id"].astype(str)
    substation_risk = substation_risk.merge(sub_coords, on="sub_id", how="left")

    export_data = {
        "geo": {
            "transmission_lines": df_lines,
            "substations": df_substations,
            "vulnerable_substations": mean_vuln_all,  # sub_id, scenario, failure_prob
        },
        "fields": {
            "mt_coords": mt_coords,
            "mt_names": mt_names,
            "e_fields": e_fields,
            "b_fields": b_fields,
            "gannon_e": gannon_e,
            "gannon_b": gannon_b,
            "halloween_e": halloween_e,
            "halloween_b": halloween_b,
            "st_patricks_e": st_patricks_e,
            "st_patricks_b": st_patricks_b,
            "hydro_quebec_e": hydro_quebec_e,
            "hydro_quebec_b": hydro_quebec_b,
        },
        "economics": {
            "ghosh_results": econ_results.get("ghosh"),
            "leontief_cons_results": econ_results.get("leontief_cons"),
            "leontief_cp_results": econ_results.get("leontief_cp"),
            "confidence_intervals": confidence_df,
        },
        "spatial_risk": {
            "substation_risk": substation_risk,  # sub_id, POP20, GDP_*, EST_*, failure_prob_*
            "voronoi_polygons": voronoi_gdf,  # substation service areas
            "aggregate_gdf": aggregate_gdf,  # raw dasymetric allocation
        },
        "metadata": {
            "scenarios": list(vuln_pivot.columns.drop("sub_id")),
            "units": {
                "e_field": "V/km",
                "b_field": "nT",
                "voltage": "V",
                "failure_prob": "probability (0-1)",
                "population": "persons",
                "gdp": "million $/day",
            },
            "crs": "EPSG:4326",
            "notes": {
                "substation_risk": "Population/GDP allocated to substations via Voronoi dasymetric interpolation",
                "pop_at_risk": "Calculate as POP20 * failure_prob for each scenario",
            },
        },
    }

    export_dir = DATA_LOC / "wsj"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / "viz_export_data.pkl"
    with open(export_path, "wb") as f:
        pickle.dump(export_data, f)

    print(f"Exported: {export_path} ({export_path.stat().st_size / 1e6:.1f} MB)")

    EXPORT_GRIDS = True
    if EXPORT_GRIDS:
        grids = {}
        for grid_file in (DATA_LOC / "viz_data").glob("grid_*.pkl"):
            with open(grid_file, "rb") as f:
                grids[grid_file.stem] = pickle.load(f)

        grid_path = DATA_LOC / "wsj" / "viz_grids_data.pkl"
        with open(grid_path, "wb") as f:
            pickle.dump(grids, f)
        print(f"Grids: {grid_path} ({grid_path.stat().st_size / 1e9:.2f} GB)")
