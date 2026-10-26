"""
Virginia-scoped version of the national paper's Figure 3 (extreme-value
geoelectric field maps + induced transmission-line voltages by return
period) -- a (storm) x (E-field, line voltage) grid of maps, restricted
to Virginia instead of CONUS.

Reuses viz/plots.py's plotting primitives (carto_e_field() etc.)
unmodified. Virginia-specific pieces are reimplemented here rather than
editing shared code in place: generate_virginia_grid_and_mask() (masks
to Virginia's boundary, not CONUS), load_va_network_lines() (reads
Virginia's own transmission_lines.csv), and the return-period data via
scripts.est_gic.load_and_process_gic_data() pointed at Virginia's h5 file.

Note: carto_e_field() reads `line_coordinates` as a module global on
viz.plots rather than a parameter, so this script monkeypatches that
global (`vplots.line_coordinates = ...`) instead of editing the function.

Run after scripts/run_virginia_gic.py and
scripts/recompute_virginia_line_voltages.py.

Usage:
    python viz/plot_virginia_hazard_maps.py
"""

import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from scipy.interpolate import griddata

import viz.plots as vplots
from viz.plot_utils import setup_map, extract_line_coordinates, savefig_with_retry
from scripts.est_gic import load_and_process_gic_data
from configs import setup_logger, setup_matplotlib
from configs.virginia_config import VA_DATA_DIR, VA_FIGURES_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/plot_virginia_hazard_maps.log")

VA_RESULTS_PATH = "statistical_analysis/va_geomagnetic_data_return_periods.h5"
VA_VIZ_DATA_DIR = VA_DATA_DIR / "viz_data"

# same three return periods + Gannon as the reference figure (matches Figure 3's panel count)
RETURN_PERIODS_TO_PLOT = [100, 150, 250]

STORM_TITLES = {
    "gannon": "2024 Gannon Storm",
    100: "1/100",
    150: "1/150",
    250: "1/250",
}


def load_va_network_lines() -> gpd.GeoDataFrame:
    """Same pattern as econ/scripts/l_prepr_data.py's load_network_data(),
    pointed at Virginia's own admittance-matrix table."""
    tl_path = VA_DATA_DIR / "admittance_matrix" / "transmission_lines.csv"
    df_lines = pd.read_csv(tl_path)
    df_lines["geometry"] = df_lines["geometry"].apply(wkt.loads)
    df_lines = gpd.GeoDataFrame(df_lines, geometry="geometry", crs="EPSG:4326")
    df_lines["name"] = df_lines["name"].astype(str)
    return df_lines


def generate_virginia_grid_and_mask(
    e_field_values: np.ndarray,
    mt_coordinates: np.ndarray,
    boundary_polygon,
    resolution: tuple = (500, 500),
    pad_deg: float = 0.3,
    filename="grid.pkl",
    method: str = "cubic",
):
    """Virginia-scoped fork of viz/plot_utils.py's generate_grid_and_mask().
    Interpolates from the full Virginia+PJM MT-site set (helps accuracy
    near the state border) but only grids/masks the area around
    `boundary_polygon`.

    method="cubic" instead of the national version's "linear" -- linear
    interpolation over this sparser site set shows visible triangulation
    facets. Cubic can overshoot into small negative values near sharp
    gradients (not physical for a magnitude), so it's clipped to 0 below.
    Pass method="linear" for the original faceted-but-no-overshoot behavior."""
    if mt_coordinates.shape[0] != e_field_values.shape[0]:
        logger.warning("Number of MT-site points and E-field values don't match.")

    minx, miny, maxx, maxy = boundary_polygon.bounds
    lon_min, lon_max = minx - pad_deg, maxx + pad_deg
    lat_min, lat_max = miny - pad_deg, maxy + pad_deg

    grid_x, grid_y = np.mgrid[
        lon_min : lon_max : complex(0, resolution[0]),
        lat_min : lat_max : complex(0, resolution[1]),
    ]

    mask = np.array(
        [boundary_polygon.contains(Point(x, y)) for x, y in zip(grid_x.ravel(), grid_y.ravel())]
    ).reshape(grid_x.shape)

    grid_z = griddata(
        mt_coordinates[:, [1, 0]], e_field_values, (grid_x, grid_y), method=method
    )
    if method != "linear":
        grid_z = np.clip(grid_z, 0, None)
    grid_z = np.ma.array(grid_z, mask=~mask)

    with open(filename, "wb") as f:
        pickle.dump((grid_x, grid_y, grid_z, e_field_values), f)
    logger.info(f"Saved grid: {filename}")


def main():
    setup_matplotlib()
    VA_VIZ_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VA_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Virginia+PJM transmission line geometry...")
    df_lines = load_va_network_lines()

    logger.info(f"Loading Virginia+PJM geomagnetic/voltage return-period data from {VA_RESULTS_PATH}...")
    df_lines, mt_coords, mt_names, e_fields, b_fields, v_fields, gannon_e, v_cols = (
        load_and_process_gic_data(VA_DATA_DIR, df_lines, VA_RESULTS_PATH)
    )

    va_boundary = get_virginia_boundary()

    logger.info("Generating Virginia-masked E-field grids...")
    grid_paths = {}
    for period in RETURN_PERIODS_TO_PLOT + ["gannon"]:
        field_values = gannon_e if period == "gannon" else e_fields[period]
        grid_path = VA_VIZ_DATA_DIR / f"va_grid_e_{period}.pkl"
        generate_virginia_grid_and_mask(field_values, mt_coords, va_boundary, filename=grid_path)
        grid_paths[period] = grid_path

    logger.info("Extracting Virginia+PJM line coordinates...")
    line_coords_file = VA_VIZ_DATA_DIR / "va_line_coords.pkl"
    line_coordinates, valid_indices = extract_line_coordinates(df_lines, filename=line_coords_file)
    # read as a module global by carto_e_field()/plot_transmission_lines(), not a parameter
    vplots.line_coordinates = line_coordinates

    stacked_e = np.abs(np.hstack([e_fields[p] for p in RETURN_PERIODS_TO_PLOT] + [gannon_e]))
    max_e, min_e = float(np.nanmax(stacked_e)), float(np.nanmin(stacked_e))

    v_cols_to_use = [f"V_{p}" for p in RETURN_PERIODS_TO_PLOT] + ["V_gannon"]
    stacked_v = np.abs(np.hstack(df_lines[v_cols_to_use].values))
    max_v = float(np.nanmax(stacked_v))
    min_v = max(float(np.nanmin(stacked_v)), 1e-10)
    logger.info(f"Global E-field range: [{min_e:.4g}, {max_e:.4g}] V/km. Voltage range: [{min_v:.4g}, {max_v:.4g}] V.")

    minx, miny, maxx, maxy = va_boundary.bounds
    pad = 0.3
    spatial_extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]

    projection = ccrs.LambertConformal(central_longitude=-78.5, central_latitude=37.5)
    fig = plt.figure(figsize=(8.5, 10.5), dpi=300)
    # extra wspace: the E-field colorbar sits between the two columns (not
    # at an outer edge like the voltage one), so it needs room or its
    # ylabel gets overdrawn by the next column's map.
    gs = gridspec.GridSpec(4, 2, figure=fig, wspace=0.45, hspace=0.2)

    num = 0
    for i, period in enumerate(["gannon"] + RETURN_PERIODS_TO_PLOT):
        title = STORM_TITLES[period]

        ax_e = fig.add_subplot(gs[i, 0], projection=projection)
        data_e = vplots.read_pickle(grid_paths[period])
        ax_e = vplots.carto_e_field(
            ax_e, label_titles={}, spatial_extent=spatial_extent, add_grid_regions=False,
            data_e=data_e, cmap="magma_r", global_max=max_e, global_min=min_e,
        )

        ax_tl = fig.add_subplot(gs[i, 1], projection=projection)
        v_col = "V_gannon" if period == "gannon" else f"V_{period}"
        ax_tl = vplots.carto_e_field(
            ax_tl, label_titles={v_col: "Voltage (V)"}, spatial_extent=spatial_extent, add_grid_regions=False,
            df_tl=df_lines, cmap="magma_r", value_column=v_col, global_min=min_v, global_max=max_v,
        )

        if i == 0:
            ax_e.text(
                0.0, 1.3, "Maximum Geoelectric Field Amplitudes", transform=ax_e.transAxes,
                ha="left", va="top", fontweight="bold", fontsize=11,
            )
            ax_tl.text(
                0.0, 1.3, " Maximum Transmission Line Voltages", transform=ax_tl.transAxes,
                ha="left", va="top", fontweight="bold", fontsize=11,
            )

        ax_e.text(0.0, 1.1, f"({chr(97 + num)}) {title}", transform=ax_e.transAxes, fontsize=11)
        num += 1
        ax_tl.text(0.0, 1.1, f"({chr(97 + num)}) {title}", transform=ax_tl.transAxes, fontsize=11)
        num += 1

    plt.tight_layout()
    out_png = VA_FIGURES_DIR / "va_hazard_maps.png"
    out_pdf = VA_FIGURES_DIR / "va_hazard_maps.pdf"
    savefig_with_retry(fig, out_png, dpi=300, bbox_inches="tight")
    savefig_with_retry(fig, out_pdf, dpi=300, bbox_inches="tight")
    logger.info(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
