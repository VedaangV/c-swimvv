"""Visualization utilities for map setup and data processing."""

import pickle
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from scipy.interpolate import griddata

from configs import setup_logger, get_data_dir, FIGURES_DIR

logger = setup_logger("plot-utils")
DATA_LOC = get_data_dir(econ=True)

# Newly incorporated econ models
ECON_MODELS = {
    "ghosh": "ghosh_results",
    "leontief_cons": "leontief_consumption_results",
    "leontief_cp": "leontief_consumption_production_results",
}

# Colorblind-friendly palettes (Okabe-Ito), one per model
ECON_PALETTES = {
    "ghosh": {
        "direct": "darkred",
        "indirect": "lightcoral",
        "label": "Ghosh (supply-driven)",
    },
    "leontief_cons": {
        "direct": "#0072B2",  # blue
        "indirect": "#56B4E9",  # sky blue
        "label": "Leontief consumption",
    },
    "leontief_cp": {
        "direct": "#009E73",  # bluish green
        "indirect": "#8FD9B6",  # lighter green
        "label": "Leontief consumption + production",
    },
}


def export_econ_comparison_table(econ_results, file_suffix=""):
    """Export a compact comparison table of the three I-O models.

    Rows are scenarios, columns are (model, metric). Metrics are total daily
    impact at mean/p5/p95 confidence and sector-total direct/indirect split.
    Units are $ billions per day. Writes CSV suitable for latex conversion.
    """
    SCALE = 1_000  # $M -> $B
    rows = []

    for model_id, df in econ_results.items():
        for scen in df["scenario"].unique():
            mean = df[(df.scenario == scen) & (df.confidence == "mean")]
            p5 = df[(df.scenario == scen) & (df.confidence == "p5")]
            p95 = df[(df.scenario == scen) & (df.confidence == "p95")]

            rows.append(
                {
                    "model": model_id,
                    "scenario": scen,
                    "direct_mean_bn": mean.direct_shock.abs().sum() / SCALE,
                    "direct_p5_bn": p5.direct_shock.abs().sum() / SCALE,
                    "direct_p95_bn": p95.direct_shock.abs().sum() / SCALE,
                    "indirect_mean_bn": mean.multiplier_effect.abs().sum() / SCALE,
                    "indirect_p5_bn": p5.multiplier_effect.abs().sum() / SCALE,
                    "indirect_p95_bn": p95.multiplier_effect.abs().sum() / SCALE,
                    "total_mean_bn": mean.total_impact.abs().sum() / SCALE,
                    "total_p5_bn": p5.total_impact.abs().sum() / SCALE,
                    "total_p95_bn": p95.total_impact.abs().sum() / SCALE,
                }
            )

    out = pd.DataFrame(rows).round(2)

    # Sort scenarios in return-period order where possible
    scenario_order = [
        "gannon-year",
        "50-year",
        "75-year",
        "100-year",
        "125-year",
        "150-year",
        "175-year",
        "200-year",
        "225-year",
        "250-year",
    ]
    out["scen_rank"] = out["scenario"].apply(
        lambda s: scenario_order.index(s) if s in scenario_order else 999
    )
    model_order = ["ghosh", "leontief_cons", "leontief_cp"]
    out["model_rank"] = out["model"].apply(
        lambda m: model_order.index(m) if m in model_order else 999
    )
    out = out.sort_values(["scen_rank", "model_rank"]).drop(
        columns=["scen_rank", "model_rank"]
    )

    out_fp = FIGURES_DIR / f"econ_comparison_table{file_suffix}.csv"
    out.to_csv(out_fp, index=False)
    logger.info(f"Saved comparison table: {out_fp}")
    return out


def savefig_with_retry(fig, path, attempts: int = 4, delay_seconds: float = 2.0, **savefig_kwargs):
    """fig.savefig() with a few retries on Windows' OSError [Errno 22]
    Invalid argument -- observed directly against real data on this
    project's own Windows machine, on more than one Virginia figure
    script, including once right after a run that had JUST successfully
    produced and saved that same file moments before. That pattern (same
    code, same path, intermittent failure) points to a transient
    environmental race -- most likely OneDrive's sync filter or Windows
    Defender's real-time scan briefly locking the file as it's created,
    not a deterministic bug in this code or in matplotlib/Pillow -- so a
    short retry-with-delay is the right fix, not another attempt to guess
    a code-level root cause. Re-raises on the last attempt so a genuine,
    persistent failure still surfaces instead of being silently
    swallowed. Originally added in viz/plot_virginia_hazard_maps.py;
    moved here once viz/plot_virginia_winding_gic_ranking.py hit the
    identical error, so a third caller doesn't need to duplicate it
    again."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            fig.savefig(path, **savefig_kwargs)
            return
        except OSError as e:
            last_err = e
            logger.warning(f"savefig({path}) failed on attempt {attempt}/{attempts}: {e}")
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise last_err


def setup_map(ax, spatial_extent=[-120, -75, 25, 50]):
    """Setup map features and styling."""
    ax.set_extent(spatial_extent, ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey")
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor="darkgrey")
    ax.add_feature(cfeature.LAKES, alpha=0.5, linewidth=0.5, edgecolor="grey")
    ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.5)
    return ax


def linestring_to_array(geom):
    """Convert LineString geometry to coordinate array."""
    if isinstance(geom, LineString):
        return np.array(geom.coords)
    else:
        return np.array([])


SUBSTATION_MAPPING = {
    "transmission": "transmission",
    "subtransmission": "transmission",
    "distribution": "distribution",
    "minor_distribution": "distribution",
    "industrial": "distribution",
    "transformer": "distribution",
    "generation": "generation",
    "generator": "generation",
    "collector": "generation",
    "switching": "switching",
    "switchyard": "switching",
    "converter": "switching",
    "compensation": "switching",
    "transition": "switching",
    "capacitor": "switching",
    None: "unknown",
    "yes": "unknown",
    "TM": "unknown",
    "traction": "unknown",
}


def process_substations(substations_gdf):
    """Process substation data and create categorized GeoDataFrame."""
    substations_gdf["SS_TYPE_CATEGORY"] = substations_gdf["SS_TYPE"].map(
        SUBSTATION_MAPPING
    )

    df = substations_gdf[
        [
            "SS_ID",
            "SS_NAME",
            "SS_OPERATOR",
            "SS_VOLTAGE",
            "SS_TYPE",
            "REGION_ID",
            "REGION",
            "lat",
            "lon",
        ]
    ].copy()

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["lat", "lon"]),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    return gdf, substations_gdf


def load_all_econ_results(suffix):
    """Load whichever of the three econ result CSVs exist on disk."""
    out = {}
    for model_id, stem in ECON_MODELS.items():
        path = FIGURES_DIR / f"{stem}{suffix}.csv"
        if path.exists():
            out[model_id] = pd.read_csv(path)
            logger.info(f"Loaded {path.name} ({len(out[model_id])} rows)")
        else:
            logger.warning(f"Missing {path.name}")
    return out


def get_conus_polygon():
    """Retrieve the polygon representing the continental United States."""
    logger.info("Retrieving CONUS polygon...")

    try:
        import cartopy.io.shapereader as shpreader

        shapename = "admin_1_states_provinces_lakes"
        us_states = shpreader.natural_earth(
            resolution="110m", category="cultural", name=shapename
        )

        conus_states = []
        for state in shpreader.Reader(us_states).records():
            if state.attributes["admin"] == "United States of America":
                if state.attributes["name"] not in ["Alaska", "Hawaii"]:
                    conus_states.append(state.geometry)

        conus_polygon = unary_union(conus_states)
        logger.info("CONUS polygon retrieved.")
        return conus_polygon
    except Exception as e:
        logger.error(f"Failed to retrieve CONUS polygon: {e}")
        return None


def generate_grid_and_mask(
    e_fields,
    mt_coordinates,
    resolution=(500, 1000),
    filename="grid.pkl",
):
    """Generate grid and mask for interpolating E-field data."""
    if mt_coordinates.shape[0] != e_fields.shape[0]:
        logger.warning("Number of points and values don't match")

    logger.info("Generating grid and mask...")
    lon_min, lon_max = np.min(mt_coordinates[:, 1]), np.max(mt_coordinates[:, 1])
    lat_min, lat_max = np.min(mt_coordinates[:, 0]), np.max(mt_coordinates[:, 0])

    grid_x, grid_y = np.mgrid[
        lon_min : lon_max : complex(0, resolution[0]),
        lat_min : lat_max : complex(0, resolution[1]),
    ]

    conus_polygon = get_conus_polygon()

    if conus_polygon is not None:
        mask = np.array(
            [
                conus_polygon.contains(Point(x, y))
                for x, y in zip(grid_x.ravel(), grid_y.ravel())
            ]
        ).reshape(grid_x.shape)
    else:
        mask = (
            (grid_x >= lon_min)
            & (grid_x <= lon_max)
            & (grid_y >= lat_min)
            & (grid_y <= lat_max)
        )

    grid_z = griddata(
        mt_coordinates[:, [1, 0]], e_fields, (grid_x, grid_y), method="linear"
    )

    grid_z = np.ma.array(grid_z, mask=~mask)

    with open(filename, "wb") as f:
        pickle.dump((grid_x, grid_y, grid_z, e_fields), f)
        logger.info("Grid and mask saved to file.")


def extract_line_coordinates(
    df, geometry_col="geometry", source_crs=None, target_crs="EPSG:4326", filename=None
):
    """Extract line coordinates from DataFrame with geometry column."""
    logger.info("Extracting line coordinates...")
    line_coordinates = []
    valid_indices = []

    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry=geometry_col)

    if df.crs is None:
        if source_crs is None:
            raise ValueError("GeoDataFrame has no CRS and source_crs is not provided")
        df = df.set_crs(source_crs, allow_override=True)

    if df.crs.to_string() != target_crs:
        df = df.to_crs(target_crs)

    for idx, geometry in enumerate(df[geometry_col]):
        if isinstance(geometry, LineString):
            coords = np.array(geometry.coords)
            if coords.ndim == 2 and coords.shape[1] >= 2:
                line_coordinates.append(coords[:, :2])
                valid_indices.append(idx)
            else:
                logger.error(
                    f"Skipping linestring at index {idx} with unexpected shape: {coords.shape}"
                )
        else:
            logger.error(f"Invalid LineString at index {idx}: {geometry}")

    logger.info("Line coordinates extracted.")

    if filename:
        with open(filename, "wb") as f:
            pickle.dump((line_coordinates, valid_indices), f)
            logger.info("Line coordinates saved to file.")

    return line_coordinates, valid_indices


def add_ferc_regions(ax, zorder=2):
    """Add grid region boundaries to the map."""
    nerc_shp_path = (
        DATA_LOC / "raw_econ_data" / "NERC Map" / "electricity_operators.shp"
    )
    grid_regions_gdf = gpd.read_file(nerc_shp_path)

    for idx, region in grid_regions_gdf.iterrows():
        ax.add_geometries(
            [region.geometry],
            crs=ccrs.PlateCarree(),
            edgecolor="grey",
            facecolor="none",
            linewidth=0.8,
            alpha=1,
            zorder=zorder,
        )

    return ax
