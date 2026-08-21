"""
Shared per-substation risk/exposure data loaders -- GIC risk, economic
(population/GDP) exposure, critical-facility counts, topology metrics.

Extracted out of scripts/deprecated_sci_islanding/compute_virginia_
criticality.py when the SCI-driven greedy islanding strategy was
deprecated (see that folder's README.md for why). These loaders
themselves were never SCI-specific -- they just read real per-substation
data (Monte Carlo GIC output, Census/BEA population and GDP, critical-
facility locations) off disk. compute_virginia_criticality.py's own
contribution was combining them into one weighted index (SCI) and using
that index to drive greedy region growth; that combination step is what
turned out to be the flawed part (see ISLANDING_MITIGATION_PLAN.md), not
the underlying data these loaders expose. Kept active and shared so
anything -- old or new selection strategy -- can use the raw signals
without depending on deprecated code.

Usage: import as a library, e.g.
    from scripts.virginia_risk_data import load_gic_risk, load_economic_exposure
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/virginia_risk_data.log")

RANK_COL = "GIC_100_mean"  # 100yr ground-GIC magnitude, from va_ground_gic_summary.csv

# Same sector taxonomy as run_virginia_areal_interpolation.py's GROUP_KEYS.
GDP_SECTOR_KEYS = (
    "AGR", "MINING", "UTIL_CONST", "MANUF", "TRADE_TRANSP",
    "INFO", "FIRE", "PROF_OTHER", "EDUC_ENT", "G",
)


def load_topology_metrics() -> pd.DataFrame:
    """sub_id, betweenness, in_main_component, latitude, longitude -- from
    Phase 1's output."""
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", "betweenness", "in_main_component", "latitude", "longitude"]]


def load_gic_risk() -> pd.DataFrame:
    """sub_id, gic_risk -- 100yr mean |ground GIC| from
    va_ground_gic_summary.csv (scripts/extract_virginia_results.py's
    summarize_ground_gic()), the same physical quantity the fragility
    Monte Carlo uses. Not va_gic_summary.csv's winding-GIC ranking, which
    is a different per-transformer quantity."""
    path = VA_ECON_DIR / "va_ground_gic_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/extract_virginia_results.py first "
            "(needs the ground_gic*.csv files scripts/run_virginia_gic.py produces)."
        )
    df = pd.read_csv(path)
    if RANK_COL not in df.columns:
        raise ValueError(f"Expected column '{RANK_COL}' in {path}; got {list(df.columns)}")
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", RANK_COL]].rename(columns={RANK_COL: "gic_risk"})


def load_economic_exposure() -> pd.DataFrame:
    """sub_id, gdp_total, population, area_km2 -- from va_substation_exposure.gpkg.

    Returns raw totals AND each substation's Voronoi service-area size
    (km2), not pre-divided into density -- callers that need density
    (e.g. an economic-impact-per-area score) should do that division
    themselves, keeping the transformation auditable at the call site.
    """
    path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_virginia_areal_interpolation.py first.")
    gdf = gpd.read_file(path)
    gdf["sub_id"] = gdf["sub_id"].astype(str)

    gdp_cols = [f"GDP_{k}" for k in GDP_SECTOR_KEYS if f"GDP_{k}" in gdf.columns]
    if not gdp_cols:
        raise ValueError(f"No GDP_* columns found in {path}; got {list(gdf.columns)}")
    if "POP20" not in gdf.columns:
        raise ValueError(f"Expected POP20 column in {path}; got {list(gdf.columns)}")

    gdf["gdp_total"] = gdf[gdp_cols].sum(axis=1)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    zero_area = (gdf["area_km2"] <= 0).sum()
    if zero_area:
        logger.warning(
            f"{zero_area} substations have a zero-area (or invalid) service "
            "polygon; excluded here since density would be undefined for them."
        )
        gdf = gdf[gdf["area_km2"] > 0].copy()

    return gdf[["sub_id", "gdp_total", "POP20", "area_km2"]].rename(columns={"POP20": "population"})


def load_critical_facility_counts() -> pd.DataFrame | None:
    """sub_id, critical_load_count -- from Phase 2b's va_critical_facilities.csv
    (scripts/fetch_virginia_critical_loads.py), point-in-polygon joined
    against the Voronoi substation service areas already stored as geometry
    in va_substation_exposure.gpkg.

    Returns None (not an error) when the fetch script hasn't been run --
    this data source is optional; callers should fall back gracefully.
    """
    facilities_path = VA_NETWORK_DIR / "va_critical_facilities.csv"
    if not facilities_path.exists():
        logger.info(
            f"{facilities_path} not found -- run scripts/fetch_virginia_critical_loads.py "
            "first to include critical-facility counts."
        )
        return None

    facilities = pd.read_csv(facilities_path)
    if facilities.empty:
        logger.warning(f"{facilities_path} exists but has 0 rows -- skipping.")
        return None

    service_areas_path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    if not service_areas_path.exists():
        raise FileNotFoundError(f"Missing {service_areas_path}. Run scripts/run_virginia_areal_interpolation.py first.")
    service_areas = gpd.read_file(service_areas_path)[["sub_id", "geometry"]]
    service_areas["sub_id"] = service_areas["sub_id"].astype(str)

    # va_substation_exposure.gpkg's geometry is in the areal-interpolation
    # raster's projected CRS, not EPSG:4326 -- build the facility points in
    # 4326 (where the fetch script saved lat/lon) and reproject to match,
    # rather than assuming the two already agree.
    points = gpd.GeoDataFrame(
        facilities,
        geometry=gpd.points_from_xy(facilities["longitude"], facilities["latitude"]),
        crs="EPSG:4326",
    ).to_crs(service_areas.crs)

    joined = gpd.sjoin(points, service_areas, predicate="within", how="inner")
    unmatched = len(points) - len(joined)
    if unmatched:
        logger.warning(
            f"{unmatched}/{len(points)} critical facilities fell outside every "
            "substation's Voronoi service area (likely a border edge effect) "
            "and are excluded from the count."
        )

    counts = joined.groupby("sub_id").size().rename("critical_load_count").reset_index()
    logger.info(
        f"{len(joined)} critical facilities matched to {len(counts)} substations. "
        f"By category: {facilities['category'].value_counts().to_dict()}"
    )
    return counts


def normalize_min_max(series: pd.Series) -> pd.Series:
    """Scale to [0, 1]. If every value is identical, return 0.5 for all of
    them -- "no variation" should read as "this signal has nothing to say
    here," not as everyone scoring lowest on that term."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def restrict_to_virginia(topology: pd.DataFrame) -> pd.DataFrame:
    """va_gic_summary.csv and va_substation_exposure.gpkg only cover
    Virginia substations -- restrict here too, using the same lat/lon +
    boundary check pattern used elsewhere in this project, so eligibility
    matches what the other signals can actually speak to."""
    va_boundary = get_virginia_boundary()
    in_va = topology.apply(
        lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1
    )
    excluded = (~in_va).sum()
    if excluded:
        logger.info(f"Excluding {excluded} non-Virginia substations.")
    return topology[in_va].copy()
