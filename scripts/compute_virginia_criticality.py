"""
Phase 2 of ISLANDING_MITIGATION_PLAN.md: Substation Criticality Index (SCI).

Combines three signals, each already computed by an earlier pipeline step,
into one ranked score per substation:

    SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i) + w_t * norm(betweenness_i)

- GIC_risk_i        <- va_ground_gic_summary.csv    (scripts/extract_virginia_results.py)
- econ_exposure_i   <- va_substation_exposure.gpkg  (scripts/run_virginia_areal_interpolation.py)
- betweenness_i     <- va_topology_metrics.csv      (scripts/build_virginia_network_graph.py)

GIC risk uses *ground* GIC (current to earth through the substation's
grounding connection), not winding GIC (current through one transformer
winding) -- ground GIC is the same physical quantity
scripts/run_virginia_fragility.py's fragility Monte Carlo already uses for
the headline economic-impact results, and it's already a single per-
substation scalar rather than something assembled by combining multiple
transformers'/windings' separately-signed values. See extract_virginia_
results.py's summarize_ground_gic() for the full reasoning.

Eligibility requires both in_main_component=True (see ISLANDING_MITIGATION_
PLAN.md's Phase 1 results -- 11 Virginia substations sit outside the main
connected component and can't be meaningfully ranked or islanded against a
network they aren't part of here) AND the substation being inside Virginia.

The Virginia restriction matters because va_gic_summary.csv and
va_substation_exposure.gpkg are each already filtered to Virginia-only
substations by the scripts that produce them (extract_virginia_results.py,
run_virginia_areal_interpolation.py) -- consistent with this project's
"model on the full PJM network, report on Virginia" convention throughout.
va_topology_metrics.csv, correctly, is NOT pre-filtered (centrality needs
the full PJM network for realistic connectivity). Left un-restricted here,
Phase 2 would try to rank ~2000 non-Virginia PJM substations that were
never in scope for GIC/econ data, and score them 0 on both -- which reads
as "modeled and confirmed zero risk" rather than the true "never modeled,"
silently corrupting the ranking. Restricting to Virginia substations first
using the same lat/lon + get_virginia_boundary() check already used
elsewhere in this pipeline avoids that.

Economic exposure is GDP + population (not establishment counts) -- decided
to keep the exposure term aligned with the $/day and population-affected
figures already reported elsewhere in the paper, rather than adding
establishment counts that are highly correlated with GDP within a sector
and would partly double-weight the same underlying signal.

GDP and population are normalized *separately* before being averaged into
one exposure score, not summed first -- GDP is in dollars (tens of
millions) and population is in raw headcounts (hundreds of thousands); a
pre-normalization sum would let population numerically dominate and the
"economic exposure" term would really just be measuring population.

econ_exposure_i and critical_load_i both use DENSITY (per km2 of the
substation's Voronoi service-area polygon), not raw totals -- found to be
necessary after a real run's #1-ranked substation turned out to be a
Damascus, VA site right on the Tennessee border, sitting in a 6,747 km2
Voronoi cell (99.5th percentile; median across all 413 VA substations is
64 km2) that had absorbed 185,417 people and 35 "critical facilities" from
across the whole surrounding region purely because the cell's polygon was
enormous, not because that substation is actually near a real concentration
of exposure. This is a known, pre-existing limitation of the areal-
interpolation step (documented in run_virginia_areal_interpolation.py's own
docstring: border substations near non-PJM neighbors get poorly-bounded,
oversized Voronoi cells since out-of-state generator points aren't
included) -- SCI's ranking is what first exposed it as load-bearing, since
nothing upstream had ever ranked individual substations by these values
before. Dividing by cell area before normalizing fixes it for every
substation, not just the most extreme outlier, and better matches what SCI
is actually meant to capture for islanding purposes: how much exposure/
critical infrastructure sits close to a given substation, not how large a
polygon Voronoi happened to draw around it. Raw totals are kept alongside
the density values in this script's output for transparency, but only the
density values feed the SCI formula.

Phase 2b (optional, droppable): a fourth term, critical_load_i, counts
hospitals + wastewater treatment plants + data centers (scripts/fetch_
virginia_critical_loads.py's va_critical_facilities.csv) inside each
substation's Voronoi service area -- the same service-area polygons
va_substation_exposure.gpkg already carries for allocating GDP/population,
so this reuses the project's existing "nearest substation" geometry rather
than inventing a new radius/distance rule. Multiple facility types are
combined as a simple total count (not weighted by category) -- decided to
avoid a second, harder-to-justify weighting assumption on top of the SCI's
own term weights. If va_critical_facilities.csv hasn't been fetched yet,
this term is silently skipped and SCI falls back to the original three
terms with 1/3 weights each -- Phase 2b was designed to be fully droppable,
so its absence must never be an error.

Usage:
    python scripts/compute_virginia_criticality.py
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/compute_virginia_criticality.log")

# Default equal weighting, three-term (Phase 2) or four-term (Phase 2b, if
# va_critical_facilities.csv is present) -- compute_sci() picks the matching
# set automatically. Worth testing alternative weight sets once the pipeline
# runs end-to-end -- report how sensitive the ranking is, don't just present
# one weight choice as ground truth.
WEIGHTS_3 = {"gic_risk": 1 / 3, "econ_exposure": 1 / 3, "betweenness": 1 / 3}
WEIGHTS_4 = {"gic_risk": 1 / 4, "econ_exposure": 1 / 4, "betweenness": 1 / 4, "critical_load": 1 / 4}

RANK_COL = "GIC_100_mean"  # 100yr ground-GIC magnitude, from va_ground_gic_summary.csv

# Same sector taxonomy as run_virginia_areal_interpolation.py's GROUP_KEYS.
GDP_SECTOR_KEYS = (
    "AGR", "MINING", "UTIL_CONST", "MANUF", "TRADE_TRANSP",
    "INFO", "FIRE", "PROF_OTHER", "EDUC_ENT", "G",
)


def load_topology_metrics() -> pd.DataFrame:
    """sub_id, betweenness, in_main_component, latitude, longitude -- from
    Phase 1's output. Keeps lat/lon (unlike the version of this function in
    an earlier revision) so eligibility can be restricted to Virginia
    without re-reading substation_info.csv separately."""
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
    is a different per-transformer quantity -- see module docstring."""
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
    (km2), deliberately not pre-divided into density here -- compute_sci()
    does that division (needs area_km2 for critical_load_i's density too,
    and keeping the raw totals visible in this function's output makes the
    transformation auditable rather than hidden inside a loader). See
    module docstring for why density, not raw totals, drives the SCI score.
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
    Phase 2b is explicitly optional and compute_sci() falls back to the
    three-term SCI in that case.
    """
    facilities_path = VA_NETWORK_DIR / "va_critical_facilities.csv"
    if not facilities_path.exists():
        logger.info(
            f"{facilities_path} not found -- skipping critical_load_i (Phase 2b "
            "is optional; run scripts/fetch_virginia_critical_loads.py first to "
            "include it). Falling back to the three-term SCI."
        )
        return None

    facilities = pd.read_csv(facilities_path)
    if facilities.empty:
        logger.warning(f"{facilities_path} exists but has 0 rows -- skipping critical_load_i.")
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
            "substation's Voronoi service area (likely a border edge effect, "
            "same as run_virginia_areal_interpolation.py's documented "
            "limitation) and are excluded from the count."
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
    boundary check pattern already used in extract_virginia_results.py and
    build_virginia_network_graph.py, so eligibility for SCI ranking matches
    what the other two signals can actually speak to."""
    va_boundary = get_virginia_boundary()
    in_va = topology.apply(
        lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1
    )
    excluded = (~in_va).sum()
    if excluded:
        logger.info(f"Excluding {excluded} non-Virginia substations from SCI ranking.")
    return topology[in_va].copy()


def compute_sci(
    topology: pd.DataFrame,
    gic: pd.DataFrame,
    econ: pd.DataFrame,
    critical_load: pd.DataFrame | None = None,
) -> pd.DataFrame:
    topology = restrict_to_virginia(topology)

    eligible = topology[topology["in_main_component"]].copy()
    excluded = len(topology) - len(eligible)
    if excluded:
        logger.info(
            f"Excluding {excluded} Virginia substations outside the main connected "
            "component from SCI ranking."
        )

    merged = eligible.merge(gic, on="sub_id", how="left").merge(econ, on="sub_id", how="left")

    missing_gic = merged["gic_risk"].isna().sum()
    missing_econ = merged[["gdp_total", "population", "area_km2"]].isna().any(axis=1).sum()
    if missing_gic:
        logger.warning(
            f"{missing_gic} eligible Virginia substations have no GIC risk data -- this is "
            "now unexpected (Virginia-only scoping already applied) and worth investigating, "
            "not just noting. Check sub_id matching between va_topology_metrics.csv and "
            "va_gic_summary.csv directly."
        )
    if missing_econ:
        logger.warning(
            f"{missing_econ} eligible Virginia substations have no economic exposure data -- "
            "this is now unexpected (Virginia-only scoping already applied) and worth "
            "investigating, not just noting. Check sub_id matching with "
            "va_substation_exposure.gpkg directly."
        )
    merged["gic_risk"] = merged["gic_risk"].fillna(0)
    merged["gdp_total"] = merged["gdp_total"].fillna(0)
    merged["population"] = merged["population"].fillna(0)
    # area_km2 is deliberately NOT filled to 0 here -- dividing by a filled
    # zero would produce inf/NaN and corrupt normalize_min_max's min/max.
    # Density is computed only where area_km2 is a valid positive number
    # (NaN > 0 is False, so merge-missing rows fall through to the 0.0
    # default below, consistent with every other "missing data scores
    # zero" case in this function) and area_km2 itself is dropped from the
    # merge-missing rows' fillna so it stays visible as NaN in the output
    # for anyone auditing why a given substation scored 0 on this term.
    valid_area = merged["area_km2"] > 0

    merged["gdp_density"] = 0.0
    merged["population_density"] = 0.0
    merged.loc[valid_area, "gdp_density"] = (
        merged.loc[valid_area, "gdp_total"] / merged.loc[valid_area, "area_km2"]
    )
    merged.loc[valid_area, "population_density"] = (
        merged.loc[valid_area, "population"] / merged.loc[valid_area, "area_km2"]
    )

    merged["gic_risk_norm"] = normalize_min_max(merged["gic_risk"])
    merged["gdp_norm"] = normalize_min_max(merged["gdp_density"])
    merged["population_norm"] = normalize_min_max(merged["population_density"])
    merged["econ_exposure_norm"] = (merged["gdp_norm"] + merged["population_norm"]) / 2
    merged["betweenness_norm"] = normalize_min_max(merged["betweenness"])

    if critical_load is not None:
        merged = merged.merge(critical_load, on="sub_id", how="left")
        merged["critical_load_count"] = merged["critical_load_count"].fillna(0)
        merged["critical_load_density"] = 0.0
        merged.loc[valid_area, "critical_load_density"] = (
            merged.loc[valid_area, "critical_load_count"] / merged.loc[valid_area, "area_km2"]
        )
        merged["critical_load_norm"] = normalize_min_max(merged["critical_load_density"])
        weights = WEIGHTS_4
        merged["SCI"] = (
            weights["gic_risk"] * merged["gic_risk_norm"]
            + weights["econ_exposure"] * merged["econ_exposure_norm"]
            + weights["betweenness"] * merged["betweenness_norm"]
            + weights["critical_load"] * merged["critical_load_norm"]
        )
        logger.info(f"SCI computed with 4 terms (weights: {weights}).")
        out_cols = [
            "sub_id", "gic_risk", "area_km2", "gdp_total", "population", "gdp_density", "population_density",
            "betweenness", "critical_load_count", "critical_load_density",
            "gic_risk_norm", "econ_exposure_norm", "betweenness_norm", "critical_load_norm", "SCI",
        ]
    else:
        weights = WEIGHTS_3
        merged["SCI"] = (
            weights["gic_risk"] * merged["gic_risk_norm"]
            + weights["econ_exposure"] * merged["econ_exposure_norm"]
            + weights["betweenness"] * merged["betweenness_norm"]
        )
        logger.info(f"SCI computed with 3 terms (weights: {weights}); critical_load_i not included.")
        out_cols = [
            "sub_id", "gic_risk", "area_km2", "gdp_total", "population", "gdp_density", "population_density",
            "betweenness", "gic_risk_norm", "econ_exposure_norm", "betweenness_norm", "SCI",
        ]

    return merged[out_cols]


def main():
    topology = load_topology_metrics()
    gic = load_gic_risk()
    econ = load_economic_exposure()
    critical_load = load_critical_facility_counts()

    sci = compute_sci(topology, gic, econ, critical_load)
    sci = sci.sort_values("SCI", ascending=False).reset_index(drop=True)

    out_path = VA_ECON_DIR / "va_substation_criticality.csv"
    sci.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")
    logger.info(f"Top 10 by SCI:\n{sci.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
