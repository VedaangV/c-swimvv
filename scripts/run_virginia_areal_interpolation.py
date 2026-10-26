"""
Map Virginia's ZCTA-level economic exposure (GDP/establishments/population)
onto Virginia+PJM substation service areas via dasymetric area interpolation,
so downstream fragility/IO modeling can attribute exposure to a substation's
service area.

Reuses econ.preprocess.p_econ_data.create_voronoi_polygons() and
tobler.dasymetric.masked_area_interpolate() unmodified, scoped to PJM's
substations/states and Virginia's own raster window (small enough to run as
a single call, no national tile chunking needed).

Known limitation: Voronoi cells right at Virginia's border may not exactly
match the true national tessellation since non-PJM neighbors aren't included
as generator points. Small, accepted edge effect.

Run after scripts/extract_virginia_results.py (needs va_socioeconomic_data.pkl).
"""

import pickle

import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from tobler.dasymetric import masked_area_interpolate

import econ.preprocess.p_econ_data as ped
from econ.preprocess.p_areal_intp import COARSE_SINGLE, _mask_from_coarse_single
from scripts.extract_virginia_results import _load_va_substation_lookup

from configs import setup_logger
from configs.virginia_config import (
    VA_ADMITTANCE_DIR,
    VA_ECON_DIR,
    get_pjm_geodataframe,
    get_virginia_boundary,
)

logger = setup_logger(log_file="logs/run_virginia_areal_interpolation.log")

GROUP_KEYS = (
    "AGR", "MINING", "UTIL_CONST", "MANUF", "TRADE_TRANSP",
    "INFO", "FIRE", "PROF_OTHER", "EDUC_ENT", "G",
)


def build_pjm_voronoi(pjm_states):
    """Voronoi cells over every PJM substation (not just Virginia's) so
    cells near Virginia's border have correct shapes — filtered down to
    Virginia-owned cells by the caller afterward."""
    ped.processed_voronoi_dir = VA_ECON_DIR / "processed_voronoi"
    ped.processed_voronoi_dir.mkdir(parents=True, exist_ok=True)

    df_substation = pd.read_csv(VA_ADMITTANCE_DIR / "substation_info.csv")
    ehv_coordinates = dict(
        zip(df_substation["name"], zip(df_substation["longitude"], df_substation["latitude"]))
    )

    logger.info(f"Building Voronoi tessellation from {len(ehv_coordinates)} PJM substations, clipped to PJM states...")
    return ped.create_voronoi_polygons(ehv_coordinates, pjm_states)


def main():
    pjm_states = get_pjm_geodataframe()
    va_boundary = get_virginia_boundary()

    voronoi_gdf = build_pjm_voronoi(pjm_states)

    va_econ_path = VA_ECON_DIR / "va_socioeconomic_data.pkl"
    if not va_econ_path.exists():
        raise FileNotFoundError(f"Missing {va_econ_path}. Run scripts/extract_virginia_results.py first.")
    with open(va_econ_path, "rb") as f:
        _, _, _, va_zcta_business_gdf, _ = pickle.load(f)

    with rasterio.open(COARSE_SINGLE) as src:
        raster_crs = src.crs

    zcta_aea = va_zcta_business_gdf.to_crs(raster_crs)
    voronoi_aea = voronoi_gdf.to_crs(raster_crs)
    voronoi_aea["sub_id"] = voronoi_aea["sub_id"].astype(str)

    sub_lookup = _load_va_substation_lookup(va_boundary)
    va_sub_names = set(sub_lookup.loc[sub_lookup["in_virginia"], "name"])
    voronoi_aea = voronoi_aea[voronoi_aea["sub_id"].isin(va_sub_names)].copy()
    logger.info(f"Virginia substation service areas: {len(voronoi_aea)} (of {len(va_sub_names)} VA substations)")
    if voronoi_aea.empty:
        raise RuntimeError("No Virginia substations survived the Voronoi clip/filter — check sub_id matching.")

    bounds = zcta_aea.total_bounds  # (minx, miny, maxx, maxy) in raster CRS
    with rasterio.open(COARSE_SINGLE) as src:
        window = from_bounds(*bounds, transform=src.transform)
        transform = src.window_transform(window)
    bbox = {"window": window, "transform": transform, "height": int(window.height), "width": int(window.width)}

    mask, trs, crs = _mask_from_coarse_single(COARSE_SINGLE, bbox)
    if mask is None or mask.sum() == 0:
        raise RuntimeError("Empty land-cover mask over Virginia's raster window.")

    mask_path = VA_ECON_DIR / "va_land_mask.tif"
    with rasterio.open(
        mask_path, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
        count=1, dtype="uint8", crs=crs, transform=trs, nodata=0,
    ) as dst:
        dst.write(mask, 1)
    logger.info(f"Saved Virginia land-cover mask: {mask_path}")

    economic_cols = [f"GDP_{k}" for k in GROUP_KEYS] + [f"EST_{k}" for k in GROUP_KEYS]
    economic_cols = [c for c in economic_cols if c in zcta_aea.columns]
    if "POP20" in zcta_aea.columns:
        economic_cols.append("POP20")
    logger.info(f"Interpolating {len(economic_cols)} economic columns: {economic_cols}")

    zs = zcta_aea.set_index("ZCTA", drop=False)
    vs = voronoi_aea.set_index("sub_id", drop=False)

    logger.info(f"Interpolating {len(zs)} Virginia ZCTAs onto {len(vs)} Virginia substation service areas...")
    result = masked_area_interpolate(
        raster=str(mask_path),
        source_df=zs,
        target_df=vs,
        extensive_variables=economic_cols,
        intensive_variables=[],
        pixel_values=[1],
        allocate_total=True,
    )

    # Dasymetric interpolation loses some mass at raster/mask edges, so
    # rescale back to the source ZCTA totals (same fix l_prepr_data.py
    # applies per-tile nationally, just a single correction here).
    for c in economic_cols:
        target = float(zcta_aea[c].sum())
        observed = float(result[c].sum())
        if observed > 0 and target > 0:
            result[c] *= target / observed

    out_path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    result.to_file(out_path, driver="GPKG")
    logger.info(f"Saved {out_path}")
    logger.info(f"Total POP20 allocated: {result['POP20'].sum() if 'POP20' in result.columns else 'n/a'}")


if __name__ == "__main__":
    main()
