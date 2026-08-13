"""
Phase 2b of ISLANDING_MITIGATION_PLAN.md: pull critical-facility locations
(hospitals, wastewater treatment plants) from public ArcGIS REST FeatureServer
APIs, filter to Virginia, and save a standardized CSV that
scripts/compute_virginia_criticality.py folds into the Substation Criticality
Index as a fourth term (critical_load_i).

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS -- NOT IN THE SANDBOX
THIS PIPELINE WAS DEVELOPED IN. That sandbox's network-egress policy blocks
every ArcGIS/GIS-hosting domain outright, confirmed directly:
hifld-geoplatform.opendata.arcgis.com and services.arcgis.com both hard-
denied the connection, and maps.nccs.nasa.gov didn't even resolve. So the
URLs in CRITICAL_LOAD_SOURCES below were sourced from web search, not
verified by actually querying them -- some may be wrong, stale, or belong to
a non-authoritative mirror. Run this script and check the per-category
record counts it logs: 0 records or a repeated HTTP error means that
category's URL needs fixing. Paste the exact log output back for a fix.

Data centers (Loudoun County) are deliberately not included -- no working
FeatureServer URL was found, only an ArcGIS Hub "Experience" map page that
wraps one without exposing it directly. Add an entry to
CRITICAL_LOAD_SOURCES if you find the underlying layer URL (Loudoun's
GeoHub -> the data-center dataset's page should have an "API"/"GeoService"
link); everything downstream already handles an arbitrary number of
categories with no other code changes needed.

Each source is queried generically -- outFields=* (not a guessed field
list) and coordinates read from each feature's own point geometry rather
than a guessed lat/lon column name, since schemas differ across sources and
couldn't be verified from this sandbox.

Usage:
    python scripts/fetch_virginia_critical_loads.py
"""

import time

import pandas as pd
import requests
from shapely.geometry import Point

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/fetch_virginia_critical_loads.log")

FETCH_USER_AGENT = "c-swim-virginia-adaptation/1.0 (+https://github.com/VedaangV/c-swimvv)"

# category -> ArcGIS FeatureServer layer query URL. Add/remove/replace
# entries freely; every category flows through the same fetch/filter path.
CRITICAL_LOAD_SOURCES = {
    "hospital": "https://services.arcgis.com/EaQ3hSM51DBnlwMq/ArcGIS/rest/services/Hospitals/FeatureServer/0/query",
    "wastewater_treatment_plant": "https://services.arcgis.com/XG15cJAlne2vxtgt/ArcGIS/rest/services/wastewater_treatment_plants_epa_frs/FeatureServer/0/query",
}

PAGE_SIZE = 2000


def _fetch_category(category: str, query_url: str, bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Query one FeatureServer layer for point features intersecting bbox
    (minx, miny, maxx, maxy in EPSG:4326), paging via resultOffset until a
    short page signals the last one. Returns category/name/latitude/
    longitude rows -- an empty DataFrame (not an exception) on failure, so
    one bad URL doesn't stop the other categories from being fetched."""
    headers = {"User-Agent": FETCH_USER_AGENT}
    minx, miny, maxx, maxy = bbox
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": PAGE_SIZE,
        "resultOffset": 0,
    }

    rows = []
    while True:
        r = None
        for attempt in range(retries):
            try:
                r = requests.get(query_url, params=params, headers=headers, timeout=60)
                if r.status_code == 200:
                    break
                logger.warning(
                    f"{category} (attempt {attempt + 1}/{retries}) -> HTTP "
                    f"{r.status_code}: {r.text[:300]!r}"
                )
            except Exception as e:
                logger.warning(f"{category} (attempt {attempt + 1}/{retries}) -> {e!r}")
                r = None
            time.sleep(2 * (attempt + 1))
        else:
            logger.error(
                f"{category}: all {retries} attempts failed at offset "
                f"{params['resultOffset']}; keeping {len(rows)} rows fetched so far."
            )
            break

        try:
            data = r.json()
        except Exception as e:
            logger.error(f"{category}: response wasn't valid JSON ({e!r}); first 300 chars: {r.text[:300]!r}")
            break

        if "error" in data:
            logger.error(f"{category}: ArcGIS service returned an error: {data['error']}")
            break

        features = data.get("features", [])
        for feat in features:
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            lon, lat = geom["coordinates"][:2]
            props = feat.get("properties", {}) or {}
            name = (
                props.get("NAME") or props.get("Name") or props.get("name")
                or props.get("FACILITY_NAME") or props.get("PRIMARY_NAME")
            )
            rows.append({"category": category, "name": name, "latitude": lat, "longitude": lon})

        if len(features) < PAGE_SIZE:
            break
        params["resultOffset"] += PAGE_SIZE

    logger.info(f"{category}: fetched {len(rows)} raw point features within the query bbox.")
    return pd.DataFrame(rows)


def fetch_and_filter_to_virginia() -> pd.DataFrame:
    va_boundary = get_virginia_boundary()
    # Small buffer for the bbox query (cheap server-side pre-filter only);
    # the precise boundary check right after is what actually decides
    # Virginia membership, same pattern used throughout this pipeline
    # (restrict_to_virginia, _log_orphan_va_membership).
    minx, miny, maxx, maxy = va_boundary.buffer(0.3).bounds

    all_rows = []
    for category, query_url in CRITICAL_LOAD_SOURCES.items():
        df = _fetch_category(category, query_url, (minx, miny, maxx, maxy))
        if df.empty:
            logger.warning(f"{category}: 0 records fetched -- URL is likely wrong; see module docstring.")
            continue
        in_va = df.apply(lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1)
        df = df[in_va].copy()
        logger.info(f"{category}: {len(df)} inside Virginia's precise boundary.")
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            "No critical-facility category returned any Virginia records. Check "
            "CRITICAL_LOAD_SOURCES URLs -- see this script's module docstring."
        )
    return pd.concat(all_rows, ignore_index=True)


def main():
    facilities = fetch_and_filter_to_virginia()
    out_path = VA_NETWORK_DIR / "va_critical_facilities.csv"
    facilities.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}: {len(facilities)} Virginia critical facilities.")
    logger.info(f"By category:\n{facilities['category'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
