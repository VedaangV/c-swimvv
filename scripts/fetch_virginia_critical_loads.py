"""
Phase 2b of ISLANDING_MITIGATION_PLAN.md: pull critical-facility locations
(hospitals, wastewater treatment plants, data centers) into a standardized
CSV that scripts/compute_virginia_criticality.py folds into the Substation
Criticality Index as a fourth term (critical_load_i).

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS -- NOT IN THE SANDBOX
THIS PIPELINE WAS DEVELOPED IN. That sandbox's network-egress policy blocks
every ArcGIS/GIS-hosting domain outright (confirmed directly), so none of
the URLs below could be verified from there.

Two ingestion paths, per category, since a live-queryable API isn't always
findable/reliable:

1. CRITICAL_LOAD_API_SOURCES: category -> ArcGIS FeatureServer query URL,
   fetched live via requests. Currently just wastewater treatment plants
   (an EPA-hosted layer that worked on the first real run).

2. CRITICAL_LOAD_LOCAL_SOURCES: category -> filename stem. Hospitals and
   data centers moved here after the API guesses either came back empty
   (hospitals: the FeatureServer URL found via search returned 0 features
   on a real run -- likely wrong layer/service, and guessing another blind
   URL wasn't a good use of another round-trip) or never had a candidate
   API in the first place (data centers: Loudoun County's data-center
   layer is wrapped in an ArcGIS Hub "Experience" app with no directly
   discoverable FeatureServer URL).

   To add a category here: download the dataset yourself (see below), save
   it under CRITICAL_LOAD_LOCAL_DIR as "<stem>.geojson"/".json"/".csv", and
   list it in CRITICAL_LOAD_LOCAL_SOURCES. No other code changes needed --
   the loader auto-detects point geometry (GeoJSON/JSON) or a lat/lon
   column pair (CSV, matched case-insensitively against a small set of
   common names) and filters to Virginia the same way the API path does.

   Where to get them:
   - Hospitals: https://hifld-geoplatform.opendata.arcgis.com/datasets/hospitals
     -> Download -> GeoJSON. This is the full national dataset (~7000
     rows); that's fine, the Virginia-boundary filter below trims it down
     the same way the API path's bbox query would have.
   - Data centers: https://geohub-loudoungis.opendata.arcgis.com -> search
     "data center" -> open the dataset page -> Download -> GeoJSON (or CSV
     if GeoJSON isn't offered).
   Save as data/virginia/network/raw_critical_loads/hospitals.geojson and
   .../data_centers.geojson (or .csv) respectively.

Usage:
    python scripts/fetch_virginia_critical_loads.py
"""

import time

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/fetch_virginia_critical_loads.log")

FETCH_USER_AGENT = "c-swim-virginia-adaptation/1.0 (+https://github.com/VedaangV/c-swimvv)"

# category -> ArcGIS FeatureServer layer query URL, fetched live.
CRITICAL_LOAD_API_SOURCES = {
    "wastewater_treatment_plant": "https://services.arcgis.com/XG15cJAlne2vxtgt/ArcGIS/rest/services/wastewater_treatment_plants_epa_frs/FeatureServer/0/query",
}

# category -> filename stem for a manually-downloaded file. See module
# docstring for exactly where to download each one from.
CRITICAL_LOAD_LOCAL_DIR = VA_NETWORK_DIR / "raw_critical_loads"
CRITICAL_LOAD_LOCAL_SOURCES = {
    "hospital": "hospitals",
    "data_center": "data_centers",
}

# Matched case-insensitively against a local CSV's column names, in order,
# to auto-detect coordinates when the schema isn't known in advance.
LAT_COL_CANDIDATES = ("latitude", "lat", "y", "point_y")
LON_COL_CANDIDATES = ("longitude", "long", "lon", "x", "point_x")
NAME_COL_CANDIDATES = ("name", "facility_name", "primary_name", "site_name")

PAGE_SIZE = 2000


def _fetch_category(category: str, query_url: str, bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Query one FeatureServer layer for point features intersecting bbox
    (minx, miny, maxx, maxy in EPSG:4326), paging via resultOffset until a
    page comes back EMPTY -- not just shorter than requested. Many ArcGIS
    services cap responses at their own server-side maxRecordCount (often
    1000) regardless of resultRecordCount, so "got fewer than asked for"
    does NOT reliably mean "that was the last page": a real run against the
    wastewater-plants source returned exactly 1000 features on page 1 and
    an earlier version of this function stopped there, silently dropping
    anything past record 1000. Paging by the actual count returned (not
    PAGE_SIZE) and stopping only on a genuinely empty page fixes that
    regardless of what cap any given server enforces.

    Returns category/name/latitude/longitude rows -- an empty DataFrame
    (not an exception) on failure, so one bad URL doesn't stop the other
    categories from being fetched."""
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
    max_pages = 500  # safety valve against a server that never returns an empty page
    for _ in range(max_pages):
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
        if not features:
            # The only reliable "last page" signal across arbitrary ArcGIS
            # services: a page with zero features. Trusting "got fewer than
            # requested" instead is what silently truncated wastewater
            # treatment plants at 1000 records on the real run that found
            # this bug -- the server's own maxRecordCount capped every page
            # at 1000 regardless of resultRecordCount=2000, so a short page
            # never actually meant "no more data."
            break

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

        params["resultOffset"] += len(features)
    else:
        # Loop ran max_pages times and every single page was non-empty --
        # never saw the natural end-of-data signal, unlike every other exit
        # path above (each of which already logged its own reason).
        logger.warning(
            f"{category}: stopped after {max_pages} consecutive non-empty pages "
            "without ever seeing an empty one -- results may be truncated."
        )

    logger.info(f"{category}: fetched {len(rows)} raw point features within the query bbox.")
    return pd.DataFrame(rows)


def _find_col(columns, candidates) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _load_local_category(category: str, stem: str) -> pd.DataFrame:
    """Read a manually-downloaded file for one category. Tries, in order,
    <stem>.geojson, <stem>.json (point geometry read via geopandas), then
    <stem>.csv (lat/lon columns auto-detected by name). Returns an empty
    DataFrame -- not an exception -- when no file exists yet, since Phase
    2b's categories are each independently optional."""
    for ext in (".geojson", ".json"):
        path = CRITICAL_LOAD_LOCAL_DIR / f"{stem}{ext}"
        if path.exists():
            gdf = gpd.read_file(path)
            gdf = gdf[gdf.geometry.type == "Point"]
            if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            name_col = _find_col(gdf.columns, NAME_COL_CANDIDATES) or _find_col(gdf.columns, ("NAME",))
            logger.info(f"{category}: loaded {len(gdf)} point features from {path}.")
            return pd.DataFrame({
                "category": category,
                "name": gdf[name_col] if name_col else None,
                "latitude": gdf.geometry.y,
                "longitude": gdf.geometry.x,
            })

    path = CRITICAL_LOAD_LOCAL_DIR / f"{stem}.csv"
    if path.exists():
        df = pd.read_csv(path)
        lat_col = _find_col(df.columns, LAT_COL_CANDIDATES)
        lon_col = _find_col(df.columns, LON_COL_CANDIDATES)
        if lat_col is None or lon_col is None:
            raise ValueError(
                f"{path}: couldn't auto-detect latitude/longitude columns. "
                f"Columns present: {list(df.columns)}. Tell me the right column "
                "names and I'll wire them into LAT_COL_CANDIDATES/LON_COL_CANDIDATES."
            )
        name_col = _find_col(df.columns, NAME_COL_CANDIDATES)
        logger.info(f"{category}: loaded {len(df)} rows from {path} (lat={lat_col!r}, lon={lon_col!r}).")
        return pd.DataFrame({
            "category": category,
            "name": df[name_col] if name_col else None,
            "latitude": df[lat_col].astype(float),
            "longitude": df[lon_col].astype(float),
        })

    logger.info(
        f"{category}: no local file at {CRITICAL_LOAD_LOCAL_DIR}/{stem}.[geojson|json|csv] "
        "-- skipping. See module docstring for where to download it."
    )
    return pd.DataFrame(columns=["category", "name", "latitude", "longitude"])


def fetch_and_filter_to_virginia() -> pd.DataFrame:
    va_boundary = get_virginia_boundary()
    # Small buffer for the API path's bbox query (cheap server-side
    # pre-filter only); the precise boundary check below is what actually
    # decides Virginia membership for every category, API or local, same
    # pattern used throughout this pipeline (restrict_to_virginia,
    # _log_orphan_va_membership).
    minx, miny, maxx, maxy = va_boundary.buffer(0.3).bounds

    all_rows = []

    for category, query_url in CRITICAL_LOAD_API_SOURCES.items():
        df = _fetch_category(category, query_url, (minx, miny, maxx, maxy))
        if df.empty:
            logger.warning(f"{category}: 0 records fetched -- URL is likely wrong; see module docstring.")
            continue
        in_va = df.apply(lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1)
        df = df[in_va].copy()
        logger.info(f"{category}: {len(df)} inside Virginia's precise boundary.")
        all_rows.append(df)

    for category, stem in CRITICAL_LOAD_LOCAL_SOURCES.items():
        df = _load_local_category(category, stem)
        if df.empty:
            continue
        in_va = df.apply(lambda r: va_boundary.contains(Point(r["longitude"], r["latitude"])), axis=1)
        df = df[in_va].copy()
        logger.info(f"{category}: {len(df)} inside Virginia's precise boundary.")
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            "No critical-facility category returned any Virginia records. Check "
            "CRITICAL_LOAD_API_SOURCES URLs and CRITICAL_LOAD_LOCAL_SOURCES files "
            "-- see this script's module docstring."
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
