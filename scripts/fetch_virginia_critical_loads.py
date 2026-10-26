"""
Phase 2b of ISLANDING_MITIGATION_PLAN.md: pulls critical-facility locations
(hospitals, wastewater treatment plants, data centers) into a standardized
CSV that compute_virginia_criticality.py folds into the SCI as a fourth
term (critical_load_i).

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS, NOT THIS PIPELINE'S
DEV SANDBOX -- its network-egress policy blocks every ArcGIS/GIS domain, so
none of the URLs below could be verified from there.

Three ingestion paths per category, since a live API isn't always
findable: CRITICAL_LOAD_API_SOURCES (ArcGIS FeatureServer query, currently
just EPA wastewater plants), CRITICAL_LOAD_OSM_SOURCES (Overpass tag
filter -- hospitals and data centers ended up here after their original
HIFLD/Loudoun-GeoHub leads turned out to be non-public or nonexistent), and
CRITICAL_LOAD_LOCAL_SOURCES (manually downloaded file, currently empty --
add a category by dropping a file under CRITICAL_LOAD_LOCAL_DIR as
"<stem>.geojson/.json/.csv" and listing it here; the loader auto-detects
geometry or lat/lon columns).

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

# Same servers/UA as build_virginia_network.py's OVERPASS_SERVERS (ru
# instance dropped there as unreachable).
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_USER_AGENT = "c-swim-virginia-adaptation/1.0 (+https://github.com/VedaangV/c-swimvv)"

# category -> Overpass QL tag filter (without brackets), fetched live via
# Overpass across node/way/relation.
CRITICAL_LOAD_OSM_SOURCES = {
    "hospital": "amenity=hospital",
    "data_center": "telecom=data_center",
}

# category -> filename stem for a manually-downloaded file. Empty for now,
# see module docstring.
CRITICAL_LOAD_LOCAL_DIR = VA_NETWORK_DIR / "raw_critical_loads"
CRITICAL_LOAD_LOCAL_SOURCES = {}

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
    services cap at their own server-side maxRecordCount regardless of
    resultRecordCount (wastewater plants came back at exactly 1000 on page
    1 once), so "fewer than asked for" doesn't mean "last page".

    Returns an empty DataFrame (not an exception) on failure, so one bad
    URL doesn't stop the other categories."""
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
            # Only reliable "last page" signal; a short page doesn't mean
            # done (see docstring, maxRecordCount capping).
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
        # Never saw an empty page -- hit max_pages instead of the natural
        # end-of-data signal.
        logger.warning(
            f"{category}: stopped after {max_pages} consecutive non-empty pages "
            "without ever seeing an empty one -- results may be truncated."
        )

    logger.info(f"{category}: fetched {len(rows)} raw point features within the query bbox.")
    return pd.DataFrame(rows)


def _query_overpass(query: str, timeout: int = 120, retries: int = 2) -> dict | None:
    """POST an Overpass query, trying each server in turn -- same pattern
    as build_virginia_network.py's _query_overpass()."""
    headers = {"User-Agent": OVERPASS_USER_AGENT}
    for url in OVERPASS_SERVERS:
        for attempt in range(retries):
            try:
                r = requests.post(url, data={"data": query}, headers=headers, timeout=timeout + 30)
                if r.status_code == 200:
                    return r.json()
                logger.warning(
                    f"Overpass {url} (attempt {attempt + 1}/{retries}) -> "
                    f"HTTP {r.status_code}: {r.text[:300]!r}"
                )
            except Exception as e:
                logger.warning(f"Overpass {url} (attempt {attempt + 1}/{retries}) -> {e!r}")
            time.sleep(3 * (attempt + 1))
    logger.error(f"All Overpass servers failed for this query (first 120 chars: {query.strip()[:120]!r})")
    return None


def _fetch_osm_category(category: str, tag_filter: str, bbox: tuple) -> pd.DataFrame:
    """Query Overpass for node/way/relation elements matching tag_filter
    (e.g. 'amenity=hospital') inside bbox. Uses `out center;` so every
    element type yields a single lat/lon."""
    minx, miny, maxx, maxy = bbox
    tag_key, tag_value = tag_filter.split("=", 1)
    tag_expr = f'["{tag_key}"="{tag_value}"]'
    query = f"""
    [out:json][timeout:90][bbox:{miny},{minx},{maxy},{maxx}];
    (
      node{tag_expr};
      way{tag_expr};
      relation{tag_expr};
    );
    out center;
    """
    data = _query_overpass(query)
    if not data or not data.get("elements"):
        logger.warning(f"{category}: 0 features from Overpass -- see module docstring / logged errors above.")
        return pd.DataFrame(columns=["category", "name", "latitude", "longitude"])

    rows = []
    for el in data["elements"]:
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center")
            if not center:
                continue
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        name = (el.get("tags") or {}).get("name")
        rows.append({"category": category, "name": name, "latitude": lat, "longitude": lon})

    logger.info(f"{category}: fetched {len(rows)} raw point features from Overpass within the query bbox.")
    return pd.DataFrame(rows)


def _find_col(columns, candidates) -> str | None:
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _load_local_category(category: str, stem: str) -> pd.DataFrame:
    """Read a manually-downloaded file for one category: tries <stem>.geojson,
    <stem>.json, then <stem>.csv (lat/lon auto-detected). Returns an empty
    DataFrame, not an exception, when no file exists yet."""
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
    # Buffer is just a cheap server-side pre-filter for the bbox query; the
    # precise boundary check below decides real Virginia membership.
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

    for category, tag_filter in CRITICAL_LOAD_OSM_SOURCES.items():
        df = _fetch_osm_category(category, tag_filter, (minx, miny, maxx, maxy))
        if df.empty:
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
            "CRITICAL_LOAD_API_SOURCES URLs, CRITICAL_LOAD_OSM_SOURCES tags, and "
            "CRITICAL_LOAD_LOCAL_SOURCES files -- see this script's module docstring."
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
