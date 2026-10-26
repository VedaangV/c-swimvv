"""
Optional refinement to substation labels used in plot_virginia_criticality.py
and plot_virginia_gic_risk.py: pulls real substation names from external
sources, since OSM's power=substation tagging often lacks a name in this
region (which is why those scripts fall back to operator/voltage/
coordinates so often).

MUST BE RUN ON A MACHINE WITH NORMAL INTERNET ACCESS -- this sandbox's
egress policy blocks essentially every external domain, ArcGIS, Wikidata
and Overpass included.

Three sources, tried separately and merged with priority HIFLD > Wikidata
> OSM Overpass when more than one matches the same substation:

1. HIFLD's Electric Substations dataset. Confirmed restricted even with
   real internet access -- returned only 3 named features for all of
   Virginia (2/2624 matched). Kept since it's harmless to also check.
2. Wikidata SPARQL, matching a fixed list of substation-type QIDs
   (WIKIDATA_SUBSTATION_CLASSES) rather than a transitive P31/P279* path --
   the transitive version hit repeated WDQS 502/504s. Also confirmed sparse
   (1/2624 matched); Wikidata just isn't an exhaustive infra inventory.
3. OSM Overpass, a live query (not tied to whatever snapshot age
   ss_df.pkl's SS_NAME column came from). No sampling cap, but still
   limited by whether a mapper bothered to tag a name.

All three matched to this project's own sub_id coordinates within
SNAP_DISTANCE_M (reusing virginia_config.py's line-endpoint snap tolerance
for the same "same substation, slightly different point" problem).

Usage:
    python scripts/fetch_virginia_substation_names.py
"""

import time

import geopandas as gpd
import pandas as pd
import requests

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, SNAP_DISTANCE_M, get_virginia_boundary

logger = setup_logger(log_file="logs/fetch_virginia_substation_names.log")

FETCH_USER_AGENT = "c-swim-virginia-adaptation/1.0 (+https://github.com/VedaangV/c-swimvv)"

HIFLD_SUBSTATIONS_URL = (
    "https://services.arcgis.com/G4S1dGvn7PIgYd6Y/ArcGIS/rest/services/"
    "HIFLD_electric_power_substations/FeatureServer/0/query"
)
# Couldn't verify the real schema from this sandbox, so try common HIFLD
# field-name conventions in order.
NAME_FIELD_CANDIDATES = ("NAME", "SUBSTATION", "SUBST_NAME", "SS_NAME")
OWNER_FIELD_CANDIDATES = ("OWNER", "UTILITY_OWNER", "OPERATOR")

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
# Fixed list instead of a wdt:P31/wdt:P279* transitive closure -- the
# transitive version hit repeated WDQS 502/504s (expensive "*" paths, geo
# filter doesn't reliably get applied first). Costs us subtypes not listed
# here, but avoids the timeouts.
WIKIDATA_SUBSTATION_CLASSES = (
    "wd:Q174814",     # electrical substation
    "wd:Q478986",     # distribution substation
    "wd:Q30945590",   # Extra High Voltage Substation
    "wd:Q1392266",    # mobile electrical substation
)
WIKIDATA_SUBSTATION_QUERY = """
SELECT ?item ?itemLabel ?coord WHERE {{
  VALUES ?class {{ {classes} }}
  ?item wdt:P31 ?class .
  SERVICE wikibase:box {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerWest "Point({minx} {miny})"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerEast "Point({maxx} {maxy})"^^geo:wktLiteral .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# node+way, power=substation and the less common power=station tag, each
# requiring a name tag (untagged features are exactly what this script is
# trying to fix, so they're no use as a source). "out center" gives ways a
# ready-made center point.
OVERPASS_SUBSTATION_QUERY = """
[out:json][timeout:180];
(
  node["power"="substation"]["name"]({miny},{minx},{maxy},{maxx});
  way["power"="substation"]["name"]({miny},{minx},{maxy},{maxx});
  node["power"="station"]["name"]({miny},{minx},{maxy},{maxx});
  way["power"="station"]["name"]({miny},{minx},{maxy},{maxx});
);
out center tags;
"""

PAGE_SIZE = 2000


def _fetch_hifld_substations(bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Same paginate-until-a-genuinely-empty-page pattern as
    fetch_virginia_critical_loads.py's _fetch_category()."""
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
    max_pages = 500
    for _ in range(max_pages):
        r = None
        for attempt in range(retries):
            try:
                r = requests.get(HIFLD_SUBSTATIONS_URL, params=params, headers=headers, timeout=60)
                if r.status_code == 200:
                    break
                logger.warning(
                    f"HIFLD substations (attempt {attempt + 1}/{retries}) -> HTTP "
                    f"{r.status_code}: {r.text[:300]!r}"
                )
            except Exception as e:
                logger.warning(f"HIFLD substations (attempt {attempt + 1}/{retries}) -> {e!r}")
                r = None
            time.sleep(2 * (attempt + 1))
        else:
            logger.error(
                f"HIFLD substations: all {retries} attempts failed at offset "
                f"{params['resultOffset']}; keeping {len(rows)} rows fetched so far."
            )
            break

        try:
            data = r.json()
        except Exception as e:
            logger.error(f"HIFLD substations: response wasn't valid JSON ({e!r}); first 300 chars: {r.text[:300]!r}")
            break
        if "error" in data:
            logger.error(f"HIFLD substations: service returned an error: {data['error']}")
            break

        features = data.get("features", [])
        if not features:
            break

        for feat in features:
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            lon, lat = geom["coordinates"][:2]
            props = feat.get("properties", {}) or {}
            name = next((props[c] for c in NAME_FIELD_CANDIDATES if props.get(c)), None)
            owner = next((props[c] for c in OWNER_FIELD_CANDIDATES if props.get(c)), None)
            if not name:
                continue  # no better than the existing OSM fallback
            rows.append({"name": name, "owner": owner, "latitude": lat, "longitude": lon})

        params["resultOffset"] += len(features)
    else:
        logger.warning(
            f"HIFLD substations: stopped after {max_pages} consecutive non-empty "
            "pages without ever seeing an empty one -- results may be truncated."
        )

    logger.info(f"HIFLD substations: fetched {len(rows)} named point features within the query bbox.")
    return pd.DataFrame(rows)


def _fetch_wikidata_substations(bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Query Wikidata Query Service for electrical-substation-like items
    (see module docstring) with a coordinate inside bbox."""
    minx, miny, maxx, maxy = bbox
    query = WIKIDATA_SUBSTATION_QUERY.format(
        classes=" ".join(WIKIDATA_SUBSTATION_CLASSES), minx=minx, miny=miny, maxx=maxx, maxy=maxy
    )
    headers = {"User-Agent": FETCH_USER_AGENT, "Accept": "application/sparql-results+json"}

    r = None
    for attempt in range(retries):
        try:
            r = requests.get(
                WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"}, headers=headers, timeout=90
            )
            if r.status_code == 200:
                break
            logger.warning(f"Wikidata SPARQL (attempt {attempt + 1}/{retries}) -> HTTP {r.status_code}: {r.text[:300]!r}")
        except Exception as e:
            logger.warning(f"Wikidata SPARQL (attempt {attempt + 1}/{retries}) -> {e!r}")
            r = None
        time.sleep(3 * (attempt + 1))
    else:
        logger.error(f"Wikidata SPARQL: all {retries} attempts failed.")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])

    try:
        data = r.json()
    except Exception as e:
        logger.error(f"Wikidata SPARQL: response wasn't valid JSON ({e!r}); first 300 chars: {r.text[:300]!r}")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])

    bindings = data.get("results", {}).get("bindings", [])
    rows = []
    for b in bindings:
        label = b.get("itemLabel", {}).get("value")
        coord = b.get("coord", {}).get("value")  # "Point(lon lat)"
        if not label or not coord:
            continue
        try:
            lon_str, lat_str = coord.replace("Point(", "").replace(")", "").split(" ")
            lon, lat = float(lon_str), float(lat_str)
        except Exception:
            continue
        rows.append({"name": label, "latitude": lat, "longitude": lon})

    logger.info(f"Wikidata: fetched {len(rows)} named substation-like points within the query bbox.")
    return pd.DataFrame(rows)


def _fetch_osm_overpass_substations(bbox: tuple, retries: int = 3) -> pd.DataFrame:
    """Live query against the current OSM database (see module docstring)."""
    minx, miny, maxx, maxy = bbox
    query = OVERPASS_SUBSTATION_QUERY.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    headers = {"User-Agent": FETCH_USER_AGENT}

    r = None
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=200)
            if r.status_code == 200:
                break
            logger.warning(f"Overpass (attempt {attempt + 1}/{retries}) -> HTTP {r.status_code}: {r.text[:300]!r}")
        except Exception as e:
            logger.warning(f"Overpass (attempt {attempt + 1}/{retries}) -> {e!r}")
            r = None
        time.sleep(5 * (attempt + 1))
    else:
        logger.error(f"Overpass: all {retries} attempts failed.")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])

    try:
        data = r.json()
    except Exception as e:
        logger.error(f"Overpass: response wasn't valid JSON ({e!r}); first 300 chars: {r.text[:300]!r}")
        return pd.DataFrame(columns=["name", "latitude", "longitude"])

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name")
        if not name:
            continue
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue
        rows.append({"name": name, "latitude": lat, "longitude": lon})

    logger.info(f"Overpass: fetched {len(rows)} named substation/station elements within the query bbox.")
    return pd.DataFrame(rows)


def _nearest_match(ours_3857: gpd.GeoDataFrame, named_points: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbor join named_points (name/latitude/longitude, EPSG:4326)
    onto ours_3857 (this project's sub_id points, already in EPSG:3857)
    within SNAP_DISTANCE_M. One row per matched sub_id, closest wins ties."""
    if named_points.empty:
        return pd.DataFrame(columns=["sub_id", "name", "match_distance_m"])

    theirs = gpd.GeoDataFrame(
        named_points[["name"]],
        geometry=gpd.points_from_xy(named_points["longitude"], named_points["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    joined = gpd.sjoin_nearest(ours_3857, theirs, max_distance=SNAP_DISTANCE_M, distance_col="match_distance_m")
    joined = joined.sort_values("match_distance_m").drop_duplicates(subset="sub_id", keep="first")
    return joined[["sub_id", "name", "match_distance_m"]]


def main():
    va_boundary = get_virginia_boundary()
    minx, miny, maxx, maxy = va_boundary.buffer(0.3).bounds

    topo_path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not topo_path.exists():
        raise FileNotFoundError(f"Missing {topo_path}. Run scripts/build_virginia_network_graph.py first.")
    topo = pd.read_csv(topo_path)
    topo["sub_id"] = topo["sub_id"].astype(str)
    ours_3857 = gpd.GeoDataFrame(
        topo[["sub_id"]],
        geometry=gpd.points_from_xy(topo["longitude"], topo["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    hifld = _fetch_hifld_substations((minx, miny, maxx, maxy))
    hifld_matches = _nearest_match(ours_3857, hifld)
    hifld_matches = hifld_matches.assign(source="hifld")
    logger.info(f"HIFLD: {len(hifld_matches)}/{len(ours_3857)} substations matched within {SNAP_DISTANCE_M}m.")

    wikidata = _fetch_wikidata_substations((minx, miny, maxx, maxy))
    wikidata_matches = _nearest_match(ours_3857, wikidata)
    wikidata_matches = wikidata_matches.assign(source="wikidata")
    logger.info(f"Wikidata: {len(wikidata_matches)}/{len(ours_3857)} substations matched within {SNAP_DISTANCE_M}m.")

    overpass = _fetch_osm_overpass_substations((minx, miny, maxx, maxy))
    overpass_matches = _nearest_match(ours_3857, overpass)
    overpass_matches = overpass_matches.assign(source="osm_overpass")
    logger.info(f"OSM Overpass: {len(overpass_matches)}/{len(ours_3857)} substations matched within {SNAP_DISTANCE_M}m.")

    # keep="first" preserves the HIFLD > Wikidata > OSM priority order above.
    combined = pd.concat([hifld_matches, wikidata_matches, overpass_matches], ignore_index=True)
    combined = combined.drop_duplicates(subset="sub_id", keep="first")

    if combined.empty:
        logger.warning(
            "No named substations matched from any source -- writing an "
            "empty result. Both viz scripts fall back cleanly to OSM-based "
            "labels when this file is empty/missing."
        )

    out_path = VA_NETWORK_DIR / "va_substation_names.csv"
    combined.to_csv(out_path, index=False)
    logger.info(
        f"Saved {out_path}: {len(combined)} substations with a real name "
        f"({(combined['source'] == 'hifld').sum()} from HIFLD, "
        f"{(combined['source'] == 'wikidata').sum()} from Wikidata, "
        f"{(combined['source'] == 'osm_overpass').sum()} from OSM Overpass)."
    )


if __name__ == "__main__":
    main()
