"""
Extracts the Virginia+PJM subset of the already-computed US storm maxima.

Per Dennies Bor, MT sites aren't filtered for the SECS/field-interpolation
step -- that still runs over all of North America (calc_storm_maxes.py).
This script just pulls out the per-site B/E field maxima that fall inside
the PJM boundary from the US-wide arrays, by index; no recomputation.

Line-level voltage maxima can't be subset the same way -- build_virginia_
network.py builds a new line set with different LINE_IDs/geometries than
the US one maxV_arr_testing_2.npy is indexed against. Those get regenerated
separately in recompute_virginia_line_voltages.py instead.
"""

import pickle

import numpy as np
import pandas as pd
from shapely.geometry import Point

from configs import setup_logger, get_data_dir
from configs.virginia_config import VA_STORM_MAXES_DIR, get_pjm_boundary

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/extract_virginia_storm_maxes.log")


def load_all_mt_sites():
    mt_sites_pickle = DATA_LOC / "EMTF" / "mt_pickle.pkl"
    with open(mt_sites_pickle, "rb") as f:
        return pickle.load(f)


def find_pjm_site_indices(MT_sites, pjm_boundary):
    pjm_indices = []
    for i, site in enumerate(MT_sites):
        if pjm_boundary.contains(Point(site.longitude, site.latitude)):
            pjm_indices.append(i)
    return np.array(pjm_indices, dtype=int)


def main():
    pjm_boundary = get_pjm_boundary()

    logger.info("Loading all US MT sites (unfiltered, per Dennies' guidance)...")
    MT_sites = load_all_mt_sites()
    logger.info(f"Total US MT sites: {len(MT_sites)}")

    pjm_indices = find_pjm_site_indices(MT_sites, pjm_boundary)
    pjm_sites = [MT_sites[i] for i in pjm_indices]
    logger.info(f"MT sites within PJM boundary: {len(pjm_sites)}")

    storm_maxes_dir = DATA_LOC / "storm_maxes"
    maxB_file = storm_maxes_dir / "maxB_arr_testing_2.npy"
    maxE_file = storm_maxes_dir / "maxE_arr_testing_2.npy"

    if not maxB_file.exists() or not maxE_file.exists():
        raise FileNotFoundError(
            f"Expected US storm maxima at {maxB_file} and {maxE_file}. "
            "Run the full US pipeline's scripts/calc_storm_maxes.py first "
            "(already done on the reference machine; these files should "
            "already exist under data/storm_maxes/)."
        )

    maxB_arr = np.load(maxB_file)  # shape: (n_sites, n_storms)
    maxE_arr = np.load(maxE_file)  # shape: (n_sites, n_storms)

    if maxB_arr.shape[0] != len(MT_sites):
        raise ValueError(
            f"maxB_arr has {maxB_arr.shape[0]} rows but mt_pickle.pkl has "
            f"{len(MT_sites)} sites — they must come from the same US run."
        )

    va_pjm_maxB = maxB_arr[pjm_indices, :]
    va_pjm_maxE = maxE_arr[pjm_indices, :]

    storm_df = pd.read_csv(DATA_LOC / "kp_ap_indices" / "storm_periods.csv")

    with open(VA_STORM_MAXES_DIR / "va_pjm_mt_sites.pkl", "wb") as f:
        pickle.dump(pjm_sites, f)
    np.save(VA_STORM_MAXES_DIR / "va_pjm_site_indices.npy", pjm_indices)
    np.save(VA_STORM_MAXES_DIR / "va_pjm_maxB.npy", va_pjm_maxB)
    np.save(VA_STORM_MAXES_DIR / "va_pjm_maxE.npy", va_pjm_maxE)
    storm_df.to_csv(VA_STORM_MAXES_DIR / "va_storm_periods.csv", index=False)

    logger.info(f"Saved PJM site subset and B/E maxima to {VA_STORM_MAXES_DIR}")
    logger.info(
        "Per-line voltage maxima were NOT extracted here — see "
        "scripts/recompute_virginia_line_voltages.py."
    )


if __name__ == "__main__":
    main()
