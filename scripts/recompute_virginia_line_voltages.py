"""
Regenerate per-line storm voltage maxima and return-period statistics for the
Virginia+PJM transmission network, and package them into an HDF5 file with
the same layout scripts/est_gic.py expects.

Why this step exists (a gap in the original adaptation plan): scripts/
est_gic.py drives its GIC solve almost entirely from *per-line* voltage
columns (V_halloween, V_st_patricks, V_gannon, V_{50,75,...,250}) that were
statistically fit against the ORIGINAL US EHV line set's storm history
(data/grid_processed/df_lines_EHV.pkl + data/grid_processed/
trans_lines_pickle.pkl, i.e. the Delaunay-weighted transmission lines used by
scripts/calc_storm_maxes.py). Those line-level voltages cannot be filtered or
reused for Virginia because scripts/build_virginia_network.py builds an
entirely different set of lines (different LINE_IDs, different geometry,
different Delaunay interpolation weights). They have to be recomputed against
the VA+PJM lines. Per-site (B/E) magnetic/electric field quantities do not
have this problem — the original plan's "no recomputation, just filter by MT
site index" instruction is correct for those (see
scripts/extract_virginia_storm_maxes.py) and is reused here for the "site"
side of the output file.

This script:
1. Builds Delaunay interpolation weights for the VA+PJM lines against the
   FULL, unfiltered set of US MT sites (never just PJM sites — same
   continent-wide-interpolation principle Dennies specified for the B/E
   step), by reusing scripts/calc_storm_maxes.read_transmission_lines()
   unmodified. Output: data/virginia/grid_processed/trans_lines_pickle.pkl
   (the last file scripts/build_admittance_matrix.py's load_and_process_data
   needs that scripts/build_virginia_network.py doesn't already produce).
2. Replays scripts/calc_storm_maxes.calculate_maxes() for every storm in
   kp_ap_indices/storm_periods.csv against those VA+PJM lines to build a
   (n_va_lines, n_storms) voltage-maxima array via multiprocessing.Pool.
   calculate_maxes() reads MT_sites/df/obs_dict/n_trans_lines as module
   globals on scripts.calc_storm_maxes, which we monkeypatch below in the
   parent process — and, since the target machine is Windows (spawn, not
   fork), also re-apply in every worker via the _init_worker Pool
   initializer, exactly the pattern the original adaptation plan called
   for. Without it, each spawned worker re-imports scripts.calc_storm_maxes
   fresh without those attributes set, calculate_maxes() raises NameError,
   process_storm-style exception handling swallows it per-storm, and the
   run "completes" with a silently all-zero maxV_arr instead of failing
   loudly. See _init_worker's docstring for the full mechanics.
3. Fits return-period statistics on that array using
   scripts/stat_analysis.py's bootstrap machinery unmodified.
4. Packages sites/mt_sites, sites/transmission_lines, events/*, and
   predictions/* into data/virginia/statistical_analysis/
   va_geomagnetic_data_return_periods.h5, matching the exact HDF5 layout
   scripts.est_gic.load_and_process_gic_data() reads.

Run after scripts/build_virginia_network.py and
scripts/extract_virginia_storm_maxes.py.
"""

import multiprocessing
import pickle
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import xarray as xr

import scripts.calc_storm_maxes as csm
import scripts.stat_analysis as sa
from configs import setup_logger, get_data_dir, LEAVE_OUT_SITES
from configs.virginia_config import VA_NETWORK_DIR, VA_STORM_MAXES_DIR, VA_STATISTICAL_ANALYSIS_DIR

DATA_LOC = get_data_dir()
logger = setup_logger(log_file="logs/recompute_virginia_line_voltages.log")

GRID_DIR = VA_NETWORK_DIR.parent / "grid_processed"

# Same constants scripts/stat_analysis.py's __main__ block uses.
SA_N_SAMPLES = 100
SA_N_YEARS = 39
SA_RETURN_PERIODS = np.arange(25, 251, 25)

HALLOWEEN_START, HALLOWEEN_END = datetime(2003, 10, 29, 0), datetime(2003, 11, 3, 0)
GANNON_START, GANNON_END = datetime(2024, 5, 9, 14), datetime(2024, 5, 12, 14)
ST_PATRICKS_START = datetime(2015, 3, 17, 0)
ST_PATRICKS_END = ST_PATRICKS_START + timedelta(days=1)
HYDRO_QUEBEC_START = datetime(1989, 3, 13, 0)
HYDRO_QUEBEC_END = HYDRO_QUEBEC_START + timedelta(days=1)


def _init_worker(mt_sites, df_, n_lines, obsdict):
    """multiprocessing.Pool initializer for Windows (spawn) workers.

    On Linux, Pool() forks — child processes inherit the parent's already
    -monkeypatched scripts.calc_storm_maxes module object (csm.MT_sites,
    csm.df, csm.obs_dict, csm.n_trans_lines set in main() below) for free.
    On Windows there is no fork; each worker starts a fresh interpreter and
    re-imports scripts.calc_storm_maxes from scratch, which does NOT have
    those attributes set (they're only assigned inside this script's own
    `if __name__ == "__main__":` block, which — correctly — never re-runs in
    spawned children). Without this initializer, _process_storm_va's call to
    csm.calculate_maxes() would raise NameError on MT_sites/df/obs_dict/
    n_trans_lines in every worker — caught by its own try/except and logged,
    so the run would silently "complete" with an all-zero maxV_arr instead
    of crashing. Re-set the same globals here so it's correct on both
    platforms."""
    csm.MT_sites = mt_sites
    csm.df = df_
    csm.n_trans_lines = n_lines
    csm.obs_dict = obsdict


def _process_storm_va(args):
    """Like scripts.calc_storm_maxes.process_storm, but always takes the
    aggregated-max branch (is_special=False) since we only need line_maxV,
    not the full per-minute voltage matrix special storms would otherwise
    return."""
    i, row = args
    try:
        _, _, _, maxV, _, _ = i, *csm.calculate_maxes(
            row["Start"], row["End"], calcV=True, is_special=False
        )
        return i, maxV
    except Exception as e:
        logger.error(f"Error processing storm {i}: {e}")
        return i, None


def build_va_trans_lines_pickle(MT_sites):
    """Delaunay-weight the VA+PJM lines against the full US MT site set."""
    site_xys = [(site.latitude, site.longitude) for site in MT_sites]
    df = csm.read_transmission_lines(
        GRID_DIR / "df_lines_EHV.pkl",
        GRID_DIR / "trans_lines_pickle.pkl",
        site_xys,
    )
    logger.info(f"VA+PJM lines with valid Delaunay weights: {len(df)}")
    return df


def compute_va_maxV_arr(storm_df, n_va_lines, n_workers=None):
    """Replay calculate_maxes() for every storm against the VA+PJM lines."""
    cache = VA_STORM_MAXES_DIR / "va_maxV_arr.npy"
    if cache.exists():
        logger.info(f"Loading cached VA line-voltage maxima: {cache}")
        return np.load(cache)

    n_storms = len(storm_df)
    maxV_arr = np.zeros((n_va_lines, n_storms))
    args = [(i, row) for i, row in storm_df.iterrows()]

    n_workers = n_workers or min(8, multiprocessing.cpu_count())
    logger.info(f"Recomputing VA line voltages for {n_storms} storms ({n_workers} workers)...")

    with multiprocessing.Pool(
        n_workers,
        initializer=_init_worker,
        initargs=(csm.MT_sites, csm.df, csm.n_trans_lines, csm.obs_dict),
    ) as pool:
        for i, maxV in pool.imap_unordered(_process_storm_va, args):
            if maxV is None:
                continue
            maxV_arr[:, i] = maxV
            np.save(cache, maxV_arr)

    return maxV_arr


def fit_v_statistics(maxV_arr):
    sa.n_samples, sa.n_years, sa.return_periods = SA_N_SAMPLES, SA_N_YEARS, SA_RETURN_PERIODS
    results_v_path = VA_STATISTICAL_ANALYSIS_DIR / "results_V_return_periods.pkl"
    results_v = sa.load_or_calculate_results(results_v_path, maxV_arr, "V")
    sa.confidence_limits(results_v, SA_RETURN_PERIODS)
    return sa.extract_confidence_medians(results_v, SA_RETURN_PERIODS)


def fit_or_reuse_site_statistics(quantity, va_site_arr, pjm_indices):
    """Prefer filtering the already-fit US-wide B/E results to PJM site
    indices (statistical fits are per-site/independent, so this is a valid,
    much cheaper filter); fall back to fitting fresh on the PJM subset if the
    US-wide cache isn't present on this machine."""
    sa.n_samples, sa.n_years, sa.return_periods = SA_N_SAMPLES, SA_N_YEARS, SA_RETURN_PERIODS
    us_cache = DATA_LOC / "statistical_analysis" / f"results_{quantity}_return_periods.pkl"

    if us_cache.exists():
        logger.info(f"Filtering existing US-wide {quantity} fit to PJM sites: {us_cache}")
        with open(us_cache, "rb") as f:
            us_results = pickle.load(f)
        results = {period: arr[pjm_indices] for period, arr in us_results.items()}
    else:
        logger.warning(
            f"No US-wide {quantity} fit at {us_cache}; fitting fresh on the "
            "PJM site subset (slower, and less data per bootstrap draw)."
        )
        va_cache = VA_STATISTICAL_ANALYSIS_DIR / f"results_{quantity}_return_periods.pkl"
        results = sa.load_or_calculate_results(va_cache, va_site_arr, quantity)

    sa.confidence_limits(results, SA_RETURN_PERIODS)
    return sa.extract_confidence_medians(results, SA_RETURN_PERIODS)


def main():
    logger.info("Loading all US MT sites (unfiltered — Delaunay weights need every site)...")
    emtf_path = DATA_LOC / "EMTF"
    MT_sites = csm.load_mt_sites(emtf_path / "mt_pickle.pkl", emtf_path)

    logger.info("Loading VA PJM-filtered MT site subset from step 3...")
    pjm_indices = np.load(VA_STORM_MAXES_DIR / "va_pjm_site_indices.npy")
    va_pjm_maxB = np.load(VA_STORM_MAXES_DIR / "va_pjm_maxB.npy")
    va_pjm_maxE = np.load(VA_STORM_MAXES_DIR / "va_pjm_maxE.npy")

    magnetic_data = xr.open_dataset(DATA_LOC / "geomag_data" / "processed_geomag_data.nc")
    obs_dict = {
        site.lower(): magnetic_data.sel(site=site)
        for site in magnetic_data.site.values
        if site.lower() not in [s.lower() for s in LEAVE_OUT_SITES]
    }

    logger.info("Building VA+PJM Delaunay-weighted transmission lines...")
    df = build_va_trans_lines_pickle(MT_sites)

    # Monkeypatch calc_storm_maxes globals: calculate_maxes()/process_storm()
    # read these as free variables, only ever set (in the original script)
    # inside its own `if __name__ == "__main__":` guard.
    csm.MT_sites = MT_sites
    csm.df = df
    csm.n_trans_lines = len(df)
    csm.obs_dict = obs_dict

    storm_df = pd.read_csv(DATA_LOC / "kp_ap_indices" / "storm_periods.csv")
    storm_df["Start"] = pd.to_datetime(storm_df["Start"])
    storm_df["End"] = pd.to_datetime(storm_df["End"])

    maxV_arr = compute_va_maxV_arr(storm_df, n_va_lines=len(df))

    logger.info("Fitting return-period statistics for VA+PJM lines (V)...")
    results_v_ci = fit_v_statistics(maxV_arr)

    logger.info("Resolving PJM site-level (B/E) return-period statistics...")
    results_b_ci = fit_or_reuse_site_statistics("B", va_pjm_maxB, pjm_indices)
    results_e_ci = fit_or_reuse_site_statistics("E", va_pjm_maxE, pjm_indices)

    def event_idx(start, end):
        idx = sa.get_event_indices(storm_df, start, end)
        if len(idx) == 0:
            logger.warning(f"No storm_df row found for event window {start}-{end}")
            return [0]
        if len(idx) > 1:
            logger.warning(f"Multiple storm_df rows match {start}-{end}, using the first")
        return [idx[0]]

    events = {
        "halloween": event_idx(HALLOWEEN_START, HALLOWEEN_END),
        "gannon": event_idx(GANNON_START, GANNON_END),
        "st_patricks": event_idx(ST_PATRICKS_START, ST_PATRICKS_END),
        "hydro_quebec": event_idx(HYDRO_QUEBEC_START, HYDRO_QUEBEC_END),
    }

    va_line_ids = df["line_id"].astype(str).to_numpy()
    va_mt_names = [MT_sites[i].name for i in pjm_indices]
    va_mt_coords = np.array([(MT_sites[i].latitude, MT_sites[i].longitude) for i in pjm_indices])

    out_path = VA_STATISTICAL_ANALYSIS_DIR / "va_geomagnetic_data_return_periods.h5"
    with h5py.File(out_path, "w") as f:
        sites = f.create_group("sites")
        events_grp = f.create_group("events")
        predictions = f.create_group("predictions")

        mt_sites_grp = sites.create_group("mt_sites")
        mt_sites_grp.create_dataset("names", data=np.array(va_mt_names, dtype="S"))
        mt_sites_grp.create_dataset("coordinates", data=va_mt_coords)

        tl_grp = sites.create_group("transmission_lines")
        tl_grp.create_dataset("line_ids", data=va_line_ids.astype("S"))

        for name, idx in events.items():
            grp = events_grp.create_group(name)
            grp.create_dataset("B", data=va_pjm_maxB[:, idx].flatten())
            grp.create_dataset("E", data=va_pjm_maxE[:, idx].flatten())
            grp.create_dataset("V", data=maxV_arr[:, idx].flatten())

        for field, ci in (("B", results_b_ci), ("E", results_e_ci), ("V", results_v_ci)):
            field_grp = predictions.create_group(field)
            for year, arr in ci.items():
                field_grp.create_dataset(f"{int(year)}_year", data=arr)

        f.attrs["description"] = (
            "Virginia+PJM MT sites, VA+PJM transmission lines, real storm "
            "events, and statistical return-period predictions"
        )
        f.attrs["date_created"] = np.bytes_(datetime.now().isoformat())

    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
