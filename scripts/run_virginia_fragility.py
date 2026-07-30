"""
Fragility Monte Carlo: convert each Virginia GIC Monte Carlo realization
into per-substation transformer failure probabilities and localized
economic exposure (population/GDP/establishments affected), for every
storm return-period scenario.

Reuses econ/preprocess/p_gic_files.py's process_gic_file(), build_merged(),
_simulate_scenario(), and process_all_files() completely unmodified —
monkeypatching only two module-level path constants that function reads as
globals rather than parameters: p_gic_files.DATA_DIR (used once, to resolve
substation_info.csv for lat/lon — redirected to VA_DATA_DIR so it resolves
Virginia's own admittance_matrix/substation_info.csv) and p_gic_files.DATA_LOC
(used for batch checkpoints and final output paths — redirected to
VA_ECON_DIR so nothing here ever touches the national econ_data tree).
OUTPUT_FILES is replaced with a fresh dict (VA-specific filenames), not
mutated in place, since that dict is shared mutable state other modules
also hold a reference to.

Uses process_gic_file()'s "Substation"-column branch (the GND_GIC_FILES=True
code path in the national pipeline) since scripts/run_virginia_gic.py's
output (ground_gic{i}.csv) matches that format exactly — Virginia has no
"effective_gic_rand_*.csv" files (a further-processed national input this
adaptation never produces).

Deliberately forced to workers=1 (sequential), even though
process_all_files() defaults to parallelizing across realizations via
ProcessPoolExecutor: that uses Windows spawn, which would re-import
econ.preprocess.p_gic_files fresh in every worker without the monkeypatches
above applied — the same silent-wrong-data failure mode fixed in
scripts/recompute_virginia_line_voltages.py, but worse here since it would
corrupt latitude/longitude in the output vulnerability table rather than
crash. Sequential is much slower but correct; a real Windows-safe
parallelization would need a custom spawn initializer, not attempted here.

WARNING: this is a genuinely heavy computation — up to 3000 realizations x
9 return-period scenarios x up to 20000 inner fragility draws each (with
early-stopping convergence checks around draw 5000+). Expect this to run
at least as long as the GIC Monte Carlo itself, likely considerably longer
given it's now sequential. It checkpoints in batches of 50 realizations
(resumable — reruns pick up where they left off) so it's safe to stop and
restart.

Untested against real data in this session (no torch environment here) —
report back exactly what happens on first run, including how far it gets
before any error.

Run after scripts/run_virginia_areal_interpolation.py and a completed
scripts/run_virginia_gic.py.
"""

import argparse

import geopandas as gpd

import econ.preprocess.p_gic_files as pgf
from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_GND_GIC_DIR, VA_DATA_DIR

logger = setup_logger(log_file="logs/run_virginia_fragility.log")


def main(limit: int | None = None):
    exposure_path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    if not exposure_path.exists():
        raise FileNotFoundError(
            f"Missing {exposure_path}. Run scripts/run_virginia_areal_interpolation.py first."
        )
    aggregate_gdf = gpd.read_file(exposure_path)
    aggregate_gdf["sub_id"] = aggregate_gdf["sub_id"].astype(str)

    gic_files = sorted(VA_GND_GIC_DIR.glob("ground_gic*.csv"))
    if not gic_files:
        raise FileNotFoundError(
            f"No ground_gic*.csv files under {VA_GND_GIC_DIR}. Run scripts/run_virginia_gic.py first."
        )
    if limit:
        gic_files = gic_files[:limit]
    logger.info(f"Processing {len(gic_files)} Virginia ground-GIC realizations")

    # Redirect the module-level path constants process_gic_file()/
    # process_all_files() read as globals — see module docstring.
    pgf.DATA_DIR = VA_DATA_DIR
    pgf.DATA_LOC = VA_ECON_DIR
    pgf.OUTPUT_FILES = {
        **pgf.OUTPUT_FILES,
        "regular_all": "va_scenario_summary_gnd_gic.nc",
        "vulnerable_regular": "va_vulnerable_substations_gnd_gic",
    }

    pgf.process_all_files(gic_files, aggregate_gdf, workers=1)

    logger.info(f"Saved {VA_ECON_DIR / 'va_scenario_summary_gnd_gic.nc'}")
    logger.info(f"Saved {VA_ECON_DIR / 'va_vulnerable_substations_gnd_gic'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virginia fragility Monte Carlo.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N ground-GIC files (for a timing sample).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
