"""
Fragility Monte Carlo: convert each Virginia GIC Monte Carlo realization
into per-substation transformer failure probabilities and localized
economic exposure (population/GDP/establishments affected), for every
storm return-period scenario.

Reuses p_gic_files.py's process_gic_file()/build_merged()/
_simulate_scenario()/process_all_files() unmodified, with DATA_DIR/DATA_LOC
redirected to Virginia's paths. Forced to workers=1 (sequential): the
default ProcessPoolExecutor path uses Windows spawn, which would re-import
p_gic_files fresh in each worker without our monkeypatches and silently
corrupt lat/lon in the output. Slower, but correct.

Heavy computation (up to 3000 realizations x 9 scenarios x up to 20000
inner draws each) -- expect it to run at least as long as the GIC Monte
Carlo. Checkpoints in batches of 50, resumable.

main() takes optional gnd_gic_dir/econ_dir overrides so the same pipeline
can run against islanded GIC output (see run_virginia_islanded_fragility.py)
instead of baseline; defaults reproduce the original baseline behavior.

SCENARIOS is also monkeypatched to add Halloween/St. Patrick's Day to the
ten scenarios this pipeline used to process.

IMPORTANT: this CANNOT be resumed from checkpoints computed under the old
SCENARIOS list (missing Halloween/St. Patrick's) -- doing so silently mixes
old and new batches and corrupts va_scenario_summary_gnd_gic.nc. Clear
`gic_processing_batches`/`gnd_gic_processing_batches` under any econ_dir
that already ran under the old scenario list before rerunning this.

Run after scripts/run_virginia_areal_interpolation.py and a completed
scripts/run_virginia_gic.py.
"""

import argparse

import geopandas as gpd

import econ.preprocess.p_gic_files as pgf
from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_GND_GIC_DIR, VA_DATA_DIR

logger = setup_logger(log_file="logs/run_virginia_fragility.log")

# Adds Halloween/St. Patrick's Day to the existing nine return periods +
# Gannon. Names must match process_gic_file()'s gic_rename_map values.
EXTRA_SCENARIOS = ["e_halloween-year-hazard A/ph", "e_st_patricks-year-hazard A/ph"]


def main(limit: int | None = None, gnd_gic_dir=VA_GND_GIC_DIR, econ_dir=VA_ECON_DIR):
    # Always baseline VA_ECON_DIR, not econ_dir: exposure is topology-independent
    # (cutting island boundary lines doesn't move anyone's house), so it's
    # shared across baseline and islanded runs rather than duplicated.
    exposure_path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    if not exposure_path.exists():
        raise FileNotFoundError(
            f"Missing {exposure_path}. Run scripts/run_virginia_areal_interpolation.py first."
        )
    aggregate_gdf = gpd.read_file(exposure_path)
    aggregate_gdf["sub_id"] = aggregate_gdf["sub_id"].astype(str)

    gic_files = sorted(gnd_gic_dir.glob("ground_gic*.csv"))
    if not gic_files:
        raise FileNotFoundError(
            f"No ground_gic*.csv files under {gnd_gic_dir}. Run scripts/run_virginia_gic.py "
            "(or scripts/run_virginia_islanded_gic.py) first."
        )
    if limit:
        gic_files = gic_files[:limit]
    logger.info(f"Processing {len(gic_files)} Virginia ground-GIC realizations from {gnd_gic_dir}")

    econ_dir.mkdir(parents=True, exist_ok=True)

    # process_gic_file()/process_all_files() read these as module globals
    pgf.DATA_DIR = VA_DATA_DIR
    pgf.DATA_LOC = econ_dir
    pgf.OUTPUT_FILES = {
        **pgf.OUTPUT_FILES,
        "regular_all": "va_scenario_summary_gnd_gic.nc",
        "vulnerable_regular": "va_vulnerable_substations_gnd_gic",
    }
    pgf.SCENARIOS = pgf.SCENARIOS + [s for s in EXTRA_SCENARIOS if s not in pgf.SCENARIOS]

    batch_dir = econ_dir / (
        "gnd_gic_processing_batches" if pgf.PROCESS_GND_FILES else "gic_processing_batches"
    )
    if batch_dir.exists() and any(batch_dir.iterdir()):
        raise RuntimeError(
            f"{batch_dir} has existing checkpoint batches from a previous run -- resuming would "
            "silently mix batches computed under the old SCENARIOS list (missing Halloween/St. "
            "Patrick's) with new ones that have them, corrupting the output. Clear this directory "
            "(and re-run scripts/run_virginia_econ_analysis.py / run_virginia_islanded_econ_analysis.py "
            "afterward) before running this script again. See module docstring."
        )

    pgf.process_all_files(gic_files, aggregate_gdf, workers=1)

    logger.info(f"Saved {econ_dir / 'va_scenario_summary_gnd_gic.nc'}")
    logger.info(f"Saved {econ_dir / 'va_vulnerable_substations_gnd_gic'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virginia fragility Monte Carlo.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N ground-GIC files (for a timing sample).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
