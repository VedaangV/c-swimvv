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

PHASE 5 (ISLANDING_MITIGATION_PLAN.md) EXTENSION: main() takes optional
gnd_gic_dir/econ_dir overrides so the same fragility Monte Carlo can run
against Phase 4's islanded GIC output instead of the baseline, writing to
a separate directory. See scripts/run_virginia_islanded_fragility.py, the
thin wrapper that calls this with VA_ISLAND_A_GND_GIC_DIR/
VA_ISLAND_A_ECON_DIR. Defaults (gnd_gic_dir=VA_GND_GIC_DIR, econ_dir=
VA_ECON_DIR) reproduce the exact original baseline behavior -- econ_dir
drives both pgf.DATA_LOC (checkpoints + final output) and the exposure
gpkg lookup, so pointing it at VA_ISLAND_A_ECON_DIR isolates both without
needing separate output filenames.

PHASE 6c/6d EXTENSION: SCENARIOS is also monkeypatched, adding Halloween
and St. Patrick's Day to the ten scenarios (nine return periods + Gannon)
this pipeline processed previously. Nothing else needed changing --
econ/preprocess/p_gic_files.py's process_gic_file() already has a rename
map for GIC_halloween/GIC_st_patricks (it was only ever the SCENARIOS
list gating which columns got processed), and every downstream script
(econ_analysis.py, compare_island_econ.py, compute_island_benefit_cost.py)
discovers scenarios dynamically from the data rather than a hardcoded
list. This closes the gap flagged since Phase 6c: Halloween and St.
Patrick's Day had real GIC data (scripts/run_virginia_gic.py always
computed GIC_halloween/GIC_st_patricks columns) but no real $/day figures,
because the fragility stage silently dropped those two columns.

IMPORTANT -- CANNOT BE RESUMED FROM EXISTING CHECKPOINTS. process_all_
files() resumes purely by counting existing batch files
(`gic_processing_batches/summary_batch_*.nc` under econ_dir) and picking
up from there -- it has no way to know those existing batches were
computed against the OLD ten-scenario SCENARIOS list. Resuming after this
change would silently combine early batches missing Halloween/St.
Patrick's with later batches that have them, corrupting
va_scenario_summary_gnd_gic.nc. Any econ_dir whose fragility Monte Carlo
already completed under the old SCENARIOS list (the baseline VA_ECON_DIR,
and any island's econ_dir that already has real results -- e.g. Island
H's) needs its `gic_processing_batches`/`gnd_gic_processing_batches`
directory cleared before rerunning this script, not resumed.
"""

import argparse

import geopandas as gpd

import econ.preprocess.p_gic_files as pgf
from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_GND_GIC_DIR, VA_DATA_DIR

logger = setup_logger(log_file="logs/run_virginia_fragility.log")

# Adds Halloween/St. Patrick's Day to the ten scenarios (nine return
# periods + Gannon) already in econ/preprocess/p_gic_files.py's
# REGULAR_SCENARIOS (via configs.settings.get_scenarios()) -- see PHASE
# 6c/6d EXTENSION above. Column names must exactly match process_gic_
# file()'s gic_rename_map values.
EXTRA_SCENARIOS = ["e_halloween-year-hazard A/ph", "e_st_patricks-year-hazard A/ph"]


def main(limit: int | None = None, gnd_gic_dir=VA_GND_GIC_DIR, econ_dir=VA_ECON_DIR):
    # Always the baseline VA_ECON_DIR, deliberately not econ_dir: exposure
    # (where population/GDP/facilities sit relative to each substation) is
    # topology-independent -- cutting Island A's boundary lines doesn't move
    # anyone's house, so there's exactly one exposure file, shared by both
    # the baseline and islanded runs, not duplicated per-island.
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

    # Redirect the module-level path constants process_gic_file()/
    # process_all_files() read as globals — see module docstring.
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
            "afterward) before running this script again. See module docstring's PHASE 6c/6d EXTENSION."
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
