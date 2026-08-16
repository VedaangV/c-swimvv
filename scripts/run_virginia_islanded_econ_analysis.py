"""
Phase 5 of ISLANDING_MITIGATION_PLAN.md (part 2 of 2): economic-impact IO
analysis on an island's fragility Monte Carlo output.

Thin wrapper around scripts/run_virginia_econ_analysis.py's now-
parameterized main() (econ_dir) -- reads/writes that island's econ dir
instead of the baseline VA_ECON_DIR. Same national IO tables, same
"regional shock through national tables" design as the baseline -- only
the regional shock itself differs (this island's islanded-topology
fragility output instead of the unmodified baseline's).

Takes an island_id ("a", "b", ...; default "a") -- reads/writes
configs.virginia_config.get_island_dirs(island_id)'s econ_dir, same
generalization as scripts/run_virginia_islanded_gic.py/run_virginia_
islanded_fragility.py.

Fast relative to the fragility Monte Carlo (IO matrix multiplication, not
a realization-by-realization simulation) -- run this right after
scripts/run_virginia_islanded_fragility.py finishes, no need to wait for
another overnight run.

Produces the whole-Virginia "Island {id} protected" $/day figures, directly
comparable to the existing baseline va_ghosh_results.csv / va_leontief_
*_results.csv (the $145.5M/day (100yr) baseline referenced throughout the
plan doc) -- same whole-VA+PJM network re-solved with the island's boundary
cut, not an island-only figure.

Run after a completed scripts/run_virginia_islanded_fragility.py (same
island_id).

Usage:
    python scripts/run_virginia_islanded_econ_analysis.py [--island b]
"""

import argparse

from configs import setup_logger
from configs.virginia_config import get_island_dirs
from scripts.run_virginia_econ_analysis import main as run_econ_main

logger = setup_logger(log_file="logs/run_virginia_islanded_econ_analysis.log")


def main(island_id: str = "a"):
    econ_dir = get_island_dirs(island_id)["econ_dir"]
    logger.info(f"Running Island {island_id} economic-impact analysis: {econ_dir}")
    run_econ_main(econ_dir=econ_dir)
    logger.info(f"Island {island_id} economic-impact analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Islanded-topology Virginia economic-impact analysis.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'b'.")
    args = parser.parse_args()
    main(island_id=args.island)
