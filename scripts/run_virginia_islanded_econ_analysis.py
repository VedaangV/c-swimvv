"""
Economic-impact IO analysis on an island's fragility Monte Carlo output.

Thin wrapper around run_virginia_econ_analysis.py's main(), pointed at a
given island's econ_dir instead of the baseline VA_ECON_DIR. Produces the
whole-Virginia "Island {id} protected" $/day figures, directly comparable
to the baseline results -- same whole-VA+PJM network, just re-solved with
the island's boundary cut.

Run after a completed run_virginia_islanded_fragility.py (same island_id).

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
