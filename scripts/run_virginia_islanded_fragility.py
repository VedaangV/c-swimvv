"""
Fragility Monte Carlo on an island's islanded GIC output.

Thin wrapper around run_virginia_fragility.py's main(), pointed at an
island's ground-GIC realizations and econ dir instead of the baseline.
Everything else (exposure data, model, sequential execution) is identical
to the baseline run.

Heavy step -- expect it to take as long as or longer than the GIC Monte
Carlo, a good overnight job. Checkpoints in batches of 50, resumable.

Run after a completed run_virginia_islanded_gic.py (same island_id).

Usage:
    python scripts/run_virginia_islanded_fragility.py [--island b] [--limit N]
"""

import argparse

from configs import setup_logger
from configs.virginia_config import get_island_dirs
from scripts.run_virginia_fragility import main as run_fragility_main

logger = setup_logger(log_file="logs/run_virginia_islanded_fragility.log")


def main(limit: int | None = None, island_id: str = "a"):
    dirs = get_island_dirs(island_id)
    logger.info(
        f"Running Island {island_id} fragility Monte Carlo: input={dirs['gnd_gic_dir']}, "
        f"output={dirs['econ_dir']}"
    )
    run_fragility_main(limit=limit, gnd_gic_dir=dirs["gnd_gic_dir"], econ_dir=dirs["econ_dir"])
    logger.info(f"Island {island_id} fragility Monte Carlo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Islanded-topology Virginia fragility Monte Carlo.")
    parser.add_argument("--island", type=str, default="a", help="Island id, e.g. 'a' or 'b'.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N ground-GIC files (for a timing sample).",
    )
    args = parser.parse_args()
    main(limit=args.limit, island_id=args.island)
