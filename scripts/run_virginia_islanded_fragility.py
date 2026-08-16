"""
Phase 5 of ISLANDING_MITIGATION_PLAN.md (part 1 of 2): fragility Monte
Carlo on Phase 4's islanded GIC output.

Thin wrapper around scripts/run_virginia_fragility.py's now-parameterized
main() (gnd_gic_dir/econ_dir) -- reads an island's ground-GIC realizations
instead of the baseline, writes to that island's econ dir instead of
VA_ECON_DIR. Everything else (the exposure gpkg, the fragility model
itself) is identical to the baseline run -- same population/GDP/facility
exposure per substation, same convergence draws, same sequential
(workers=1) execution for the same Windows-safety reason documented in
run_virginia_fragility.py's module docstring.

Takes an island_id ("a", "b", ...; default "a") -- reads/writes
configs.virginia_config.get_island_dirs(island_id)'s paths, same
generalization and reasoning as scripts/run_virginia_islanded_gic.py.

WARNING (same as the underlying script): this is the heavy step. Expect it
to take as long as or longer than the GIC Monte Carlo itself -- a good
overnight job, same as run_virginia_islanded_gic.py was. It checkpoints in
batches of 50 realizations and is resumable if interrupted.

Run after a completed scripts/run_virginia_islanded_gic.py (same island_id).

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
