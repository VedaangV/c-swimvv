"""
Phase 5 of ISLANDING_MITIGATION_PLAN.md (part 2 of 2): economic-impact IO
analysis on Island A's fragility Monte Carlo output.

Thin wrapper around scripts/run_virginia_econ_analysis.py's now-
parameterized main() (econ_dir) -- reads/writes VA_ISLAND_A_ECON_DIR
instead of the baseline VA_ECON_DIR. Same national IO tables, same "regional
shock through national tables" design as the baseline -- only the regional
shock itself differs (Island A's islanded-topology fragility output instead
of the unmodified baseline's).

Fast relative to the fragility Monte Carlo (IO matrix multiplication, not
a realization-by-realization simulation) -- run this right after
scripts/run_virginia_islanded_fragility.py finishes, no need to wait for
another overnight run.

Produces the whole-Virginia "Island A protected" $/day figures, directly
comparable to the existing baseline va_ghosh_results.csv / va_leontief_
*_results.csv (the $145.5M/day (100yr) baseline referenced throughout the
plan doc) -- same whole-VA+PJM network re-solved with Island A's boundary
cut, not an island-only figure. See ISLANDING_MITIGATION_PLAN.md's Phase 5
section for what still needs to be computed on top of this output (the
switching-cost and island's-own-temporary-outage-cost benefit-cost
accounting -- not yet built, deliberately: needs these real numbers first).

Run after a completed scripts/run_virginia_islanded_fragility.py.

Usage:
    python scripts/run_virginia_islanded_econ_analysis.py
"""

from configs import setup_logger
from configs.virginia_config import VA_ISLAND_A_ECON_DIR
from scripts.run_virginia_econ_analysis import main as run_econ_main

logger = setup_logger(log_file="logs/run_virginia_islanded_econ_analysis.log")


def main():
    logger.info(f"Running Island A economic-impact analysis: {VA_ISLAND_A_ECON_DIR}")
    run_econ_main(econ_dir=VA_ISLAND_A_ECON_DIR)
    logger.info("Island A economic-impact analysis complete.")


if __name__ == "__main__":
    main()
