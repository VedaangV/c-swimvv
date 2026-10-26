"""
Systematic search for more individually-positive islands, now that A+B
combined showed island effects are roughly additive -- stacking a
net-negative island onto a net-positive one erased the win rather than
averaging out. So the path forward is finding more individually-positive
candidates, not combining indiscriminately.

For each new island id in --islands (e.g. c d e), in order, runs the full
pipeline: select (excluding prior islands' members + boundary neighbors,
plus any just picked this run), re-solve GIC, fragility Monte Carlo, econ
analysis, then the diagnostic comparisons. Sequential and one candidate
fully through before the next, since each stage is itself resumable, so
an interrupted overnight run still leaves finished candidates usable.

Usage:
    python scripts/search_candidate_islands.py --islands c d e
    python scripts/search_candidate_islands.py --islands c d e --exclude a b
"""

import argparse
import time

from configs import setup_logger
import scripts.deprecated_sci_islanding.select_virginia_island as sel
import scripts.run_virginia_islanded_gic as gic
import scripts.run_virginia_islanded_fragility as frag
import scripts.run_virginia_islanded_econ_analysis as econ
import scripts.compare_island_gic as cmp_island
import scripts.compare_virginia_gic as cmp_va

logger = setup_logger(log_file="logs/search_candidate_islands.log")


def main(new_island_ids: list, base_exclude: list):
    already_excluded = list(base_exclude)
    completed = []
    failed = []

    for island_id in new_island_ids:
        t0 = time.time()
        try:
            logger.info(f"===== Candidate {island_id}: selecting (excluding {already_excluded}) =====")
            sel.main(island_id=island_id, exclude_island_ids=already_excluded)
            already_excluded = already_excluded + [island_id]

            logger.info(f"===== Candidate {island_id}: GIC Monte Carlo =====")
            gic.main(island_id=island_id)

            logger.info(f"===== Candidate {island_id}: fragility Monte Carlo =====")
            frag.main(island_id=island_id)

            logger.info(f"===== Candidate {island_id}: econ analysis =====")
            econ.main(island_id=island_id)

            logger.info(f"===== Candidate {island_id}: comparison diagnostics =====")
            cmp_island.main(island_id=island_id)
            cmp_va.main(island_id=island_id)

            elapsed = time.time() - t0
            logger.info(f"===== Candidate {island_id} COMPLETE in {elapsed/60:.1f} minutes =====")
            completed.append(island_id)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"===== Candidate {island_id} FAILED after {elapsed/60:.1f} minutes: {e!r} ====="
            )
            logger.error(
                f"Stopping here -- {island_id} is already in the exclusion list, so re-running "
                f"with the remaining islands won't re-select something overlapping it. "
                f"Completed so far: {completed}."
            )
            failed.append(island_id)
            break

    logger.info(f"Search finished. Completed: {completed}. Failed/stopped at: {failed}.")
    logger.info(
        "For each completed candidate, check data/virginia/econ/island_{id}/va_ghosh_results.csv "
        "against the baseline and against Island A/B's own results to see which candidates are "
        "individually positive for realistic-severity scenarios."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequentially select and fully evaluate multiple new candidate islands."
    )
    parser.add_argument("--islands", type=str, nargs="+", required=True, help="New island ids to generate, e.g. --islands c d e")
    parser.add_argument("--exclude", type=str, nargs="+", default=["a", "b"], help="Already-selected island ids to exclude footprints of.")
    args = parser.parse_args()
    main(new_island_ids=args.islands, base_exclude=args.exclude)
