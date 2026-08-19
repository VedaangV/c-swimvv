"""
Phase 6 extension: systematic search for more individually-positive
islands (like Island B was for the real Gannon storm), now that the A+B
combined run confirmed island effects are ~additive (see
ISLANDING_MITIGATION_PLAN.md's Phase 6 section -- AB's diff matched A's
diff + B's diff to within ~0.5 percentage points across all 30 scenarios).
That kills "protect more regions to be safe" as a strategy -- stacking a
net-negative island (A) onto a net-positive one (B) didn't average out,
it erased B's win. The path to a consistently-positive whole-Virginia
result is finding MORE individually-positive candidates and combining
only those (which additivity says should compound favorably), not
combining indiscriminately.

For each new island id in --islands (e.g. c d e), in order, fully:
1. Select it (scripts/select_virginia_island.py), excluding every
   previously-selected island's members + boundary neighbors -- both
   already on disk (--exclude) AND the ones this run itself just picked,
   so every candidate stays mutually non-adjacent. Keeps the additivity
   property clean if a promising combination gets tested later, and
   avoids one candidate accidentally growing into another's territory.
2. Re-solve GIC with its boundary removed (scripts/run_virginia_islanded_
   gic.py) -- heavy, ~as long as the original baseline Monte Carlo.
3. Fragility Monte Carlo (scripts/run_virginia_islanded_fragility.py) --
   also heavy, comparable weight.
4. Econ IO analysis (scripts/run_virginia_islanded_econ_analysis.py) --
   fast.
5. Both diagnostic comparisons (scripts/compare_island_gic.py, scripts/
   compare_virginia_gic.py) -- fast, so morning-after results are
   immediately readable (local protection + redistribution hotspots)
   without any further manual steps.

Sequential, one candidate fully through before starting the next -- each
stage is itself checkpointed/resumable (the fragility Monte Carlo in
particular), so an interruption partway through an overnight run still
leaves whichever candidates finished with complete, usable results.

Usage:
    python scripts/search_candidate_islands.py --islands c d e
    python scripts/search_candidate_islands.py --islands c d e --exclude a b
"""

import argparse
import time

from configs import setup_logger
import scripts.select_virginia_island as sel
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
                f"Stopping here -- {island_id} was added to the exclusion list before failing, so "
                "re-running this script with the remaining islands (or after fixing whatever broke) "
                f"won't re-select something overlapping it. Completed so far: {completed}."
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
