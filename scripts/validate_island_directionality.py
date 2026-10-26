"""
Real-physics directional check on a handful of finalist island candidates,
using all twelve available storm scenarios, before committing any of them
to the expensive full GIC Monte Carlo + fragility + econ pipeline.

Background: current_flow_betweenness_centrality (an earlier scoring proxy)
has no storm direction in it -- it's an isotropic average over all current
flow directions. Real storms aren't direction-neutral: of the twelve storm
scenarios this pipeline can solve for, only three (gannon, halloween,
st_patricks) are genuinely independent real directional forcings measured
from magnetotelluric data; the nine return-period scenarios share Gannon's
polarity and vary only in magnitude. This was discovered concretely:
Island G's own members' GIC increased under real Gannon forcing but
decreased under the (non-independent) 100-year scenario, with a completely
different dominant hotspot substation between the two. An isotropic proxy
can't see this; only an actual admittance/GIC solve forced by a real
storm's E-field can.

The search now scores candidates directly against the three real storms
(see select_island_by_redistribution_sensitivity.py), so this script's
remaining job is a final sanity check: one single-iteration real solve per
finalist candidate (cheap -- ~1/N_MONTE_CARLO the cost of a full run),
reporting all twelve scenarios for continuity with the rest of the
project's results. A candidate whose real effect keeps the same sign
across gannon/halloween/st_patricks is "directionally consistent"; one
that flips sign is "directionally fragile" and shouldn't be committed to
a full run without treating that as an open question.

This does not replace the full pipeline -- it's single-iteration, not
Monte Carlo, and carries no fragility/$/day figure. It's a cheap filter to
avoid spending an overnight run on an obviously direction-fragile candidate.

Run after scripts/select_island_by_redistribution_sensitivity.py has
produced one or more candidate va_island_{id}.json files.

Usage:
    python scripts/validate_island_directionality.py --candidates g h i
"""

import argparse
import json

from configs import setup_logger
from configs.virginia_config import VA_NETWORK_DIR, get_directional_check_dirs
from scripts.real_storm_gic import (
    REAL_HISTORICAL_STORMS,
    ALL_STORM_PERIODS,
    solve_single_iteration,
    ensure_baseline_single_iteration,
    real_storm_redistribution,
)
from scripts.run_virginia_gic import load_gic_context
from scripts.run_virginia_islanded_gic import load_island
from scripts.select_island_by_redistribution_sensitivity import load_risk_weights

logger = setup_logger(log_file="logs/validate_island_directionality.log")


def evaluate_directional_consistency(per_storm: dict) -> dict:
    """True if redistribution_cost_real's sign agrees across all three
    real, independently-directional storms, False if it flips between
    them, None if no real-storm column was present."""
    real_signs = {
        p: (per_storm[p]["redistribution_cost_real"] > 0)
        for p in REAL_HISTORICAL_STORMS if p in per_storm
    }
    directionally_consistent = len(set(real_signs.values())) <= 1 if real_signs else None
    return {"directionally_consistent_across_real_storms": directionally_consistent, "real_storm_signs": real_signs}


def main(island_ids: list):
    risk_weights = load_risk_weights()
    ctx = load_gic_context()
    baseline_df = ensure_baseline_single_iteration(ctx)

    results = {}
    for island_id in island_ids:
        island = load_island(island_id)
        members = set(str(m) for m in island["members"])
        boundary_line_ids = {str(x) for x in island["boundary_line_ids"]}
        dirs = get_directional_check_dirs(island_id)
        logger.info(
            f"Island {island_id}: running single-iteration real solve with "
            f"{len(boundary_line_ids)} boundary lines cut..."
        )
        candidate_df = solve_single_iteration(ctx, boundary_line_ids, dirs["gic_dir"], dirs["gnd_gic_dir"])
        per_storm = real_storm_redistribution(baseline_df, candidate_df, members, risk_weights, storms=ALL_STORM_PERIODS)
        results[island_id] = {"per_storm": per_storm, **evaluate_directional_consistency(per_storm)}

    out_path = VA_NETWORK_DIR / f"va_directional_check_{'_'.join(island_ids)}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {out_path}")

    for island_id, r in results.items():
        logger.info(f"--- Island {island_id} ---")
        logger.info(
            f"Directionally consistent across gannon/halloween/st_patricks: "
            f"{r['directionally_consistent_across_real_storms']} "
            f"(redistribution_cost_real > 0 -- worse -- for: "
            f"{[p for p, positive in r['real_storm_signs'].items() if positive]})"
        )
        for period in ALL_STORM_PERIODS:
            if period not in r["per_storm"]:
                continue
            s = r["per_storm"][period]
            tag = "REAL" if period in REAL_HISTORICAL_STORMS else "return-period"
            logger.info(
                f"  {period:>12} ({tag:>13}): own_effect={s['own_effect']:+.4f}  "
                f"redistribution_cost_real={s['redistribution_cost_real']:+.4f}"
            )
        if r["directionally_consistent_across_real_storms"] is False:
            logger.warning(
                f"Island {island_id} is DIRECTIONALLY FRAGILE: net_score's isotropic proxy gave one "
                "verdict, but the real network-wide effect flips sign across gannon/halloween/"
                "st_patricks. Do not commit this candidate to a full Phase 4/5 run without treating "
                "that as an open question, not a settled result."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cheap real-physics directional check on finalist island candidates "
                     "(Phase 6c) -- one single-iteration admittance/GIC solve per candidate, "
                     "compared across the three real historical storms plus the nine "
                     "return-period scenarios, before committing any candidate to a full "
                     "Phase 4/5 Monte Carlo run."
    )
    parser.add_argument(
        "--candidates", type=str, nargs="+", required=True,
        help="Island id(s) to check, e.g. --candidates g h -- each must already have a "
             "va_island_{id}.json from select_island_by_redistribution_sensitivity.py.",
    )
    args = parser.parse_args()
    main(island_ids=args.candidates)
