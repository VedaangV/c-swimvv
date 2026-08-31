"""
Phase 6c of ISLANDING_MITIGATION_PLAN.md: a cheap, real-physics directional
check on a handful of finalist candidates before committing any of them to
the expensive full Phase 4/5 pipeline (N_MONTE_CARLO-iteration GIC Monte
Carlo + fragility + econ).

WHY THIS EXISTS
---------------
scripts/select_island_by_redistribution_sensitivity.py's `redistribution_cost`
is deliberately cheap: it recomputes current_flow_betweenness_centrality on
the rest of the network after a candidate cut and compares it to a one-time
baseline. That betweenness computation has no storm in it at all -- it
models current flowing between every pair of nodes in proportion to
conductance, the same regardless of which direction a real storm's
geoelectric field happens to point. It is, in effect, an *average over all
possible directions*, not a measurement of what any one real storm would
do.

Real storms are not direction-neutral. `scripts/run_virginia_gic.py`'s own
data pipeline (`load_and_process_gic_data` in `scripts/est_gic.py`) makes
this precise: the three real historical events (Gannon 2024, Halloween
2003, St. Patrick's Day 2015) each carry their own independently-measured
per-line E-field, magnitude AND polarity, straight from magnetotelluric
data -- genuinely different directional forcings. The nine statistical
return-period scenarios (50-250yr) are NOT independently directional in
the same way: their per-line MAGNITUDE comes from a lognormal return-period
fit, but their per-line SIGN is deliberately forced to match Gannon's own
observed polarity (`sgn_full` in `load_and_process_gic_data`) rather than
drawn independently. So of the twelve storm scenarios this pipeline can
solve for, only three -- gannon, halloween, st_patricks -- are genuinely
independent real directional forcings; the rest share Gannon's directional
signature and vary only in overall magnitude and in the *relative*
magnitude profile across lines (their own statistical fit, not Gannon's
real one).

This was discovered empirically before it was found in the code: Island G's
own members' GIC INCREASED under the real Gannon forcing (+56.23A) but
DECREASED under the 100-year return-period forcing (-59.01A), and the
single dominant redistribution hotspot substation was completely different
between the two (`96396693` for 100yr vs `96284749` for Gannon) -- despite
100yr sharing Gannon's per-line polarity. A cheap, isotropic, direction-
blind proxy cannot see this. Only an actual admittance/GIC solve, forced by
a real storm's actual E-field, can.

THE MIDDLE GROUND
------------------
Re-running full Phase 4/5 (a multi-thousand-iteration Monte Carlo, "a good
overnight job" per run_virginia_islanded_gic.py's own docstring) for every
candidate the cheap search produces defeats the point of having a cheap
search at all. This script is the middle step: ONE single-iteration real
admittance/GIC solve per finalist candidate (start_idx=0, end_idx=1 --
~1/N_MONTE_CARLO the cost of a full run), compared against the same
single iteration's baseline (no boundary cut), across all twelve available
storm forcings -- with the three genuinely-independent real historical
events (gannon, halloween, st_patricks) as the actual directional test, and
the nine return-period scenarios reported alongside for continuity with the
rest of the project's results.

A candidate whose real network-wide effect (this script's
`redistribution_cost_real`) keeps the same sign across gannon, halloween,
AND st_patricks is "directionally consistent" -- the cheap proxy's verdict
plausibly generalizes. A candidate that flips sign between them (as Island
G did between gannon and the -- non-independent -- 100yr scenario) is
"directionally fragile": net_score's isotropic proxy said one thing, but
the real answer depends on which way the storm is actually oriented, which
cannot be known in advance (see VIRGINIA_ADAPTATION.md Section 5.6 for why
-- no forecasting capability gives regional geoelectric-field direction more
than about an hour of lead time, per LiveWire/Hu et al. 2024 and the LSTM
GIC-forecasting literature).

This does NOT replace Phase 4/5 -- a directionally-consistent verdict here
is still just three single-iteration solves, not a Monte Carlo, and carries
no fragility/economic $/day figure. It's a cheap filter: don't spend an
overnight Phase 4/5 run on a candidate this check already shows is
direction-fragile.

Run after scripts/select_island_by_redistribution_sensitivity.py has
produced one or more candidate va_island_{id}.json files.

Usage:
    python scripts/validate_island_directionality.py --candidates g h i
"""

import argparse
import json

import pandas as pd

from configs import setup_logger
from configs.virginia_config import (
    VA_NETWORK_DIR,
    VA_GIC_DIR,
    VA_GND_GIC_DIR,
    get_directional_check_dirs,
)
from scripts.est_gic import return_periods
from scripts.run_virginia_gic import main as run_gic_main
from scripts.run_virginia_islanded_gic import load_island
from scripts.select_island_by_redistribution_sensitivity import load_risk_weights

logger = setup_logger(log_file="logs/validate_island_directionality.log")

# The only three genuinely independent real directional forcings this
# pipeline has -- each with its own measured per-line E-field magnitude AND
# polarity (see module docstring). The real directional test.
REAL_HISTORICAL_STORMS = ["gannon", "halloween", "st_patricks"]

# Magnitude-only variation on Gannon's own polarity (see module docstring) --
# reported for continuity with the rest of the project's 10-scenario econ
# results, but NOT treated as an independent directional data point.
RETURN_PERIOD_STORMS = [str(p) for p in return_periods]

ALL_STORM_PERIODS = REAL_HISTORICAL_STORMS + RETURN_PERIOD_STORMS


def ensure_baseline_single_iteration() -> pd.DataFrame:
    """Iteration 0 of the unmodified-topology baseline. Reuses
    VA_GND_GIC_DIR/ground_gic0.csv if the full baseline Monte Carlo has
    already produced it (iteration 0 of a completed run is identical to
    iteration 0 of a fresh single-iteration run -- same seed_base+0 draw --
    so there's no reason to recompute it); otherwise runs just that one
    iteration."""
    path = VA_GND_GIC_DIR / "ground_gic0.csv"
    if not path.exists():
        logger.info(f"{path} not found -- running a single baseline iteration.")
        run_gic_main(start_idx=0, end_idx=1, gic_dir=VA_GIC_DIR, gnd_gic_dir=VA_GND_GIC_DIR)
    else:
        logger.info(f"Reusing existing baseline iteration 0: {path}")
    df = pd.read_csv(path)
    df["Substation"] = df["Substation"].astype(str)
    return df


def run_candidate_single_iteration(island_id: str) -> pd.DataFrame:
    """One real admittance/GIC solve (iteration 0 only) with this
    candidate's boundary cut -- same seed_base+0 draw as the baseline, so
    any difference between the two is attributable to the topology change,
    not random noise."""
    island = load_island(island_id)
    boundary_line_ids = {str(x) for x in island["boundary_line_ids"]}
    dirs = get_directional_check_dirs(island_id)
    logger.info(
        f"Island {island_id}: running single-iteration real solve with "
        f"{len(boundary_line_ids)} boundary lines cut..."
    )
    run_gic_main(
        start_idx=0, end_idx=1, exclude_line_ids=boundary_line_ids,
        gic_dir=dirs["gic_dir"], gnd_gic_dir=dirs["gnd_gic_dir"],
    )
    df = pd.read_csv(dirs["gnd_gic_dir"] / "ground_gic0.csv")
    df["Substation"] = df["Substation"].astype(str)
    return df


def compare_real_storms(
    baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, members: set, risk_weights: dict,
) -> dict:
    """For every available storm period, compute the same two quantities
    evaluate_cut() computes with the cheap proxy -- own_effect (change in
    |GIC| summed over the island's own members, risk-weighted) and
    redistribution_cost_real (the same, summed over the REST of the
    network) -- but from an actual admittance/GIC solve forced by that
    storm's real E-field, not from current-flow betweenness. Absolute
    value is used (not signed GIC) because it's |GIC| that damages
    transformers in the fragility model this project's $/day figures come
    from -- a redistribution that flips a line's current direction without
    changing its magnitude is not a benefit."""
    merged = baseline_df.merge(candidate_df, on="Substation", how="inner", suffixes=("_before", "_after"))
    dropped = (len(baseline_df) - len(merged), len(candidate_df) - len(merged))
    if dropped != (0, 0):
        logger.warning(
            f"Inner join on Substation dropped {dropped[0]} baseline / {dropped[1]} candidate rows "
            "(likely nodes isolated by the boundary cut, or by keep_nodes_with_ground's non-zero-row "
            "filtering) -- excluded from both own_effect and redistribution_cost_real below."
        )

    is_member = merged["Substation"].isin(members)
    rest = merged.loc[~is_member]
    own = merged.loc[is_member]

    per_storm = {}
    for period in ALL_STORM_PERIODS:
        col = f"GIC_{period}"
        before_col, after_col = f"{col}_before", f"{col}_after"
        if before_col not in merged.columns or after_col not in merged.columns:
            logger.warning(f"Missing {col} in single-iteration output -- skipping period {period}.")
            continue

        own_delta = (own[after_col].abs() - own[before_col].abs())
        rest_delta = (rest[after_col].abs() - rest[before_col].abs())
        own_effect = float((own_delta * own["Substation"].map(risk_weights).fillna(0.0)).sum())
        redistribution_cost_real = float((rest_delta * rest["Substation"].map(risk_weights).fillna(0.0)).sum())

        per_storm[period] = {
            "own_effect": own_effect,
            "redistribution_cost_real": redistribution_cost_real,
            "is_real_historical": period in REAL_HISTORICAL_STORMS,
        }

    real_signs = {
        p: (per_storm[p]["redistribution_cost_real"] > 0)
        for p in REAL_HISTORICAL_STORMS if p in per_storm
    }
    directionally_consistent = len(set(real_signs.values())) <= 1 if real_signs else None

    return {
        "per_storm": per_storm,
        "directionally_consistent_across_real_storms": directionally_consistent,
        "real_storm_signs": real_signs,
    }


def main(island_ids: list):
    risk_weights = load_risk_weights()
    baseline_df = ensure_baseline_single_iteration()

    results = {}
    for island_id in island_ids:
        island = load_island(island_id)
        members = set(str(m) for m in island["members"])
        candidate_df = run_candidate_single_iteration(island_id)
        results[island_id] = compare_real_storms(baseline_df, candidate_df, members, risk_weights)

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
            tag = "REAL" if s["is_real_historical"] else "return-period"
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
