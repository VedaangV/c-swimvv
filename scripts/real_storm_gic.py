"""
Shared real-storm, single-iteration GIC scoring -- the actual-physics
replacement for current_flow_betweenness_centrality's isotropic,
direction-blind proxy (see VIRGINIA_ADAPTATION.md Section 5.5-5.6 for why
that proxy can't see a real storm's directional forcing, and Section 5.4
of ISLANDING_MITIGATION_PLAN.md's Phase 6c/6d for the empirical case --
Island G's own members' GIC flipped sign between the real Gannon storm
and the (non-independently-directional) 100-year return-period scenario).

Used by two different callers, at two different points in the pipeline:
- scripts/validate_island_directionality.py -- a post-hoc filter on one
  already-selected finalist candidate.
- scripts/select_island_by_redistribution_sensitivity.py -- IN-LOOP
  scoring, replacing current_flow_betweenness_centrality inside
  evaluate_cut() so real-storm robustness drives which candidate gets
  selected in the first place, not just a check applied after the fact.

REAL_HISTORICAL_STORMS is the actual directional test: gannon, halloween,
and st_patricks are the only three scenarios in this pipeline with their
own independently-measured per-line E-field magnitude AND polarity (see
scripts/est_gic.py's load_and_process_gic_data -- the nine return-period
scenarios share Gannon's own polarity, scaled only in magnitude, so they
are not independent directional evidence).

WORST-CASE, NOT AVERAGE
-------------------------
A candidate's redistribution cost is scored as the MAX (worst) of its
real redistribution cost across the three real storms, not their mean.
The goal stated directly by the person running this project was "a method
that is positive for all cases" -- an average can be small while one
storm direction is quietly terrible; a max can't hide that. A candidate
only scores well here if it doesn't hurt badly under ANY of the three
real directions.

COST, BLUNTLY
---------------
One single-iteration real solve already computes ALL twelve storm
columns at once (one admittance-matrix factorization, many injection
vectors) -- see run_monte_carlo() in scripts/run_virginia_gic.py -- so
scoring against three real storms costs the SAME as scoring against one:
one solve per candidate, not three. That solve still isn't free (~20-40s
observed on real data, dominated by admittance-matrix construction and
the per-period GIC calculation, not by load_gic_context()'s one-time
setup). A search that evaluates dozens of candidates (n_seed_candidates
seeds, plus every boundary-adjacent candidate at every growth step) will
take proportionately longer than the old current-flow-betweenness proxy,
which was cheap enough not to think about. Lower --n-seed-candidates
and/or --max-size if a full search run is impractically slow.
"""

import pandas as pd

from configs import setup_logger
from scripts.est_gic import return_periods

logger = setup_logger(log_file="logs/real_storm_gic.log")

REAL_HISTORICAL_STORMS = ["gannon", "halloween", "st_patricks"]
RETURN_PERIOD_STORMS = [str(p) for p in return_periods]
ALL_STORM_PERIODS = REAL_HISTORICAL_STORMS + RETURN_PERIOD_STORMS


def solve_single_iteration(ctx: dict, exclude_line_ids: set, gic_dir, gnd_gic_dir) -> pd.DataFrame:
    """One real admittance/GIC solve (iteration 0 only) with
    exclude_line_ids cut, using an already-loaded context
    (scripts.run_virginia_gic.load_gic_context()). Returns the resulting
    ground-GIC dataframe (Substation + GIC_<period> columns, all twelve
    periods).

    Deletes any existing iteration-0 output in gic_dir/gnd_gic_dir before
    solving: run_monte_carlo() skips re-solving an iteration whose output
    files already exist (the right behavior for resuming an interrupted
    Monte Carlo run), but that's wrong here -- gic_dir/gnd_gic_dir is
    reused across many different exclude_line_ids during a search, and a
    stale ground_gic0.csv from a DIFFERENT candidate would otherwise be
    silently reused instead of solved fresh.

    Imports run_virginia_gic lazily to avoid a circular import
    (select_island_by_redistribution_sensitivity.py -> this module ->
    run_virginia_gic, while run_virginia_islanded_gic.py already imports
    run_virginia_gic directly)."""
    from scripts.run_virginia_gic import run_monte_carlo

    gnd_out_path = gnd_gic_dir / "ground_gic0.csv"
    winding_out_path = gic_dir / "winding_gic_rand_0.csv"
    gnd_out_path.unlink(missing_ok=True)
    winding_out_path.unlink(missing_ok=True)

    run_monte_carlo(
        ctx, start_idx=0, end_idx=1, exclude_line_ids=exclude_line_ids,
        gic_dir=gic_dir, gnd_gic_dir=gnd_gic_dir,
    )
    df = pd.read_csv(gnd_out_path)
    df["Substation"] = df["Substation"].astype(str)
    return df


def ensure_baseline_single_iteration(ctx: dict) -> pd.DataFrame:
    """Iteration 0 of the unmodified-topology baseline, reused by every
    caller that needs one (validate_island_directionality.py's one-off
    check; select_island_by_redistribution_sensitivity.py's search, once
    per search run, reusing the same ctx as every candidate evaluation).
    Reuses VA_GND_GIC_DIR/ground_gic0.csv if the full baseline Monte
    Carlo already produced it -- iteration 0 of a completed run is
    identical to a fresh single-iteration run (same seed_base+0 draw) --
    otherwise runs just that one iteration. Unlike solve_single_iteration
    above, deliberately does NOT delete/overwrite: this is the one
    unmodified-topology case, so a pre-existing file here is always
    correct to reuse, never stale."""
    from scripts.run_virginia_gic import run_monte_carlo
    from configs.virginia_config import VA_GIC_DIR, VA_GND_GIC_DIR

    path = VA_GND_GIC_DIR / "ground_gic0.csv"
    if not path.exists():
        run_monte_carlo(ctx, start_idx=0, end_idx=1, gic_dir=VA_GIC_DIR, gnd_gic_dir=VA_GND_GIC_DIR)
    df = pd.read_csv(path)
    df["Substation"] = df["Substation"].astype(str)
    return df


def real_storm_redistribution(
    baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, members: set, risk_weights: dict,
    storms: list = REAL_HISTORICAL_STORMS,
) -> dict:
    """Per-storm own_effect / redistribution_cost_real -- the real-physics
    analogs of evaluate_cut()'s benefit/redistribution_cost, using |GIC|
    change (magnitude drives transformer fragility, not direction) summed
    over the island's own members (own_effect) or the rest of the network
    (redistribution_cost_real), risk-weighted the same way the topology
    proxy was. Same math as validate_island_directionality.py's
    compare_real_storms -- kept here as the one shared implementation
    both that post-hoc filter and the in-loop search scorer use."""
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
    for period in storms:
        before_col, after_col = f"GIC_{period}_before", f"GIC_{period}_after"
        if before_col not in merged.columns or after_col not in merged.columns:
            continue
        own_delta = own[after_col].abs() - own[before_col].abs()
        rest_delta = rest[after_col].abs() - rest[before_col].abs()
        per_storm[period] = {
            "own_effect": float((own_delta * own["Substation"].map(risk_weights).fillna(0.0)).sum()),
            "redistribution_cost_real": float((rest_delta * rest["Substation"].map(risk_weights).fillna(0.0)).sum()),
        }
    return per_storm


def worst_case_redistribution_cost(per_storm: dict) -> float:
    """max over the real storms' redistribution_cost_real -- see module
    docstring's WORST-CASE, NOT AVERAGE. 0.0 if per_storm is empty (no
    real-storm columns present -- shouldn't happen against real data, but
    scoring as "no worse than baseline" rather than crashing is the right
    failure mode for a search loop)."""
    if not per_storm:
        return 0.0
    return max(v["redistribution_cost_real"] for v in per_storm.values())
