"""
Shared real-storm, single-iteration GIC scoring -- the actual-physics
replacement for current_flow_betweenness_centrality's direction-blind proxy
(see VIRGINIA_ADAPTATION.md 5.5-5.6 for why that proxy misses a storm's
directional forcing; Island G's members' GIC flipped sign between the real
Gannon storm and the 100-year scenario, per ISLANDING_MITIGATION_PLAN.md
5.4/Phase 6c-6d).

Used by validate_island_directionality.py (post-hoc filter on one finalist)
and select_island_by_redistribution_sensitivity.py (in-loop scoring,
replacing current_flow_betweenness_centrality inside evaluate_cut()).

REAL_HISTORICAL_STORMS = gannon, halloween, st_patricks -- the only three
scenarios with independently measured E-field polarity (the nine
return-period scenarios just scale Gannon's polarity in magnitude, so they
aren't independent directional evidence; see est_gic.py's
load_and_process_gic_data).

Scored as the MAX redistribution cost across the three storms, not the
mean -- the goal is "positive for all cases," and an average can hide one
bad direction.

Cost note: one single-iteration solve already computes all twelve storm
columns at once (run_monte_carlo() in run_virginia_gic.py), so scoring
three real storms costs the same as scoring one -- but that solve still
isn't free (~20-40s observed), so a search over many candidates will be
slower than the old cheap betweenness proxy. Lower --n-seed-candidates
and/or --max-size if a full search is too slow.
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
    (run_virginia_gic.load_gic_context()). Returns the ground-GIC
    dataframe (Substation + GIC_<period> columns, all twelve periods).

    Deletes any existing iteration-0 output first: run_monte_carlo() skips
    re-solving if output already exists (right for resuming an
    interrupted run, wrong here since gic_dir/gnd_gic_dir gets reused
    across many different exclude_line_ids during a search).

    Imports run_virginia_gic lazily to avoid a circular import."""
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
    caller that needs one. Reuses VA_GND_GIC_DIR/ground_gic0.csv if the
    full baseline Monte Carlo already produced it (iteration 0 of a
    completed run is identical to a fresh single-iteration run), else
    runs just that one iteration. Unlike solve_single_iteration, does NOT
    delete/overwrite -- this is the one case where an existing file is
    always correct to reuse, never stale."""
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
    change (magnitude drives fragility, not direction) summed over the
    island's members (own_effect) or the rest of the network
    (redistribution_cost_real), risk-weighted the same way the topology
    proxy was. Same math as validate_island_directionality.py's
    compare_real_storms, shared here between both callers."""
    merged = baseline_df.merge(candidate_df, on="Substation", how="inner", suffixes=("_before", "_after"))
    dropped = (len(baseline_df) - len(merged), len(candidate_df) - len(merged))
    if dropped != (0, 0):
        logger.warning(
            f"Inner join on Substation dropped {dropped[0]} baseline / {dropped[1]} candidate rows "
            "(likely isolated by the cut, or by keep_nodes_with_ground filtering) -- excluded below."
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


def load_fragility_exposure() -> pd.DataFrame:
    """va_substation_exposure.gpkg (POP20 + the ten GDP_*/EST_* sector
    columns per substation, built by run_virginia_areal_interpolation.py)
    -- the same table run_virginia_fragility.py feeds into the real
    fragility Monte Carlo. Loaded once per search run, not per candidate.

    Virginia-only by construction, so fragility_weighted_redistribution
    below only scores substations with real economic exposure -- closer
    to the actual whole-VA $/day objective than real_storm_redistribution's
    PJM-wide raw-|GIC| sum."""
    import geopandas as gpd
    from configs.virginia_config import VA_ECON_DIR

    path = VA_ECON_DIR / "va_substation_exposure.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_virginia_areal_interpolation.py first.")
    exposure = gpd.read_file(path)
    exposure["sub_id"] = exposure["sub_id"].astype(str)
    return exposure


# Much smaller than the real fragility Monte Carlo's defaults (20000
# iterations, 2000 batch, early-stop checking from draw 5000) -- trades
# precision for speed since this runs twice per storm per candidate.
# max_iter/batch both get set to `draws`, so _simulate_scenario does
# exactly one batch and returns; keep draws under 5000 so its convergence
# check (which starts at i>=5000) never kicks in and `tol` stays inert.
# Callers needing less noise (e.g. Phase 6i's hard filter) can raise
# `draws`, but should stay under 5000.
CHEAP_FRAGILITY_DRAWS = 300


def fragility_weighted_per_substation(
    gic_by_sub: pd.Series, exposure: pd.DataFrame, draws: int = CHEAP_FRAGILITY_DRAWS,
) -> pd.DataFrame:
    """Runs p_gic_files.py's own fragility function (_simulate_scenario,
    the real econ pipeline's lognormal Pf model) at a small draw count,
    reused directly rather than reimplemented so this can't drift out of
    sync with the real model. Returns each substation's failure_prob
    times its total GDP exposure (summed across the ten GDP_* sectors) --
    an "expected $ GDP-affected" scalar, cheap enough for a search loop.

    gic_by_sub: Series of raw GIC values indexed by sub_id (whatever the
    index is actually named, e.g. "Substation")."""
    import econ.preprocess.p_gic_files as pgf
    from configs import GDP_COLUMNS, get_simulation_config

    gic_frame = gic_by_sub.rename("_gic_val").reset_index()
    gic_frame.columns = ["sub_id", "_gic_val"]
    merged = exposure.merge(gic_frame, on="sub_id", how="inner")
    merged["latitude"] = 0.0
    merged["longitude"] = 0.0

    config = get_simulation_config()
    _, res, tbl = pgf._simulate_scenario(
        "_gic_val", merged, theta0=config["theta0"], tol=config["tolerance"],
        max_iter=draws, batch=draws,
    )
    if tbl is None:
        return pd.DataFrame(columns=["sub_id", "expected_gdp_affected"])

    gdp_total = merged.set_index("sub_id")[GDP_COLUMNS].sum(axis=1)
    tbl = tbl.set_index("sub_id")
    tbl["expected_gdp_affected"] = tbl["failure_prob"] * gdp_total.reindex(tbl.index)
    return tbl[["expected_gdp_affected"]].reset_index()


def fragility_weighted_redistribution(
    baseline_df: pd.DataFrame, candidate_df: pd.DataFrame, members: set, exposure: pd.DataFrame,
    storms: list = REAL_HISTORICAL_STORMS, draws: int = CHEAP_FRAGILITY_DRAWS,
) -> dict:
    """Fragility-weighted analog of real_storm_redistribution() above --
    same own_effect/redistribution_cost_real shape, but each substation's
    contribution is its fragility-model expected-GDP-affected delta
    instead of a raw |GIC| delta. Closes a real gap: raw |GIC| agreement
    doesn't guarantee the actual fragility-priced $/day outcome agrees
    (Island H's Halloween result had a favorable |GIC| sign but a
    negative real $/day, which motivated this).

    Phase 6i's hard real-storm filter passes a higher `draws` than the
    default to tighten Monte Carlo noise, since it needs to be trustworthy
    enough to hard-reject candidates on."""
    per_storm = {}
    for period in storms:
        before_col, after_col = f"GIC_{period}", f"GIC_{period}"
        if before_col not in baseline_df.columns or after_col not in candidate_df.columns:
            continue

        before_gic = baseline_df.set_index("Substation")[before_col]
        after_gic = candidate_df.set_index("Substation")[after_col]

        before_tbl = fragility_weighted_per_substation(before_gic, exposure, draws=draws)
        after_tbl = fragility_weighted_per_substation(after_gic, exposure, draws=draws)
        merged = before_tbl.merge(after_tbl, on="sub_id", how="outer", suffixes=("_before", "_after")).fillna(0.0)

        is_member = merged["sub_id"].isin(members)
        delta = merged["expected_gdp_affected_after"] - merged["expected_gdp_affected_before"]
        per_storm[period] = {
            "own_effect": float(delta[is_member].sum()),
            "redistribution_cost_real": float(delta[~is_member].sum()),
        }
    return per_storm


def worst_case_redistribution_cost(per_storm: dict) -> float:
    """max over the real storms' redistribution_cost_real (see module
    docstring). 0.0 if per_storm is empty -- shouldn't happen against real
    data, but scoring as "no worse than baseline" beats crashing."""
    if not per_storm:
        return 0.0
    return max(v["redistribution_cost_real"] for v in per_storm.values())
