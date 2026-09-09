"""
Run Virginia's economic-impact IO analysis (Ghosh value-added shock, plus
two Leontief consumption/production variants) — the piece that actually
produces $/day-style headline numbers, comparable in *kind* to the national
paper's, driven by Virginia's own fragility Monte Carlo output.

Reuses econ/scripts/econ_analysis.py's run_ghosh_value_added(),
run_leontief_consumption(), and run_leontief_consumption_production()
completely unmodified. Those read the national SAM/technology/final-demand
tables via econ_analysis.py's own (deliberately untouched) DATA_LOC —
exactly the "national tables, regional shock" design the SAM-scope decision
in VIRGINIA_ADAPTATION.md calls for. The regional shock itself comes from
econ.scripts.l_prepr_data.load_gic_results(), whose *own* DATA_LOC is
monkeypatched to Virginia's tree so it reads
scripts/run_virginia_fragility.py's output instead of the national
gic_processing_batches/ directory.

PRECISION BUG, FOUND AGAINST REAL DATA: get_confidence_intervals() does
NOT get reused here anymore -- see get_confidence_intervals_unrounded()
below for why. Its shared-code counterpart ends with `return df.round(2)`,
rounding the "mean" column (the actual value run_ghosh_value_added() etc.
read as the shock input) to 2 decimal places BEFORE it reaches the IO
model. For a large intervention (Islands A/B/F/G/H) the true difference
between baseline and islanded mean GDP-affected almost always exceeds
0.01, so this was invisible. For Island J (a single-substation cut, a
much smaller lever) it wasn't: baseline's and Island J's true mean
GDP_AGR_affected for the real Gannon storm both rounded to the identical
2-decimal value, so the (deterministic, linear) Ghosh propagation produced
bit-for-bit identical total_impact for the two topologies -- confirmed
directly: the fully-combined 3000-realization dataset (recomputed fresh
via the same xr.concat load_gic_results() does) shows real, varying
per-realization values (e.g. realization 19's GDP_AGR_affected =
0.17338990542618557, matching the raw fragility batch output exactly, not
equal to baseline's realization 19), so the data going in is correct --
the rounding was destroying real signal on the way out.

Important: econ_analysis.py's _load_national_totals() (called internally by
the Leontief functions to get rho/phi) is intentionally left untouched —
it needs the true *national* population and sector GDP as the denominator
for "Virginia's share of the national economy," which is the entire point
of driving national IO tables with a regional shock rather than building a
Virginia-specific SAM.

Does not touch econ.models.cge_data_model's IPOPT-based CGE path —
econ_analysis.py's own main() doesn't call it either; it's a separate, more
advanced model not needed for these three headline analyses. (The IPOPT
path fixes made earlier — configs/settings.py and econ_analysis.py's
hardcoded solver path — still matter if that CGE path is ever exercised,
just not required for this script.)

Untested against real data in this session. Run after
scripts/run_virginia_fragility.py.

PHASE 5 (ISLANDING_MITIGATION_PLAN.md) EXTENSION: main() takes an optional
econ_dir override so this same IO analysis can run against Phase 5's
islanded fragility output (scripts/run_virginia_islanded_fragility.py's
VA_ISLAND_A_ECON_DIR) instead of the baseline VA_ECON_DIR, both reading
lpd.load_gic_results() from and writing its four output CSVs to that same
directory. See scripts/run_virginia_islanded_econ_analysis.py, the thin
wrapper that calls this with econ_dir=VA_ISLAND_A_ECON_DIR. Default
(econ_dir=VA_ECON_DIR) reproduces the exact original baseline behavior.
The national SAM/technology/final-demand tables and _load_national_totals()
stay untouched regardless -- see the "national tables, regional shock"
reasoning above, unaffected by which regional shock (baseline or islanded)
is being driven through them.
"""

import numpy as np
import pandas as pd

import econ.scripts.l_prepr_data as lpd
from econ.scripts.econ_analysis import (
    run_ghosh_value_added,
    run_leontief_consumption,
    run_leontief_consumption_production,
)

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR


logger = setup_logger(log_file="logs/run_virginia_econ_analysis.log")


def get_confidence_intervals_unrounded(ds) -> pd.DataFrame:
    """Fork of econ/scripts/econ_analysis.py's get_confidence_intervals()
    -- same computation as its non-alpha-beta branch (Virginia never sets
    USE_ALPHA_BETA_SCENARIO, so that branch is the only one this project
    ever exercises), with the shared version's final `df.round(2)` removed.
    See this module's docstring's PRECISION BUG section for why: that
    rounding is applied to the "mean" column BEFORE it becomes the IO
    model's shock input, and for a small enough intervention two genuinely
    different mean GDP-affected values can round to the identical 2-decimal
    figure, producing identical downstream $/day output for two different
    topologies. Reimplemented here rather than editing the shared file in
    place, same convention scripts/run_virginia_gic.py already follows for
    scripts/est_gic.py's loop body."""
    results = []
    for scenario in ds.scenario.values:
        scenario_data = ds.sel(scenario=scenario)

        gdp_vars = [v for v in ds.data_vars if v.startswith("GDP_") and v.endswith("_affected")]
        for gdp_var in gdp_vars:
            sector = gdp_var.replace("GDP_", "").replace("_affected", "")
            vals = scenario_data[gdp_var].values
            results.append({
                "scenario": scenario, "variable": f"GDP_{sector}",
                "mean": np.mean(vals), "p5": np.percentile(vals, 5), "p95": np.percentile(vals, 95),
            })

        est_vars = [v for v in ds.data_vars if v.startswith("EST_") and v.endswith("_affected")]
        total_est = sum(scenario_data[v].values for v in est_vars)
        results.append({
            "scenario": scenario, "variable": "EST_TOTAL",
            "mean": np.mean(total_est), "p5": np.percentile(total_est, 5), "p95": np.percentile(total_est, 95),
        })

        pop_vals = scenario_data["mean_pop_affected"].values
        results.append({
            "scenario": scenario, "variable": "POP_AFFECTED",
            "mean": np.mean(pop_vals), "p5": np.percentile(pop_vals, 5), "p95": np.percentile(pop_vals, 95),
        })

    df = pd.DataFrame(results)
    df["ci_width"] = df["p95"] - df["p5"]
    df["uncertainty_pct"] = (df["ci_width"] / df["mean"]) * 100
    return df


def main(econ_dir=VA_ECON_DIR):
    lpd.DATA_LOC = econ_dir

    logger.info(f"Loading Virginia fragility Monte Carlo results from {econ_dir}...")
    combined_ds, combined_vuln, vuln_table = lpd.load_gic_results()

    logger.info("Computing confidence intervals across Virginia's GIC realizations (full precision, not rounded)...")
    confidence_df = get_confidence_intervals_unrounded(combined_ds)

    logger.info("Running Ghosh value-added shock (national IO tables, Virginia GDP shock)...")
    ghosh_df = run_ghosh_value_added(confidence_df)

    logger.info("Running Leontief consumption shock (Virginia's share of national demand)...")
    leontief_cons_df = run_leontief_consumption(confidence_df)

    logger.info("Running Leontief consumption + production shock...")
    leontief_cp_df = run_leontief_consumption_production(confidence_df)

    econ_dir.mkdir(parents=True, exist_ok=True)
    confidence_df.to_csv(econ_dir / "va_confidence_intervals.csv", index=False)
    ghosh_df.to_csv(econ_dir / "va_ghosh_results.csv", index=False)
    leontief_cons_df.to_csv(econ_dir / "va_leontief_consumption_results.csv", index=False)
    leontief_cp_df.to_csv(
        econ_dir / "va_leontief_consumption_production_results.csv", index=False
    )

    logger.info(f"Saved Virginia economic-impact results to {econ_dir}")


if __name__ == "__main__":
    main()
