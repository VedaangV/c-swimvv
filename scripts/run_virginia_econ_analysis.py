"""
Run Virginia's economic-impact IO analysis: a Ghosh value-added shock plus
two Leontief consumption/production variants, driven by Virginia's own
fragility Monte Carlo output against the national SAM/IO tables.

Reuses econ_analysis.py's run_ghosh_value_added(), run_leontief_consumption(),
and run_leontief_consumption_production() unmodified; the regional shock
comes from l_prepr_data.load_gic_results() with DATA_LOC monkeypatched to
Virginia's tree.

Note: get_confidence_intervals_unrounded() below exists because the shared
get_confidence_intervals() rounds its "mean" column to 2 decimals before it
hits the IO model -- fine for large interventions, but it silently made
Island J's shock identical to baseline's (both rounded the same), producing
identical total_impact for two different topologies. Don't swap this back
to the shared rounded version.

main() takes an optional econ_dir override so the same analysis can run
against islanded fragility output (see run_virginia_islanded_econ_analysis.py);
default reproduces the original baseline behavior.
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
    """Fork of econ_analysis.py's get_confidence_intervals() (non-alpha-beta
    branch only, the only one Virginia exercises) with the final
    `df.round(2)` dropped -- see module docstring for why."""
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
