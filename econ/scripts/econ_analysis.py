"""Economic analysis module for GIC impact assessment.
Author: Dennies Bor & Edward Oughton

Runs three I-O loss models on the Monte Carlo scenario outputs from
p_gic_files.py:

    run_ghosh_value_added            Supply-driven Ghosh shock on sector VA.
    run_leontief_consumption         Demand-driven Leontief shock on PCE +
                                     government final demand only.
    run_leontief_consumption_production
                                     Demand-driven Leontief shock on PCE +
                                     government + investment + exports.

All three write CSVs with the same schema.
"""

import copy
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from pyomo.environ import value
from pyomo.opt import SolverFactory

from configs import (
    USE_ALPHA_BETA_SCENARIO,
    FIGURES_DIR,
    setup_logger,
    PROCESS_GND_FILES,
    get_data_dir,
    IPOPT_EXEC,
)

from econ.scripts.l_prepr_data import load_gic_results
from econ.models.io_model import InputOutputModel
from econ.models.cge_data_model import run_cge_example

logger = setup_logger("econ impact assessment")

DATA_LOC = get_data_dir(econ=True)


def _load_national_totals():
    """Return (national population, dict of national daily GDP by sector).

    These are the denominators for rho (population share) and phi
    (sector GDP share). POP20 is taken from the same decennial file
    that p_gic_files samples from so rho is internally consistent.
    """
    socio_pkl = DATA_LOC / "processed_econ" / "socioeconomic_data.pkl"
    with open(socio_pkl, "rb") as f:
        _, zcta_pop_20, _, zcta_business_gdf, _ = pickle.load(f)

    pop_total = int(zcta_pop_20["POP20"].sum())

    gdp_cols = [c for c in zcta_business_gdf.columns if c.startswith("GDP_")]
    national_daily_gdp = {
        c.replace("GDP_", ""): float(zcta_business_gdf[c].sum()) for c in gdp_cols
    }
    return pop_total, national_daily_gdp


def get_confidence_intervals(ds, alpha_beta_scenario=USE_ALPHA_BETA_SCENARIO):
    """Collapse the Monte Carlo dataset into per-scenario mean / p5 / p95
    for GDP, EST, and affected population."""
    results = []

    if alpha_beta_scenario:
        return_periods = [
            "75yr",
            "100yr",
            "125yr",
            "150yr",
            "175yr",
            "200yr",
            "225yr",
            "250yr",
        ]

        for rp in return_periods:
            lower_scenario = f"gic_{rp}_conf_68_lower"
            mean_scenario = f"gic_{rp}_mean_prediction"
            upper_scenario = f"gic_{rp}_conf_68_upper"
            scenario_group = f"gic_{rp}"

            lower_data = (
                ds.sel(scenario=lower_scenario)
                if lower_scenario in ds.scenario.values
                else None
            )
            mean_data = (
                ds.sel(scenario=mean_scenario)
                if mean_scenario in ds.scenario.values
                else None
            )
            upper_data = (
                ds.sel(scenario=upper_scenario)
                if upper_scenario in ds.scenario.values
                else None
            )

            if mean_data is None:
                continue

            gdp_vars = [
                v
                for v in ds.data_vars
                if v.startswith("GDP_") and v.endswith("_affected")
            ]
            for gdp_var in gdp_vars:
                sector = gdp_var.replace("GDP_", "").replace("_affected", "")
                results.append(
                    {
                        "scenario": scenario_group,
                        "variable": f"GDP_{sector}",
                        "mean": float(mean_data[gdp_var].values),
                        "p5": (
                            float(lower_data[gdp_var].values)
                            if lower_data is not None
                            else float(mean_data[gdp_var].values)
                        ),
                        "p95": (
                            float(upper_data[gdp_var].values)
                            if upper_data is not None
                            else float(mean_data[gdp_var].values)
                        ),
                    }
                )

            est_vars = [
                v
                for v in ds.data_vars
                if v.startswith("EST_") and v.endswith("_affected")
            ]
            est_mean = sum(float(mean_data[v].values) for v in est_vars)
            est_lower = (
                sum(float(lower_data[v].values) for v in est_vars)
                if lower_data is not None
                else est_mean
            )
            est_upper = (
                sum(float(upper_data[v].values) for v in est_vars)
                if upper_data is not None
                else est_mean
            )
            results.append(
                {
                    "scenario": scenario_group,
                    "variable": "EST_TOTAL",
                    "mean": est_mean,
                    "p5": est_lower,
                    "p95": est_upper,
                }
            )

            results.append(
                {
                    "scenario": scenario_group,
                    "variable": "POP_AFFECTED",
                    "mean": float(mean_data["mean_pop_affected"].values),
                    "p5": (
                        float(lower_data["mean_pop_affected"].values)
                        if lower_data is not None
                        else float(mean_data["mean_pop_affected"].values)
                    ),
                    "p95": (
                        float(upper_data["mean_pop_affected"].values)
                        if upper_data is not None
                        else float(mean_data["mean_pop_affected"].values)
                    ),
                }
            )

    else:
        for scenario in ds.scenario.values:
            scenario_data = ds.sel(scenario=scenario)

            gdp_vars = [
                v
                for v in ds.data_vars
                if v.startswith("GDP_") and v.endswith("_affected")
            ]
            for gdp_var in gdp_vars:
                sector = gdp_var.replace("GDP_", "").replace("_affected", "")
                vals = scenario_data[gdp_var].values
                results.append(
                    {
                        "scenario": scenario,
                        "variable": f"GDP_{sector}",
                        "mean": np.mean(vals),
                        "p5": np.percentile(vals, 5),
                        "p95": np.percentile(vals, 95),
                    }
                )

            est_vars = [
                v
                for v in ds.data_vars
                if v.startswith("EST_") and v.endswith("_affected")
            ]
            total_est = sum(scenario_data[v].values for v in est_vars)
            results.append(
                {
                    "scenario": scenario,
                    "variable": "EST_TOTAL",
                    "mean": np.mean(total_est),
                    "p5": np.percentile(total_est, 5),
                    "p95": np.percentile(total_est, 95),
                }
            )

            pop_vals = scenario_data["mean_pop_affected"].values
            results.append(
                {
                    "scenario": scenario,
                    "variable": "POP_AFFECTED",
                    "mean": np.mean(pop_vals),
                    "p5": np.percentile(pop_vals, 5),
                    "p95": np.percentile(pop_vals, 95),
                }
            )

    df = pd.DataFrame(results)
    df["ci_width"] = df["p95"] - df["p5"]
    df["uncertainty_pct"] = (df["ci_width"] / df["mean"]) * 100
    return df.round(2)


def _clean_scenario_name(name):
    """Strip the prefix/suffix wrappers applied in p_gic_files."""
    return str(name).replace("e_", "").replace("-hazard A/ph", "")


def run_ghosh_value_added(confidence_df):
    """Supply-driven Ghosh shock.

    Treats affected sector GDP as a value-added loss and propagates it
    through the Ghosh inverse to obtain total output impacts.
    """
    io = InputOutputModel("10sector")
    results = []

    for scenario in confidence_df["scenario"].unique():
        scen = confidence_df[confidence_df["scenario"] == scenario]
        gdp_rows = scen[scen["variable"].str.startswith("GDP_")]

        for conf in ["mean", "p5", "p95"]:
            va_shock = pd.Series(0.0, index=io.sectors)
            for _, row in gdp_rows.iterrows():
                sector = row["variable"].replace("GDP_", "")
                if sector in io.sectors:
                    va_shock[sector] = -row[conf]

            output_impacts = io.total_output_from_value_added(va_shock)

            for sector in io.sectors:
                results.append(
                    {
                        "scenario": _clean_scenario_name(scenario),
                        "confidence": conf,
                        "sector": sector,
                        "direct_shock": va_shock[sector],
                        "total_impact": output_impacts[sector],
                        "multiplier_effect": output_impacts[sector] - va_shock[sector],
                    }
                )

    return pd.DataFrame(results)


def _build_leontief_shock(rho, phi, fd_df, include_production_channel):
    """Construct the daily final demand shock vector Delta_f ($ millions).

    Consumption-only:
        Delta_f = -(1/365) * rho * (F010 + F100)

    Consumption + production:
        Delta_f = -(1/365) * [rho * (F010 + F100) + phi * (F020 + F040)]

    F030 (inventory change) is not shocked: it is an accounting residual,
    not a behavioral demand category. The 1/365 converts annual Use-table
    columns to a daily flow.
    """
    f_pc = fd_df["F010"].to_numpy(float)
    f_gov = fd_df["F100"].to_numpy(float)

    consumption_channel = rho * (f_pc + f_gov)

    if include_production_channel:
        f_inv = fd_df["F020"].to_numpy(float)
        f_exp = fd_df["F040"].to_numpy(float)
        production_channel = phi * (f_inv + f_exp)
    else:
        production_channel = 0.0

    return -(1.0 / 365.0) * (consumption_channel + production_channel)


def _run_leontief(confidence_df, include_production_channel):
    """Shared Leontief routine. Iterates scenarios x confidence levels,
    builds Delta_f, and computes Delta_x = L @ Delta_f."""
    io = InputOutputModel("10sector")

    fd_df = pd.read_csv(
        DATA_LOC / "10sector" / "final_demand.csv", index_col=0
    ).reindex(io.sectors)

    pop_total, national_daily_gdp = _load_national_totals()

    results = []
    for scenario in confidence_df["scenario"].unique():
        scen = confidence_df[confidence_df["scenario"] == scenario]
        gdp_rows = scen[scen["variable"].str.startswith("GDP_")]
        pop_rows = scen[scen["variable"] == "POP_AFFECTED"]

        for conf in ["mean", "p5", "p95"]:
            pop_aff = float(pop_rows[conf].iloc[0]) if not pop_rows.empty else 0.0
            rho = pop_aff / pop_total if pop_total > 0 else 0.0

            phi = np.zeros(len(io.sectors))
            for j, sector in enumerate(io.sectors):
                match = gdp_rows[gdp_rows["variable"] == f"GDP_{sector}"]
                if match.empty:
                    continue
                national = national_daily_gdp.get(sector, 0.0)
                if national <= 0:
                    continue
                phi[j] = float(match[conf].iloc[0]) / national

            delta_f = _build_leontief_shock(
                rho=rho,
                phi=phi,
                fd_df=fd_df,
                include_production_channel=include_production_channel,
            )
            delta_x = io.L @ delta_f

            for j, sector in enumerate(io.sectors):
                results.append(
                    {
                        "scenario": _clean_scenario_name(scenario),
                        "confidence": conf,
                        "sector": sector,
                        "rho": rho,
                        "phi": float(phi[j]),
                        "direct_shock": float(delta_f[j]),
                        "total_impact": float(delta_x[j]),
                        "multiplier_effect": float(delta_x[j] - delta_f[j]),
                    }
                )

    return pd.DataFrame(results)


def run_leontief_consumption(confidence_df):
    """Leontief shock on consumption only: households (F010) and
    government (F100). Investment and exports are assumed to be absorbed
    by inventory drawdown over a one-day outage horizon and are not
    shocked."""
    return _run_leontief(confidence_df, include_production_channel=False)


def run_leontief_consumption_production(confidence_df):
    """Leontief shock on consumption + production: households (F010),
    government (F100), investment (F020), and exports (F040). Relevant
    for longer outages where production shortfalls can no longer be
    covered by inventory."""
    return _run_leontief(confidence_df, include_production_channel=True)


def run_policy(model_instance, policy_params, scenario_name="Policy Simulation"):
    """Run a policy simulation using the standard CGE model."""
    baseline = {
        "Y": {i: model_instance.Y[i].value for i in model_instance.i},
        "Z": {i: model_instance.Z[i].value for i in model_instance.i},
        "Q": {i: model_instance.Q[i].value for i in model_instance.i},
        "Xp": {i: model_instance.Xp[i].value for i in model_instance.i},
        "Xg": {i: model_instance.Xg[i].value for i in model_instance.i},
        "Xv": {i: model_instance.Xv[i].value for i in model_instance.i},
        "E": {i: model_instance.E[i].value for i in model_instance.i},
        "M": {i: model_instance.M[i].value for i in model_instance.i},
        "F": {
            (h, i): model_instance.F[h, i].value
            for h in model_instance.h
            for i in model_instance.i
        },
        "pq": {i: model_instance.pq[i].value for i in model_instance.i},
        "pf": {h: model_instance.pf[h].value for h in model_instance.h},
        "pz": {i: model_instance.pz[i].value for i in model_instance.i},
        "Td": model_instance.Td.value,
        "Sp": model_instance.Sp.value,
        "Sg": model_instance.Sg.value,
        "epsilon": model_instance.epsilon.value,
        "welfare": model_instance.obj.expr(),
    }

    policy_model = copy.deepcopy(model_instance)

    logger.info(f"Applying policy scenario: {scenario_name}")
    for param_name, param_value in policy_params.items():
        if "." in param_name:
            base_param, index = param_name.split(".")
            if hasattr(policy_model, base_param):
                if index in getattr(policy_model, base_param):
                    setattr(
                        getattr(policy_model, base_param)[index], "value", param_value
                    )
                else:
                    logger.warning(f"Index {index} not found in {base_param}")
            else:
                logger.warning(f"Parameter {base_param} not found in model")
        else:
            if hasattr(policy_model, param_name):
                setattr(policy_model, param_name, param_value)
            else:
                logger.warning(f"Parameter {param_name} not found in model")

    solver = SolverFactory("ipopt", executable=IPOPT_EXEC)
    solver.options["max_iter"] = 5000
    solver.options["tol"] = 1e-6

    logger.info("Solving policy scenario...")
    results = solver.solve(policy_model, tee=False)

    baseline_gdp = 0
    policy_gdp = 0
    for h in policy_model.h:
        for i in policy_model.i:
            baseline_gdp += baseline["F"][(h, i)]
            policy_gdp += policy_model.F[h, i].value
    baseline["gdp"] = baseline_gdp

    policy_results = {
        "Y": {i: policy_model.Y[i].value for i in policy_model.i},
        "Z": {i: policy_model.Z[i].value for i in policy_model.i},
        "Q": {i: policy_model.Q[i].value for i in policy_model.i},
        "Xp": {i: policy_model.Xp[i].value for i in policy_model.i},
        "Xg": {i: policy_model.Xg[i].value for i in policy_model.i},
        "Xv": {i: policy_model.Xv[i].value for i in policy_model.i},
        "E": {i: policy_model.E[i].value for i in policy_model.i},
        "M": {i: policy_model.M[i].value for i in policy_model.i},
        "F": {
            (h, i): policy_model.F[h, i].value
            for h in policy_model.h
            for i in policy_model.i
        },
        "pq": {i: policy_model.pq[i].value for i in policy_model.i},
        "pf": {h: policy_model.pf[h].value for h in policy_model.h},
        "pz": {i: policy_model.pz[i].value for i in policy_model.i},
        "Td": policy_model.Td.value,
        "Sp": policy_model.Sp.value,
        "Sg": policy_model.Sg.value,
        "epsilon": policy_model.epsilon.value,
        "welfare": policy_model.obj.expr(),
        "gdp": policy_gdp,
    }

    pct_changes = {}
    for var in ["Y", "Z", "Q", "Xp", "Xg", "Xv", "E", "M", "pq", "pz"]:
        pct_changes[var] = {
            i: (
                ((policy_results[var][i] / baseline[var][i] - 1) * 100)
                if baseline[var][i] != 0
                else float("nan")
            )
            for i in policy_model.i
        }
    pct_changes["pf"] = {
        h: (
            ((policy_results["pf"][h] / baseline["pf"][h] - 1) * 100)
            if baseline["pf"][h] != 0
            else float("nan")
        )
        for h in policy_model.h
    }
    pct_changes["F"] = {
        (h, i): (
            ((policy_results["F"][(h, i)] / baseline["F"][(h, i)] - 1) * 100)
            if baseline["F"][(h, i)] != 0
            else float("nan")
        )
        for h in policy_model.h
        for i in policy_model.i
    }
    for var in ["Td", "Sp", "Sg", "epsilon", "welfare", "gdp"]:
        pct_changes[var] = (
            ((policy_results[var] / baseline[var] - 1) * 100)
            if baseline[var] != 0
            else float("nan")
        )

    logger.info(f"RESULTS FOR {scenario_name}")

    return {
        "baseline": baseline,
        "policy": policy_results,
        "pct_changes": pct_changes,
        "solver_status": {
            "status": results.solver.status,
            "termination_condition": results.solver.termination_condition,
        },
    }


def apply_cge_to_confidence_intervals(confidence_df, sam_type="us"):
    """Apply CGE model to confidence intervals."""
    baseline = run_cge_example(sam_type=sam_type, display_results=False)

    va0 = {i: sum(value(baseline.F[h, i]) for h in baseline.h) for i in baseline.i}
    b0 = {i: value(baseline.b[i]) for i in baseline.i}

    out = []
    for sc in confidence_df["scenario"].unique():
        df_sc = confidence_df[
            (confidence_df["scenario"] == sc)
            & (confidence_df["variable"].str.startswith("GDP_"))
        ]

        for lev in ["mean", "p5", "p95"]:
            policy_params, direct = {}, {}
            for _, r in df_sc.iterrows():
                i = r["variable"].replace("GDP_", "")
                if i in va0 and va0[i] > 0:
                    pct = float(r[lev]) / va0[i]
                    pct = max(0.0, min(0.99, pct))
                    policy_params[f"b.{i}"] = b0[i] * (1.0 - pct)
                    direct[i] = -float(r[lev])

            res = run_policy(baseline, policy_params, scenario_name=f"{sc}_{lev}")

            for i in baseline.i:
                dZ = res["policy"]["Z"][i] - res["baseline"]["Z"][i]
                out.append(
                    {
                        "scenario": _clean_scenario_name(sc),
                        "confidence": lev,
                        "sector": i,
                        "direct_shock": direct.get(i, 0.0),
                        "total_impact": dZ,
                        "multiplier_effect": dZ - direct.get(i, 0.0),
                        "price_effect": res["pct_changes"].get("pq", {}).get(i, 0.0),
                        "gdp_contribution": sum(
                            res["policy"]["F"][(h, i)] - res["baseline"]["F"][(h, i)]
                            for h in baseline.h
                        ),
                    }
                )

    return pd.DataFrame(out).round(2)


def _suffix():
    """Filename suffix matching the Ghosh convention used in the rest of
    the pipeline."""
    if USE_ALPHA_BETA_SCENARIO:
        return "_alpha_beta"
    if PROCESS_GND_FILES:
        return "_gnd_gic"
    return ""


def main():
    """Main analysis function."""
    logger.info("Loading and Processing Data")
    combined_ds, combined_vuln, vuln_table = load_gic_results()

    logger.info("Calculating Confidence Intervals")
    confidence_df = get_confidence_intervals(
        combined_ds, alpha_beta_scenario=USE_ALPHA_BETA_SCENARIO
    )

    logger.info("Running Ghosh value-added shock")
    ghosh_df = run_ghosh_value_added(confidence_df)

    logger.info("Running Leontief consumption shock")
    leontief_cons_df = run_leontief_consumption(confidence_df)

    logger.info("Running Leontief consumption + production shock")
    leontief_cp_df = run_leontief_consumption_production(confidence_df)

    suffix = _suffix()
    ghosh_df.to_csv(FIGURES_DIR / f"ghosh_results{suffix}.csv", index=False)
    leontief_cons_df.to_csv(
        FIGURES_DIR / f"leontief_consumption_results{suffix}.csv", index=False
    )
    leontief_cp_df.to_csv(
        FIGURES_DIR / f"leontief_consumption_production_results{suffix}.csv",
        index=False,
    )
    confidence_df.to_csv(FIGURES_DIR / f"confidence_intervals{suffix}.csv", index=False)

    logger.info(f"Saved all results with suffix '{suffix}'")

    return {
        "ghosh_results": ghosh_df,
        "leontief_consumption_results": leontief_cons_df,
        "leontief_consumption_production_results": leontief_cp_df,
        "confidence_intervals": confidence_df,
    }


if __name__ == "__main__":
    results = main()
    logger.info("Economic analysis completed successfully!")
