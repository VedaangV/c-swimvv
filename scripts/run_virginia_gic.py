"""
Run the Virginia+PJM GIC Monte Carlo simulation.

Mirrors scripts/est_gic.py's main() loop (the regular, non-gannon-only
path) almost line for line, reusing every numerical routine from
scripts.build_admittance_matrix and scripts.est_gic directly and unmodified
— only the data locations differ. scripts/est_gic.py's own main() isn't
called directly because it hardcodes US paths in several places that aren't
exposed as parameters (the statistical-analysis HDF5 name, the
trans_lines_pickle.pkl path, and the winding_gic_rand_*.csv output
directory), so this script reimplements its loop body with those paths
pointed at Virginia's output tree instead.

Deviation from the original adaptation plan worth flagging explicitly: the
plan assumed scripts/est_gic.py used multiprocessing.Pool with a Windows
`_init_worker` fix that needed to be carried over. The version of
scripts/est_gic.py actually in this repository has no multiprocessing at
all — it iterates Monte Carlo samples sequentially and parallelizes only the
per-return-period GIC calculation within one iteration via
ThreadPoolExecutor (numerical work happens through torch, not numpy, and
each iteration is already fairly heavy). This script follows that same
sequential-iteration structure rather than inventing a multiprocessing
pattern the reference implementation doesn't have.

Run after scripts/run_virginia_admittance.py.

PHASE 4 (ISLANDING_MITIGATION_PLAN.md) EXTENSION: main() takes an optional
exclude_line_ids set and gic_dir/gnd_gic_dir overrides, so the same Monte
Carlo loop can re-solve GIC with Island A's boundary lines removed, writing
to a separate output directory rather than overwriting the baseline. See
scripts/run_virginia_islanded_gic.py, the thin wrapper that calls this
with those parameters set -- kept as parameters on the existing function
rather than a separate near-duplicate 150-line loop, since
run_virginia_gic.py is itself already Virginia-adaptation code (not the
shared/national est_gic.py this project avoids editing in place), so
extending it directly here is the right level to make this change at.
Defaults (exclude_line_ids=None, gic_dir=VA_GIC_DIR, gnd_gic_dir=
VA_GND_GIC_DIR) reproduce the exact original baseline behavior.

Line removal happens once, on the full df_lines, before the Monte Carlo
loop starts -- deterministic topology edit, not a per-sample stochastic
effect like apply_random_line_dc_blocking()'s per-iteration DC-blocker
mask (which still applies on top, now operating on the reduced line set).
Removing a line here means it contributes nothing to either the admittance
matrix (network_admittance()/random_admittance_matrix() iterate whatever
df_lines they're handed) or the geoelectric injection currents
(get_injection_currents() sums per-line contributions from the same
df_lines) -- both consume the identical filtered table, so no other
structure (buses, transformers, grounding) needs to change for the
substations that remain connected via other paths.
"""

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import scripts.build_admittance_matrix as bam
from scripts.build_admittance_matrix import (
    random_admittance_matrix,
    randomize_grounding_resistance,
    apply_random_line_dc_blocking,
)
from scripts.est_gic import (
    load_and_process_gic_data,
    get_injection_currents,
    nodal_voltage_calculation,
    keep_nodes_with_ground,
    solve_total_nodal_gic,
    calculate_GIC,
    calc_trafo_gic,
    can_use_float16,
    return_periods,
    P_LINE_BLOCK,
    seed_base,
)
from configs import setup_logger
from configs.virginia_config import VA_DATA_DIR, VA_GIC_DIR, VA_GND_GIC_DIR, N_MONTE_CARLO

logger = setup_logger(log_file="logs/run_virginia_gic.log")

VA_RESULTS_PATH = "statistical_analysis/va_geomagnetic_data_return_periods.h5"


def main(
    start_idx: int = 0,
    end_idx: int | None = None,
    exclude_line_ids: set | None = None,
    gic_dir=VA_GIC_DIR,
    gnd_gic_dir=VA_GND_GIC_DIR,
):
    # get_transformer_samples()/process_substation_buses() read/write a
    # module-level DATA_LOC in build_admittance_matrix.py, not a parameter.
    bam.DATA_LOC = VA_DATA_DIR

    substation_buses, bus_ids_map, sub_look_up, df_lines, df_substations_info = (
        bam.process_substation_buses(VA_DATA_DIR)
    )

    df_lines.drop(columns=["geometry"], inplace=True)
    df_lines["name"] = df_lines["name"].astype(str)

    trafos_data = bam.get_transformer_samples(
        substation_buses, n_samples=N_MONTE_CARLO, new_trafos=False
    )

    df_lines, mt_coords, mt_names, e_fields, b_fields, v_fields, gannon_e, v_cols = (
        load_and_process_gic_data(VA_DATA_DIR, df_lines, VA_RESULTS_PATH)
    )

    if exclude_line_ids:
        exclude_line_ids = {str(x) for x in exclude_line_ids}
        before = len(df_lines)
        df_lines = df_lines[~df_lines["name"].astype(str).isin(exclude_line_ids)].reset_index(drop=True)
        removed = before - len(df_lines)
        logger.info(
            f"Islanding: removed {removed}/{before} lines from df_lines "
            f"(boundary_line_ids had {len(exclude_line_ids)})."
        )
        if removed != len(exclude_line_ids):
            logger.warning(
                f"Removed {removed} lines but exclude_line_ids had {len(exclude_line_ids)} "
                "entries -- some IDs didn't match df_lines['name'] (duplicates, or an ID "
                "mismatch between va_island_a.json and this run's transmission_lines.csv). "
                "Investigate before trusting the islanded results."
            )

    sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))
    n_nodes = len(sub_look_up)

    logger.info(f"Starting VA+PJM GIC Monte Carlo ({len(trafos_data)} samples)...")
    for i, trafo_data in enumerate(trafos_data):
        if i < start_idx:
            continue
        if end_idx is not None and i >= end_idx:
            break

        out_path = gic_dir / f"winding_gic_rand_{i}.csv"
        gnd_out_path = gnd_gic_dir / f"ground_gic{i}.csv"
        if out_path.exists() and gnd_out_path.exists():
            continue

        logger.info(f"[{i}] Building random admittance matrix...")
        df_ss_iter = randomize_grounding_resistance(df_substations_info, seed=seed_base + i)
        df_lines_iter = apply_random_line_dc_blocking(df_lines, p_block=P_LINE_BLOCK, seed=seed_base + i)

        Y_n, Y_e, df_transformers = random_admittance_matrix(
            substation_buses, trafo_data, bus_ids_map, sub_look_up, df_lines_iter, df_ss_iter,
        )

        dtype = np.float32 if (can_use_float16(Y_n) and can_use_float16(Y_e)) else np.float64
        Y_n = Y_n.astype(dtype, copy=False)
        Y_e = Y_e.astype(dtype, copy=False)

        row_active = np.any(Y_n != 0, axis=1) | np.any(Y_e != 0, axis=1)
        nz = np.flatnonzero(row_active)
        keep = keep_nodes_with_ground(Y_n[np.ix_(nz, nz)], Y_e[np.ix_(nz, nz)])
        non_zero_indices = nz[keep]

        Y_n = Y_n[np.ix_(non_zero_indices, non_zero_indices)]
        Y_e_reduced = Y_e[np.ix_(non_zero_indices, non_zero_indices)]
        Y_total = Y_n + Y_e_reduced
        del Y_n
        gc.collect()

        injections_data = get_injection_currents(
            df_lines_iter, n_nodes, non_zero_indices, sub_look_up, VA_DATA_DIR,
        )
        nodal_voltages = nodal_voltage_calculation(Y_total, injections_data)

        df_lines_copy = df_lines_iter.copy()
        df_lines_copy["from_bus"] = df_lines_copy["from_bus"].apply(sub_look_up.get)
        df_lines_copy["to_bus"] = df_lines_copy["to_bus"].apply(sub_look_up.get)

        def _calc_period(period):
            try:
                if f"V_{period}" not in nodal_voltages:
                    return period, None, None
                V_nodal = nodal_voltages[f"V_{period}"]
                df_gic = calculate_GIC(
                    df_lines_copy.copy(), V_nodal, f"V_{period}", non_zero_indices, n_nodes,
                )
                gic = calc_trafo_gic(
                    sub_look_up, df_transformers.copy(), V_nodal, sub_ref,
                    n_nodes, non_zero_indices, f"{period}-year-hazard",
                )
                return period, df_gic, gic
            except Exception as e:
                logger.error(f"[{i}] Error calculating GIC for period {period}: {e}")
                return period, None, None

        gic_data = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_calc_period, p): p for p in ["gannon", *return_periods]}
            for fut in as_completed(futures):
                period, df_gic_res, gic_res = fut.result()
                if df_gic_res is not None and gic_res is not None:
                    gic_data[period] = gic_res

        if not gic_data:
            logger.error(f"[{i}] No GIC results computed, skipping.")
            continue

        if not out_path.exists():
            winding_gic_df_list = []
            for period, gic_values in gic_data.items():
                rows = [
                    (trafo, winding, gic)
                    for trafo, windings in gic_values.items()
                    for winding, gic in windings.items()
                ]
                if not rows:
                    continue
                winding_gic_df_list.append(
                    pd.DataFrame(rows, columns=["Transformer", "Winding", f"{period}-year-hazard A/ph"])
                )

            if winding_gic_df_list:
                merged = pd.concat(winding_gic_df_list, axis=1)
                merged = merged.loc[:, ~merged.columns.duplicated()]
                df_transformers["Transformer"] = df_transformers["name"]
                merged = merged.merge(
                    df_transformers[["sub_id", "Transformer", "latitude", "longitude"]],
                    on="Transformer", how="inner",
                )
                if not merged.empty:
                    merged.to_csv(out_path, index=False)
                    logger.info(f"[{i}] Saved {out_path}")

        if not gnd_out_path.exists():
            v_keys = [c for c in v_cols if c in nodal_voltages and nodal_voltages[c] is not None]
            if v_keys:
                ig_all = np.vstack(
                    [solve_total_nodal_gic(Y_e_reduced, nodal_voltages[k], non_zero_indices, n_nodes) for k in v_keys]
                ).T
                idx_series = df_ss_iter["name"].map(sub_look_up)
                mask = idx_series.notna()
                valid_substations = df_ss_iter.loc[mask, "name"].to_numpy()
                valid_indices = idx_series[mask].astype(int).to_numpy()
                gic_values_T = ig_all[valid_indices].T
                data = {"Substation": valid_substations}
                for col, vals in zip(v_keys, gic_values_T):
                    data[f"GIC_{col.split('V_')[1]}"] = vals
                pd.DataFrame(data).to_csv(gnd_out_path, index=False)
                logger.info(f"[{i}] Saved {gnd_out_path}")

        del Y_e, Y_total
        gc.collect()

    logger.info("Virginia+PJM GIC Monte Carlo complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Virginia+PJM GICs.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    main(start_idx=args.start, end_idx=args.end)
