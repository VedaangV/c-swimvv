"""
Run the Virginia+PJM GIC Monte Carlo simulation.

Mirrors scripts/est_gic.py's main() loop closely, reusing its numerical
routines (and build_admittance_matrix's) unmodified -- just with Virginia's
data paths swapped in, since est_gic.py hardcodes US paths in a few places
that aren't exposed as parameters.

Note: est_gic.py itself has no multiprocessing -- it iterates Monte Carlo
samples sequentially and only parallelizes the per-return-period GIC calc
within an iteration via ThreadPoolExecutor. This script follows the same
structure.

main() takes an optional exclude_line_ids set and gic_dir/gnd_gic_dir
overrides so the same Monte Carlo loop can re-solve GIC with a set of
boundary lines removed (islanding), writing to a separate output directory.
See run_virginia_islanded_gic.py. Defaults reproduce the original baseline
behavior. Line removal is a one-time deterministic topology edit applied
before the iterations start, on top of which the per-iteration DC-blocker
mask still applies.

main()'s body is split into load_gic_context() (the expensive,
exclude_line_ids-independent setup: network topology, transformer samples,
storm E-field data -- ~10-20s) and run_monte_carlo(ctx, ...) (the actual
per-iteration solve). This lets callers that need many solves per process
(e.g. real_storm_gic.py's candidate-cut search) load the context once and
reuse it across calls instead of paying the load cost every time.

Run after scripts/run_virginia_admittance.py.
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


def load_gic_context() -> dict:
    """Load everything the Monte Carlo loop needs up front (network
    topology, transformer samples, storm E-field data) independent of
    exclude_line_ids or which iterations will run. Pass the result to
    run_monte_carlo() as many times as needed."""
    # get_transformer_samples()/process_substation_buses() read/write
    # DATA_LOC as a module global in build_admittance_matrix.py.
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

    sub_ref = dict(zip(df_substations_info.name, df_substations_info.buses))
    n_nodes = len(sub_look_up)

    return {
        "substation_buses": substation_buses,
        "bus_ids_map": bus_ids_map,
        "sub_look_up": sub_look_up,
        "df_lines": df_lines,
        "df_substations_info": df_substations_info,
        "trafos_data": trafos_data,
        "sub_ref": sub_ref,
        "n_nodes": n_nodes,
        "v_cols": v_cols,
    }


def run_monte_carlo(
    ctx: dict,
    start_idx: int = 0,
    end_idx: int | None = None,
    exclude_line_ids: set | None = None,
    gic_dir=VA_GIC_DIR,
    gnd_gic_dir=VA_GND_GIC_DIR,
):
    """The per-iteration solve loop, operating on an already-loaded ctx.
    exclude_line_ids filtering happens on a local copy of df_lines each
    call -- ctx itself is never mutated, so it's safe to reuse across
    calls with different candidate cuts."""
    substation_buses = ctx["substation_buses"]
    bus_ids_map = ctx["bus_ids_map"]
    sub_look_up = ctx["sub_look_up"]
    df_lines = ctx["df_lines"]
    df_substations_info = ctx["df_substations_info"]
    trafos_data = ctx["trafos_data"]
    sub_ref = ctx["sub_ref"]
    n_nodes = ctx["n_nodes"]
    v_cols = ctx["v_cols"]

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


def main(
    start_idx: int = 0,
    end_idx: int | None = None,
    exclude_line_ids: set | None = None,
    gic_dir=VA_GIC_DIR,
    gnd_gic_dir=VA_GND_GIC_DIR,
):
    """A caller that needs many solves in one process (e.g. island
    selection) should call load_gic_context() once and run_monte_carlo()
    directly instead of this."""
    ctx = load_gic_context()
    run_monte_carlo(
        ctx, start_idx=start_idx, end_idx=end_idx, exclude_line_ids=exclude_line_ids,
        gic_dir=gic_dir, gnd_gic_dir=gnd_gic_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Virginia+PJM GICs.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    main(start_idx=args.start, end_idx=args.end)
