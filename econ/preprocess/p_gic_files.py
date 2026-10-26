"""
Monte Carlo reliability analysis for transformer failures under geomagnetic storm scenarios.
"""

import os
import warnings
from glob import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm, weibull_min
from numpy.random import default_rng
from tqdm import tqdm
import torch

from configs import (
    setup_logger,
    get_data_dir,
    USE_ALPHA_BETA_SCENARIO,
    get_scenarios,
    GDP_COLUMNS,
    EST_COLUMNS,
    DROP_COLUMNS,
    CSV_DTYPES,
    get_simulation_config,
    ALPHA_BETA_GIC_FILE,
    EFF_GIC_DIR,
    GND_GIC_DIR,
    OUTPUT_FILES,
    DATA_DIR,
    PROCESS_GND_FILES,
)
from econ.scripts.l_prepr_data import load_and_aggregate_tiles

warnings.filterwarnings("ignore")

DATA_LOC = get_data_dir(econ=True)
logger = setup_logger(log_file="logs/p_gic_files_gpu")
_rng = default_rng(seed=42)

SCENARIOS = get_scenarios()
GDP_COLS = GDP_COLUMNS
EST_COLS = EST_COLUMNS
DROP_COLS = DROP_COLUMNS
DTYPE_CSV = CSV_DTYPES
device = None


def _get_device():
    global device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return device


def one_draw_vec(batch, n_tr, pop, gic_ref, gdp_df, est_df, theta0=75.0):
    dev = _get_device()
    pop_gpu = torch.as_tensor(pop, device=dev, dtype=torch.float32)
    gic_gpu = torch.as_tensor(gic_ref, device=dev, dtype=torch.float32)
    gdp_gpu = torch.stack(
        [
            torch.as_tensor(gdp_df[c].values, device=dev, dtype=torch.float32)
            for c in GDP_COLS
        ],
        dim=1,
    )
    est_gpu = torch.stack(
        [
            torch.as_tensor(est_df[c].values, device=dev, dtype=torch.float32)
            for c in EST_COLS
        ],
        dim=1,
    )

    beta_frag = 0.25 + 0.25 * torch.rand((batch, 1), device=dev)
    Cpen = (
        torch.ones((batch, 1), device=dev)
        if USE_ALPHA_BETA_SCENARIO
        else 0.60 + 0.40 * torch.rand((batch, 1), device=dev)
    )

    sel = torch.rand((batch, n_tr), device=dev)
    age = torch.where(
        sel < 0.55,
        33.0 + 17.0 * torch.rand((batch, n_tr), device=dev),
        1.0 + 31.0 * torch.rand((batch, n_tr), device=dev),
    )
    beta_age = 1.0 + 2.0 * torch.rand((batch, 1), device=dev)
    eta_age = 30.0 + 20.0 * torch.rand((batch, 1), device=dev)

    F_age = 1.0 - torch.exp(-torch.pow((age / eta_age).clamp_min(1e-12), beta_age))
    theta = theta0 * (1.0 - 0.4 * F_age)

    log_gic = torch.log(torch.abs(gic_gpu).clamp_min(1e-12))[None, :]
    z = (torch.log(Cpen) + log_gic - torch.log(theta.clamp_min(1e-12))) / beta_frag
    Pf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

    fails = (torch.rand((batch, n_tr), device=dev) < Pf).to(torch.float32)
    pop_aff = fails @ pop_gpu
    gdp_aff_mat = fails @ gdp_gpu
    est_aff_mat = fails @ est_gpu

    gdp_aff = {
        c: gdp_aff_mat[:, i].detach().cpu().numpy() for i, c in enumerate(GDP_COLS)
    }
    est_aff = {
        c: est_aff_mat[:, i].detach().cpu().numpy() for i, c in enumerate(EST_COLS)
    }
    return (
        pop_aff.detach().cpu().numpy(),
        gdp_aff,
        est_aff,
        fails.detach().cpu().numpy().astype(bool),
    )


def _simulate_scenario(scenario, merged, theta0, tol, max_iter, batch):
    """Simulates reliability impacts for a single GIC scenario with convergence checking."""
    v = merged.dropna(subset=[scenario, "POP20"])
    if v.empty:
        return scenario, None, None

    n_tr = len(v)
    pop = v["POP20"].values
    gic = v[scenario].values

    pop_s = np.empty(max_iter)
    gdp_s = {c: np.empty(max_iter) for c in GDP_COLS}
    est_s = {c: np.empty(max_iter) for c in EST_COLS}
    fail_c = np.zeros(n_tr, dtype=int)

    i = 0
    while i < max_iter:
        b = min(batch, max_iter - i)
        p, g, e, f = one_draw_vec(b, n_tr, pop, gic, v[GDP_COLS], v[EST_COLS], theta0)
        sl = slice(i, i + b)
        pop_s[sl] = p
        for c in GDP_COLS:
            gdp_s[c][sl] = g[c]
        for c in EST_COLS:
            est_s[c][sl] = e[c]
        fail_c += f.sum(axis=0)
        i += b

        if i >= 5000:
            m = pop_s[:i].mean()
            if m:
                hw = 1.96 * pop_s[:i].std(ddof=1) / np.sqrt(i)
                if hw / m < tol:
                    break
            elif i > 10000:
                break

    res = {"mean_pop_affected": pop_s[:i].mean()}
    for c in GDP_COLS:
        res[f"{c}_affected"] = gdp_s[c][:i].mean()
    for c in EST_COLS:
        res[f"{c}_affected"] = est_s[c][:i].mean()

    tbl = pd.DataFrame(
        {
            "sub_id": v["sub_id"].values,
            "latitude": v["latitude"].values,
            "longitude": v["longitude"].values,
            "scenario": scenario,
            "failure_prob": fail_c / i,
        }
    )
    return scenario, res, tbl


def process_gic_file(df: pd.DataFrame) -> pd.DataFrame:
    """Processes GIC data by taking maximum values across time periods and extracting coordinates."""
    if "Substation" in df.columns:
        df = df.rename(columns={"Substation": "sub_id"})

        gic_rename_map = {
            "GIC_halloween": "e_halloween-year-hazard A/ph",
            "GIC_st_patricks": "e_st_patricks-year-hazard A/ph",
            "GIC_gannon": "e_gannon-year-hazard A/ph",
            "GIC_50": "e_50-year-hazard A/ph",
            "GIC_75": "e_75-year-hazard A/ph",
            "GIC_100": "e_100-year-hazard A/ph",
            "GIC_125": "e_125-year-hazard A/ph",
            "GIC_150": "e_150-year-hazard A/ph",
            "GIC_175": "e_175-year-hazard A/ph",
            "GIC_200": "e_200-year-hazard A/ph",
            "GIC_225": "e_225-year-hazard A/ph",
            "GIC_250": "e_250-year-hazard A/ph",
        }

        df = df.rename(columns=gic_rename_map)

        gic_cols = [c for c in df.columns if c.endswith("-year-hazard A/ph")]
        df[gic_cols] = df[gic_cols].abs()
        df[gic_cols] = df[gic_cols] / 3.0

        df["sub_id"] = df["sub_id"].astype(str)

        df_substation = pd.read_csv(
            DATA_DIR / "admittance_matrix" / "substation_info.csv"
        )

        df_substation["name"] = df_substation["name"].astype(str)
        df_substation.rename(columns={"name": "sub_id"}, inplace=True)

        df = df.merge(
            df_substation[["sub_id", "latitude", "longitude"]], on="sub_id", how="left"
        )

        return df

    else:
        gic_cols = [c for c in df.columns if "year-hazard A/ph" in c]
        df[gic_cols] = df[gic_cols].abs()
        gic_max = df.groupby("sub_id", observed=True)[gic_cols].max().reset_index()
        coords = (
            df.groupby("sub_id", observed=True)[["latitude", "longitude"]]
            .first()
            .reset_index()
        )
        out = gic_max.merge(coords, on="sub_id")
        out["sub_id"] = out["sub_id"].astype(str)
        return out


def build_merged(agg_df, gic_df):
    """Merges economic aggregation data with GIC scenario data."""
    return agg_df.merge(
        gic_df[["sub_id"] + SCENARIOS + ["latitude", "longitude"]], on="sub_id"
    )


def _worker(arg):
    """Worker function for parallel processing of GIC files."""
    ridx, path, agg_df, theta0, tol, max_iter, batch, metrics = arg
    gic_df = process_gic_file(pd.read_csv(path, dtype=DTYPE_CSV))
    merged = build_merged(agg_df, gic_df)

    scen_res, vuln_tbls = {}, []
    for sc in SCENARIOS:
        _, res, tbl = _simulate_scenario(sc, merged, theta0, tol, max_iter, batch)
        if res is not None:
            scen_res[sc] = res
            vuln_tbls.append(tbl)

    summ_df = pd.DataFrame.from_dict(scen_res, orient="index")
    summ_df = summ_df.reindex(index=SCENARIOS)[metrics]

    ds = xr.Dataset(
        {m: (["scenario"], summ_df[m].to_numpy()) for m in metrics},
        coords={"scenario": SCENARIOS, "gic_realization": [ridx]},
    )
    for tbl in vuln_tbls:
        tbl["gic_realization"] = ridx

    return ds, pa.Table.from_pandas(pd.concat(vuln_tbls, ignore_index=True))


def process_all_files(
    gic_files,
    aggregate_gdf,
    theta0=None,
    tol=None,
    max_iter=None,
    batch=None,
    workers=os.cpu_count(),
    save_batch=None,
):
    """Processes all GIC files in batches with resume capability and memory-efficient output handling."""
    config = get_simulation_config()
    theta0 = theta0 or config["theta0"]
    tol = tol or config["tolerance"]
    max_iter = max_iter or config["max_iterations"]
    batch = batch or config["batch_size"]
    save_batch = save_batch or config["save_batch_size"]

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        workers = 1

    if PROCESS_GND_FILES:
        batch_dir_name = "gnd_gic_processing_batches"
    else:
        batch_dir_name = "gic_processing_batches"

    batch_dir = DATA_LOC / batch_dir_name
    batch_dir.mkdir(exist_ok=True)

    existing_summaries = sorted(batch_dir.glob("summary_batch_*.nc"))
    existing_vulns = sorted(batch_dir.glob("vuln_batch_*.parquet"))
    start_batch = len(existing_summaries)

    if start_batch > 0:
        logger.info(
            f"Found {start_batch} existing batches, resuming from batch {start_batch}"
        )

    first_gic = process_gic_file(pd.read_csv(gic_files[0], dtype=DTYPE_CSV))
    first_mrg = build_merged(aggregate_gdf, first_gic)
    _, first_metrics_dict, _ = _simulate_scenario(
        SCENARIOS[0], first_mrg, theta0, tol, 5000, batch
    )
    metrics = list(first_metrics_dict.keys())

    total_batches = (len(gic_files) + save_batch - 1) // save_batch

    for batch_idx in range(start_batch, total_batches):
        start_idx = batch_idx * save_batch
        end_idx = min(start_idx + save_batch, len(gic_files))
        batch_files = gic_files[start_idx:end_idx]

        logger.info(
            f"Processing batch {batch_idx+1}/{total_batches}: files {start_idx}-{end_idx}"
        )

        args = (
            (i, p, aggregate_gdf, theta0, tol, max_iter, batch, metrics)
            for i, p in enumerate(batch_files, start=start_idx)
        )

        if workers == 1:
            ds_list, vuln_tables = [], []
            for args_item in tqdm(
                (
                    (i, p, aggregate_gdf, theta0, tol, max_iter, batch, metrics)
                    for i, p in enumerate(batch_files, start=start_idx)
                ),
                total=len(batch_files),
                desc=f"Batch {batch_idx+1}",
            ):
                ds, vt = _worker(args_item)
                ds_list.append(ds)
                vuln_tables.append(vt)
        else:
            ds_list, vuln_tables = [], []
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for ds, vt in tqdm(
                    ex.map(_worker, args),
                    total=len(batch_files),
                    desc=f"Batch {batch_idx+1}",
                ):
                    ds_list.append(ds)
                    vuln_tables.append(vt)

        batch_ds = xr.concat(ds_list, dim="gic_realization")
        batch_ds.to_netcdf(batch_dir / f"summary_batch_{batch_idx:03d}.nc")

        batch_vuln = pa.concat_tables(vuln_tables)
        batch_vuln.to_pandas().to_parquet(
            batch_dir / f"vuln_batch_{batch_idx:03d}.parquet"
        )

        logger.info(f"Saved batch {batch_idx+1}: files {start_idx}-{end_idx}")

    logger.info("Combining all batches into final files...")

    summary_files = sorted(batch_dir.glob("summary_batch_*.nc"))
    datasets = []
    try:
        name_to_save = OUTPUT_FILES["regular_all"]
        OUT_PATH = DATA_LOC / name_to_save
        if OUT_PATH.exists():
            OUT_PATH.unlink()
        for f in summary_files:
            ds = xr.open_dataset(f)
            datasets.append(ds)
        combined_ds = xr.concat(datasets, dim="gic_realization")
        combined_ds.to_netcdf(OUT_PATH, format="NETCDF4", mode="w")
    finally:
        for ds in datasets:
            ds.close()

    vuln_files = sorted(batch_dir.glob("vuln_batch_*.parquet"))

    vuln_output_path = DATA_LOC / OUTPUT_FILES["vulnerable_regular"]
    if vuln_output_path.exists():
        import shutil

        shutil.rmtree(vuln_output_path)

    chunk_size = 10
    all_vuln_dfs = []

    for i in range(0, len(vuln_files), chunk_size):
        chunk_files = vuln_files[i : i + chunk_size]
        chunk_dfs = []

        for f in chunk_files:
            df = pd.read_parquet(f)
            chunk_dfs.append(df)

        if chunk_dfs:
            chunk_combined = pd.concat(chunk_dfs, ignore_index=True)
            all_vuln_dfs.append(chunk_combined)

    if all_vuln_dfs:
        combined_vuln = pd.concat(all_vuln_dfs, ignore_index=True)

        # Scenario values contain "A/ph" -- partitioning directly on that
        # turns the "/" into a path separator and pyarrow can't read the
        # column back. Partition on a sanitized copy instead.
        combined_vuln["scenario_partition"] = combined_vuln["scenario"].str.replace(
            "/", "-", regex=False
        )
        vuln_table = pa.Table.from_pandas(combined_vuln)

        logger.info(f"Writing vulnerable substations to {vuln_output_path}")

        parquet_fmt = pa.dataset.ParquetFileFormat()
        write_opts = parquet_fmt.make_write_options(compression="snappy")

        pa.dataset.write_dataset(
            data=vuln_table,
            base_dir=str(vuln_output_path),
            format=parquet_fmt,
            file_options=write_opts,
            partitioning=["scenario_partition"],
            max_rows_per_group=50_000,
            existing_data_behavior="overwrite_or_ignore",
        )

    logger.info("Final files created successfully")


def main():
    """Main function that runs reliability analysis for either alpha-beta uncertainty or multi-file scenarios."""
    aggregate_gdf = load_and_aggregate_tiles()
    aggregate_gdf["sub_id"] = aggregate_gdf["sub_id"].astype(str)

    config = get_simulation_config()

    logger.info(
        f"Using {'alpha-beta uncertainty' if USE_ALPHA_BETA_SCENARIO else 'regular multi-file'} scenario processing"
    )

    if USE_ALPHA_BETA_SCENARIO:
        import geopandas as gpd

        gic_file = gpd.read_file(ALPHA_BETA_GIC_FILE)

        gic_file = gic_file.rename(columns={"name": "sub_id"})
        gic_file["sub_id"] = gic_file["sub_id"].astype(str)

        gic_file[SCENARIOS] = gic_file[SCENARIOS] / 3.0
        logger.info("Converted total GIC to per-phase GIC by dividing by 3")

        if "latitude" not in gic_file.columns and "longitude" not in gic_file.columns:
            gic_file["longitude"] = gic_file.geometry.x
            gic_file["latitude"] = gic_file.geometry.y

        gic_df = pd.DataFrame(gic_file)
        required_cols = ["sub_id", "latitude", "longitude"] + SCENARIOS
        gic_df = gic_df[required_cols]

        gic_df[SCENARIOS] = gic_df[SCENARIOS].abs()

        merged = build_merged(aggregate_gdf, gic_df)

        scen_res, vuln_tbls = {}, []

        logger.info(
            f"Processing {len(SCENARIOS)} scenarios with uncertainty-quantified GIC"
        )

        for sc in SCENARIOS:
            logger.info(f"Processing scenario: {sc}")
            _, res, tbl = _simulate_scenario(
                sc,
                merged,
                config["theta0"],
                config["tolerance"],
                config["max_iterations"],
                config["batch_size"],
            )
            if res is not None:
                scen_res[sc] = res
                vuln_tbls.append(tbl)
            else:
                logger.warning(f"No results for scenario {sc}")

        if scen_res:
            summ_df = pd.DataFrame.from_dict(scen_res, orient="index")
            metrics = list(scen_res[list(scen_res.keys())[0]].keys())
            summ_df = summ_df.reindex(index=SCENARIOS)[metrics]

            ds = xr.Dataset(
                {m: (["scenario"], summ_df[m].to_numpy()) for m in metrics},
                coords={"scenario": SCENARIOS, "gic_realization": [0]},
            )
            OUT_PATH = DATA_LOC / OUTPUT_FILES["alpha_beta_uncertainty"]

            if OUT_PATH.exists():
                OUT_PATH.unlink()
            ds.to_netcdf(OUT_PATH, format="NETCDF4", mode="w")

            if vuln_tbls:
                for tbl in vuln_tbls:
                    tbl["gic_realization"] = 0
                combined_vuln = pd.concat(vuln_tbls, ignore_index=True)
                combined_vuln.to_parquet(
                    DATA_LOC / OUTPUT_FILES["vulnerable_alpha_beta_uncertainty"]
                )

            logger.info(
                "Alpha-beta uncertainty scenario processing completed successfully"
            )
            print("\nSummary Results:")
            print(summ_df)
        else:
            logger.error("No valid scenario results generated")

    else:
        gic_files = []

        if PROCESS_GND_FILES:
            files = sorted(GND_GIC_DIR.glob("ground_gic*.csv"))
            if files:
                gic_files.extend(files)
                logger.info(f"Found {len(files)} ground_gic files in: {GND_GIC_DIR}")
            else:
                logger.warning(f"No gnd_gic files found in: {GND_GIC_DIR}")
        else:
            for gic_dir_str in EFF_GIC_DIR:
                gic_dir = Path(gic_dir_str).expanduser()
                potential_files = sorted(
                    glob(str(gic_dir / "effective_gic_rand_*.csv"))
                )
                if potential_files:
                    gic_files = potential_files
                    logger.info(f"Found GIC files in: {gic_dir}")
                    break

        if not gic_files:
            logger.error("No GIC files found in any configured directory. Exiting.")
            return

        logger.info("Processing %d GIC realisations", len(gic_files))
        process_all_files(gic_files, aggregate_gdf)


if __name__ == "__main__":
    main()
