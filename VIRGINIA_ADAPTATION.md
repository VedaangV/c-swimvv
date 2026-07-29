# Virginia C-SWIM Adaptation

Adapts the C-SWIM pipeline to Virginia, using the PJM region (VA + 12
neighboring states/DC) as the modeling domain and filtering to Virginia only
for reporting. This document supersedes the original planning doc
(`VIRGINIA_ADAPTATION_V2.md`) where the two disagree — the plan was written
before the actual scripts it needed to reuse (`build_admittance_matrix.py`,
`calc_storm_maxes.py`, `est_gic.py`, `stat_analysis.py`,
`preprocess/06_dl_power_grid_update.py`, `preprocess/08_p_power_grid.py`) had
been read in full, and several of its assumptions about their schemas and
behavior turned out not to hold. Confirmed scientific decisions from the
original plan (run SECS across all of North America, never filter MT sites,
model the full PJM footprint, filter to Virginia only for output) are
unchanged.

## Key design decision: `VA_DATA_DIR` mirrors the US `DATA_DIR` layout

`scripts/build_admittance_matrix.py`, `scripts/calc_storm_maxes.py`, and
`scripts/est_gic.py` are written as standalone pipelines against a single
`DATA_LOC` (== `configs.DATA_DIR`), with subdirectories named
`grid_processed/`, `admittance_matrix/`, `gic/`, `gnd_gic/`,
`statistical_analysis/`. Rather than reimplementing their numerics, every
Virginia script points these unmodified functions at
`configs.virginia_config.VA_DATA_DIR` in place of `DATA_DIR`, and
`VA_DATA_DIR` is laid out with the exact same subdirectory names. This is
why `VA_ADMITTANCE_DIR`, `VA_GIC_DIR`, `VA_GND_GIC_DIR`, and
`VA_STATISTICAL_ANALYSIS_DIR` in `configs/virginia_config.py` are simply
`VA_DATA_DIR / "<name>"` — they have to match what the reused functions
construct internally.

The exceptions are `data/EMTF/mt_pickle.pkl` (all 1696 US MT sites) and
`data/geomag_data/processed_geomag_data.nc` (observatory data): these are
read directly from the original `DATA_DIR`, never copied into
`VA_DATA_DIR`, because Dennies' guidance is to never filter or duplicate
them — the SECS interpolation and Delaunay line-weighting always use the
full site set.

## Deviations discovered from the original plan

1. **`grid_mapping.csv` is a required, hand-annotated file, not a generated
   one.** `build_admittance_matrix.load_and_process_data()` unconditionally
   reads `<DATA_LOC>/grid_mapping.csv` (a per-substation transformer-role
   table produced by manual GIS review on the reference machine). It is
   only actually *used* on the `evan_data=True` code path
   (`get_transformer_data_evan`); the default `evan_data=False` path
   (`build_substation_buses`, a Monte Carlo synthetic transformer-type
   estimate — what this adaptation uses throughout) never reads its parsed
   values. `scripts/run_virginia_admittance.py` writes an empty, schema-valid
   stub so the loader doesn't crash, and documents why that's safe.

2. **`trans_lines_pickle.pkl` is not a network-build artifact — it's
   produced by the storm-maxima step.** The plan assumed
   `build_virginia_network.py` (Step 2) would produce every file
   `build_admittance_matrix.py` needs. In the real pipeline,
   `scripts/calc_storm_maxes.read_transmission_lines()` builds this file
   from `df_lines_EHV.pkl` by attaching Delaunay interpolation weights
   against the MT site list — it's part of the SECS/electric-field step, not
   the grid-topology step. `scripts/recompute_virginia_line_voltages.py`
   (see deviation 4) produces it for Virginia.

3. **`preprocess/06`/`preprocess/08` filter geographically via a FERC-region
   spatial join (`data/nerc_gdf.geojson`), not a bounding box.**
   `scripts/build_virginia_network.py` reimplements the two loader functions
   that did the FERC sjoin (`load_and_process_transmission_lines`,
   `load_and_process_substations`) with a PJM-boundary spatial filter
   instead, stubbing the `REGION_ID`/`REGIONS` columns downstream code reads
   by name (`get_EHV_lines()`'s FERC-filter step) with a constant `"PJM"`
   value. Everything else in `preprocess/08_p_power_grid.py`
   (`graph_node_edges`, `get_nodes_edges`, `get_EHV_lines`,
   `prepare_substations_for_grid_analysis`) is geography-agnostic and reused
   unmodified via `importlib` (the files' numeric-prefixed names aren't
   valid `import` targets).

4. **Per-line voltage return periods can't be filtered from the US run —
   they have to be recomputed against the new VA+PJM line set.** This is
   the biggest gap in the original plan. Site-level B/E maxima
   (`maxB_arr`/`maxE_arr`) are a valid index-based filter because each row
   is one MT site, independent of network topology
   (`scripts/extract_virginia_storm_maxes.py`). But line-level voltage
   maxima (`maxV_arr`, and the `V_*` columns `est_gic.py` merges onto
   `df_lines`) are specific to the *original* US `LINE_ID`s and their
   Delaunay weights — Virginia's brand-new line set has neither.
   `scripts/recompute_virginia_line_voltages.py` replays
   `calc_storm_maxes.calculate_maxes()` for every storm in
   `storm_periods.csv` against the VA+PJM lines (still weighted against the
   *full* MT site list, per Dennies' guidance), fits return-period
   statistics with `stat_analysis.py`'s bootstrap machinery, and packages
   the result into `va_geomagnetic_data_return_periods.h5` with the same
   layout `est_gic.load_and_process_gic_data()` reads. This is a real,
   non-trivial compute cost (~90 storms × a full SECS+Delaunay pass each) —
   there is no way to avoid it without changing the line topology decision
   in Step 2, which Dennies confirmed should stay PJM-scoped.

5. **`est_gic.py` has no multiprocessing, but `calc_storm_maxes.py` does —
   and its `Pool` has no initializer, which matters because the target
   machine is Windows.** `est_gic.py` iterates Monte Carlo samples
   sequentially (parallelizing only per-return-period GIC calc within one
   iteration, via `ThreadPoolExecutor` — threads share the parent's memory
   regardless of platform, so this is fine as-is); `scripts/run_virginia_gic.py`
   follows the same sequential structure rather than inventing
   multiprocessing the reference implementation doesn't have.
   `calc_storm_maxes.py`, however, *does* use a bare `multiprocessing.Pool`
   whose worker function (`process_storm`) reads `MT_sites`/`df`/`obs_dict`/
   `n_trans_lines` as module globals — set only inside the script's own
   `if __name__ == "__main__":` guard. An early version of
   `scripts/recompute_virginia_line_voltages.py` (which reuses
   `calculate_maxes()` the same way) assumed `Pool()` forks and inherits
   those monkeypatched globals for free. That's true on Linux, but **not on
   Windows** — the reference machine's platform — where `Pool()` defaults to
   `spawn`: each worker re-imports `scripts.calc_storm_maxes` fresh, without
   the monkeypatched attributes, `calculate_maxes()` raises `NameError`, and
   `process_storm`'s own `try/except` silently swallows it per storm. The
   run "completes" with an all-zero `maxV_arr` instead of crashing —
   exactly the failure mode the original plan's `_init_worker` warning was
   about. Fixed in `recompute_virginia_line_voltages.py` with a proper
   `Pool(initializer=_init_worker, initargs=(...))` that re-applies the same
   globals inside every worker. **`calc_storm_maxes.py`'s own `main()` was
   left untouched** (out of scope here, and it's the already-run, "do not
   modify" reference pipeline) **but appears to have this same latent bug on
   Windows** — worth Dennies confirming whether the original US run actually
   went through this code path or a different, since-diverged version,
   since a silent all-empty result wouldn't necessarily have been obvious.

6. **Functions that need a data directory but don't take one as a
   parameter** (`build_admittance_matrix.get_transformer_samples()`,
   which reads its own module-level `DATA_LOC` for the cache path) are
   monkeypatched — e.g. `bam.DATA_LOC = VA_DATA_DIR` before calling — exactly
   as the original plan anticipated for cases where "functions don't accept
   arguments."

7. **Economic-model hookup is bounded, not complete — and the SAM-scope
   question has now been decided, provisionally.** `run_econ.py`'s analysis
   step depends on a much longer chain (`econ/preprocess/p_gic_files.py` →
   `p_areal_intp.py` → SAM/technology tables → `econ/scripts/econ_analysis.py`)
   than "filter ZCTA data, run `run_econ.py analysis`" suggested, and the
   real schema differs from what reading the source implied — the
   `STABBR` column `scripts/extract_virginia_results.py` originally assumed
   on `zcta_pop_20` doesn't exist in the actual pickle on the reference
   machine (confirmed via `scripts/_inspect_socioeconomic_data.py`); the fix
   derives Virginia's ZCTA set from `naics_est_gdp`/`regions_pop_df`'s
   `GeoFips` (county FIPS, first two digits = state FIPS) instead.

   **Decision (national SAM/technology tables, driven by a
   Virginia-localized shock):** rather than building a bespoke
   Virginia-regional SAM — which would require its own supply-use tables
   and econometric balancing, a separate research project on its own —
   Virginia's IO impact analysis should reuse the existing national
   SAM/technology tables and drive them with Virginia's own GIC exposure
   (population/business counts from Virginia ZCTAs, attributed via a
   Voronoi tessellation of *Virginia+PJM* substations rather than the
   national substation set). This is standard practice in sub-national
   disaster-economics impact studies. Flag for Dennies to override if he
   disagrees, but it's the only tractable choice on the ASSIP timeline.

   **What's actually still needed to wire this up, and why it isn't done
   yet:** unlike everything else in this document, the remaining chain
   isn't just a path-substitution problem.
   `econ/scripts/econ_analysis.py` solves the Leontief/Ghosh IO models via
   `pyomo` + an external `ipopt` solver binary, and `configs/settings.py`
   hardcodes that binary's path to a Linux location
   (`IPOPT_EXEC = "/home/pve_ubuntu/.../bin/ipopt"`) that will not exist on
   the reference Windows machine without a real install + config fix.
   `econ/preprocess/p_areal_intp.py` does raster-based dasymetric
   interpolation (`tobler.dasymetric.masked_area_interpolate`, tiled
   `rasterio` reads over the national NLCD land-cover raster) to map
   economic exposure onto substation Voronoi cells — a real geospatial
   subsystem, not more pandas. `scripts/_check_econ_pipeline_deps.py` checks
   whether IPOPT, `pyomo`, `rasterio`/`tobler`, the NLCD raster, and the
   SAM/technology CSVs are actually present and working before any of this
   is worth writing code against — run it and share the output before
   asking for the Virginia-scoped adaptation of `p_areal_intp.py`/
   `econ_analysis.py`.

## Repository layout (this repo, not the Windows path in the original plan)

```
configs/virginia_config.py                  Step 1 — PJM/VA parameters, output dirs
scripts/build_virginia_network.py           Step 2 — VA+PJM EHV network from HIFLD + OSM
scripts/extract_virginia_storm_maxes.py     Step 3a — PJM MT-site B/E filter (index-based)
scripts/recompute_virginia_line_voltages.py Step 3b — VA+PJM line V recompute + stat fit + h5
scripts/run_virginia_admittance.py          Step 4 — VA+PJM admittance matrix + MC samples
scripts/run_virginia_gic.py                 Step 5 — VA+PJM GIC Monte Carlo
scripts/extract_virginia_results.py         Step 6 — filter GIC + econ ZCTAs to Virginia
dc_blocker/README.md                        Placeholder — not implemented
```

Outputs land under `data/virginia/` (gitignored, same as every other `data/`
subdirectory) and `figures/virginia/`.

## Execution order

```bash
export PYTHONPATH="$(pwd)"

# Prerequisite (already done on the reference machine, per the original plan):
#   data/EMTF/mt_pickle.pkl, data/geomag_data/processed_geomag_data.nc,
#   data/storm_maxes/{maxB,maxE}_arr_testing_2.npy, data/kp_ap_indices/storm_periods.csv,
#   data/Electric__Power_Transmission_Lines/*.shp, data/econ_data/raw_econ_data/tl_2022_us_state.zip

python scripts/build_virginia_network.py
python scripts/extract_virginia_storm_maxes.py
python scripts/recompute_virginia_line_voltages.py   # expensive: ~90 storms recomputed
python scripts/run_virginia_admittance.py
python scripts/run_virginia_gic.py                    # expensive: N_MONTE_CARLO iterations
python scripts/extract_virginia_results.py
```

`configs/virginia_config.N_MONTE_CARLO` defaults to 500 (vs. 3000 for the US
run) given the ASSIP timeline — confirm the final sample size with Dennies
before treating results as final.

## Open items for Dennies

- `N_MONTE_CARLO` is set to 3000 (matching the national pipeline) as of the
  first full run against real data — confirm this is an acceptable final
  sample size, or a different number, for reported results.
- The SAM-scope question (item 7 above) has a provisional decision —
  national SAM/technology tables, driven by a Virginia-localized GIC
  shock — made without Dennies' sign-off, given the ASSIP timeline.
  Confirm or override before treating any resulting $/day figures as
  final.
- `IPOPT_EXEC` in `configs/settings.py` needs a real Windows path (or a
  working `ipopt` on `PATH`) before `econ/scripts/econ_analysis.py` can run
  at all — check with `scripts/_check_econ_pipeline_deps.py`.
- `Aaditya Bose` is building a parallel VA+PJM network independently per the
  original plan — worth a methodology sync given `build_virginia_network.py`
  makes several concrete choices (FERC-sjoin replaced with a PJM boundary
  filter, 150m substation-line buffer distance, Overpass server list) that
  another independent build may not match exactly.
