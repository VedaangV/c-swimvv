"""
Build the Virginia+PJM admittance matrix inputs.

Reuses scripts/build_admittance_matrix.py's process_substation_buses() and
get_transformer_samples() completely unmodified, pointed at
configs.virginia_config.VA_DATA_DIR instead of the US data/ directory.

Two adaptations were necessary and are called out explicitly here rather
than done silently:

1. build_admittance_matrix.load_and_process_data() unconditionally reads
   `<DATA_LOC>/grid_mapping.csv` — a hand-annotated table of per-substation
   transformer roles that, on the reference machine, was produced by manual
   GIS review, not by any script in this repo. It is only actually consumed
   when process_substation_buses(..., evan_data=True) is used (the
   alternate "Evan's verified data" transformer path); the default
   evan_data=False path this script uses (build_substation_buses(), a
   synthetic/Monte-Carlo transformer-type estimate) never reads the
   parsed values. Since Virginia has no such manual annotation, this script
   writes an empty-but-valid stub grid_mapping.csv so the loader doesn't
   crash, and documents that the values are unused for our path.

2. get_transformer_samples() writes its output to `<module-level DATA_LOC>
   / sample_network.pkl` — a module global, not a function parameter — so
   it has to be monkeypatched to VA_DATA_DIR before calling, exactly as the
   original adaptation plan anticipated for functions that "don't accept
   arguments" for their data location.

Run after scripts/build_virginia_network.py and
scripts/recompute_virginia_line_voltages.py (which together populate
VA_DATA_DIR/grid_processed/ with all six files load_and_process_data()
expects).
"""

import pandas as pd

import scripts.build_admittance_matrix as bam
from configs import setup_logger
from configs.virginia_config import VA_DATA_DIR, VA_ADMITTANCE_DIR, N_MONTE_CARLO

logger = setup_logger(log_file="logs/run_virginia_admittance.log")

REQUIRED_GRID_FILES = [
    "unique_sub_voltage_pairs.pkl",
    "df_lines_EHV.pkl",
    "trans_lines_pickle.pkl",
    "substation_to_line_voltages.pkl",
    "ss_df.pkl",
    "trans_lines_within_FERC_filtered.pkl",
]


def _check_inputs():
    grid_dir = VA_DATA_DIR / "grid_processed"
    missing = [f for f in REQUIRED_GRID_FILES if not (grid_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing VA+PJM network inputs under {grid_dir}: {missing}. "
            "Run scripts/build_virginia_network.py and "
            "scripts/recompute_virginia_line_voltages.py first."
        )


def _write_grid_mapping_stub():
    """See module docstring point 1: an empty, schema-valid grid_mapping.csv
    so load_and_process_data() can load it without erroring, given Virginia
    has no manual transformer-role annotation and doesn't need one for the
    evan_data=False (synthetic transformer sampling) code path used here."""
    path = VA_DATA_DIR / "grid_mapping.csv"
    if path.exists():
        return
    pd.DataFrame(columns=["Marker Label", "SS_ID", "Attributes"]).to_csv(path, index=False)
    logger.info(f"Wrote stub grid_mapping.csv (no manual transformer annotation for VA): {path}")


def main():
    _check_inputs()
    _write_grid_mapping_stub()

    # get_transformer_samples() reads a module-level DATA_LOC, not a
    # parameter — redirect it to Virginia's output tree.
    bam.DATA_LOC = VA_DATA_DIR

    logger.info(f"Building VA+PJM substation buses from {VA_DATA_DIR}...")
    substation_buses, bus_ids_map, sub_look_up, df_lines, substations_df = (
        bam.process_substation_buses(VA_DATA_DIR)
    )

    logger.info(f"Generating {N_MONTE_CARLO} Monte Carlo transformer samples...")
    bam.get_transformer_samples(
        substation_buses,
        n_samples=N_MONTE_CARLO,
        new_trafos=True,
    )

    n_nodes = len(sub_look_up)
    logger.info(f"Substations: {len(substations_df)}")
    logger.info(f"Buses: {len(bus_ids_map)}")
    logger.info(f"Transmission lines: {len(df_lines)}")
    logger.info(f"Admittance matrix dimensions: {n_nodes} x {n_nodes}")
    logger.info(f"Cached under: {VA_ADMITTANCE_DIR}")


if __name__ == "__main__":
    main()
