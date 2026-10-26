"""
Build the Virginia+PJM admittance matrix inputs.

Reuses build_admittance_matrix.py's process_substation_buses() and
get_transformer_samples() unmodified, pointed at VA_DATA_DIR instead of the
US data/ directory. Writes a stub grid_mapping.csv since Virginia has no
manual transformer-role annotation (unused on the synthetic-sampling path
we use here anyway). Run after scripts/build_virginia_network.py and
scripts/recompute_virginia_line_voltages.py.
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
    """Empty, schema-valid grid_mapping.csv so load_and_process_data()
    doesn't error -- Virginia has no manual transformer-role annotation."""
    path = VA_DATA_DIR / "grid_mapping.csv"
    if path.exists():
        return
    pd.DataFrame(columns=["Marker Label", "SS_ID", "Attributes"]).to_csv(path, index=False)
    logger.info(f"Wrote stub grid_mapping.csv (no manual transformer annotation for VA): {path}")


def main():
    _check_inputs()
    _write_grid_mapping_stub()

    # get_transformer_samples() reads DATA_LOC as a module global, not a param
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
