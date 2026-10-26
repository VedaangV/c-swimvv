"""
Virginia-scoped version of viz/plots.py's plot_vuln_trafos() -- substation
transformer failure-probability maps (100/150/200/250-year storms), sized
and colored by mean failure probability, with lines colored by voltage.

Reuses plot_vuln_trafos() unmodified, passing Virginia's own extent/
projection instead of the CONUS defaults. FIGURES_DIR is monkeypatched to
VA_FIGURES_DIR since that function reads it as a module global, not a
parameter.

Line geometry comes from plot_virginia_hazard_maps.load_va_network_lines();
vulnerability data is reduced from run_virginia_fragility.py's per-
realization parquet output via econ/scripts/l_prepr_data.py's
process_vulnerability_chunks() (same function the national pipeline uses).

Run after scripts/run_virginia_fragility.py.

Usage:
    python viz/plot_virginia_vulnerable_trafos.py
"""

import pandas as pd
import cartopy.crs as ccrs

import viz.plots as vplots
from viz.plot_virginia_hazard_maps import load_va_network_lines
from econ.scripts.l_prepr_data import process_vulnerability_chunks
from configs import setup_logger, setup_matplotlib
from configs.virginia_config import VA_ECON_DIR, VA_FIGURES_DIR, get_virginia_boundary

logger = setup_logger(log_file="logs/plot_virginia_vulnerable_trafos.log")

VULN_DATA_PATH = VA_ECON_DIR / "va_vulnerable_substations_gnd_gic"


def load_virginia_vulnerability_data() -> pd.DataFrame:
    """Reads the per-realization parquet dataset from run_virginia_fragility.py
    and reduces it with the same process_vulnerability_chunks() the national
    pipeline uses."""
    if not VULN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {VULN_DATA_PATH}. Run scripts/run_virginia_fragility.py first."
        )
    combined_vuln = pd.read_parquet(VULN_DATA_PATH)
    combined_vuln["sub_id"] = combined_vuln["sub_id"].astype(str)
    logger.info(f"Loaded {len(combined_vuln)} per-realization vulnerability rows from {VULN_DATA_PATH}.")

    mean_vuln_all = process_vulnerability_chunks(combined_vuln)
    logger.info(f"Reduced to {len(mean_vuln_all)} sub_id x scenario mean-failure-probability rows.")
    return mean_vuln_all


def main():
    setup_matplotlib()

    logger.info("Loading Virginia transmission line geometry...")
    df_lines = load_va_network_lines()

    logger.info(f"Loading Virginia vulnerability data from {VULN_DATA_PATH}...")
    mean_vuln_all = load_virginia_vulnerability_data()

    va_boundary = get_virginia_boundary()
    minx, miny, maxx, maxy = va_boundary.bounds
    pad = 0.3
    va_extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
    va_projection = ccrs.LambertConformal(central_longitude=-78.5, central_latitude=37.5)

    # plot_vuln_trafos() reads this as a module global, not a parameter
    vplots.FIGURES_DIR = VA_FIGURES_DIR

    vplots.plot_vuln_trafos(
        mean_vuln_all, df_lines, file_suffix="virginia", extent=va_extent, projection=va_projection,
    )
    logger.info(
        f"Saved {VA_FIGURES_DIR / 'vulnerable_trafos_virginia.png'} and "
        f"{VA_FIGURES_DIR / 'vulnerable_trafos_virginia.pdf'}"
    )


if __name__ == "__main__":
    main()
