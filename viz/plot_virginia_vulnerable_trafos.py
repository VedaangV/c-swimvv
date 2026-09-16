"""
Virginia-scoped version of viz/plots.py's plot_vuln_trafos() -- substation
transformer failure-probability maps (100/150/200/250-year storms), sized
and colored by mean failure probability, with transmission lines colored
by voltage class.

Reuses plot_vuln_trafos() completely unmodified apart from two parameters
that function now accepts specifically so a regional caller doesn't need
to duplicate its ~180-line plotting loop: `extent` (this script passes
Virginia's own bounding box instead of the function's CONUS default) and
`projection` (a LambertConformal centered on Virginia instead of the
national center, same center used by viz/plot_virginia_hazard_maps.py).
FIGURES_DIR is monkeypatched to VA_FIGURES_DIR the same way viz/plot_
virginia_hazard_maps.py monkeypatches vplots.line_coordinates -- that
function reads FIGURES_DIR as a module global on viz.plots, not a
parameter, and this is a standalone script process so there's no need to
restore it afterward.

Two Virginia-specific inputs, each reimplemented rather than editing
shared code in place (this project's established convention):
1. Transmission line geometry + voltage class -- reuses viz/plot_
   virginia_hazard_maps.py's load_va_network_lines() unmodified (reads
   data/virginia/admittance_matrix/transmission_lines.csv, which already
   has the same "V" voltage-class column plot_vuln_trafos() expects,
   built by the same scripts/build_admittance_matrix.py logic as the
   national table).
2. Per-substation mean failure probability -- scripts/run_virginia_
   fragility.py's output (va_vulnerable_substations_gnd_gic, a parquet
   dataset partitioned by scenario) is per-realization; econ/scripts/
   l_prepr_data.py's process_vulnerability_chunks() (the exact function
   the national pipeline uses for the same reduction) collapses it to
   the sub_id/scenario/mean_failure_prob/latitude/longitude shape
   plot_vuln_trafos() expects -- reused unmodified here too.

Run after scripts/run_virginia_fragility.py. Untested against real data
in this session (no torch/cartopy environment here) -- report back
exactly what happens on first run, including how far it gets before any
error, same caution every other real-data Virginia script in this
project has needed on first run.

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
    """Reads the per-realization parquet dataset scripts/run_virginia_
    fragility.py wrote, then reduces it with the same process_vulnerability_
    chunks() the national pipeline uses -- see module docstring point 2."""
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

    # plot_vuln_trafos() reads this as a module global on viz.plots, not a
    # parameter -- see module docstring.
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
