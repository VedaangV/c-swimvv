"""
Virginia C-SWIM Configuration
Geographic parameters and output paths for the Virginia+PJM adaptation.
Every other Virginia script imports from this module.
"""
import geopandas as gpd
from pathlib import Path

from configs import DATA_DIR

# PJM member states (Regional Transmission Organization covering Virginia).
# GIC flows across state lines through the interconnected PJM network, so the
# simulation domain is PJM, not just Virginia (confirmed by Dennies Bor).
PJM_STATES = ["VA", "MD", "DE", "NJ", "PA", "OH", "IN", "IL", "MI", "WV", "NC", "KY", "DC"]
PJM_FIPS = ["51", "24", "10", "34", "42", "39", "18", "17", "26", "54", "37", "21", "11"]

VA_FIPS = "51"
VA_STATE_ABBR = "VA"

# US state boundaries, already downloaded during econ preprocessing.
_STATES_PATH = DATA_DIR / "econ_data" / "raw_econ_data" / "tl_2022_us_state.zip"


def _load_states() -> gpd.GeoDataFrame:
    return gpd.read_file(_STATES_PATH)


def get_virginia_boundary():
    """Return the Virginia state boundary polygon (EPSG:4326)."""
    states = _load_states()
    return states[states["STUSPS"] == VA_STATE_ABBR].geometry.iloc[0]


def get_pjm_boundary():
    """Return the PJM region boundary as the union of member state polygons (EPSG:4326)."""
    states = _load_states()
    pjm = states[states["STUSPS"].isin(PJM_STATES)]
    return pjm.geometry.unary_union


def get_pjm_geodataframe() -> gpd.GeoDataFrame:
    """Return PJM member states as a GeoDataFrame (EPSG:4326)."""
    states = _load_states()
    return states[states["STUSPS"].isin(PJM_STATES)].copy()


# Approximate PJM bounding box (south, west, north, east), used for Overpass
# queries and any other bbox-scoped download. Derived from the member-state
# footprint: ME/southern-Great-Lakes edge down to the NC/KY/TN border, from
# the Atlantic coast to central Illinois/Indiana.
PJM_BBOX = (36.5, -95.0, 47.5, -73.0)  # (south, west, north, east)

# Virginia-specific output directories. Kept entirely separate from the
# existing US results under DATA_DIR — nothing here ever writes into the
# original data/grid_processed, data/admittance_matrix, data/gic, etc.
VA_DATA_DIR = DATA_DIR / "virginia"
VA_NETWORK_DIR = VA_DATA_DIR / "network"
VA_STORM_MAXES_DIR = VA_DATA_DIR / "storm_maxes"
VA_ADMITTANCE_DIR = VA_DATA_DIR / "admittance_matrix"
VA_GIC_DIR = VA_DATA_DIR / "gic"
VA_GND_GIC_DIR = VA_DATA_DIR / "gnd_gic"
VA_ECON_DIR = VA_DATA_DIR / "econ"
VA_STATISTICAL_ANALYSIS_DIR = VA_DATA_DIR / "statistical_analysis"
VA_FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "virginia"

# Phase 4 (ISLANDING_MITIGATION_PLAN.md) -- islanded-topology GIC re-solve
# output, kept entirely separate from the baseline va_gic/va_gnd_gic
# directories above so Island A's results never overwrite or get confused
# with the unmodified-topology baseline.
VA_ISLAND_A_GIC_DIR = VA_DATA_DIR / "gic_island_a"
VA_ISLAND_A_GND_GIC_DIR = VA_DATA_DIR / "gnd_gic_island_a"

# Phase 5 -- islanded-topology fragility/econ output. A dedicated
# subdirectory (not just different filenames within VA_ECON_DIR) because
# econ/preprocess/p_gic_files.py's DATA_LOC controls both checkpoint
# batches and final output paths together -- pointing it at a separate
# directory isolates both at once, matching the plan doc's "va_island_a_econ/
# subdirectory" design.
VA_ISLAND_A_ECON_DIR = VA_ECON_DIR / "island_a"


def get_directional_check_dirs(island_id: str) -> dict:
    """gic_dir/gnd_gic_dir for scripts/validate_island_directionality.py's
    single-iteration real-storm check (Phase 6c, ISLANDING_MITIGATION_PLAN.md)
    -- deliberately separate from get_island_dirs(island_id) below, which
    holds a candidate's full N_MONTE_CARLO-iteration Phase 4/5 output. This
    check only ever writes iteration 0 (one real admittance/GIC solve, not
    a full Monte Carlo), and is meant to run BEFORE a candidate has earned
    the expensive full run -- keeping it in its own directory means a
    single-iteration file can never be mistaken for (or collide with) a
    real, complete Phase 4 run for the same island_id."""
    dirs = {
        "gic_dir": VA_DATA_DIR / f"gic_directional_check_{island_id}",
        "gnd_gic_dir": VA_DATA_DIR / f"gnd_gic_directional_check_{island_id}",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_island_dirs(island_id: str) -> dict:
    """gic_dir/gnd_gic_dir/econ_dir for a given island id ("a", "b", ...) --
    island_id="a" resolves to exactly VA_ISLAND_A_GIC_DIR/VA_ISLAND_A_GND_GIC_DIR/
    VA_ISLAND_A_ECON_DIR (kept as standalone constants too, since Island A's
    already-computed results and a good deal of already-written code
    reference them directly -- this is for Phase 6's Island B and beyond,
    added once a second real island made copy-pasting three more constants
    and three more near-identical wrapper scripts per island clearly the
    wrong call vs. parameterizing once)."""
    dirs = {
        "gic_dir": VA_DATA_DIR / f"gic_island_{island_id}",
        "gnd_gic_dir": VA_DATA_DIR / f"gnd_gic_island_{island_id}",
        "econ_dir": VA_ECON_DIR / f"island_{island_id}",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


for _d in (
    VA_DATA_DIR,
    VA_NETWORK_DIR,
    VA_STORM_MAXES_DIR,
    VA_ADMITTANCE_DIR,
    VA_GIC_DIR,
    VA_GND_GIC_DIR,
    VA_ECON_DIR,
    VA_STATISTICAL_ANALYSIS_DIR,
    VA_FIGURES_DIR,
    VA_ISLAND_A_ECON_DIR,
    VA_ISLAND_A_GIC_DIR,
    VA_ISLAND_A_GND_GIC_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# EHV voltage threshold (same as the full US pipeline's cut_off_volt/230kV convention).
EHV_VOLTAGE_THRESHOLD = 230  # kV

# Same constants as configs/settings.py LINE_RESISTANCE / TRAFO_WINDING_R —
# duplicated here so this file has no other Virginia-side dependents beyond
# `configs`, matching the original plan. scripts/build_admittance_matrix.py
# has its own copies it actually uses; these are for network-build scripts
# that need a resistance lookup before the admittance step.
LINE_RESISTANCE = {
    765: 0.010,
    500: 0.0141,
    345: 0.0283,
    230: 0.0500,
    161: 0.0800,
    138: 0.0900,
}

TRAFO_WINDING_R = {
    "GY-GY": {"pri": 0.04, "sec": 0.06},
    "Auto": {"pri": 0.04, "sec": 0.06},
    "GY-GY-D": {"pri": 0.20, "sec": 0.10},
    "GY-D": {"pri": 0.05, "sec": float("inf")},
    "GSU": {"pri": 0.15, "sec": float("inf")},
}
RG_SYNTHETIC = 10.0  # grounding resistance for artificial (low-voltage) buses
RG_REAL = 0.2  # grounding resistance for real substations

# Max distance (meters) to snap a line endpoint to the nearest substation.
SNAP_DISTANCE_M = 1500

# Monte Carlo iterations for the GIC simulation. Matches the full US
# pipeline's 3000 for directly comparable statistical rigor (confidence
# intervals on par with the national paper) — bumped up from an initial 500
# used for early pipeline validation, now that the pipeline is proven
# end-to-end against real data.
N_MONTE_CARLO = 3000
