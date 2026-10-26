"""
Unified configuration for C-SWIM + Economic Analysis
Authors: Dennies Bor, Ed Oughton
"""

import os
import sys
import shutil
import logging
from pathlib import Path

import matplotlib.pyplot as plt

APP_NAME = "c-swim"
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
ECON_DATA_DIR = DATA_DIR / "econ_data"
DEFAULT_DATA_DIR = ECON_DATA_DIR

FIGURES_DIR = ROOT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

GANNON_GND_GIC_DIR = DATA_DIR / "gannon_gnd_gic"
GANNON_GND_GIC_DIR.mkdir(parents=True, exist_ok=True)

ADMITTANCE_DIR = DATA_DIR / "admittance_matrix"
GND_GIC_DIR = DATA_DIR / "gnd_gic"
HORTON_GRID_DIR = DATA_DIR / "horton_grid"

EFF_GIC_DIR = [
    DATA_DIR / "gic_eff",
    DATA_DIR / "final_gic" / "gic_eff",
]

ALPHA_BETA_GIC_FILE = (
    DATA_DIR / "regression" / "substations_with_gic_uncertainty_scaled.geojson"
)

LEAVE_OUT_SITES = ["GUA", "HON"]  # ALE is closed
P_TRAFO_BD = 0.01
cut_off_volt = 160
USE_OLD_SUB_DATA = True  # Use OSM data as of the Gannon storm

# Prefer whatever's on PATH (works across conda envs/OSes); fall back to the
# known Linux install path otherwise.
IPOPT_EXEC = shutil.which("ipopt") or "/home/pve_ubuntu/miniconda3/envs/spw-env/bin/ipopt"

PROCESS_GND_FILES = False

USE_ALPHA_BETA_SCENARIO = (
    os.getenv("USE_ALPHA_BETA_SCENARIO", "false").lower() == "true"
)

ALPHA_BETA_SCENARIOS = [
    "gic_75yr_conf_68_lower",
    "gic_75yr_mean_prediction",
    "gic_75yr_conf_68_upper",
    "gic_100yr_conf_68_lower",
    "gic_100yr_mean_prediction",
    "gic_100yr_conf_68_upper",
    "gic_125yr_conf_68_lower",
    "gic_125yr_mean_prediction",
    "gic_125yr_conf_68_upper",
    "gic_150yr_conf_68_lower",
    "gic_150yr_mean_prediction",
    "gic_150yr_conf_68_upper",
    "gic_175yr_conf_68_lower",
    "gic_175yr_mean_prediction",
    "gic_175yr_conf_68_upper",
    "gic_200yr_conf_68_lower",
    "gic_200yr_mean_prediction",
    "gic_200yr_conf_68_upper",
    "gic_225yr_conf_68_lower",
    "gic_225yr_mean_prediction",
    "gic_225yr_conf_68_upper",
    "gic_250yr_conf_68_lower",
    "gic_250yr_mean_prediction",
    "gic_250yr_conf_68_upper",
]

REGULAR_SCENARIOS = [
    "e_50-year-hazard A/ph",
    "e_75-year-hazard A/ph",
    "e_gannon-year-hazard A/ph",
    "e_100-year-hazard A/ph",
    "e_125-year-hazard A/ph",
    "e_150-year-hazard A/ph",
    "e_175-year-hazard A/ph",
    "e_200-year-hazard A/ph",
    "e_225-year-hazard A/ph",
    "e_250-year-hazard A/ph",
]

GDP_COLUMNS = [
    "GDP_AGR",
    "GDP_MINING",
    "GDP_UTIL_CONST",
    "GDP_MANUF",
    "GDP_TRADE_TRANSP",
    "GDP_INFO",
    "GDP_FIRE",
    "GDP_PROF_OTHER",
    "GDP_EDUC_ENT",
    "GDP_G",
]

EST_COLUMNS = [
    "EST_AGR",
    "EST_MINING",
    "EST_UTIL_CONST",
    "EST_MANUF",
    "EST_TRADE_TRANSP",
    "EST_INFO",
    "EST_FIRE",
    "EST_PROF_OTHER",
    "EST_EDUC_ENT",
    "EST_G",
]

DROP_COLUMNS = ["n_samples", "n_substations", "total_pop"]
CSV_DTYPES = {"sub_id": "category"}

DEFAULT_THETA0 = 75.0
DEFAULT_TOLERANCE = 0.1
DEFAULT_MAX_ITERATIONS = 20000
DEFAULT_BATCH_SIZE = 2000
DEFAULT_SAVE_BATCH_SIZE = 50

OUTPUT_FILES = {
    "alpha_beta_uncertainty": "scenario_summary_alpha_beta_uncertainty.nc",
    "alpha_beta_regular": "scenario_summary_alpha_beta.nc",
    "regular_all": (
        "scenario_summary_gnd_gic.nc"
        if PROCESS_GND_FILES
        else "scenario_summary_eff_gic.nc"
    ),
    "vulnerable_alpha_beta_uncertainty": "vulnerable_substations_alpha_beta_uncertainty.parquet",
    "vulnerable_alpha_beta_regular": "vulnerable_substations_alpha_beta.parquet",
    "vulnerable_regular": (
        "vulnerable_substations_gnd_gic"
        if PROCESS_GND_FILES
        else "vulnerable_substations_eff_gic"
    ),
}

# LUCY_DATA_LOC and SWERVE_DIR point to external data repos -- set these for your own setup.

LUCY_DATA_LOC = Path("/data/archives/nfs/2024-May-Storm-data")
SWERVE_DIR = Path("/data/archives/nfs/SWERVE/")

nerc_gic = LUCY_DATA_LOC / "nerc" / "gic"
nerc_mag_folder = LUCY_DATA_LOC / "nerc" / "mag"
tva_folder = LUCY_DATA_LOC / "tva"
tva_gic = tva_folder / "gic"
tva_mag = tva_folder / "mag"

peak_times_path = DATA_DIR / "peak_times_1.npy"
ground_gic_folder = DATA_DIR / "gannon_gnd_gic"
cache_file = DATA_DIR / "gannon_gic_processed" / "gic_data.npz"

VAL_FIGURES_DIR = FIGURES_DIR

DEFAULT_SITE_IDS = [
    10052,
    10076,
    10099,
    10238,
    10255,
    10587,
    10618,
    10619,
    10622,
    10063,
    10077,
    10079,
    10107,
    10112,
    10113,
    10114,
    10115,
    10402,
    10428,
    10438,
    10659,
    10660,
    10693,
    10181,
    10182,
    10184,
    10185,
    10186,
    10187,
    10195,
    10197,
    10200,
    10201,
    10203,
    10204,
    10207,
    10208,
    10212,
    10220,
    10249,
    10250,
    50100,
    50127,
    50112,
    50131,
    50132,
    50103,
    50104,
    50109,
    50115,
    50116,
    50117,
    50118,
    50119,
    50120,
    50122,
]

TVA_NAME_MAP = {
    "anderson": None,
    "montgomery": "Montgomery",
    "widowscreek2": "Widows Creek 2",
    "gleason": "Gleason",
    "raccoonmountain": "Raccoon Mountain",
    "johnsonville": "Johnsonville",
    "pinhook": "Pinhook",
    "paradise3": "Paradise",
    "southaven": "Southaven",
    "shelby": "Shelby",
    "eastpoint": "East Point",
    "bullrun": "Bull Run",
    "sullivan": "Sullivan",
    "weakley": "Weakley",
    "union": "Union",
    "rutherford": "Rutherford",
    "bradley": "Bradley",
}

DEFAULT_START_TIME = "2024-05-10T12:00:00"
DEFAULT_END_TIME = "2024-05-12T12:00:00"


def get_scenarios():
    return ALPHA_BETA_SCENARIOS if USE_ALPHA_BETA_SCENARIO else REGULAR_SCENARIOS


def get_simulation_config():
    return {
        "theta0": DEFAULT_THETA0,
        "tolerance": DEFAULT_TOLERANCE,
        "max_iterations": DEFAULT_MAX_ITERATIONS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "save_batch_size": DEFAULT_SAVE_BATCH_SIZE,
    }


def get_data_dir(subdir=None, econ=False):
    base = ECON_DATA_DIR if econ else DATA_DIR
    data_dir = base / subdir if subdir else base
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def setup_logger(name=APP_NAME, log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def setup_matplotlib():
    """Configure global matplotlib defaults for all plots."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


# Reproducible mapping configurations
REP_DATA_DIR = DATA_DIR / "rep_data"
TVA_DIR = REP_DATA_DIR / "tva"
UIUC_DIR = REP_DATA_DIR / "uiuc150"
TVA_BOUNDARY = REP_DATA_DIR / "tva_boundaries" / "TVA_Power_Service_Area_clean.geojson"
UIUC_XLSX = UIUC_DIR / "uiuc-150bus[1].xlsx"

# GIC solver constants
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
RG_REAL = 0.2
RG_SYNTHETIC = 10.0
POOL_GEN = ["GSU", "GY-D"]
POOL_TRANS = ["GY-GY", "Auto", "GY-GY-D"]

VOLTAGE_THRESHOLD = 229
SNAP_DISTANCE = 1500
KEEP_SYNTHETIC_ABOVE = 230
CRS_PROJ = "EPSG:3857"
CRS_GEO = "EPSG:4326"

for d in (TVA_DIR, UIUC_DIR, REP_DATA_DIR / "tva_boundaries"):
    d.mkdir(parents=True, exist_ok=True)
