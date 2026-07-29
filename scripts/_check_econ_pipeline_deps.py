"""
One-off diagnostic: check whether the heavier econ-pipeline dependencies
(IPOPT solver, tobler/rasterio raster stack, and the NLCD raster file
itself) are actually available on this machine before investing in a
Virginia-scoped adaptation of econ/preprocess/p_areal_intp.py and
econ/scripts/econ_analysis.py.

Usage: python scripts\\_check_econ_pipeline_deps.py
"""

import os
import shutil

from configs import IPOPT_EXEC, ECON_DATA_DIR

print("=== IPOPT solver ===")
print(f"configs.IPOPT_EXEC = {IPOPT_EXEC}")
print(f"  exists at that path: {os.path.exists(IPOPT_EXEC)}")
found_on_path = shutil.which("ipopt")
print(f"  found on PATH: {found_on_path or 'NOT FOUND'}")

print("\n=== pyomo + solver ===")
try:
    from pyomo.opt import SolverFactory

    opt = SolverFactory("ipopt")
    print(f"  pyomo SolverFactory('ipopt').available(): {opt.available()}")
except Exception as e:
    print(f"  FAILED: {e!r}")

print("\n=== Raster / dasymetric stack ===")
for pkg in ("rasterio", "tobler", "geopandas"):
    try:
        mod = __import__(pkg)
        print(f"  {pkg}: OK (version {getattr(mod, '__version__', '?')})")
    except Exception as e:
        print(f"  {pkg}: FAILED - {e!r}")

print("\n=== NLCD raster data ===")
nlcd_path = ECON_DATA_DIR / "land_mask" / "Annual_NLCD_LndCov_2023_CU_C1V0.tif"
coarse_mask = ECON_DATA_DIR / "land_mask" / "coarse" / "nlcd_coarse_mask.tif"
print(f"  {nlcd_path}: exists = {nlcd_path.exists()}")
print(f"  {coarse_mask}: exists = {coarse_mask.exists()}")

print("\n=== SAM / technology tables ===")
for rel in [
    "supply_use_tables/use_tables.csv",
    "supply_use_tables/supply_tables.csv",
    "gross_output/gross_output.csv",
    "sam/us_balanced_sam.csv",
    "10sector/direct_requirements.csv",
]:
    p = ECON_DATA_DIR / rel
    print(f"  {p}: exists = {p.exists()}")
