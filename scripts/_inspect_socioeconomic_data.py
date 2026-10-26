"""
One-off diagnostic: prints the actual structure of socioeconomic_data.pkl so
filter_socioeconomic_data() in extract_virginia_results.py can be fixed to
match reality instead of guessing from p_econ_data.py's source.

Usage: python scripts\\_inspect_socioeconomic_data.py
"""

import pickle

from configs import ECON_DATA_DIR

path = ECON_DATA_DIR / "processed_econ" / "socioeconomic_data.pkl"
print(f"Loading {path}")

with open(path, "rb") as f:
    obj = pickle.load(f)

names = ["naics_est_gdp", "zcta_pop_20", "regions_pop_df", "zcta_business_gdf", "states_gdf"]

if isinstance(obj, tuple):
    print(f"Top-level: tuple of {len(obj)} objects (expected {len(names)}: {names})")
    items = list(enumerate(obj))
else:
    print(f"Top-level: NOT a tuple, it's a {type(obj)}")
    items = [(0, obj)]

for i, item in items:
    label = names[i] if i < len(names) else f"item_{i}"
    print(f"\n=== [{i}] {label} ===")
    print(f"type: {type(item)}")
    if hasattr(item, "columns"):
        print(f"columns: {list(item.columns)}")
        print(f"shape: {item.shape}")
        print(item.head(3))
    elif hasattr(item, "keys"):
        print(f"keys: {list(item.keys())[:20]}")
    else:
        print(repr(item)[:500])
