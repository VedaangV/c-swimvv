import geopandas as gpd

gdf = gpd.read_file(r"data\virginia\econ\va_substation_exposure.gpkg")
gdf["sub_id"] = gdf["sub_id"].astype(str)
gdf["area_km2"] = gdf.geometry.area / 1e6

row = gdf[gdf["sub_id"] == "707088636"]
print("707088636 cell area (km2):", row["area_km2"].values[0])
print(gdf["area_km2"].describe())
print("Percentile rank vs all substations:", (gdf["area_km2"] < row["area_km2"].values[0]).mean())