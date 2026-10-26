import pandas as pd

sci = pd.read_csv(r"data\virginia\econ\va_substation_criticality.csv")
topo = pd.read_csv(r"data\virginia\network\va_topology_metrics.csv")
topo["sub_id"] = topo["sub_id"].astype(str)
sci["sub_id"] = sci["sub_id"].astype(str)
merged = sci.merge(topo[["sub_id", "latitude", "longitude"]], on="sub_id")

# the SW Virginia point from the map is near (36.86, -81.42)
row = merged.loc[((merged["latitude"] - 36.86).abs() < 0.05) & ((merged["longitude"] + 81.42).abs() < 0.05)]
print(row.T)