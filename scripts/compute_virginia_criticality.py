"""
Phase 2 of ISLANDING_MITIGATION_PLAN.md: Substation Criticality Index (SCI).

Combines three signals, each already computed by an earlier pipeline step,
into one ranked score per substation:

    SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i) + w_t * norm(betweenness_i)

- GIC_risk_i        <- va_gic_summary.csv           (scripts/extract_virginia_results.py)
- econ_exposure_i   <- va_substation_exposure.gpkg  (scripts/run_virginia_areal_interpolation.py)
- betweenness_i     <- va_topology_metrics.csv      (scripts/build_virginia_network_graph.py)

Only substations with in_main_component=True in va_topology_metrics.csv are
eligible -- see ISLANDING_MITIGATION_PLAN.md's Phase 1 results for why (11
Virginia substations sit outside the main connected component and can't be
meaningfully ranked or islanded against a network they aren't part of here).

TODO markers below are the parts left for you to implement. The file
boilerplate (paths, column names, join keys) is wired up against what the
earlier pipeline steps actually produce, verified against
run_virginia_areal_interpolation.py and build_virginia_network_graph.py's
real output -- so you shouldn't need to go rediscover those.

Usage:
    python scripts/compute_virginia_criticality.py
"""

import geopandas as gpd
import pandas as pd

from configs import setup_logger
from configs.virginia_config import VA_ECON_DIR, VA_NETWORK_DIR

logger = setup_logger(log_file="logs/compute_virginia_criticality.log")

# Default equal weighting. Worth testing alternative weight sets once the
# pipeline runs end-to-end -- report how sensitive the ranking is, don't
# just present one weight choice as ground truth.
WEIGHTS = {"gic_risk": 1 / 3, "econ_exposure": 1 / 3, "betweenness": 1 / 3}

RANK_COL = "100-year-hazard A/ph_mean"  # same column plot_virginia_gic_risk.py ranks on
ECON_COLS = None  # TODO: which columns from va_substation_exposure.gpkg count as "exposure"?
                  # (hint: GDP_* + POP20 + EST_* all exist -- decide whether EST_* double-counts
                  # against GDP_* or adds independent signal, and document the choice)


def load_topology_metrics() -> pd.DataFrame:
    """sub_id, betweenness, in_main_component -- from Phase 1's output."""
    path = VA_NETWORK_DIR / "va_topology_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_virginia_network_graph.py first.")
    df = pd.read_csv(path)
    df["sub_id"] = df["sub_id"].astype(str)
    return df[["sub_id", "betweenness", "in_main_component"]]


def load_gic_risk() -> pd.DataFrame:
    """sub_id, gic_risk -- from va_gic_summary.csv's 100yr mean GIC column.

    TODO: read VA_ECON_DIR / "va_gic_summary.csv", pull out `RANK_COL`,
    return a two-column frame (sub_id, gic_risk). Check extract_virginia_
    results.py's summarize_and_rank() if you need to confirm the exact
    column names available -- va_gic_summary.csv also has a "_p95" variant
    per hazard column if you decide mean alone isn't the right signal.
    """
    raise NotImplementedError


def load_economic_exposure() -> pd.DataFrame:
    """sub_id, econ_exposure -- from va_substation_exposure.gpkg.

    TODO: read VA_ECON_DIR / "va_substation_exposure.gpkg" with
    geopandas.read_file(), sum whichever columns you decided on in
    ECON_COLS into one econ_exposure value per substation, return a
    two-column frame (sub_id, econ_exposure). The file is indexed/keyed by
    sub_id already (see run_virginia_areal_interpolation.py's
    `vs = voronoi_aea.set_index("sub_id", drop=False)`).
    """
    raise NotImplementedError


def normalize_min_max(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]. TODO: implement.

    Watch for the degenerate case where every value is identical
    (max == min) -- decide what that should return rather than dividing by
    zero, and note the choice.
    """
    raise NotImplementedError


def compute_sci(topology: pd.DataFrame, gic: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    """The actual index. TODO: implement.

    Steps:
    1. Filter `topology` to in_main_component == True.
    2. Join topology, gic, econ on sub_id (decide: inner join, or left join
       from topology with missing values filled/logged? A substation
       missing from one of the three sources is worth knowing about, not
       silently excluding).
    3. Normalize each of the three raw columns with normalize_min_max().
    4. Combine via WEIGHTS into one `SCI` column.
    5. Return a DataFrame with sub_id, the three raw values, the three
       normalized values, and SCI -- keep the intermediate columns rather
       than only the final score, so the ranking is auditable later.
    """
    raise NotImplementedError


def main():
    topology = load_topology_metrics()
    gic = load_gic_risk()
    econ = load_economic_exposure()

    sci = compute_sci(topology, gic, econ)
    sci = sci.sort_values("SCI", ascending=False).reset_index(drop=True)

    out_path = VA_ECON_DIR / "va_substation_criticality.csv"
    sci.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path}")
    logger.info(f"Top 10 by SCI:\n{sci.head(10).to_string(index=False)}")


if __name__ == "__main__":
    main()
