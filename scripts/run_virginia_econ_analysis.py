"""
Run Virginia's economic-impact IO analysis (Ghosh value-added shock, plus
two Leontief consumption/production variants) — the piece that actually
produces $/day-style headline numbers, comparable in *kind* to the national
paper's, driven by Virginia's own fragility Monte Carlo output.

Reuses econ/scripts/econ_analysis.py's get_confidence_intervals(),
run_ghosh_value_added(), run_leontief_consumption(), and
run_leontief_consumption_production() completely unmodified. Those read the
national SAM/technology/final-demand tables via econ_analysis.py's own
(deliberately untouched) DATA_LOC — exactly the "national tables, regional
shock" design the SAM-scope decision in VIRGINIA_ADAPTATION.md calls for.
The regional shock itself comes from
econ.scripts.l_prepr_data.load_gic_results(), whose *own* DATA_LOC is
monkeypatched to Virginia's tree so it reads
scripts/run_virginia_fragility.py's output instead of the national
gic_processing_batches/ directory.

Important: econ_analysis.py's _load_national_totals() (called internally by
the Leontief functions to get rho/phi) is intentionally left untouched —
it needs the true *national* population and sector GDP as the denominator
for "Virginia's share of the national economy," which is the entire point
of driving national IO tables with a regional shock rather than building a
Virginia-specific SAM.

Does not touch econ.models.cge_data_model's IPOPT-based CGE path —
econ_analysis.py's own main() doesn't call it either; it's a separate, more
advanced model not needed for these three headline analyses. (The IPOPT
path fixes made earlier — configs/settings.py and econ_analysis.py's
hardcoded solver path — still matter if that CGE path is ever exercised,
just not required for this script.)

Untested against real data in this session. Run after
scripts/run_virginia_fragility.py.
"""

import econ.scripts.l_prepr_data as lpd
from econ.scripts.econ_analysis import (
    get_confidence_intervals,
    run_ghosh_value_added,
    run_leontief_consumption,
    run_leontief_consumption_production,
)

from configs import setup_logger, USE_ALPHA_BETA_SCENARIO
from configs.virginia_config import VA_ECON_DIR


logger = setup_logger(log_file="logs/run_virginia_econ_analysis.log")


def main():
    lpd.DATA_LOC = VA_ECON_DIR

    logger.info("Loading Virginia fragility Monte Carlo results...")
    combined_ds, combined_vuln, vuln_table = lpd.load_gic_results()

    logger.info("Computing confidence intervals across Virginia's GIC realizations...")
    confidence_df = get_confidence_intervals(
        combined_ds, alpha_beta_scenario=USE_ALPHA_BETA_SCENARIO
    )

    logger.info("Running Ghosh value-added shock (national IO tables, Virginia GDP shock)...")
    ghosh_df = run_ghosh_value_added(confidence_df)

    logger.info("Running Leontief consumption shock (Virginia's share of national demand)...")
    leontief_cons_df = run_leontief_consumption(confidence_df)

    logger.info("Running Leontief consumption + production shock...")
    leontief_cp_df = run_leontief_consumption_production(confidence_df)

    confidence_df.to_csv(VA_ECON_DIR / "va_confidence_intervals.csv", index=False)
    ghosh_df.to_csv(VA_ECON_DIR / "va_ghosh_results.csv", index=False)
    leontief_cons_df.to_csv(VA_ECON_DIR / "va_leontief_consumption_results.csv", index=False)
    leontief_cp_df.to_csv(
        VA_ECON_DIR / "va_leontief_consumption_production_results.csv", index=False
    )

    logger.info(f"Saved Virginia economic-impact results to {VA_ECON_DIR}")


if __name__ == "__main__":
    main()
