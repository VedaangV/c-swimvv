"""
Diagnostic: prints scripts/select_island_by_redistribution_sensitivity.py's
seed_candidate_trace as a readable table -- the comparison across all
evaluated seed candidates (benefit, redistribution_cost, switching_cost,
net_score) that only ever gets saved in va_island_{id}.json, never
printed to the console log.

Usage:
    python scripts/show_seed_candidate_trace.py --island test_growth
"""

import argparse
import json

import pandas as pd

from configs.virginia_config import VA_NETWORK_DIR


def main(island_id: str = "a"):
    path = VA_NETWORK_DIR / f"va_island_{island_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with open(path) as f:
        island = json.load(f)

    trace = island.get("seed_candidate_trace")
    if not trace:
        raise ValueError(
            f"{path} has no seed_candidate_trace -- this island wasn't produced by "
            "select_island_by_redistribution_sensitivity.py (the deprecated SCI method "
            "doesn't record this)."
        )

    df = pd.DataFrame(trace).sort_values("net_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df["chosen"] = df["candidate"] == island["seed"]

    print(f"\n=== Island {island_id}: {len(df)} seed candidates evaluated, sorted by net_score ===")
    print(df.to_string(index=False))
    print(f"\nChosen seed: {island['seed']} (member_count ended at {island['member_count']})")

    stop_reason = island.get("stop_reason")
    if stop_reason:
        print(f"Growth stopped: {stop_reason}")
        rejected = island.get("rejected_candidate")
        if rejected:
            print(
                f"  Best rejected growth candidate: {rejected['candidate']} -- would-be net_score "
                f"{rejected['net_score']:.4f} (benefit={rejected['benefit']:.4f}, "
                f"redistribution_cost={rejected['redistribution_cost']:.4f}, "
                f"switching_cost={rejected['switching_cost']:.2f}) -- WAS considered, made things "
                "worse, not just 'no better'."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show a redistribution-sensitivity island's seed_candidate_trace.")
    parser.add_argument("--island", type=str, default="a")
    args = parser.parse_args()
    main(island_id=args.island)
