# DC Blocker Optimization (Planned Extension)

This module will implement optimal DC blocker placement for Virginia's power grid.

## Planned approach
Given Virginia GIC results from the main pipeline, find the minimum number
of DC blockers (capacitive devices installed at transformer neutrals) that
reduce GIC below safe thresholds at the highest-risk substations.

## Algorithm (greedy baseline)
1. Load Virginia GIC results
2. Rank substations by mean GIC across Monte Carlo iterations
3. For N = 5, 10, 20 blockers:
   - Place at highest-risk unprotected substation
   - Re-solve admittance matrix with grounding resistance → ∞
   - Update rankings
   - Repeat until N blockers placed
4. Plot risk vs. number of blockers curve

## Potential extensions (if time permits before August 7)
- GNN-based fast admittance matrix re-evaluation
- RL-based optimization replacing greedy search

## Status: NOT STARTED — implement after Virginia baseline is complete
