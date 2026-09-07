# Virginia C-SWIM Adaptation — Project Documentation

This document is a comprehensive, end-to-end account of the Virginia
adaptation of C-SWIM (Coupled Space Weather Impact Model): the pipeline
that was built, the grid-vulnerability and economic-impact results it
produced, and the multi-strategy islanding mitigation investigation
carried out on top of it. It's written to be readable on its own, without
needing the national C-SWIM codebase's internals memorized first. Where a
number, script name, or finding is stated, it is grounded in real code or
real output in this repository, not projected — anything still pending
real-data confirmation is flagged explicitly as such.

**Paper this adapts**: Oughton, E.J., Bor, D.K., Weigel, R., Gaunt, C.T.,
Dogan, R., Huang, L., Love, J.J., & Wiltberger, M. (2024). "Major Space
Weather Risks Identified via Coupled Physics-Engineering-Economic
Modeling." *arXiv:2412.18032*. That paper builds the national-scale
pipeline (all of CONUS); this adaptation is the first state-level
application of the same methodology, scoped to Virginia within the PJM
interconnection.

---

## Contents

1. [Project overview](#1-project-overview)
2. [Pipeline architecture: from national C-SWIM to Virginia+PJM](#2-pipeline-architecture-from-national-c-swim-to-virginiapjm)
3. [Grid vulnerability assessment](#3-grid-vulnerability-assessment)
4. [Economic impact assessment](#4-economic-impact-assessment)
5. [Islanding mitigation investigation](#5-islanding-mitigation-investigation)
6. [Pipeline adaptation details (deviations from the original plan)](#6-pipeline-adaptation-details-deviations-from-the-original-plan)
7. [Repository guide](#7-repository-guide)
8. [Open items](#8-open-items)

---

## 1. Project overview

Geomagnetically induced currents (GICs) — quasi-DC currents driven into
the power grid by geomagnetic storms — pose a risk to high-voltage
transformers: current above a threshold saturates a transformer's core,
causing hot-spot heating and, in the worst case, permanent damage that
can take 12-24+ months to repair. National-scale risk assessments (like
the C-SWIM paper above) establish that this risk is real and material,
but they aren't built to answer state-level, actionable questions: which
specific substations in a given state are most exposed, what a storm
actually costs that state's economy per day, and whether a concrete
mitigation action (in this project's case, temporarily islanding part of
the grid during a storm) can reduce that cost.

This project answers those questions for Virginia, using the real PJM
interconnection topology (Virginia plus the 12 neighboring
states/DC that share its synchronous grid) as the electrical domain,
filtering to Virginia-specific reporting throughout. Three phases:

1. **Adapt the pipeline** (Section 2) — rebuild every stage of national
   C-SWIM (network topology, admittance/GIC solve, storm statistics,
   economic IO models) against a Virginia+PJM-scoped dataset instead of
   the full national one, reusing the original numerical code wherever
   possible rather than reimplementing it.
2. **Assess grid vulnerability and economic impact** (Sections 3-4) —
   run the adapted pipeline against 92 real historical storms
   (1985-2024) plus statistically-derived return-period hazard curves
   (25- to 250-year), rank Virginia's substations by GIC risk, and
   translate storm-driven transformer damage into whole-Virginia $/day
   economic loss via Ghosh and Leontief input-output models.
3. **Investigate islanding as a mitigation strategy** (Section 5) — the
   large, still-active part of this project. Explore whether temporarily
   disconnecting a chosen region of Virginia's grid during a storm's
   peak-risk window (protective islanding, not resilience/local-generation
   islanding — see Section 5.1) can reduce that economic loss. This
   involved building and testing two entirely different substation
   *selection* strategies, six-plus real candidate islands, a literature
   review that explained a real, reproducible failure mode, and a from-
   scratch redistribution-aware replacement algorithm — currently mid-
   validation against real data.

**Headline results established so far**:
- A 100-year return-period storm is projected to cost Virginia
  **$145.5M/day**; a 250-year event, **$230.6M/day** (baseline, no
  mitigation).
- The first islanding selection strategy tried (a criticality-index-driven
  greedy method) was tested against six real candidate regions and found
  to be *unreliable* — mostly net-negative for whole-Virginia economic
  impact — for a diagnosable, literature-confirmed reason (Section 5.3).
- A second, redistribution-sensitivity-aware selection algorithm was
  built to fix that specific flaw (Section 5.4); it is mechanically
  validated (its own real-data test run correctly avoided a substation
  independently confirmed to be a real GIC-redistribution hotspot from
  the first strategy's results) but **has not yet been run through the
  real fragility/economic pipeline** — that is the immediate next step,
  not yet a reported result.

---

## 2. Pipeline architecture: from national C-SWIM to Virginia+PJM

National C-SWIM is a six-stage pipeline (collect geomagnetic/grid data →
derive extreme geoelectric field scenarios statistically → build an
electrical admittance model of the EHV grid → simulate GIC under
synthetic and historical storms → validate against real measurements →
socio-economic impact modeling via IO/CGE models), run once across all
of CONUS. The Virginia adaptation reruns every stage of that same
pipeline against a smaller, Virginia+PJM-scoped dataset, reusing the
original numerical code (SECS field interpolation, the nodal-admittance
GIC solve, the IO economic models) as directly as possible rather than
reimplementing any of it — the changes are almost entirely about *what
data* feeds those functions, not the functions themselves. Section 6
documents every specific deviation this required in detail; the shape
of the adaptation is:

| Stage | National C-SWIM | Virginia adaptation |
|---|---|---|
| Network topology | All of CONUS | PJM interconnection footprint (VA + 12 neighboring states/DC), via a spatial filter against `data/nerc_gdf.geojson` |
| Storm/field data | 1616 US MT sites, full continental SECS interpolation | Same full site set and interpolation (never filtered — geomagnetic field interpolation needs the whole spatial context) |
| Line-level voltage statistics | Precomputed for the national line set | **Recomputed from scratch** against Virginia+PJM's own (different) line set — the single most expensive adaptation step |
| GIC solve | National admittance matrix | Virginia+PJM admittance matrix (same Y-matrix/Ohm's-law physics, smaller network) |
| Economic model | National SAM/technology IO tables, national population/GDP shock | **Same national SAM/technology tables** (building a bespoke Virginia-regional SAM was judged out of scope), but the shock itself is Virginia-localized (Virginia ZCTA population/GDP, attributed to Virginia+PJM substation service areas) |
| Reporting | All of CONUS | Filtered to Virginia only |

The GIC physics itself is a quasi-DC resistive circuit problem: a storm's
geoelectric field induces a voltage along each transmission line
proportional to its length and orientation; solving Ohm's/Kirchhoff's
laws over the network's nodal admittance (Y) matrix gives the resulting
current at every substation and transformer. This is exactly the same
model class the islanding investigation's redistribution-sensitivity
algorithm later reuses directly (Section 5.4) — cutting a transmission
line is a real, physical edit to that same Y-matrix, not a separate
approximation.

See Section 6 for the specific, itemized deviations this adaptation
required (grid_mapping.csv handling, the FERC-sjoin replacement, the
Windows multiprocessing bug found and fixed, the SAM-scope decision, and
more) — that section is written for someone extending or auditing this
codebase, while this overview is written for understanding the project's
shape and results.

---

## 3. Grid vulnerability assessment

### 3.1 Storm data: 92 real historical storms, statistically extended

`data/kp_ap_indices/storm_periods.csv` contains **92 real historical
geomagnetic storm periods**, spanning 1985-04-19 through 2024-05-09 (the
last entry is the real May 2024 "Gannon" storm). `scripts/calc_storm_maxes.py`
(reused from the national pipeline, via SECS field interpolation against
all 1616 US MT sites) computes the maximum magnetic field (B), electric
field (E), and line voltage (V) each of these 92 storms would have
produced. `scripts/recompute_virginia_line_voltages.py` is the Virginia-
specific step that replays this line-voltage calculation against
Virginia+PJM's own transmission-line set (a real, non-trivial recompute —
Virginia's lines are a brand-new set with no relationship to the national
line IDs the original per-storm voltage data was computed against).

Two distinct kinds of storm scenario are derived from this same 92-storm
catalog, and downstream results (Sections 4-5) consistently label them
differently:

- **Statistically-derived return-period scenarios** (labeled `50-year`,
  `75-year`, `100-year`, ... `250-year` in every downstream output; the
  full range is `np.arange(25, 251, 25)`). `scripts/stat_analysis.py`
  bootstrap-resamples the 92 storms' magnitudes (`n_samples=100`) and
  fits a lognormal distribution to the pooled sample (`powerlaw.Fit(...)
  .lognormal`), then reads off the magnitude at each return period's
  exceedance probability. **No individual historical storm is "labeled"
  as a return period** — all 92 are pooled into one statistical
  distribution, and the return-period values are extrapolated from that
  fitted curve.
- **Real named historical events**, run through the same pipeline using
  their *actual* computed magnitude (not a statistical estimate):
  Halloween (2003), Gannon (2024), and St. Patrick's Day. These are
  pulled out of the 92-storm catalog by date-matching
  (`stat_analysis.get_event_indices()`). Only `gannon-year` is carried
  through as a labeled scenario in the final economic-impact output
  (`va_ghosh_results.csv`); Halloween and St. Patrick's Day appear in the
  raw GIC-comparison diagnostics used during the islanding investigation
  but not in the 10-scenario econ Monte Carlo scenario set.

### 3.2 GIC risk simulation

`scripts/run_virginia_admittance.py` builds the Virginia+PJM nodal
admittance (Y) matrix from real transmission-line resistance data
(`transmission_lines.csv`'s `R` column) and synthesizes a Monte Carlo
sample of plausible transformer configurations per substation (since
exact winding/grounding configuration isn't public data at the needed
granularity). `scripts/run_virginia_gic.py` solves the resulting
resistive-circuit GIC problem for every (storm scenario × Monte Carlo
realization) combination — `N_MONTE_CARLO = 3000`, matching the national
pipeline's sample size — producing both winding GIC (per-transformer)
and **ground GIC** (the current flowing to earth through a substation's
grounding connection — the physical quantity that drives transformer
heating risk, and the one used throughout the fragility/risk-ranking
work below). `scripts/extract_virginia_results.py` filters this Monte
Carlo output down to Virginia-only substations for reporting.

### 3.3 Substation risk ranking

Two distinct risk-ranking approaches were used across this project, at
different points:

- **Raw ground-GIC magnitude** (`va_ground_gic_summary.csv`'s
  `GIC_100_mean` column — 100-year mean ground GIC per substation) is the
  simplest, most directly physical risk signal, and is what the current
  (post-deprecation) islanding selection algorithm uses directly
  (Section 5.4).
- **Substation Criticality Index (SCI)** — a weighted composite index
  (GIC risk + economic exposure + betweenness centrality + optionally
  critical-load density) built specifically to drive the first islanding
  selection strategy. SCI is documented in full, including why it was
  ultimately retired, in Section 5.2-5.3 and
  `scripts/deprecated_sci_islanding/README.md`. It is no longer part of
  the active pipeline, though the underlying raw signals it was built
  from (GIC risk, economic exposure) remain in active use via
  `scripts/virginia_risk_data.py`.

---

## 4. Economic impact assessment

### 4.1 Methodology

Storm-driven GIC is translated into economic loss via a fragility →
input-output (IO) chain, reusing the national pipeline's economic
modeling code unmodified (`econ/scripts/econ_analysis.py`,
`econ/preprocess/p_gic_files.py`) against Virginia-localized inputs:

1. **Fragility** (`scripts/run_virginia_fragility.py`): converts each GIC
   Monte Carlo realization into a per-substation transformer failure
   probability, using continuous fragility curves (probability of damage
   as a function of GIC magnitude) — not a hard threshold. This is a
   deliberate modeling choice with real consequences discussed in Section
   5.3: a continuous curve prices *every* redistributed amp, sub-
   threshold or not, unlike a threshold-based damage model.
2. **Economic exposure** (`scripts/run_virginia_areal_interpolation.py`):
   maps Virginia ZCTA-level population and GDP data onto each Virginia+
   PJM substation's own Voronoi service-area polygon (dasymetric
   interpolation against NLCD land-cover data, via `tobler`/`rasterio`),
   establishing how much of Virginia's economic activity is "exposed" at
   each substation.
3. **IO impact models** (`scripts/run_virginia_econ_analysis.py`): three
   distinct national-SAM-table-based models, all driven by the Virginia-
   localized shock from steps 1-2:
   - **Ghosh value-added shock** — supply-side, "what happens to
     production when a supply disruption hits."
   - **Leontief consumption shock** — demand-side, "what happens to
     downstream consumption."
   - **Leontief consumption + production shock** — combined.

   Results are reported across 10 economic sectors (AGR, MINING,
   UTIL_CONST, MANUF, TRADE_TRANSP, INFO, FIRE, PROF_OTHER, EDUC_ENT, G)
   and three confidence levels (mean, p5, p95) per scenario.

**Why national SAM tables, not a bespoke Virginia-regional one**:
building a proper Virginia-specific social accounting matrix would
require its own supply-use tables and econometric balancing — a separate
research project. The adaptation instead reuses the existing national
SAM/technology tables (same ones the national C-SWIM paper uses) and
drives them with a Virginia-localized shock — standard practice in sub-
national disaster-economics impact studies, though it does mean the
*ratio* logic ("Virginia's share of the national economy") is baked into
the model, not something Virginia-specific results can fully escape.

### 4.2 Headline results

- **100-year return-period storm: $145.5M/day** (baseline, mean
  confidence, no mitigation).
- **250-year return-period storm: $230.6M/day**.
- Losses scale monotonically with storm severity across the full 25- to
  250-year range, as expected from the fragility-curve mechanism (higher
  GIC → higher failure probability → more transformer damage → larger
  supply/demand shock).

These are the **baseline** (unmitigated) figures every islanding
candidate in Section 5 is compared against — an islanding result is
reported as the *change* in this whole-Virginia $/day figure once a
region's boundary is (hypothetically, temporarily) disconnected, not as
a standalone number.

---

## 5. Islanding mitigation investigation

### 5.1 Motivation and scope: protective, not resilience, islanding

The core idea: during a storm's peak-risk window, deliberately switch
open a chosen set of transmission lines, temporarily disconnecting a
region of the grid. This breaks the conductive path GIC needs to reach
that region's transformers, protecting them from damage, at the cost of
a short, planned power outage for that region (hours, not days) — a
tradeoff worth making if the alternative is a transformer failure that
can take 12-24+ months to repair. This is **protective islanding**, and
it is the *only* thing modeled anywhere in this project, for every
candidate discussed below: the GIC/fragility/econ simulation has no
concept of a disconnected region continuing to operate on local
generation (**resilience islanding**) — cutting a boundary changes
current *redistribution* in the physics, nothing more. Local generation,
where it enters this project at all (Section 5.2's `require_local_
generation`, Section 5.4's benefit-cost script), only ever affects which
substations get *selected*, or discounts an *assumed* outage cost after
the fact — never what the simulation itself computes.

This framing was clarified against real literature precedent: Oughton et
al. 2017 (Space Weather) and MacManus 2023/2025 establish protective
islanding as an operational GIC mitigation practice (New Zealand's grid
operator has used it), and arXiv:2507.12495 (Oughton, Renton, Mac Manus,
Bor, Rodger) is the closest direct precedent for combining islanding
with an economic-impact assessment — its own strongest scenario
(Scenario 5) is anchored on Manapouri, a real hydro generation site
directly supplying a protected load (the Tiwai Point aluminum smelter),
a materially different and more effective mechanism than protective-only
islanding (see Section 5.2's Island F).

Islanding-specific precedent combining it directly with GIC risk doesn't
exist as a dedicated paper — an honest, named gap, not a hidden one; this
project's actual contribution is combining separately-established ideas
(topology reconfiguration for GIC mitigation, generally: arXiv:1701.01469;
protective islanding as an operational practice) into an under-explored
specific application, and rigorously testing it against real data.

### 5.2 Phase 1: SCI-driven greedy selection (retired)

**Design.** A Substation Criticality Index (SCI) was computed per
eligible (Virginia, main-connected-component) substation as a weighted
blend:

```
SCI_i = w_e * norm(GIC_risk_i) + w_x * norm(econ_exposure_i)
      + w_t * norm(betweenness_i) [+ w_c * norm(critical_load_i)]
```

(equal weights, three-term or four-term depending on whether
critical-facility data was available). A seeded greedy region-growing
algorithm (ancestry: Clauset 2005, "Finding local community structure in
networks") then: seeded on `argmax(SCI)`, initialized the island as that
one substation, and iteratively added whichever boundary-adjacent
substation most improved the ratio of SCI-weighted value (summed SCI
over island members) to voltage-weighted boundary cost (kV/100 per
physical line cut), stopping the moment no addition improved that ratio.

**Six real candidates were selected and fully evaluated** (topology-
edited GIC re-solve → fragility Monte Carlo → Ghosh/Leontief econ
impact, compared against the unmitigated baseline):

| Island | Members | Real result |
|---|---|---|
| A | 3 (seed `708152212`) | Net-negative for whole-Virginia $/day in **every** scenario checked |
| B | 4 | Mostly negative; genuinely positive for the real Gannon storm scenario |
| A+B combined | 7 | Worse than either individually — confirmed additivity (AB's diff ≈ A's diff + B's diff, within ~0.5pp across 30 scenarios), meaning stacking a net-negative island onto a net-positive one *erased* the win rather than averaging out |
| C, D, E | — | Negligible effect in either direction |
| F (generation-anchored, `require_local_generation`) | 11 (seed `81595538`, 175 MW nearby EIA-860 generation) | 6/10 scenarios net-positive, **including the real Gannon storm (+$1.15M/day)**, with a coherent severity-scaling pattern (losses at low severity, growing benefit at high severity) |

Island F's positive result initially looked like evidence that
generation-anchored seeding was the fix — but two follow-up
investigations revised that conclusion:

1. **The physics never modeled generation continuity at all** (Section
   5.1) — `require_local_generation` only changed *which* 11 substations
   got selected as seed and grown, landing the *same* SCI-greedy
   algorithm in a structurally different, better-performing region of
   the graph than Island A's pure-SCI seed. That's a real effect, but a
   different claim than "generation kept the lights on."
2. **A missing benefit-cost term was implemented and found to be
   second-order.** The plan had always intended a proper benefit-cost
   accounting (switching-labor cost + the island's own temporary-outage
   cost, discounted by matched generation coverage) but it had never
   actually been coded — `scripts/compute_island_benefit_cost.py` fixed
   that gap. Run against real data: Island A's total one-time cost
   ($0.059M) and Island F's ($0.324M, only 62% generation-discounted
   since whole-island generation coverage, 38%, is far weaker than the
   seed substation's own 11.2x) were both **3-4 orders of magnitude
   smaller** than the fragility-driven benefit swings (hundreds to
   thousands of $M once integrated over a plausible transformer-repair
   duration) — meaning whether an island helps or hurts is determined
   almost entirely by the raw redistribution physics, not by the cost
   accounting or by generation.

### 5.3 Diagnosis: why SCI-driven greedy selection failed

Two compounding problems, established over the course of testing all six
candidates above:

1. **SCI's betweenness term used the wrong physics.** It was shortest-
   path betweenness (Freeman 1977) — "how much traffic takes the single
   cheapest route through this node." GIC doesn't route that way: it's
   Ohm's-law current splitting across every parallel path simultaneously,
   in proportion to conductance. `scripts/deprecated_sci_islanding/
   check_redistribution_proxy.py` built a current-flow-betweenness proxy
   (Newman 2005; Brandes & Fleischer 2005 — the electrically-faithful
   analog, provably equivalent to modeling the graph as a resistor
   network) to test this; it showed real but incomplete predictive power.

2. **The deeper problem: no term anywhere in SCI or the greedy growth
   ratio ever modeled where current goes once a boundary is cut.** Every
   real run showed the same pattern — cutting a boundary concentrated
   redistributed current onto a small, consistent set of substations
   elsewhere (`96396693` for Island A, `96284749` for Island F, both 25+
   km away — too far for a local boundary tweak to route around) large
   enough to flip the whole-Virginia result negative.

This turned out to be a **known, named problem** in the GIC mitigation
literature, confirmed via a targeted literature review, not a Virginia-
specific fluke. Nagarajan et al.'s work on the GIC blocking-device
placement problem states directly: *"installing a blocking device at one
location can exacerbate GIC levels at others across the power grid,"*
and that *"simplistic strategies like greedy methods are likely
inadequate"* for this problem class as a direct result
(arXiv:2310.09409; review at arXiv:2402.07302). The field's actual
methods — exact optimization directly over the GIC circuit
(arXiv:1701.01469's MIQCP formulation), sensitivity-based placement, and
robust optimization under uncertain storm severity (arXiv:2006.02461,
arXiv:2104.00204's distributionally robust optimization) — share one
property SCI-greedy selection lacked: every candidate intervention is
scored by its measured effect on the *whole* network, never by a local
index at the cut site alone. An existing open-source tool,
**PowerModelsGMD.jl** (LANL-ANSI), already implements optimal
transmission switching under GMD as a real optimization problem, for
reference.

**As a result, the entire SCI-driven method — its computation, its
selection algorithm, and the two diagnostics built around it — was
retired**, moved to `scripts/deprecated_sci_islanding/` (kept working,
not deleted, with a README explaining this reasoning in full) rather
than continuing to iterate on an approach the literature had already
identified as structurally inadequate for this problem class.

### 5.4 Phase 2: redistribution-sensitivity-guided selection (current)

**Design** (`scripts/select_island_by_redistribution_sensitivity.py`).
Same seeded-region-growing *shape* as the retired method (so it's a
drop-in producer of the same `va_island_{id}.json` schema every
downstream Phase-4/5 script already consumes) — what changed is the
*score*. Every candidate (seed choice, and every growth step) is scored
by **actually cutting it and measuring the real effect**:

```
net_score = benefit
          - redistribution_cost_weight * redistribution_cost
          - switching_cost_weight * switching_cost
```

- `benefit` = sum of a two-term risk weight (0.5 × normalized GIC risk +
  0.5 × normalized economic exposure — deliberately *no* static
  betweenness term, since redistribution is now measured dynamically
  instead of approximated once upfront) over the candidate's members.
- `redistribution_cost` = the *measured* change in risk-weighted
  current-flow betweenness (Newman 2005; Brandes & Fleischer 2005 — same
  math as the diagnostic in Section 5.3) summed over every remaining
  substation, comparing a one-time network-wide baseline against the
  network with this specific candidate's boundary actually cut. This is
  the direct fix: instead of assuming redistribution away, it's computed
  for every real candidate considered.
- `switching_cost` = voltage-weighted physical boundary cost, unchanged
  from the retired method.

Growth stops the moment no candidate improves `net_score` — and, as of a
recent fix, the algorithm now records and reports *why* it stopped
(`stop_reason`: no eligible neighbors left vs. a candidate was genuinely
evaluated and rejected, with that candidate's full score breakdown
saved) rather than looking identical either way.

**Two real calibration bugs were found and fixed against actual Virginia
data** — both by the same method (checking real observed output
magnitudes, not just synthetic fixtures):

1. `redistribution_cost` was originally computed as a **mean** across
   ~2475 remaining substations, out of a (mistaken) concern that summing
   over thousands of nodes would make it *dominate* `benefit`. Real data
   showed the opposite: `current_flow_betweenness_centrality`'s
   normalized per-node values are already tiny at real network scale, so
   dividing by ~2475 crushed the signal to ~1e-5 — numerically
   irrelevant next to `benefit` (~0.1-0.5) and `switching_cost` (~2-15),
   silently degenerating `net_score` into "pick highest benefit, penalize
   switching cost" — reintroducing the exact blindness this algorithm
   exists to fix, with extra computation spent to get there. **Fixed**:
   switched to a **sum**.
2. `switching_cost_weight` (originally 0.1) made a single 230kV line's
   weighted contribution (0.23) alone exceed every real candidate's
   `benefit` (observed range 0.15-0.34) — every multi-line candidate
   scored `net_score < 0` regardless of its actual redistribution
   profile. **Recalibrated to 0.015** using the same real observed
   magnitudes; checked directly against the trace that motivated it, all
   10 real seed candidates that were uniformly negative at the old
   weight became positive and meaningfully differentiated at the new
   one.

A third addition, `--min-lines`, restricts *seed* choice (not growth) to
substations with at least N physical lines — added because the top-
scoring seed candidate by every other criterion kept turning out to be a
genuine degree-1 dead-end substation: real risk, minimal switching cost,
and (correctly, not a bug) ~zero redistribution effect, since nothing
routes through a true leaf — a legitimate, defensible finding on its
own, but one that can never grow past itself, which isn't useful for
testing the growth mechanism.

**Current real-data status (Island G, `--exclude a b c d e f
--min-lines 2`)**: seed `96396692`, grew to 2 members (`+638514006`),
`net_score = 0.16`. A striking, independent validation surfaced directly
from this run: the best-scoring *rejected* growth candidate was
`96396693` — the **exact same substation** independently identified,
months earlier from real Monte Carlo GIC results, as **Island A's #1
real-storm redistribution hotspot** (Section 5.3). The new algorithm
flagged that same substation as a bad addition (`redistribution_cost =
0.0248`, ~10x any other candidate in the trace) using nothing but a
cheap graph computation, with no prior knowledge of Island A's results —
real evidence the metric has genuine predictive validity, not just
internal consistency.

**Pushed through the real fragility/economic pipeline (Phase 4/5): Island
G is 9/10 scenarios net-positive.** The one exception is the real Gannon
storm scenario, which is net-negative — `diff_per_day_musd = -1.089293`,
own-outage cost $0.0129M (0% generation coverage, undiscounted), switching
cost $0.05M (2 lines), `net_musd` (mid_365d variant) = **-397.66**. This
makes Island G the strongest real result of any candidate produced so
far — a higher net-positive fraction than Island F's 6/10, and without
Island F's generation-anchoring constraint on seed choice. It's also the
clearest concrete demonstration of Section 5.5's directional-blindness
finding: the one scenario where G loses is the one genuinely independent
real storm in the 10-scenario set, and Section 5.6's real-storm check
later confirmed (see below) that Halloween and St. Patrick's Day — the
other two real, independently-directional storms — pattern-match Gannon's
own-member/rest-of-network signature rather than the nine return-period
scenarios', suggesting the true real-storm win rate may be closer to 0/3
than 9/10 once (if) those two are pushed through the full econ pipeline
too, not just checked at the raw-GIC level.

### 5.5 Directional blindness: why an "average" scenario is not a real storm

`redistribution_cost` (Section 5.4) is deliberately cheap — it recomputes
current-flow betweenness (Newman 2005; Brandes & Fleischer 2005) on the
rest of the network after a candidate cut. That computation has no storm
direction in it at all: it models current flowing between every pair of
nodes in proportion to conductance, isotropically — effectively an
*average over all possible directions*, not a measurement of what any one
real storm would do to this specific topology.

Real storms are not direction-neutral, and this pipeline's own data makes
that precise. `scripts/est_gic.py`'s `load_and_process_gic_data` treats the
twelve storm scenarios it can solve for very differently:

- **Three real historical events** — Gannon (2024), Halloween (2003), St.
  Patrick's Day (2015) — each carry their own independently-measured
  per-line E-field, magnitude *and* polarity, straight from magnetotelluric
  data. These are genuinely different directional forcings.
- **Nine statistical return-period scenarios** (50-250yr, the ones used
  throughout Section 4's econ Monte Carlo) get their *magnitude* from a
  lognormal return-period fit, but their per-line *sign* is deliberately
  forced to match Gannon's own observed polarity (`sgn_full`) rather than
  drawn independently. They are not a fourth, fifth, ... source of real
  directional information — they are magnitude variations on Gannon's own
  direction.

This was found empirically before it was found in the code. Pushing Island
G through the real single-storm GIC solve (not the topology-only proxy)
showed its own members' GIC **increasing** under the real Gannon forcing
(+56.23A) but **decreasing** under the 100-year return-period forcing
(-59.01A) — opposite signs, despite 100yr sharing Gannon's polarity — and
the single dominant redistribution-hotspot substation was completely
different between the two (`96396693` for 100yr vs. `96284749` for
Gannon). A candidate whose `net_score` looks good against the isotropic
proxy, or even against several of the return-period scenarios, has not
been checked against a real storm's actual direction — the two can and do
disagree.

Practically, there is no way around this by picking the storm's direction
in advance: no forecasting capability gives a specific incoming storm's
regional geoelectric-field direction more than roughly an hour of lead
time (real-time L1 solar wind/IMF monitoring via ACE/DSCOVR), and precise,
substation-resolved directional geoelectric-field forecasting beyond that
is an open research problem (Kelly et al.; the LSTM/"LiveWire" GIC-
forecasting literature, Hu et al. 2024). So the right response is not
"predict the direction and select accordingly" but "check whether a
candidate holds up across the range of real directions actually observed,"
which is what Section 5.6 below does.

### 5.6 Phase 6c: cheap real-storm directional validation (the middle ground)

**`scripts/validate_island_directionality.py`.** Re-running the full
Phase 4/5 Monte Carlo (thousands of iterations, "a good overnight job")
for every candidate the cheap search produces would defeat the point of
having a cheap search — but trusting `net_score` alone is exactly the
blindness Section 5.5 just described. The middle ground: for each finalist
candidate, run **one** real admittance/GIC solve (iteration 0 only,
`start_idx=0, end_idx=1` — ~1/`N_MONTE_CARLO` the cost of a full run) with
that candidate's boundary actually cut, using the *same* seed draw as an
equivalent single-iteration baseline (so any difference is attributable to
the topology change, not random noise). Comparing the two across all
twelve available storm forcings gives, per storm:

- `own_effect` — the risk-weighted change in `|GIC|` summed over the
  candidate's own members (the real-physics analog of `benefit`).
- `redistribution_cost_real` — the same, summed over the rest of the
  network (the real-physics analog of `redistribution_cost`), using
  `|GIC|` rather than signed GIC since it's magnitude that drives
  transformer fragility, not direction.

A candidate is flagged **directionally consistent** if
`redistribution_cost_real`'s sign agrees across all three genuinely
independent real storms — gannon, halloween, st_patricks (the nine
return-period scenarios are reported alongside for continuity with
Section 4's results, but are not treated as independent directional
evidence, per Section 5.5). A candidate that flips sign between the three
real storms is **directionally fragile** — exactly the failure mode Island
G's Gannon-vs-100yr divergence demonstrated — and should not be committed
to a full Phase 4/5 run without treating that as an open question, not a
settled result.

This does not replace Phase 4/5, and produces no $/day figure — it is a
cheap filter between the topology-only search and the expensive real
pipeline, reusing the exact admittance/GIC machinery
(`scripts/run_virginia_gic.py`) Phase 4 already uses, just for one
iteration instead of a full Monte Carlo.

**Run against real data on Island G**: `directionally_consistent_across_
real_storms = True` (gannon, halloween, st_patricks all agree with each
other), but the more important result is in the numbers, not the flag.
Every one of the three real storms shows `own_effect` positive and
`redistribution_cost_real` negative (island's own members get worse, rest
of network gets better); every one of the nine return-period scenarios
shows the opposite (own members better, rest of network worse) — a clean,
unanimous split between the two groups, not noise. Since 9 of the 10
scenarios in Island G's econ result (Section 5.4/Section 8) are drawn
from the return-period family, this raises a real possibility that the
"9/10 positive" econ verdict is dominated by scenarios systematically
unlike any real storm, while the one real scenario in that set (Gannon)
came back negative for the reason this section predicts. Halloween and
St. Patrick's Day have not yet been pushed through the full fragility/econ
pipeline to get real `net_musd` figures — that's the next concrete step,
not just running this check on further candidates.

### 5.7 Phase 6d: generation removed; the search scored by real storms

**Generation-anchoring removed from active code, not just deprecated.**
`scripts/check_island_generation.py` and `scripts/virginia_demand_proxy.py`
moved to `scripts/deprecated_sci_islanding/`; `compute_island_benefit_
cost.py`'s own-outage cost is unconditional again, no generation-coverage
discount. Settled by direct comparison: Island G, selected with zero
generation requirement, reached 9/10 net-positive, beating
generation-anchored Island F's 6/10.

**The search itself is now scored by real storms, not
`current_flow_betweenness_centrality`.** Section 5.6's real-storm check
was a post-hoc filter applied *after* selection already picked a winner
with the isotropic proxy — exactly how Island G could pass selection and
still lose on the one real, independent-direction storm (Gannon) in its
econ set. `evaluate_cut()` now scores every candidate by one real
single-iteration admittance/GIC solve against gannon/halloween/
st_patricks, taking the **worst case across the three, not the
average** — directly targeting "a method that is positive for all
cases" rather than positive on average. `scripts/run_virginia_gic.py`
was split into a reusable `load_gic_context()` (expensive, one-time) and
`run_monte_carlo()` (the actual per-iteration solve) so a search
evaluating dozens of candidates pays the setup cost once, not per
candidate; `scripts/real_storm_gic.py` holds the shared real-storm
scoring logic used by both the search and Section 5.6's check (which was
refactored onto it too, removing duplication).

One real admittance solve already computes all twelve storm columns at
once, so scoring against three real storms costs the same as scoring
against one — but that one solve still costs ~20-40s on real data,
versus the old proxy's near-instant graph computation, so a full search
run now takes meaningfully longer. `SWITCHING_COST_WEIGHT`/
`REDISTRIBUTION_COST_WEIGHT` need a third real-data calibration pass —
the new redistribution_cost lives in raw risk-weighted `|GIC|` amps, a
completely different scale than the old normalized betweenness units.

**Run against real data: Island H, 10/10 scenarios net-positive, including
the real Gannon storm.** Seed `107946151`, grown to 3 members
(`107946150`, `108016246`), 4 boundary lines, weights unchanged
(`SWITCHING_COST_WEIGHT=0.015`, `REDISTRIBUTION_COST_WEIGHT=1.0` — no
recalibration needed this time: `benefit` (~1.26) and the redistribution
term's contribution (~0.83) came out comparable orders of magnitude on
real data, unlike the two prior scale-mismatch bugs). All three real
storms showed the same favorable sign pattern
(`redistribution_cost_real` negative for gannon/halloween/st_patricks:
-3.50/-2.98/-0.83; `own_effect` small and positive: +0.92/+0.21/+0.21) —
`directionally_consistent_across_real_storms = True`, and unlike Island
G, the small cost to H's own members is dominated by the larger benefit
to the rest of the network under every real storm, not the reverse.

Pushed through Phase 4/5: **`net_musd` (mid_365d) = +18.49 for
gannon-year** (benefit-cost ratio ~118) — the first candidate to clear
the one real storm in the standard scenario set. All nine return-period
scenarios are also positive, scaling monotonically with severity (50yr:
+$58.4M, ratio 370 -> 250yr: +$455.8M, ratio 2,884), the same physically
sensible shape Island F showed. This is a real, direct validation of the
cheap real-storm check's predictive value: the raw-GIC comparison above
predicted a net-positive Gannon result *before* the expensive Monte Carlo
ran, on the strength of H's own-cost/network-benefit ratio being the
opposite of G's -- and the full pipeline confirmed it.

**Resolved, and the result is more nuanced than hoped: real dollar
figures now exist for all three real storms, and only one is positive.**
Halloween and St. Patrick's Day were wired into `run_virginia_fragility.
py`/`econ_analysis.py` (Phase 6d's `SCENARIOS` monkeypatch). Real result:
`gannon-year` +$18.49M (positive, as before), `halloween-year` **−$4.51M
(negative)**, `st_patricks-year` **$0.00M** (exactly zero benefit, so
net −$0.16M once switching/own-outage cost is included). Whole-scenario-set
tally: **10/12 net-positive** (all nine return periods plus Gannon), the
best result of any candidate, but the real-storm tally specifically is
**1 of 3 positive**, not 3 of 3.

This is a sharper finding than "not yet validated" — it's a concrete
demonstration that the raw-GIC directional check (Section 5.6) is not
sufficient on its own. That check found all three real storms favorable
at the `|GIC|`-magnitude level (`redistribution_cost_real` negative for
all three) and reported `directionally_consistent_across_real_storms =
True` — yet Halloween's real $/day sign came out negative anyway. The
gap: the fragility curve is nonlinear and the whole-Virginia econ diff
depends on *which specific substations* see the GIC change and their
population/GDP exposure, not just an aggregated network-wide `|GIC|`
sum. Sign agreement in the cheap check is evidence worth having (it
correctly predicted Gannon's sign) but is not proof of $/day-level
agreement — a real, demonstrated limitation, not just a theoretical
caveat, and a stronger basis for Section 4.7/5.6's argument than the
prediction alone would have been.

St. Patrick's Day's exact $0.00M is plausibly a genuine floor effect
(a materially weaker storm than Gannon, plausibly not crossing the
fragility failure threshold for any Virginia substation in either
topology) rather than a data artifact, but worth a direct check of
`va_ghosh_results.csv`'s raw `total_impact` for that scenario in both
the baseline and Island H directories before treating it as settled.

**Honest headline for the paper**: Island H is real-storm-positive for
the most severe historical event in this dataset (Gannon) and the full
return-period family, negative for one weaker real storm (Halloween),
and neutral for another (St. Patrick's). No candidate produced by this
project has achieved unanimous real-storm positivity. Reporting that
precisely -- not just the 10/12 headline -- is the more defensible
claim, and arguably a better result for the paper than a clean 3/3
would have been: it demonstrates the validation methodology (Sections
5.5-5.7) is doing real work, not rubber-stamping whatever the search
produces.

**Confirmed the limitation is the scoring function, not search breadth.**
A much larger search (10 seed candidates, max size 15, still no
`--exclude`) reproduced Island H exactly -- same seed, same 3 members,
same net_score. A wider search finding the identical answer rules out
"the budget was too small"; the ceiling is `redistribution_cost`'s
definition. **Phase 6e (`ISLANDING_MITIGATION_PLAN.md`) implements the
fix**: `redistribution_cost` is now scored via a fragility-weighted
expected-GDP-affected delta -- reusing `econ/preprocess/p_gic_files.py`'s
own fragility function directly at a small draw count, not raw `|GIC|`
magnitude -- since raw-`|GIC|` sign agreement (which Halloween had) is
demonstrably not sufficient evidence of $/day-level agreement (which
Halloween didn't have). Implemented, not yet run against real data; the
scoring weights need a fourth real-data calibration pass once it is,
since the new `redistribution_cost` lives in a different (likely much
smaller) unit scale than the raw-`|GIC|` version.

### 5.8 Status and honest framing for the paper

As of this writing: **the SCI-driven method is a completed, negative
result** (tested against six real candidates, found unreliable, root-
caused, and confirmed against independent literature) — a legitimate,
publishable finding on its own, not a failure of the investigation. **The
redistribution-sensitivity method is mechanically validated but not yet
economically validated** — real code, real bug-fixes grounded in real
data, a real independent-validation signal (the `96396693` match), but
no real $/day result yet. If Island G's real Phase 4/5 run comes back
negative or negligible too, that is *also* a legitimate, well-evidenced
finding worth reporting honestly — consistent with the structural
differences this project has already identified between its own setup
and the one real precedent (arXiv:2507.12495) where islanding
demonstrably worked: that scenario was generation-anchored (not just
protective), threshold-based in its damage model (not continuous), one
piece of a much larger expert-engineered switching plan (not a
standalone algorithmic cut), and used a small labor-only cost
denominator (not a redistribution cost netted out the way this project's
is). None of that makes this project's methodology wrong — it means a
negative result here says something specific and defensible about
protective-only, algorithmically-selected islanding under a continuous
damage model, not about islanding as a mitigation class in general.

See `ISLANDING_MITIGATION_PLAN.md` for the full phase-by-phase working
log this section summarizes, including exact commands, real console
output, and every intermediate finding in chronological order.

---

## 6. Pipeline adaptation details (deviations from the original plan)

This section is the original engineering deviation log, preserved for
anyone extending or auditing the pipeline-adaptation code specifically —
Sections 1-5 above are the project-level narrative; this section is the
implementation-level reference.

### Key design decision: `VA_DATA_DIR` mirrors the US `DATA_DIR` layout

`scripts/build_admittance_matrix.py`, `scripts/calc_storm_maxes.py`, and
`scripts/est_gic.py` are written as standalone pipelines against a single
`DATA_LOC` (== `configs.DATA_DIR`), with subdirectories named
`grid_processed/`, `admittance_matrix/`, `gic/`, `gnd_gic/`,
`statistical_analysis/`. Rather than reimplementing their numerics, every
Virginia script points these unmodified functions at
`configs.virginia_config.VA_DATA_DIR` in place of `DATA_DIR`, and
`VA_DATA_DIR` is laid out with the exact same subdirectory names. This is
why `VA_ADMITTANCE_DIR`, `VA_GIC_DIR`, `VA_GND_GIC_DIR`, and
`VA_STATISTICAL_ANALYSIS_DIR` in `configs/virginia_config.py` are simply
`VA_DATA_DIR / "<name>"` — they have to match what the reused functions
construct internally.

The exceptions are `data/EMTF/mt_pickle.pkl` (all 1696 US MT sites) and
`data/geomag_data/processed_geomag_data.nc` (observatory data): these are
read directly from the original `DATA_DIR`, never copied into
`VA_DATA_DIR`, because Dennies' guidance is to never filter or duplicate
them — the SECS interpolation and Delaunay line-weighting always use the
full site set.

### Deviations discovered from the original plan

1. **`grid_mapping.csv` is a required, hand-annotated file, not a generated
   one.** `build_admittance_matrix.load_and_process_data()` unconditionally
   reads `<DATA_LOC>/grid_mapping.csv` (a per-substation transformer-role
   table produced by manual GIS review on the reference machine). It is
   only actually *used* on the `evan_data=True` code path
   (`get_transformer_data_evan`); the default `evan_data=False` path
   (`build_substation_buses`, a Monte Carlo synthetic transformer-type
   estimate — what this adaptation uses throughout) never reads its parsed
   values. `scripts/run_virginia_admittance.py` writes an empty, schema-valid
   stub so the loader doesn't crash, and documents why that's safe.

2. **`trans_lines_pickle.pkl` is not a network-build artifact — it's
   produced by the storm-maxima step.** The plan assumed
   `build_virginia_network.py` (Step 2) would produce every file
   `build_admittance_matrix.py` needs. In the real pipeline,
   `scripts/calc_storm_maxes.read_transmission_lines()` builds this file
   from `df_lines_EHV.pkl` by attaching Delaunay interpolation weights
   against the MT site list — it's part of the SECS/electric-field step, not
   the grid-topology step. `scripts/recompute_virginia_line_voltages.py`
   (see deviation 4) produces it for Virginia.

3. **`preprocess/06`/`preprocess/08` filter geographically via a FERC-region
   spatial join (`data/nerc_gdf.geojson`), not a bounding box.**
   `scripts/build_virginia_network.py` reimplements the two loader functions
   that did the FERC sjoin (`load_and_process_transmission_lines`,
   `load_and_process_substations`) with a PJM-boundary spatial filter
   instead, stubbing the `REGION_ID`/`REGIONS` columns downstream code reads
   by name (`get_EHV_lines()`'s FERC-filter step) with a constant `"PJM"`
   value. Everything else in `preprocess/08_p_power_grid.py`
   (`graph_node_edges`, `get_nodes_edges`, `get_EHV_lines`,
   `prepare_substations_for_grid_analysis`) is geography-agnostic and reused
   unmodified via `importlib` (the files' numeric-prefixed names aren't
   valid `import` targets).

4. **Per-line voltage return periods can't be filtered from the US run —
   they have to be recomputed against the new VA+PJM line set.** This is
   the biggest gap in the original plan. Site-level B/E maxima
   (`maxB_arr`/`maxE_arr`) are a valid index-based filter because each row
   is one MT site, independent of network topology
   (`scripts/extract_virginia_storm_maxes.py`). But line-level voltage
   maxima (`maxV_arr`, and the `V_*` columns `est_gic.py` merges onto
   `df_lines`) are specific to the *original* US `LINE_ID`s and their
   Delaunay weights — Virginia's brand-new line set has neither.
   `scripts/recompute_virginia_line_voltages.py` replays
   `calc_storm_maxes.calculate_maxes()` for every storm in
   `storm_periods.csv` against the VA+PJM lines (still weighted against the
   *full* MT site list, per Dennies' guidance), fits return-period
   statistics with `stat_analysis.py`'s bootstrap machinery, and packages
   the result into `va_geomagnetic_data_return_periods.h5` with the same
   layout `est_gic.load_and_process_gic_data()` reads. This is a real,
   non-trivial compute cost (~90 storms × a full SECS+Delaunay pass each) —
   there is no way to avoid it without changing the line topology decision
   in Step 2, which Dennies confirmed should stay PJM-scoped.

5. **`est_gic.py` has no multiprocessing, but `calc_storm_maxes.py` does —
   and its `Pool` has no initializer, which matters because the target
   machine is Windows.** `est_gic.py` iterates Monte Carlo samples
   sequentially (parallelizing only per-return-period GIC calc within one
   iteration, via `ThreadPoolExecutor` — threads share the parent's memory
   regardless of platform, so this is fine as-is); `scripts/run_virginia_gic.py`
   follows the same sequential structure rather than inventing
   multiprocessing the reference implementation doesn't have.
   `calc_storm_maxes.py`, however, *does* use a bare `multiprocessing.Pool`
   whose worker function (`process_storm`) reads `MT_sites`/`df`/`obs_dict`/
   `n_trans_lines` as module globals — set only inside the script's own
   `if __name__ == "__main__":` guard. An early version of
   `scripts/recompute_virginia_line_voltages.py` (which reuses
   `calculate_maxes()` the same way) assumed `Pool()` forks and inherits
   those monkeypatched globals for free. That's true on Linux, but **not on
   Windows** — the reference machine's platform — where `Pool()` defaults to
   `spawn`: each worker re-imports `scripts.calc_storm_maxes` fresh, without
   the monkeypatched attributes, `calculate_maxes()` raises `NameError`, and
   `process_storm`'s own `try/except` silently swallows it per storm. The
   run "completes" with an all-zero `maxV_arr` instead of crashing —
   exactly the failure mode the original plan's `_init_worker` warning was
   about. Fixed in `recompute_virginia_line_voltages.py` with a proper
   `Pool(initializer=_init_worker, initargs=(...))` that re-applies the same
   globals inside every worker. **`calc_storm_maxes.py`'s own `main()` was
   left untouched** (out of scope here, and it's the already-run, "do not
   modify" reference pipeline) **but appears to have this same latent bug on
   Windows** — worth Dennies confirming whether the original US run actually
   went through this code path or a different, since-diverged version,
   since a silent all-empty result wouldn't necessarily have been obvious.

6. **Functions that need a data directory but don't take one as a
   parameter** (`build_admittance_matrix.get_transformer_samples()`,
   which reads its own module-level `DATA_LOC` for the cache path) are
   monkeypatched — e.g. `bam.DATA_LOC = VA_DATA_DIR` before calling — exactly
   as the original plan anticipated for cases where "functions don't accept
   arguments."

7. **Economic-model hookup is bounded, not complete — and the SAM-scope
   question has now been decided, provisionally.** `run_econ.py`'s analysis
   step depends on a much longer chain (`econ/preprocess/p_gic_files.py` →
   `p_areal_intp.py` → SAM/technology tables → `econ/scripts/econ_analysis.py`)
   than "filter ZCTA data, run `run_econ.py analysis`" suggested, and the
   real schema differs from what reading the source implied — the
   `STABBR` column `scripts/extract_virginia_results.py` originally assumed
   on `zcta_pop_20` doesn't exist in the actual pickle on the reference
   machine (confirmed via `scripts/_inspect_socioeconomic_data.py`); the fix
   derives Virginia's ZCTA set from `naics_est_gdp`/`regions_pop_df`'s
   `GeoFips` (county FIPS, first two digits = state FIPS) instead.

   **Decision (national SAM/technology tables, driven by a
   Virginia-localized shock):** rather than building a bespoke
   Virginia-regional SAM — which would require its own supply-use tables
   and econometric balancing, a separate research project on its own —
   Virginia's IO impact analysis should reuse the existing national
   SAM/technology tables and drive them with Virginia's own GIC exposure
   (population/business counts from Virginia ZCTAs, attributed via a
   Voronoi tessellation of *Virginia+PJM* substations rather than the
   national substation set). This is standard practice in sub-national
   disaster-economics impact studies. Flag for Dennies to override if he
   disagrees, but it's the only tractable choice on the ASSIP timeline.

   **What was actually blocking this, and what's fixed:** unlike everything
   else in this document, the remaining chain isn't just a
   path-substitution problem — but `scripts/_check_econ_pipeline_deps.py`
   run against the reference Windows machine found the underlying tools all
   present and working (IPOPT on `PATH`, `rasterio`/`tobler`/`geopandas`
   importable, the NLCD raster + coarse mask + all SAM/technology CSVs on
   disk). Two real cross-platform bugs did surface and are fixed:
   `configs/settings.py`'s `IPOPT_EXEC` hardcoded a Linux path
   (`/home/pve_ubuntu/.../bin/ipopt`) — now resolved via `shutil.which("ipopt")`
   first, falling back to that literal only if not found on `PATH` — and
   `econ/scripts/econ_analysis.py` had that *same* Linux path hardcoded
   inline a second time (bypassing the config entirely), now using the
   `IPOPT_EXEC` constant like the rest of the codebase already does. Neither
   fix is Virginia-specific; both were live bugs in the shared national
   pipeline whenever run on Windows.
   `econ/preprocess/p_areal_intp.py` does raster-based dasymetric
   interpolation (`tobler.dasymetric.masked_area_interpolate`, tiled
   `rasterio` reads over the national NLCD land-cover raster) to map
   economic exposure onto substation Voronoi cells — a real geospatial
   subsystem, not more pandas. `scripts/_check_econ_pipeline_deps.py`
   confirmed IPOPT, `pyomo`, `rasterio`/`tobler`, the NLCD raster, and the
   SAM/technology CSVs are all present and working on the reference machine.

   **What's built, three scripts completing the chain:**
   - `scripts/run_virginia_areal_interpolation.py` — rebuilds the Voronoi
     tessellation from Virginia+PJM's own substations (not the national
     set) clipped to PJM's states (not all of CONUS — achieved by feeding
     `create_voronoi_polygons()` a pre-filtered `states_gdf`, so its
     CONUS-exclusion filter becomes a no-op and the union of what's left
     *is* the PJM footprint), then runs `masked_area_interpolate()` once
     over Virginia's own raster window (small enough to skip the national
     pipeline's 64-tile chunking entirely) to map Virginia ZCTA economic
     data onto Virginia substation service areas, with the same mass-loss
     rescaling `l_prepr_data.load_and_aggregate_tiles()` applies nationally.
   - `scripts/run_virginia_fragility.py` — reuses
     `p_gic_files.py`'s `process_gic_file()`/`build_merged()`/
     `_simulate_scenario()`/`process_all_files()` unmodified (via
     monkeypatched `DATA_DIR`/`DATA_LOC`/`OUTPUT_FILES`) to convert every
     Virginia GIC Monte Carlo realization into per-substation transformer
     failure probabilities and exposure. Forced to `workers=1` rather than
     the library's default `ProcessPoolExecutor` parallelization — that
     uses Windows spawn, which would silently corrupt latitude/longitude in
     the output by re-importing `p_gic_files` fresh (unpatched) in every
     worker, the same failure mode fixed in
     `recompute_virginia_line_voltages.py` but harder to notice here since
     it wouldn't crash. Sequential is much slower but correct.
   - `scripts/run_virginia_econ_analysis.py` — runs `econ_analysis.py`'s
     Ghosh value-added and two Leontief shock models completely unmodified
     against Virginia's fragility output. Only
     `l_prepr_data.py`'s `DATA_LOC` is redirected (to read Virginia's
     fragility batches); `econ_analysis.py`'s *own* `DATA_LOC` is
     deliberately left pointed at the national tables, since
     `_load_national_totals()` needs the true national population/GDP as
     the denominator for "Virginia's share of the national economy" — that
     ratio *is* the national-tables-plus-regional-shock design, not
     something to redirect.

   Does not touch `econ.models.cge_data_model`'s IPOPT-based CGE path —
   `econ_analysis.py`'s own `main()` doesn't call it either; it's a
   separate, more advanced model not needed for these three headline
   analyses. The IPOPT path fixes still matter if that's ever exercised.

### Visualization: substation naming and figure parity with the national report

`scripts/build_admittance_matrix.py` renames each substation's `SS_ID`
(itself just the raw OSM node/way `osmid`, since most PJM-region
`power=substation` OSM features carry no `name` tag) to `name`, and every
downstream file — `substation_info.csv`, the GIC CSVs, `va_gic_summary.csv`
— inherits that as `sub_id`. The real OSM name, when one exists, survives
only in `grid_processed/ss_df.pkl`'s `SS_NAME` column, which nothing joins
back in for reporting. `viz/plot_virginia_gic_risk.py` loads
`ss_df.pkl` and labels each substation with `SS_NAME` when OSM has one,
falling back to `"{operator} {voltage} substation ({lat}, {lon})"` or
`"Unnamed substation ({lat}, {lon})"` otherwise — no code up the chain
changes, since `sub_id` is still what every merge key is built from.

`viz/plot_virginia_econ_impact.py` mirrors `viz/plots.py`'s
`plot_econo_naics()` layout and color palette (one panel per storm
severity, direct+indirect stacked bars per sector) reading the three
`scripts/run_virginia_econ_analysis.py` output CSVs directly, but plots
in `$M/day` instead of the national script's hardcoded `Bn $/day` —
Virginia's totals (tens to low hundreds of $M/day) would otherwise
render as barely-visible slivers under the national scale.

---

## 7. Repository guide

### Core pipeline (Sections 2-4)

```
configs/virginia_config.py                  Step 1 — PJM/VA parameters, output dirs
scripts/build_virginia_network.py           Step 2 — VA+PJM EHV network from HIFLD + OSM
scripts/extract_virginia_storm_maxes.py     Step 3a — PJM MT-site B/E filter (index-based)
scripts/recompute_virginia_line_voltages.py Step 3b — VA+PJM line V recompute + stat fit + h5
scripts/run_virginia_admittance.py          Step 4 — VA+PJM admittance matrix + MC samples
scripts/run_virginia_gic.py                 Step 5 — VA+PJM GIC Monte Carlo
scripts/extract_virginia_results.py         Step 6 — filter GIC + econ ZCTAs to Virginia
scripts/run_virginia_areal_interpolation.py Econ — VA substation-exposure Voronoi/dasymetric interpolation
scripts/run_virginia_fragility.py           Econ — VA fragility Monte Carlo (effective GIC + vulnerability)
scripts/run_virginia_econ_analysis.py       Econ — VA Ghosh/Leontief IO analysis ($/day results)
viz/plot_virginia_network.py                Viz — VA+PJM EHV network map
viz/plot_virginia_gic_risk.py               Viz — VA substation GIC risk map + ranking
viz/plot_virginia_econ_impact.py            Viz — VA sector-level $/day impact bars (3 IO models)
```

### Islanding investigation — active

```
scripts/virginia_risk_data.py                          Shared: raw per-substation data loaders (GIC risk, econ exposure, critical facilities, topology)
scripts/island_graph_utils.py                           Shared: graph loading, boundary-cut math, conductance attachment
scripts/virginia_demand_proxy.py                        Shared: population-based demand proxy (order-of-magnitude only)
scripts/check_island_generation.py                      EIA-860 generation-capacity matching (per substation or substation pool)
scripts/select_island_by_redistribution_sensitivity.py  Current selection algorithm (Section 5.4)
scripts/show_seed_candidate_trace.py                    Diagnostic: full seed-candidate comparison table for a sensitivity-selected island
scripts/run_virginia_islanded_gic.py                    Phase 4 — topology-edited GIC re-solve for any island_id
scripts/run_virginia_islanded_fragility.py              Phase 5 — fragility Monte Carlo for any island_id
scripts/run_virginia_islanded_econ_analysis.py          Phase 5 — econ IO analysis for any island_id
scripts/compare_island_gic.py                           Diagnostic — GIC at an island's own members, baseline vs. islanded
scripts/compare_virginia_gic.py                         Diagnostic — GIC across all VA substations, baseline vs. islanded (redistribution hotspots)
scripts/compare_island_econ.py                          Diagnostic — $/day diff, island vs. baseline, by scenario
scripts/compute_island_benefit_cost.py                  Full benefit-cost accounting (switching cost + generation-discounted own-outage cost)
scripts/combine_islands.py                              Combine multiple already-selected islands into one for joint testing
viz/plot_virginia_island.py                             Sanity-check figure for any island's members/boundary/growth trace
```

### Islanding investigation — deprecated (`scripts/deprecated_sci_islanding/`)

Kept working, not deleted; see that folder's own `README.md` for the full
retirement rationale (Section 5.3 above summarizes it).

```
compute_virginia_criticality.py    Phase 2 — SCI computation
select_virginia_island.py          Phase 3 — SCI-driven seeded greedy region growing (incl. require_local_generation)
check_redistribution_proxy.py      Diagnostic — current-flow-betweenness proxy that first flagged SCI's flaw
check_generation_sci_overlap.py    Diagnostic — SCI-vs-generation overlap check (Island F's real seed validation)
search_candidate_islands.py        Orchestrates select-and-evaluate for multiple SCI-selected candidates (used for C, D, E)
plot_virginia_criticality.py       SCI ranking map (moved from viz/)
plot_virginia_island.py            SCI-island sanity-check figure (moved from viz/; superseded by the active viz/plot_virginia_island.py)
```

### Placeholder, not implemented

```
dc_blocker/README.md               DC-blocker mitigation was never built in this project; do not cite it as completed work
```

Outputs land under `data/virginia/` (gitignored, same as every other
`data/` subdirectory) and `figures/virginia/`. Real islanding candidates
are saved as `data/virginia/network/va_island_{id}.json`.

### Execution order

```bash
export PYTHONPATH="$(pwd)"

# Core pipeline
python scripts/build_virginia_network.py
python scripts/extract_virginia_storm_maxes.py
python scripts/recompute_virginia_line_voltages.py   # expensive: ~90 storms recomputed
python scripts/run_virginia_admittance.py
python scripts/run_virginia_gic.py                    # expensive: N_MONTE_CARLO iterations
python scripts/extract_virginia_results.py
python scripts/run_virginia_areal_interpolation.py
python scripts/run_virginia_fragility.py               # expensive: sequential, likely the longest step
python scripts/run_virginia_econ_analysis.py

# Figures
python viz/plot_virginia_network.py
python viz/plot_virginia_gic_risk.py
python viz/plot_virginia_econ_impact.py

# Islanding: select a candidate, then push it through the same shape of pipeline
python scripts/select_island_by_redistribution_sensitivity.py --island <id> --exclude <prior ids> --min-lines 2
python scripts/show_seed_candidate_trace.py --island <id>
python scripts/run_virginia_islanded_gic.py --island <id>
python scripts/run_virginia_islanded_fragility.py --island <id>
python scripts/run_virginia_islanded_econ_analysis.py --island <id>
python scripts/compare_island_gic.py --island <id>
python scripts/compare_virginia_gic.py --island <id>
python scripts/compare_island_econ.py --island <id>
python scripts/compute_island_benefit_cost.py --island <id>
```

`configs/virginia_config.N_MONTE_CARLO` is set to 3000, matching the US
run's sample size.

---

## 8. Open items

- `N_MONTE_CARLO` is set to 3000 (matching the national pipeline) as of the
  first full run against real data — confirm this is an acceptable final
  sample size, or a different number, for reported results.
- The SAM-scope decision (Section 6, item 7) — national SAM/technology
  tables, driven by a Virginia-localized GIC shock — was made without
  Dennies' sign-off, given the ASSIP timeline. Confirm or override before
  treating any resulting $/day figures as final.
- `Aaditya Bose` is building a parallel VA+PJM network independently per
  the original plan — worth a methodology sync given
  `build_virginia_network.py` makes several concrete choices (FERC-sjoin
  replaced with a PJM boundary filter, 150m substation-line buffer
  distance, Overpass server list) that another independent build may not
  match exactly.
- **Resolved: Island G's real fragility/economic result is 9/10
  scenarios net-positive** (Section 5.4) — the strongest real result of
  any candidate so far, and no longer an open item on its own. The open
  question shifted, not closed: the one negative scenario is the real
  Gannon storm, and `scripts/validate_island_directionality.py` (run
  against real data, Section 5.6) confirms Gannon is not an isolated
  fluke — Halloween and St. Patrick's Day, the only other genuinely
  independent real storms available, both pattern-match Gannon's
  own-member/rest-of-network signature at the raw-GIC level rather than
  the nine return-period scenarios' pattern. **Halloween and St.
  Patrick's Day have not been pushed through the full fragility/econ
  pipeline** (only Gannon has a real `net_musd` figure) — this is now the
  single most important open item: without it, "9/10 positive" and "0/3
  real storms positive" are both consistent with the same underlying
  data, and only one of them is knowable right now.
- Generation-anchoring (Island F) is no longer worth further investment:
  Island G, selected with no generation requirement at all, beat F's 6/10
  with 9/10 — restricting seed choice to generation-backed substations
  (Section 5.4's finding that best-SCI and generation-backed substations
  barely overlap in Virginia) looks like a net cost to candidate quality,
  not a benefit, once the redistribution-sensitivity method itself is
  doing the selecting.
