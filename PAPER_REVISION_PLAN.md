# Paper Revision Plan — Response to Ed's Review (2026-10-01)

Tracks every item from Ed's feedback email, who does it, and what's needed to
unblock it. Numbers below are computed directly from the submitted PDF
(`pdftotext` + figure/table caption count), not estimated.

## 0. The headline problem: PU budget

AGU Space Weather's limit is 25 Publication Units (1 PU = 500 words, or one
figure, or one table). Current draft, measured directly:

| Item | Count | PU |
|---|---|---|
| Body text (abstract through conclusion, excl. references) | 9,575 words | ~19.2 |
| Figures | 14 | 14 |
| Tables | 7 | 7 |
| **Total** | | **~40.2 PU** |

That's **~15 PU over the limit** before any new content (research questions,
restoration scenarios, more abstract numbers) is added. Consolidation isn't
optional polish here — it's required to submit at all. Section 5 below
re-tallies the PU count after every proposed consolidation.

## 1. Items Claude can do directly in code (no new analysis design needed)

| # | Item | File(s) | Status |
|---|---|---|---|
| 1a | Figure 3: add basemap, inset locator map, substation dots in legend, resize/recolor dots (e.g. by voltage) | `viz/plot_virginia_network.py` | Not started. Currently has no basemap, no inset, and `subs.plot()` (line 53) has no `label=` so it's silently excluded from `ax.legend()`. Will reuse the exact inset-locator pattern already built for `plot_island_maps.py` (query the main axes' real post-draw position, place the inset relative to it). |
| 1b | Figure 8 (winding-GIC ranking) still shows raw `[SS ####]` tags in the submitted PDF | `viz/plot_virginia_winding_gic_ranking.py` | This is a **stale figure, not a new bug** — it reuses `_load_substation_labels()` from `plot_virginia_gic_risk.py`, which was already fixed earlier this project to drop the SS-ID fallback tier. The figure in the PDF was rendered before that fix. Just needs a rerun. Will also fix this file's own docstring, which still describes the old 4-tier priority. |
| 1c | Combine Figures 12, 13, 14 (island maps) into one multi-panel figure | `viz/plot_island_maps.py` | Will add a combined-output mode (3-panel, each island keeping its own inset locator) so this becomes 1 PU instead of 3. |
| 1d | Replace Tables 5, 6, 7 (per-island benefit-cost) with one results figure | New script, e.g. `viz/plot_island_benefit_cost_summary.py` | Ed's explicit preference: "Graphs are better... show magnitudes without needing to read each number." Will build a grouped chart (benefit-cost ratio or net benefit per scenario, grouped by island A/B/C) replacing all 3 tables with 1 figure. Raw tables can still go in supplementary. |
| 1e | PU budget worksheet (this doc, Section 5) | — | Done below; will keep updating as changes land. |
| 1f | Flag AI-writing tells with line numbers | — | Already scanned (Section 4) — concrete hit list, not vague advice. |

## 2. Items needing a document source Claude doesn't have yet

**Blocker: there is no `.tex`/`.bib` source in this repo** — the paper is being
built/maintained elsewhere (Overleaf, presumably). Claude can only read the
compiled PDF right now. The following all require the LaTeX source to be
shared (e.g. add Claude as an Overleaf collaborator, or export and drop the
source into this repo under something like `paper/`):

- Combining Tables 1, 2, 3 into one table (currently 3 PU → 1 PU)
- Removing Figure 1 and reworking Figure 2 into the single detailed
  pipeline graphic Ed wants (see 2a below — needs new diagram source too)
- Combining Figures 4, 5, 6 into one 7×10in composite (needs the current
  Mermaid/diagram sources for those three flowcharts — these were built
  earlier in this chat session but aren't persisted as files anywhere
  Claude can currently reach; if you still have the Mermaid source or
  exported SVGs, send them so they aren't rebuilt from scratch)
- Inserting the research question(s) into the intro and threading them
  through a new/merged discussion section
- The line-by-line "de-AI-ify" pass (hyphenation, rule-of-three, stock
  connector words) — this is literally a text edit, needs the source
- Moving material to a Supplementary Information section (standard AGU
  structure)

**What I need from you:** the LaTeX project (Overleaf link with edit access,
or exported `.tex`/`.bib`/figure-source `.zip` dropped into this repo). Once
I have it, everything in this section becomes something I can do directly,
the same way I've been doing the figure/table/citation work all along.

### 2a. On Figure 2 specifically
Ed wants it to become *the* figure — "really amazing quality... showing all
model inputs, processing steps and outputs." This is worth treating as its
own small design project (like the three method flowcharts earlier), not a
quick touch-up: I'd want to iterate on it with you the same way we did for
Figures 4-6 originally (a round of drafts, your feedback on
abstraction level/detail, repeat). Let's schedule that as its own thread
once Figure 1 removal is confirmed and I have a source format to publish
into (Mermaid → image export, or native TikZ if you want it to live in the
LaTeX directly for cleaner PU/quality control).

## 3. Items needing new analysis/modeling (design decisions needed first)

This is the biggest substantive piece: **total-event-impact via restoration
scenarios**, which Ed has now asked for twice.

**What exists today:** `scripts/compute_island_benefit_cost.py` converts the
fragility-driven `$/day` diff into a one-time number by multiplying by a
*flat* assumed repair duration (180/365/730 days) — i.e. it assumes the full
daily loss rate persists unchanged for N days, then stops instantly. That's
a crude step function, not a restoration curve.

**What Ed is asking for:** a genuine temporal restoration model — daily loss
starts at the full fragility-driven rate right after the storm and declines
over time as repairs/replacement progress, integrated to a true "total event
impact in $" rather than `rate × flat days`. This is the same total-impact
framing his own NZ paper (arXiv:2507.12495) uses, and it directly produces
the richer abstract/conclusion numbers he's asking for in points 3 and 9.

**What I can do:** design and implement the restoration-curve integration
(new script, probably `scripts/compute_total_event_impact.py`, sitting
between the existing `$/day` diff and the benefit-cost script), and
re-run it for every scenario/island to get real total-impact and revised
CBA numbers.

**What needs your (and Ed's) input before I build it** — these are genuine
judgment calls, not something I should silently pick:
1. **Restoration curve shape.** Options, roughly in order of complexity:
   - Step function (current behavior) — rate is flat for N days, zero after.
   - Linear ramp-down — rate declines linearly from full to zero over N days
     (captures partial restoration/staged transformer replacement).
   - Exponential decay — rate decays with some time constant (common in
     outage-restoration literature, e.g. FEMA/utility restoration curves).
   I can pull candidate restoration-curve parameterizations from the outage-
   restoration literature if useful, but the choice should be something you
   and Ed sign off on, since it directly drives every total-impact and CBA
   number in the paper.
2. **Does this replace or supplement the existing 180/365/730-day variants?**
   My read: the three variants become three *different restoration-curve
   speeds* (fast/moderate/slow) rather than three flat windows — same
   framing, more realistic function. Confirm that's the intent.
3. Once the curve is chosen, I need your/Ed's go-ahead that the resulting
   numbers (total $ impact per storm, revised CBA ratios) are what should
   replace the current `$/day`-only reporting in the abstract/conclusion.

**Sequencing note:** this should happen *before* the abstract/conclusion
rewrite (Section 4 below), since those need the real numbers it produces.

## 4. Writing/style items (your voice, Claude can draft + flag)

- **Research question(s) before the intro's final paragraph.** I can draft
  candidates once I've seen how Ed structures this in his own paper — I
  can't fetch arXiv:2507.12495 directly (this sandbox's network proxy
  blocks arxiv.org), so either paste me its intro/discussion section, or
  describe the structure you want mirrored, and I'll draft options against
  it.
- **Discussion section (standalone, or merged discussion+conclusion)**
  revisiting the research question(s) explicitly. Same dependency as above.
- **Abstract rewrite** — more numbers, less Ghosh-vs-Leontief methodological
  detail (present as a range instead), plus the new total-impact and CBA
  numbers once Section 3 lands. I'll draft this directly once those numbers
  exist; this is a natural place for me to produce a tight first draft for
  you to edit into your own voice.
- **Conclusion rewrite** — needs the same new numbers (total impact per
  event + CBA results per island/event), not just the current `$/day`
  figures.
- **AI-writing pass.** Concretely found already (not vague — exact counts
  from the submitted PDF):
  - "Critically" used 5 times as a sentence-opener, "Moreover" 3 times —
    classic stock-connector overuse.
  - At least a dozen "X, Y, and Z" triadic-list sentences (e.g. "a
    reproducible framework..., an actionable assessment..., and a proposed
    method...", "Methodological assumptions, methods of tracing economic
    cascade, and geographic scope...") — Ed's "presents everything in
    threes" complaint, confirmed in the actual text.
  - Hyphenation is heavy throughout (`physics-engineering-economic`,
    `topology-independent`, `data-driven`, etc.) — some of these are
    legitimate domain terms (`input-output`, `benefit-cost`), but several
    read as AI's reflexive-compounding habit rather than established
    terminology; worth a pass to cut the ones that aren't standard.
  I can produce a full line-by-line list with suggested rewrites once I
  have the LaTeX source (Section 2), rather than hunting through PDF text.

## 5. PU re-tally after proposed changes

| Change | Figures | Tables |
|---|---|---|
| Start | 14 | 7 |
| Remove Fig 1 | 13 | 7 |
| Combine Figs 4+5+6 → 1 | 11 | 7 |
| Combine Figs 12+13+14 → 1 (or move to Supp — then 0 in main) | 9 (or 8 if moved to Supp) | 7 |
| Combine Tables 1+2+3 → 1 | 9 | 5 |
| Replace Tables 5+6+7 with 1 new results figure (tables → Supp) | 10 | 2 |
| **Projected main-text total** | **10 figures** | **2 tables** |

→ 12 PU from figures+tables (down from 21), leaving **13 PU for body text**
(~6,500 words) if we want real margin under 25 — current body text is
~9,575 words (~19.2 PU) *before* adding the research-question/discussion
material, so a genuine prose trim (not just moving things to Supplementary)
is still needed alongside the structural changes. Worth tracking word count
explicitly as the rewrite happens rather than discovering the overage again
at the end.

## 6. Suggested order of operations

1. **You:** share LaTeX source (Overleaf access or export) — unblocks
   almost everything in Section 2.
2. **You + Ed:** confirm restoration-curve approach (Section 3) — unblocks
   the new numbers everything else depends on.
3. **Claude:** Figure 3 fix, Figure 8 regeneration, combined island-map
   figure, combined benefit-cost results figure — all independent of 1/2,
   can start immediately.
4. **Claude:** build total-event-impact / restoration-curve analysis once
   3's design is confirmed.
5. **Claude:** combine Tables 1-3, draft Figure 2 redesign, draft research
   question(s) + discussion section — once source (1) is available.
6. **Claude:** abstract + conclusion rewrite with real new numbers — once
   4 is done.
7. **You:** voice/style pass on all drafted prose; final call on AI-tell
   edits; review every figure iteration.
8. **Both:** re-run the PU tally against the actual revised source before
   submission.
