# AnchorDynamics2026 — figure flow

## Figure plan

Settled structure. Earlier drafts numbered these differently; this is the reference.

| # | Subject | Status |
|---|---|---|
| **1** | Behaviour and anchoring | Built — `fig1_session_units_behaviour_v2.pdf`. Example session, MEC population anchoring with PC1, fraction of cells locked, hit rate by state, ADI. |
| **2** | Anatomy of grid cells, spatial cells, and anchoring score (PC1) | Built — `fig2_anatomy_identity.pdf`. Best-fit slice; PC1 following by region and by cell type; medial/lateral split and the cross-axis test. |
| **3** | Pupil dilation and its relation to anchoring state | Partly built — `eye_anchoring_all_sessions.pdf`, `eye_anchoring_transitions.pdf`, `eye_anchoring_track_profiles.pdf`, and the single-trial example `eye_trial_example_M21D19.pdf`. |
| **4** | Communication between brain regions (XGBoost prediction) | Not built. `cross_region_xgboost_coupling.ipynb` exists with `RUN=False`; `n_filters` still 100 (should be 10) and shank assignment is broken (`coord_probe_x` has 2 unique values). |
| **5** | LFP | Built — `fig5_lfp_anchoring.pdf`, `fig5_lfp_example_{M25D23,M27D23,M26D18,M21D22,M20D14}.pdf` (five examples spanning balance and effect size, built by `lfp_examples.py`), `fig5_speed_theta_coupling.pdf`, `figure5_lfp_anchoring.ipynb`. Raw LFP arrived in `clark2025_lfp` (NOT `clark2025`, which is theta phase only). Theta power is lower when anchored, speed-matched, and **weakens rather than shifts** (amplitude −4.1%, p = 4 × 10⁻⁶; frequency −0.04 Hz). The speed–theta **coupling halves** when anchored (slope 0.250 → 0.091, p = 0.002), surviving common-speed-support trimming, a ΔSD dose–response test and a circular-shift null. Sliding-window version (`fig5_speed_theta_timecourse.pdf`) resolves **across** sessions (window r +0.147 anchored vs +0.265, p = 5 × 10⁻⁴) but **not within** one (2/55 sessions beat their own shift null) — state blocks are long and few, so single-session traces are illustration only. **Mechanism tested and the obvious one excluded** (`fig5_speed_tuning_by_state.pdf`): speed tuning of 3,719 speed cells is unchanged (gain p = 0.28), speed decodes slightly better when anchored, and sensitivity to *unexpected* speed (position partialled out of both sides) is preserved (−4.7%, p = 0.046; null after rate normalisation). At matched timescale (250 ms bins, position partialled out) theta's speed sensitivity falls ~3× more than the cells' (−15.6% vs −4.7%), so anchoring changes **entrainment** more than the cellular speed code. Theta tracks *unexpected* speed no better than total speed (−15.6%, p = 0.002), so the prediction-error explanation is excluded. Spike–theta phase locking is **unchanged** (`fig5_phase_locking_by_state.pdf`; MRL 0.1414 vs 0.1425, p = 0.79, 7,542 cells, counts and speed matched) and its speed dependence *strengthens* when anchored (p = 1.3 × 10⁻⁵) — so the entrainment reading fails too. **Not MEC-specific and not laminar** (`region_slope.py`, run off a local microSD copy — the lab mount cannot survive this read): per-session slope change −0.00114 in MEC (p = 0.051), −0.00507 in pre/para/post/sub (p = 0.0079, *larger* than MEC), −0.00128 in visual (p = 0.111); layers null (Kruskal p = 0.71 on session medians). ⚠ Four mechanistic accounts now tested and failed (velocity gain, prediction error, entrainment, localised drive to MEC); the empirical dissociation is solid but the interpretation is open — don't assert a mechanism in the Discussion. ⚠ Cerebellum has only 3–5 usable sessions, so the intended distant negative control does not exist. ⚠ The trial-level slope halving and the bin-level −15% are different statistics — do not quote one against the other. ⚠ Do not re-run ΔpR²(speed | position) — speed is 4.5× more redundant with position when anchored, which confounds it. Organised by **layer** (ENTm1 ≈ 4× ENTm2/3/5), **not** mediolaterally (p = 0.17) — but the laminar profile *disagrees* with the cell-level one (cells peak in ENTm2), so it cannot be presented as confirming it. See the open question on the broadband decrease. |
| **6** | Multi-context | Not built for this structure. `multi_context_vr_*` notebooks exist; the medial/lateral context-decoding result was null on 7 of 7 tests. |

> **⚠ Figures 1–3 all rest on the population anchoring state, which is defined over ENTm cells only** (`REGION_PREFIX` in `lick_raster_by_trial_type.ipynb`). Any cached table regenerated with that filter disabled silently changes every one of them. This has already happened once: `eye_anchoring_trials.csv` was overwritten with an all-cell version, and an example figure was built on M28 D18 — a session with **zero** ENTm spatial cells, whose anchoring state was therefore defined entirely by visual cortex. Check the session count (58, not 80) before trusting that file.

> **⚠ The regional specificity claim is not currently supported.** Cells outside MEC follow the MEC anchoring axis at least as strongly as cells within it, and pre/para/postsubiculum exceeds MEC. See the TODO in Methods under *Per-cell agreement with the population axis*. This affects the framing of Figures 2 and 4 in particular.

This lays out the 7-figure structure for the paper, built largely from
what's already validated in this collection (see `HYPOTHESES.md` for the
full per-notebook methods/results this draws on). The narrative arc:

**tool is trustworthy → the phenomenon is real and population-wide → where
it lives anatomically → it coordinates across regions, selectively → that
coordination has a behavioural/physiological basis linking two completely
different systems → it generalizes beyond MEC entirely → it's dissociable
from whether the animal has behaviourally mastered the task.**

*Reality (tool) → reality (phenomenon) → location → coordination →
mechanism → generality → boundary condition.* Each figure raises the
stakes of the one before it; Figures 4 and 5 in particular are a
deliberate puzzle/resolution pair (4 raises "why does MEC agree with a
sensory region as much as a spatial one?", 5 answers it). Figure 7 is a
deliberate final complication rather than a tidy bow: it was built
expecting to confirm that anchoring strengthens with learning, and instead
found it doesn't — reported as found, not reframed to fit.

Each figure lists panels, which notebook(s) each draws on, and what's
built vs. new work.

---

## Figure 1 — Anchoring is a population-wide state of the cognitive map

*(merges the former Figures 1 and 2: premise, tool validation, and the
population-state evidence now sit in one figure, backed by three independent
sources — a recurrent network with known ground truth, MEC recordings, and
hippocampal imaging with experimenter-controlled ground truth.)*

**Premise.** A cognitive map is only functionally a map if it stays coupled to
the environment it represents. An internally consistent spatial code that has
silently drifted from ground truth is not a degraded map — it has stopped being
a map in the functional sense, because it can no longer guide behaviour
relative to the real world. Anchoring is therefore not a side-phenomenon layered
on the cognitive map; it is the candidate mechanism that keeps the map useful at
all.

- **A.** Conceptual schematic: a path-integrated map drifts smoothly away from
  the environment (internally coherent, externally wrong); with anchoring it is
  periodically re-locked to landmarks and stays coupled. **New — needs drawing.**
  The RNN's own drift-vs-corrected trajectories can serve as the underlying
  panel.
- **B.** What anchored vs non-anchored means at the single-cell level —
  `trial_cluster_anchoring.ipynb`'s method, redrawn as a clean schematic.
- **C.** **RNN ground truth.** A -> C -> A cue-loss session: decode error rises
  when the cue is removed, and the classifier recovers the true per-trial labels
  it never saw. Median accuracy **0.990** across 1024 units, sensitivity 0.993,
  specificity 0.884 (`anchoring_classifier_validation.ipynb`).
- **D.** **MEC population.** PC1-sorted rasters for a high- and a low-coordination
  session, side by side — the clearest visual statement of what a shared state
  looks like and what its absence looks like.
- **E.** **MEC quantification.** PC1 of the cell x trial label matrix against a
  per-cell circular-shift null (which preserves each cell's anchored fraction
  *and* its autocorrelation, destroying only cross-cell alignment):
  **significant in 71/75 sessions, median z = +11.2**; 59/60 once the two
  non-MEC animals are excluded.
- **F.** **Hippocampus, independent ground truth.** Pettit et al. calcium
  imaging: classifier population state vs experimenter-controlled cue
  visibility, best session r = 0.78, p = 4.2e-39
  (`pettit_hippocampal_anchoring.ipynb`). Three systems, one phenomenon.

**Status**: C, D, E, F built. A is new (schematic, no new analysis). B is a
redraw.

> **The effect is reliable but small, and the figure must say so.**
> Against the marginal-preserving null, raw PC1 explained variance is a median
> of **0.097 against a null expectation of 0.058** — roughly 4 percentage points
> of shared variance above chance. Mean pairwise Cohen's kappa between cells is
> **0.036**, exceeding 0.05 in only 26/75 sessions. So most trial-to-trial
> label variance is *not* the shared factor: cells largely anchor and un-anchor
> independently.
>
> "Anchoring states are population-wide events" overstates this. The supportable
> claim is **a consistently detectable population component of modest size**.
> Lead with the z-score (robust, 71/75) and report raw PC1 and kappa alongside as
> effect sizes. A reader who checks panel E will find 0.097; better that the
> figure states it.
>
> **If the stronger claim is wanted, PC1 is the wrong instrument.** It measures a
> single linear factor across all cells and all trials. If anchoring is
> population-wide but *episodic* — coherent blocks separated by mixed periods —
> PC1 dilutes it badly. A block-structure or change-point measure could show
> strong coordination that PC1 cannot see. That analysis does not exist yet and
> is the single most valuable thing that could be added to this figure.

> **⚠ Possible artefact in the population claim: coordinated silence.**
> Trials on which a cell emits no spikes have no profile to correlate and are
> **forced to the non-anchored class** rather than excluded. That affects a median
> of **12.3% of trials per cell** (mean 19.9%); every cell has some, 57.5% have
> >10%, 10.1% have >50%. And silence is **strongly correlated across cells** —
> mean pairwise r = +0.63 to +0.88, with PC1 of the silence matrix alone reaching
> **0.53-0.91**, against ~0.10 for the anchoring labels. Some trials are silent in
> every cell (the animal not running).
>
> So a coordinated, non-anchoring signal is injected directly into the matrix PC1
> is computed from. **The circular-shift null does not catch this**: shifting
> preserves each cell's marginal, so the null keeps the same proportion of forced
> labels while the observed matrix keeps their alignment.
>
> This does not explain the whole effect — silence touches ~12% of cell-trials
> while PC1 is 0.10 — but it is the right order of magnitude to matter. **Test
> before Figure 1E is finalised**: recompute with invalid trials held as missing
> and excluded pairwise. If the population component survives, the claim is
> stronger for having been tested; if it collapses, the panel is measuring
> co-silence.

> **Confound to declare: coordination measures track how much cells switch.**
> Raw PC1 vs per-cell switch rate rho = **-0.805**; mean kappa **-0.770**; the
> z-score is best but still **-0.540**. Sessions whose cells rarely switch score
> as more coordinated. Part of this is a genuine statistical limit — few
> transitions means little to align on — but it means **the session ranking by
> coordination cannot be read as a biological gradient** without disentangling
> it. This applies to Figure 2D's anatomy panel in particular.

> **Two animals are not recording MEC.** M22 (0% ENTs, 0% ENTd) and M27 (0%
> ENTs, 0% ENTd, 40% cerebellum) supply 15 of 75 sessions and 8 of the 12 lowest
> raw-PC1 sessions. They should be excluded here on the same grounds they were
> excluded from the MCVR analysis. Note their PC1 *z-scores* are indistinguishable
> from the rest (9.7 vs 11.4, p = 0.50), and kappa does not correlate with the
> fraction of cells in MEC (rho = -0.043, p = 0.72) — so **the shared state is not
> demonstrably MEC-specific in these data**, which Figure 3 needs to know before
> building a cross-region specificity argument.
>
> Separately, the eight lowest-kappa sessions among the MEC animals are all
> **M20** (kappa 0.003-0.010, 95-100% MEC, high switch rates). One animal
> contributing 13 sessions with near-chance inter-cell agreement needs explaining
> before it is averaged in.

> **The cached label matrices are stale.** All 75 (`all_sessions_rasters/*_cache.npz`,
> built 29 Aug) were computed from `cell_classifications.csv` — **v1**: shift-optimised
> grid scores, the superseded grid-score definition, different curation. The anchoring
> labels themselves are fine (trial-correlation classifier, not spectral), but the
> cell set feeding every PC1 matrix predates the v2 rebuild. **Rebuild on v2 before
> this figure is finalised**, taking region assignments from the clark2025 cluster
> metadata rather than v1's `brain_region`.

### Classifier validation detail (Figure 1C, and Methods)

Full validation lives in `anchoring_classifier_validation.ipynb`, five panels from
the cached simulation: worked examples with true and inferred label strips; accuracy
distribution and confusion matrix; the phase-aliasing median filter on/off; the
resolution floor; and generality against gridness.

- The size-5 median filter earns its place: specificity **0.808 -> 0.884** at
  negligible cost to sensitivity (0.990 -> 0.993).
- **It also imposes a hard 3-trial resolution floor**, because it erases any run
  shorter than 3 by construction. Non-anchored recall is 0.02 for a one-trial
  epoch and 0.07 for two — effectively invisible — then 0.64 at three trials,
  plateauing above 0.80 from five. So the classifier has per-trial *labelling*
  with a 3-trial minimum detectable epoch. Still 3-4x finer than the eleven-trial
  support of the spectral classifier, but the honest number is 3.
- **Generality holds but is not flat**: accuracy correlates with gridness
  (Spearman r = +0.219, p = 1.5e-12), though small in magnitude — least grid-like
  quintile 0.970 vs 0.990 for the most grid-like. Caveat: low-gridness RNN units
  are not ramping cells, so this supports "accuracy does not depend strongly on
  periodic structure" rather than "identical for ramping cells".
- **Unvisited position bins are interpolated over, not zero-filled.** Occupancy
  patterns are shared across simultaneously recorded cells, so zero-filling can
  make trials correlate through the animal's trajectory rather than its firing —
  the exact quantity being clustered. Accuracy is unchanged on simulated data
  (0.990 either way) but the treatment is correct and may matter on real
  recordings with patchier coverage.

> **The spectral classifier is excluded from the paper entirely.** It was built
> for grid cells (harmonic structure defined as fields per trial, averaged over
> the session's grid cells) and is temporally coarse. Its logic is not inherently
> grid-restricted — a cell that repeats every trial is periodic at the
> fundamental — but it was designed, tuned and validated on grid cells, and this
> paper needs one instrument applied equally to grid and non-grid populations.
>
> **This breaks a dependency.** `xgboost_anchor_assay.py` derives its labels from
> exactly that method (spectrogram, peak bins at harmonics [12, 28, 44, 60, 76]).
> Anything built on it must be regenerated with trial-correlation labels from
> `spatial_manifolds/anchoring.py` — including Figure 3 Part B, and any existing
> Figure 6 task-anchoring result. `get_task_anchored_labels_in_time` in
> `detect_grids.py` is the other consumer to audit.

---

## Figure 2 — Anatomy of spatial cell types and where anchoring dynamics live

- **A.** Anatomical distribution of grid cells and non-grid spatial cells across
  region and the ML/AP/DV axes (`grid_cell_anatomy_v2.ipynb`).
- **B.** Which sessions record multiple regions simultaneously, and what
  co-occurs — region-combination chart plus an example multi-region session.
  This is where Figure 3's region pairs come from.
- **C.** A DV-sorted, region-labelled anchoring raster for that session
  (`trial_cluster_anchoring.ipynb` Part D).
- **D.** Does anchoring strength vary with DV position or region? **Not yet
  built.** Raw PC1 correlates with DV position (rho = -0.516, p = 2e-6, surviving
  exclusion of the non-MEC animals at -0.492) — but see the switching confound
  above: DV also predicts cell-type composition, and coordination measures track
  switch rate. **This panel must use the switching-normalised measure, not raw
  PC1**, or it will report a confound as anatomy.

**Status**: A–C built (A needs re-running on v2 — the notebook now reads the v2
datasheet through a compatibility shim, and the shift-optimised columns it used
to plot no longer exist); D is new and gated on the confound being resolved.

### Grid module identification (Methods support for A)

Modules are identified **per session** — they are defined over simultaneously
recorded cells, so a cluster pooled across sessions would at best be a spacing
band shared by different animals. Curated cells are embedded by UMAP on their
spatial autocorrelograms and clustered with DBSCAN, eps set by the k-distance
knee (silhouette collapses to the smallest radius offered: 12/28 sessions at the
sweep floor, median 10 clusters, 34% noise).

A cluster becomes a module only if grid-enriched **and** geometrically coherent:
>= 5 grid cells, >= 2x the session's own grid-cell rate, spacing IQR <= 10 cm,
orientation circular SD <= 10 degrees. Enrichment alone is insufficient — a
cluster can be grid-dominated while mixing modules.

**Yield: 31 modules in 24 of 28 sessions; 480/962 grid cells (49.9%) labelled**,
the rest left explicitly unassigned. Cross-session reproducibility of the labels
is 1.000 (median, 9 testable days). Module counts are a **lower bound**: only
6/28 sessions yield two or more, and sessions yielding none are the low-grid-cell
ones, so a null there reflects sampling rather than absence of modularity.
Implemented once in `spatial_manifolds/grid_modules.py`.

---

## Figure 3 — Anchoring coordinates across simultaneously-recorded regions, selectively

Two complementary lines of evidence for the same region pairs: do regions'
anchoring *labels* agree, and does the *functional coupling* between their cells
strengthen during population-anchored epochs? If both tell the same story for
the same pairs — especially the negative control staying flat on both — that is
much stronger than either alone.

**Part A — label agreement (built):** cross-region Cohen's kappa heatmap,
row-normalised confusion matrices, the PC1-vs-agreement correlation (r = 0.45),
and session-to-session consistency (`population_anchoring_agreement.ipynb`).

> **Report the n for each pair in the figure, not a footnote.** Region inventory
> across the 75 cached sessions: ENTs 5003 cells / 60 sessions; ENTd 2473 / 42;
> VIS 1464 / 45; PARA 750 / 31; CERE 487 / 11; **HPF 108 cells / 4 sessions**.
> MEC–HPF is quoted as the strongest agreement (kappa = 0.52) and rests on 108
> cells; MEC–CERE carries the specificity argument on 11 sessions. The pairs with
> real support are **MEC–VIS (45 sessions)** and **MEC–PARA (31)**. Lead with
> those; present HPF and CERE as preliminary or drop them from the headline.

**Part B — functional coupling (new, not built):** does predictive coupling
between cells strengthen during population-anchored epochs? Uses the existing
XGBoost assay design — target cell predicted from position alone vs position plus
other cells, ΔpR² compared between anchored and non-anchored trial subsets — but
with **the conditioning variable changed from the target cell's own label to the
population state**. That removes a circularity: conditioning a cell's
predictability on its own anchoring label means both are computed from the same
trial-by-trial rate map.

- **B1. Within-region (MEC->MEC)** — cheapest; same pipeline, swapped conditioning
  variable.
- **B2. Cross-region** — covariate pool from the other simultaneously recorded
  region. This is where the parked **covariate-pool-size sweep** belongs:
  subsample at fixed sizes (5/10/20 cells) and plot ΔpR² against pool size rather
  than picking one threshold blind, since region pairs differ enormously in yield.
- **B3.** Three informative outcomes: coupling strengthens within and across
  regions (shared computation); within MEC only (local coherence); nowhere
  (a shared external gate correlating independent codes — "labels agree, coupling
  doesn't", itself a clean result).

> **Prerequisites before B is worth running**: (i) the population state must be
> established on v2 (see Figure 1); (ii) `xgboost_anchor_assay.py` must be moved
> off the spectral classifier; (iii) the finding that kappa does not correlate
> with MEC fraction needs explaining, since it weakens the premise that this is a
> MEC-centred phenomenon.

---

## Figure 4 — Engagement links MEC and hippocampus, and resolves Figure 3's puzzle

Figure 3 raises a puzzle it cannot resolve alone: why does MEC agree with visual
cortex about as strongly as with parasubiculum? This figure tests a
shared-engagement account using two independent measurement systems.

- **A.** MEC pupil and gaze: smaller, less variable pupils and steadier gaze on
  anchored trials (trial-level r = -0.26 pupil, -0.17 gaze) —
  `eye_position_task_anchoring.ipynb`.
- **B.** Restate the puzzle explicitly as the thing being explained.
- **C.** Hippocampal `engagement_cluster` — Pettit et al.'s independent,
  licking-derived engagement label, separate from the cue-visibility ground
  truth. Does the classifier's population state track it too? **New.**
  **Do first**: check whether cue visibility and engagement are themselves
  correlated in that dataset — if blind trials are also disengaged trials, the
  "two independent measures" claim needs restating.
- **D.** If C holds: convergent evidence from two systems (pupil in MEC,
  licking in hippocampus) for one construct.

**Status**: A built; C–D new.

---

## Figure 5 — Anchoring generalizes beyond MEC

- **A.** Full Pettit replication: all 6 sessions with enough blind trials are
  positive and significant (r = 0.31–0.78, median 0.58, 3 mice). The
  trial-number-alignment bug fix deserves a methods footnote — a real, catchable
  failure mode for anyone reusing that dataset.
- **B.** PC1 comparison, hippocampus vs MEC: 9% (39 sessions) vs 15% (75) —
  **recompute both against the marginal-preserving null**, since the raw values
  carry the switching confound.
- **C.** Apply Figure 1's permutation test to the hippocampal sessions.

> **Decision needed**: if the headline hippocampal panel moves into Figure 1 as
> convergent evidence, this figure becomes the depth treatment rather than the
> reveal. That is defensible, but it is a weaker closing figure than it was.
> Consider whether the paper closes here or on Figure 6.

**Status**: A built; B needs the null-corrected recompute; C reruns existing code
on already-converted data.

---

## Figure 6 — Anchoring is dissociable from behavioural mastery of the task

The `harry_multi_context_vr` dataset (6 mice, 30 usable sessions): every session
alternates in ~20-trial blocks between a long-familiar context (`rz1`) and one
introduced as novel (`rz2`). No open-field session exists, so there is no
grid/non-grid identity — every panel is whole-population, split only by region.

Built to test a clean expectation — population anchoring should be weak in `rz2`
early and strengthen as the context is mastered — which **did not hold**, and is
reported that way rather than reframed. Panels K–N ask why: re-indexing trials by
exposure count rather than day shows there mostly is not a multi-day learning
curve to track, since most mice reach competence within their first block of
`rz2` exposure. That converts the figure from "anchoring fails to track learning"
to "anchoring is consistent with there being no slow learning process to track".

> **M22 and M27 supply no usable MEC cells** and should be excluded here too, as
> in the medial/lateral analysis.

**Status**: A–N built.

---

## Open-field anchoring — a new capability, not yet placed in the flow

The trial-based classifier needs repeated traversals of the same track. An
open-field session has none, and any time window samples a different partial
subset of the arena — so window-against-window correlation confounds *did the
code drift* with *did the animal go somewhere else*. Scoring each window against
a **whole-session reference map**, over the bins that window actually visited,
removes it: the reference has full coverage, and this is the true analogue of the
trial version (each trial scored by its correlation with the consensus).

Validated on a 2D RNN cue-loss session (`open_field_anchoring.ipynb`): decode
error 6.3 cm anchored vs 109.7 cm blind; **accuracy 1.000 for 30-90 s windows**,
100% of units. Detection survives down to **~17 cm of drift in a 2.2 m arena**,
and holds for non-grid units — so the generality of the trial-based classifier
carries over. Implemented as `open_field_anchoring()` in
`spatial_manifolds/anchoring.py`.

**A tested hypothesis that did not hold.** Anchoring score does *not* decline
across an open-field session. Across 168 sessions, 24,070 cells and 402,687
windows: median per-session slope **+0.052 for grid cells (p = 0.154)** and
**+0.088 for non-grid spatial cells (p = 0.0013)** — flat to slightly *positive*,
with no difference between cell types (p = 0.31). The positive drift is likely a
firing-rate artefact (rate rises across the session, p = 2e-18, and the anchoring
slope correlates with the rate slope, rho = +0.318).

What is real is at the *start*: the first decile sits well below the rest for both
populations (z = -0.157 vs +0.019, paired p = 2.5e-12), flat thereafter. Coverage
is *higher* in that first decile, so it is not a coverage artefact. That reads as
re-anchoring on entry to the arena rather than disengagement over time — the
inverse of the engagement-decline framing, and worth considering as a panel in
Figure 1 or 4.

---

## What's fully built vs. what's left

| Figure | Built | Genuinely new work |
|---|---|---|
| 1 | RNN ground truth + full classifier validation (C), MEC rasters (D), PC1 vs null (E), hippocampal example (F) | conceptual schematic (A), single-cell schematic redraw (B); **rebuild label matrices on v2**; block-structure/change-point measure if the stronger population claim is wanted |
| 2 | anatomy, multi-region inventory, example DV raster (A–C) | anchoring-vs-DV/region summary (D) — **gated on the switching confound**; re-run A on v2 |
| 3 | label agreement (Part A) | functional-coupling XGBoost test (Part B) — **blocked on moving off the spectral classifier**; pool-size sweep; per-pair n reported honestly |
| 4 | MEC pupil/gaze (A) | hippocampal engagement test (C–D) + the correlation check that must precede it |
| 5 | Pettit replication, PC1 comparison | null-corrected PC1 recompute (B); hippocampal permutation test (C) |
| 6 | all panels (A–N) | exclude M22/M27 |
| OF | classifier + RNN 2D validation; session-trend null | placement in the figure flow; coverage-matched null for real sessions |

Real gaps, in rough order of scientific payoff:

**(1) Rebuild the population evidence on v2.** Everything in Figure 1E, Figure 2D
and Figure 3 inherits from 75 cached matrices built on the superseded v1
classification. This is cheap and gates the paper's opening claim.

**(2) A coordination measure that is not confounded by switch rate.** Both PC1
and kappa track how often cells change state (rho = -0.81 and -0.77). Until this
is disentangled, no session ranking or anatomical gradient built on them can be
interpreted. A block-structure measure may solve the effect-size problem and this
one at once.

**(3) Move `xgboost_anchor_assay.py` off the spectral classifier.** A
prerequisite for Figure 3 Part B, and a correctness fix for anything already
built on it.

**(4) Figure 3 Part B and Figure 4C together** — the two tests that distinguish
the mechanistic hypotheses rather than describing the phenomenon further.

**(5) Explain M20.** Thirteen sessions, 95-100% MEC, near-chance inter-cell
agreement (kappa 0.003-0.010). Either a real property of that animal or a
signal-quality issue; either way it should not be silently averaged in.

**(6) Explain why kappa does not track MEC fraction** (rho = -0.043, p = 0.72).
If the shared state is not MEC-specific, the framing of Figure 3 needs revisiting.

---

### A new result, not yet placed in the figure flow: does the spatial code itself generalize across contexts?

Everything above asks whether *behaviour* or *anchoring* track learning.
A newly-added Part E in the same notebook asks a more direct neural
question: train a Bayesian position decoder on `rz1` trials only, test it
on `rz2` — does the population use one context-invariant spatial code, or
genuinely distinct ones per context (classic remapping)?

**A methods correction, caught mid-analysis**: the VR corridor is
circular (teleports from 230cm back to 0cm), so position 0 and 230 are
the same seam, not opposite ends of a line. The first pass used a plain
linear error metric that didn't know this, which badly inflated errors
for edge-adjacent trials and produced a strikingly clean-looking
"compressive" directional bias — decoded position over-shooting early on
the track, under-shooting late, crossing zero near `rz2`'s reward zone.
That pattern mostly **disappeared** once wraparound-aware (circular)
distance was used instead, so it isn't being reported as a finding.

**Now run across the full 30-session batch** (up from one example
session) — and the corrected effect holds up strongly, not weakly: median
within-context error 9.1cm, cross-context 28.1cm, shuffle null 57.6cm
(median fold-change 2.4x), and cross-context error is worse than
within-context in **29/30 sessions** (Wilcoxon paired, p=3.5e-8). That's
one of the most consistent effects anywhere in this notebook. One session is shown in illustrative detail, picked
programmatically from the `best` variable computed in the batch cell
(lowest within-context error among sessions with >200 cells and a >3x
cross-context fold-change) rather than hardcoded, so this panel can't
silently drift out of sync with what the batch actually found —
currently `M29` D29 (n=288 cells, ~4.2x gap). The notebook's usual
running-example session (`M22` D42) turns out to be one of the *weaker*
sessions for this specific analysis (26th of 30 by decoding quality),
which is why a different session was used here. **What still doesn't survive**: any claim about a *specific*
directional bias pattern — the position-binned signed-error analysis is
still single-session only, not yet run across the batch the way the
headline error-magnitude result now is.

**Why it isn't Figure 7 Panel O yet, even with the batch behind it**:
this is a genuinely different kind of claim (population code
generalization) from the rest of Figure 7 (anchoring-vs-behaviour), so it
doesn't obviously slot into the existing panel sequence even though it
now meets the same batch-validation bar as everything else in the figure.
Two things worth doing before it gets a number: (1) run the position-
binned bias analysis across the batch too, the same way the headline
error-magnitude result now is, so any claim about *where* or *how* the
remapping shows up is as solid as the claim that it happens; (2) check
whether cross-context decoding accuracy itself improves across `rz2`
exposures — the direct neural analogue of Part D's exposure-indexed
behavioural curves. Given the strength of the batch result, this is now a
strong candidate for its own short figure (spatial code generalization is
a distinct enough claim to carry one) rather than folding into Figure 7.

---

## Hypothetical mechanisms of anchoring

Four candidate mechanisms, not mutually exclusive — the likely true story
is layered (something gates whether correction happens; something else
describes what the code does once it doesn't):

- **A. Sensory/landmark correction.** Anchoring = whether available visual
  input is being used to periodically reset accumulated path-integration
  error. Literally the RNN's mechanism (net7's place-cell-driven
  hidden-state reset); most directly *supported* — it's what Figure 6's
  ground-truth manipulation tests, and it holds (r up to 0.78).
- **B. Global arousal/neuromodulatory gating** (LC-noradrenergic or
  cholinergic tone). A diffuse state modulating sensory gain and code
  stability across many regions at once — explains why MEC agrees with
  PARA, HPF, *and* VIS roughly equally rather than VIS being specially
  privileged. Pupil dilation (Figure 5A) is a textbook readout of exactly
  this.
- **C. Intrinsic attractor-network dynamics.** Describes *what happens* to
  the code once correction lapses, not *why* — MEC's grid code drifting
  coherently under a continuous-attractor account, consistent with the RNN
  showing smoothly rising decode error rather than instant phase
  scrambling. Compatible with, not competing against, A or B.

  > **DRIFT IS REFUTED, AND SO IS DISPLACEMENT (tested 2026-10-04).** The
  > RNN's smoothly rising decode error is not what MEC does. Three tests on
  > 494 grid cells / 57 sessions (`drift_vs_remap.py`), each excluding a
  > different account:
  >
  > - *Not a slow drift between trials.* Adjacent non-anchored trials align
  >   within 10 cm only 13.4% of the time, against a 10% independent-offset
  >   floor and 24.7% for anchored pairs, with no decay from separation 1 to
  >   20. A random walk predicts adjacent trials almost always aligned, and
  >   alignment decaying with separation. Neither holds.
  > - *Not a coherent field somewhere else.* Best-of-offsets correlation on
  >   non-anchored trials is +0.557 against an UNRELATED-CELL floor of
  >   +0.538 under the same search — only +0.019 above it, where anchored
  >   trials sit +0.051 above. This reproduces the control already in
  >   MANUSCRIPT.md §51 (+0.029 above a +0.500 floor). The best-of-offsets
  >   search is optimistically biased and an r_best near 0.56 means nothing
  >   without this floor. **Any account requiring a displaced-but-intact
  >   field — including a failed trial-onset reset — is excluded here.**
  > - *Not a drift within a trial either.* Matching to the anchored template
  >   window by window along the track is flat on non-anchored trials
  >   (+0.049 / +0.059 / +0.055 / +0.024 / +0.050 across 0–200 cm; first vs
  >   last window p = 0.53). A phase set at trial onset that then drifts
  >   predicts high early and falling. Anchored trials do decline slightly
  >   (+0.315 → +0.281, p = 0.0002), so the measure can see such a gradient.
  >
  > What survives is the manuscript's existing characterisation: on
  > non-anchored trials the cell loses spatial correspondence to the track,
  > uniformly along it, with no coherent field at any offset. C is left
  > with no mechanism — the drift it was invoked to describe is absent at
  > both timescales, and the attractor-phase framing of anchoring is not
  > supported by these data. A phase decorrelating much faster than one
  > traversal would also look like this and is not excluded, but testing it
  > needs within-trial phase estimation, not trial maps.
- **D. Task engagement/motivation, distinct from generic arousal.** The
  Pettit paper's own framing (a licking-derived engagement measure, not a
  physiological arousal proxy) — the one candidate with independent
  published support from a different lab and system (Figure 5C).

- **E. Cholinergic gating of a theta-nested two-loop circuit, with
  anchoring as the reliability of a trial-onset phase reset.** SPECULATIVE
  AND UNTESTED — queued for the next budget. This is an attempt to make
  A, B and C into one circuit rather than three candidates, and it is
  written out in full below because every piece of it is falsifiable.

### E in full — the two-loop account

*Status: speculation. Nothing in this subsection has been tested. It was
built to be consistent with the LFP figure, the monosynaptic figure, the
pupil figure and the drift result simultaneously, which is the only thing
currently recommending it.*

**The starting observation.** Gamma is generated *by* fast-spiking
interneurons — ING (mutual inhibition among FS cells) and PING (E→I→E),
with the frequency set by the kinetics of the inhibitory limb. Two gamma
bands are therefore plausibly **two loops with different kinetics**, not
one rhythm with variable power. The monosynaptic figure already shows the
PING motif directly: interneuron→grid is the strongest pathway in the
dataset (3.93%, 5.1x over chance), grid→grid the strongest excitatory one,
and inhibition is the slower limb (3.0 vs 1.0 ms).

**Why the FS cells are the hinge.** Three facts about the same population:
they generate the gamma bands; they are the dominant inhibitory input to
grid cells; and they are the speed cells. In attractor models the recurrent
connectivity maintaining the bump is inhibitory and a velocity signal
shifts it — so one population supplies the attractor's recurrent
inhibition, the velocity input for path integration, and the rhythm whose
frequency indexes which loop is running. The interneurons are not
bystanders reporting the state; they are the machinery whose balance *is*
the state. This is a better reason for them to follow it than "they are
coupled to grid cells".

**The loops.** Slow gamma (30–48 Hz) = the recurrent loop: grid↔FS, slower
inhibitory kinetics, the attractor path-integrating, plus hippocampal input
(which in the Colgin framework also arrives in slow gamma). Fast gamma
(60–100 Hz) = feedforward sensory drive entraining a faster, input-locked
inhibitory loop.

**The polarity, which is the part that initially looks wrong.** Under
Hasselmo, high ACh favours feedforward sensory input and suppresses
recurrent transmission; low ACh favours recurrent dynamics. The anchored
state is the *constricted* pupil — lower cholinergic tone — and should
therefore be recurrent-dominant: slow gamma up, fast gamma down and
uncoupled from theta. **That is what the LFP figure shows.** The direction
that reads as wrong under "landmarks pin the map" is the predicted
direction once the gate is cholinergic.

**The mechanism.** Engagement sets which loop dominates. In the engaged
state the recurrent loop runs and the entorhinal code stays bound to the
task reference frame; feedforward visual drive is down-weighted and no
longer theta-locked, which is *desirable* here because the corridor is
visually repetitive and the reward zone is unmarked on uncued trials,
making vision an actively misleading position cue. In the disengaged state
feedforward drive is strong and theta-locked, and the code loses its
correspondence to the track.

> **WHAT THIS SUBSECTION NO LONGER CLAIMS (2026-10-04).** An earlier version
> said anchoring was the reliability of a grid-phase RESET at trial onset —
> bump set at the teleport, path-integrated down the track, landing at an
> arbitrary phase when the reset failed. The drift tests above exclude it:
> a failed reset predicts a coherent field at a wrong offset, and once the
> unrelated-cell floor is applied there is no coherent field at any offset,
> nor any gradient along the track. The phase-level story is gone. What
> survives is the *circuit* claim — two interneuron-generated loops whose
> balance is set neuromodulatorily — which does not depend on where the
> grid phase sits on a non-anchored trial, and the band-direction argument
> below. Do not reinstate the reset mechanism without new evidence.

**What it already accounts for that the alternatives do not.**

  - *The speed-cell null.* FS speed cells carry the same speed code in both
    states. If they are the velocity input, path-integration gain is
    unchanged — so the integrator *should* be intact and only the reset
    should fail. The result currently written up as mechanistically
    unexplained is what this account requires.
  - *The no-redistribution constraint.* ACh acts on feedforward
    transmission and on recurrent gain through different targets, so the
    two bands move independently rather than as a seesaw — predicting the
    uncorrelated magnitudes across sessions (rho = -0.08) that rule out the
    tidier switch model.
  - *The connectivity null.* Co-membership of the attractor manifold, not
    the edges, is what two cells share. Connected pairs need not share the
    state more than matched unconnected ones.

**Independent support for the premise, from this same task.** The weakest
link above is the claim that suppressing visual input is *adaptive* rather
than merely odd — it only makes sense if the animal's position estimate is
already self-motion dominated. Tennant et al. (2018) tested exactly that in
this task: when the speed gain between the treadmill and the visual scene
was manipulated, mice followed their **motor** information rather than the
visual scene. So the animal's own strategy privileges self-motion over
vision when the two conflict, which is independent, published, same-task
evidence for the premise rather than something assumed to make the model
work. It also sharpens what the trial-onset reset has to do: if motor
integration carries position, the only thing that needs setting at the
teleport is the *starting point*, which is precisely the variable the drift
result says is unreliable on non-anchored trials.

This also reframes the LFP direction. Vision being down-weighted in the
engaged state is not a puzzle to be explained away — it is the neural
counterpart of a behavioural preference already demonstrated in this task.

**Tests, cheapest first.** All three run on data already in hand.

  1. **Which structure does each band cohere with?** THE TEST THAT MATTERS,
     because it is the one that earns the band assignment in entorhinal
     cortex instead of importing it. The slow=internal / fast=external gloss
     comes from CA1 (Colgin et al. 2009), where fast gamma indexes *MEC
     input to CA1* — a definition that cannot be carried into MEC, since
     there the band cannot index its own input. But this collection has
     sessions recording MEC, the subicular complex and visual cortex
     simultaneously, with raw LFP in `clark2025_lfp`. If MEC fast gamma
     coheres preferentially with visual cortex and MEC slow gamma with the
     subicular/hippocampal side, the assignment is established *here*,
     in entorhinal cortex, from these recordings. If it does not, E's
     band interpretation fails and should be abandoned rather than hedged.
  2. **Does preferred theta phase of grid and FS spiking shift between
     states?** If the dominant loop changes, the spikes should move
     relative to theta. `clark2025` carries theta phase, which is all this
     needs. NOTE: this shows that *something* about the loop balance
     changes; on its own it does NOT establish which band carries which
     input stream. That is test 1's job, and an earlier version of this
     document wrongly credited test 2 with it.
  3. **Is inhibitory latency/strength in monosynaptic grid–interneuron
     pairs unchanged between states**, while the feedforward signature
     changes? The attractor should be intact; only its input should differ.

**The decisive experiment, which needs new data.** This collection has no
gain-manipulation sessions, so the following is a future experiment and NOT
part of the test queue above. Tennant et al. showed mice follow motor over
visual information *on average*; this account says that balance should
**depend on anchoring state** — visual capture should be stronger on
non-anchored trials, and near-absent on anchored ones. A gain manipulation
run during recording would test the model's central claim (that the state
*is* the motor/visual balance) at both the behavioural and the grid-phase
level in one manipulation. Worth noting as the experiment this whole account
most wants, and the reason to keep the account explicit rather than vague.

**Known limits.** The slow=internal / fast=external assignment is borrowed
from hippocampal work and the manuscript deliberately resists it — test 2
is what would earn it. Waveform classification gives FS cells, not PV vs
SST, so "which interneuron implements which loop" is beyond these data and
should stay at the level of motif rather than cell type. And the account is
built to fit results already in hand, so it is at present a synthesis
rather than a prediction — tests 1–3 are what would change that.

**Existing evidence against pure local circuit reconfiguration**: the
pre-existing NGS→GC XGBoost result (`task_anchoring_influence_of_xgboost.ipynb`)
found anchoring state doesn't change how well local non-grid spatial cells
predict grid cell firing — a strike against local rewiring, pushing toward
an upstream-gated or global-state account (A/B/D).

**What would separate them**: Figure 4 Part B (does cross-region
*coupling*, not just label agreement, strengthen) and Figure 5C (does
hippocampal engagement predict anchoring even among visible trials, i.e.
with cue availability held constant) are the two most direct tests
currently scoped. If Figure 5's arousal/engagement panels explain away
Figure 4's MEC-VIS kappa, that favours B/D over A being sufficient alone.
Figure 7 already bears on this: if **D** (task engagement/motivation) were
doing most of the work, anchoring should track behavioural competence at
getting the reward — instead it's strong and significant from the first
session in a context regardless of whether that mouse ever learns where
the reward is, which favours **A** (landmark availability alone is enough
to sustain anchoring) over engagement/motivation being the primary driver
— though this is one dataset's group-level pattern, not a strong causal
test on its own. Figure 7 Panel G adds a timescale wrinkle to this: within
a single day, anchoring *does* decline together with hit rate across
blocks (sharpest in the hardest trial condition) — exactly the kind of
coupling a state-dependent gate (B, arousal/motivation) would produce, just
operating too fast to explain the (null) multi-day pattern. The two
findings aren't contradictory if **A** sets the multi-day floor
(landmark-driven correction available regardless of learning) while **B**
modulates it on top, minute-to-minute within a session.

---

## Future direction — reframing Alzheimer's-model deficits as anchoring changes

Several published AD-mouse-model studies use exactly the ephys/imaging +
VR-linear-track paradigm this collection's tooling was built for, and
report findings — MEC-hippocampal desynchronization, place-field
instability/enlargement, disrupted replay and reduced offline map
stabilization — that are currently framed as independent phenomena. Worth
testing directly whether the anchoring pipeline (PC1 population state,
permutation significance, cross-region agreement) applied to one of these
datasets shows those deficits are reducible to a change in anchoring
rate/quality, rather than separate mechanisms — that would be a strong,
disease-relevant generalization test beyond Figure 6's healthy-mouse
hippocampal replication.

Candidate datasets flagged from a literature pass (Sept 2026):
- Medial entorhinal-hippocampal desynchronization in an AD mouse model —
  VR linear track, dual 256-ch silicon probes, MEC+hippocampus
  simultaneously; closest match to this collection's own recording setup.
- Aberrant MEC dynamics linked to tau pathology and spatial memory
  impairment (bioRxiv) — MEC-specific, complements the desynchronization
  paper above.
- Disrupted hippocampal replay / reduced offline map stabilization in an
  AD mouse model — natural pairing with Figure 2's population-state
  framing, if trial-level anchoring labels can be reconstructed from their
  data.

Not scoped into the 6-figure structure above — this is a follow-up-paper
idea, not a gap in the current one.
