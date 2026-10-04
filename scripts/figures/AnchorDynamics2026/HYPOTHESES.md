# Task-anchoring hypotheses

This collection has a narrower focus than the rest of `scripts/figures/`: it's
about validating and applying a **task-anchoring classifier** (does a cell's
spatial tuning stay locked to landmarks/track structure on a given trial, or
does it drift?), then asking what that anchoring state correlates with —
behaviourally, physiologically, and anatomically. Each notebook below tests a
specific hypothesis; together they build from "does the classifier work" to
"what does anchoring state actually mean for the animal."

## 1. `net7_1D_track_simulation.ipynb` — does a task-anchoring classifier recover known ground truth?

**Hypothesis**: a classifier that labels each trial "anchored" or
"non-anchored" from a cell's spatial firing pattern alone can correctly
recover the *true* anchoring state, when that state is known by construction
(not assumed).

**Method**: an RNN trained on a 1D VR track is run through a controlled
A→C→A session (cue-visible → cue-blind → cue-visible again), so the true
anchored/non-anchored phase of every trial is known exactly. A
phase-coherence-gated spectral classifier (dominant-frequency power +
cross-trial phase agreement at track-length harmonics) is validated against
this ground truth.

**Key findings**: the naive spectral classifier lagged the true transition by
~10-15 trials — traced to two causes: (1) genuine multi-trial spectrogram
window blending at boundaries, and (2) more fundamentally, raw spectral
*power* can't distinguish phase-locked firing from same-frequency
random-phase firing. Gating on phase coherence (not just power) fixed most of
this. This notebook is also where the trial-clustering alternative
(notebook 2) was first validated against the same ground truth, since it's
the only place ground truth exists.

**Status**: validated; established the RNN simulation + ground-truth
framework the rest of this collection builds on.

---

## 2. `trial_cluster_anchoring.ipynb` — a simpler, assumption-free anchoring classifier, and does it reveal population structure?

**Hypotheses** (four, building on each other):
1. A correlation-based classifier — cluster trials by how well each trial's
   spatial firing profile correlates with every other trial, no frequency-
   domain assumptions — can match or exceed the spectral method's accuracy
   against the same simulated ground truth.
2. The method transfers to **real biological VR recordings**, where there is
   no ground truth, producing well-separated (high-silhouette) clusters
   rather than an arbitrary 50/50 split.
3. Anchoring is not purely a single-cell phenomenon: a population-wide,
   coordinated anchored/non-anchored state exists and is detectable via PC1
   of the population's trial x label matrix.
4. This population structure is not idiosyncratic to one example session —
   it should be checkable, and vary in strength, across the full set of
   recorded VR sessions.

**Method**: per-trial correlation matrix → 2-means clustering on each
trial's mean correlation with all others → median-filter correction for a
specific failure mode (grid-cell periodicity causing chance phase-aligned
false positives during genuinely non-anchored trials). Validated on the
net7 ground truth, then applied to M25D24, then to all 75 sessions in
`cell_classifications.csv` with brain-region and dorsoventral-position
context.

**Key findings**: 95-100% accuracy against ground truth after the
phase-aliasing fix (median filter, size 5); ~99% median accuracy across 7
test units. On real data, GC and NGS populations both show >0.3 silhouette
in 100% of cells (M25D24). PC1 of the label matrix explains 15-22% of
variance in example sessions and 6-33% across all 75 sessions — i.e. a real
but variable degree of population-wide coordination in every session
checked.

**Status**: validated on simulated + real data; scaled to all sessions.

---

## 3. `eye_position_task_anchoring.ipynb` — does anchoring state show up in gaze and pupil behaviour?

**Hypotheses**:
1. Gaze position (pupil centroid, DLC-tracked) systematically differs
   between anchored and non-anchored trials — e.g. the animal looks
   somewhere different, or more variably, when not anchored to the task.
2. Pupil dilation (an arousal/attention proxy independent of *where* the eye
   points) also differs by anchoring state.

**Method**: M25D24's eye-tracking channels (`eye_x`, `eye_y`,
`eye_dilation`, ~30Hz DLC) are binned onto the same trial x position-bin
grid as the neural rate maps and compared against the population anchoring
state from notebook 2's cached labels — at both trial resolution (deviation
from the session's typical gaze profile) and raw per-frame resolution.

**Key findings**: no shift in *average* gaze location between states, but
gaze is modestly less variable during anchored trials (trial-level Spearman
r=-0.17, p=0.009; raw frame-level std lower on anchored frames, p=1.8e-24
for eye_y), and pupils are smaller and less variable during anchored trials
(trial-level r=-0.26, p<0.001; raw frame-level p=1.7e-240). All effects are
small in absolute terms (~10-15%) but consistent in direction across three
independent metrics — more suggestive of a general arousal/engagement
co-variate than a "looking somewhere else" effect.

**Important caveat baked into the analysis**: a real bug in pynapple's
`compute_1d_tuning_curves_continuous` (silently zero-fills position bins with
no eye-camera samples instead of marking them NaN) inflated the first-pass
effect size 5-7x and even flipped its sign before being caught and fixed.
Any new eye-tracking analysis reusing this binning approach should apply the
same exact-zero-to-NaN correction.

**Status**: single session (M25D24) only. Worth checking whether the
direction/magnitude of these effects replicates across the other 74 cached
sessions in `trial_cluster_anchoring.ipynb`'s `all_sessions_rasters/`
before treating this as a general finding.

---

## 4. `grid_cell_anatomy_v2.ipynb` (copy) — where are anchored/anchoring-relevant cells anatomically, and which sessions let us compare regions directly?

**Hypotheses**:
1. Grid cells and NGS cells are non-uniformly distributed across brain
   regions and across the ML/AP/DV anatomical axes (tested in the original,
   unmodified sections of this notebook — copied here unchanged).
2. (New in this copy) A non-trivial fraction of sessions record from more
   than one brain region *simultaneously* (not just across separate
   sessions), which matters for any future analysis that wants to compare
   regions without the confound of recording on different days.

**Method (new section)**: for every session in `cell_classifications.csv`,
identify the set of distinct macro brain regions recorded (MEC/PARA/PRE/
HPF/CERE/SUB/VIS), then separately re-check with MEC split into superficial
(ENTm1-3) and deep (ENTm5-6, ENTl6a) sublayers.

**Key findings**: 23/69 sessions (33%) simultaneously recorded >=2 distinct
macro brain regions — 19 sessions combine MEC+PARA (mice 25-29) and 4
combine MEC+HPF (mouse 21 only). No session recorded 3+ distinct macro
regions. The best-balanced example (M26D15, 57 MEC + 63 PARA cells) is
plotted anatomically as a demonstration. Splitting MEC into superficial/deep
sublayers shows nearly all sessions record from both simultaneously (27/50
MEC-only sessions), which is a within-region layer distinction, not a
multi-region one, and is reported separately for that reason.

**Status**: new analysis, single pass — the region categorization here
(from `cell_classifications.csv`, curated/classified cells only) is
necessarily narrower than the one used in notebooks 2-3 (from
`all_cluster_brain_locations_chris.csv`, every recorded cell) since it only
covers cells that passed spatial-cell classification; a session could record
from a region with zero classified cells and not show up here. (Notebook 5
below redoes this multi-region identification from the broader "every
recorded cell" population and finds substantially more multi-region
sessions as a result — see there for the fuller picture.)

---

## 5. `population_anchoring_agreement.ipynb` — is anchoring a population-wide state, and do brain regions agree on it?

**Hypotheses**:
1. How much of a session's anchoring dynamics is explained by a single,
   population-wide factor (PC1, notebook 2) — quantified across every
   cached session, not just the M25D24 example.
2. For sessions recording multiple brain regions simultaneously, each
   region's population-level anchoring state should agree with the other's
   more than chance — and more so for regions plausibly linked to the task
   (MEC, parasubiculum, hippocampal formation) than for one that isn't
   (cerebellum).
3. Sessions with stronger within-region population coordination (higher
   PC1) should also show stronger cross-region agreement, if both reflect
   the same underlying shared state.

**Method**: reuses notebook 2's cached PC1 explained variance directly (no
recomputation). For cross-region agreement, region-level "population
consensus" per trial (majority vote across a region's cells, threshold 0.5)
is compared between simultaneously-recorded regions via Cohen's kappa
(chance-corrected, since anchored/non-anchored base rates differ by region
and session) — pooled across all sessions sharing a region pair, plus
row-normalized 2×2 confusion matrices for the headline pairs. Multi-region
sessions are identified from the anchoring cache's own "every recorded
cell" population (same cells the label matrices are built from), which
turned out to be substantially broader than notebook 4's classified-cells-
only view: 53/75 sessions (71%) record ≥2 macro regions, spanning
MEC/PARA/VIS/HPF/CERE/SUB/PRE combinations — including 3 sessions with
simultaneous MEC+CEREBELLUM recording, which notebook 4's narrower dataset
didn't have at all.

**Also fixed here**: HPF (hippocampal formation) was silently falling into
the catch-all `OTHER` region bucket in the shared anchoring cache
(`all_sessions_rasters/*.npz`, used by notebooks 2 and 3 too) — 108 cells
across mouse 21's sessions were affected. Fixed by remapping the cache in
place; `trial_cluster_anchoring.ipynb`'s region legend/colors were updated
to match.

**Key findings**:
- PC1 explained variance: mean=15%, median=15%, range=[6%, 33%] across all
  75 sessions, uncorrelated with session cell count (r=-0.00) — every
  session shows *some* shared anchoring signal, but it's a minority of the
  variance, not a dominant factor.
- Cross-region agreement (Cohen's kappa, pooled): HPF-MEC 0.52, MEC-VIS
  0.49, MEC-PARA 0.44, PARA-VIS 0.38, CERE-PARA 0.28, CERE-VIS 0.27,
  **CERE-MEC 0.19**, MEC-SUB 0.01 (n=1 session, unreliable). The
  row-normalized confusion matrices show strong diagonal dominance (68-85%)
  for MEC vs PARA/HPF/VIS, breaking down for MEC vs CERE (61-64%, close to
  chance) — directly confirming the "high MEC-PARA, low MEC-CEREBELLUM"
  prediction that motivated this notebook.
- Unexpected: MEC-VIS agreement is as strong as MEC-PARA, despite visual
  cortex having no obvious direct role in spatial/task anchoring — read
  together with notebook 3's pupil/gaze findings, this points toward a
  shared arousal/engagement signal underlying anchoring state across
  regions, not something intrinsic only to spatially-tuned circuits.
- Spearman corr(session PC1 var, cross-region kappa) = 0.45 (p<0.001,
  n=104 region-pair-sessions): sessions with more within-region population
  coordination also show more cross-region agreement, consistent with both
  reflecting one shared underlying signal.

**Status**: validated; directly answers this collection's second
cross-cutting open question below. Low-session-count pairs (CERE-MEC n=3,
HPF-MEC n=4, MEC-SUB n=1) should be treated as preliminary until more
sessions accumulate.

---

## 6. `session_validity_pc1_significance.ipynb` — which sessions are actually valid to subset on population structure?

**Hypothesis**: raw PC1 explained variance (notebooks 2 and 5) can't by
itself distinguish a session with real, shared population-wide anchoring
structure from one where the same PC1 value is just what chance alignment
of independently-switching cells would produce — because PC1 of pure noise
is never zero, and how far from zero depends on the session's own cell and
trial count. A proper per-session significance test is needed before
treating "high vs. low PC1" as a meaningful distinction, or before pooling/
subsetting sessions for further population-level analysis (as notebook 5
does).

**Method**: for every session, build a null distribution for PC1 explained
variance by circularly shifting each cell's anchoring-label sequence by an
independent random offset (1000 shuffles) — this preserves each cell's own
run-length/switching statistics exactly while destroying trial-by-trial
alignment *between* cells, isolating the one thing being tested. Real PC1
is compared to this null via both an empirical p-value and a parametric
(normal-approximation) p-value from the z-score, the latter used for a
Bonferroni comparison across all 75 sessions since the empirical p-value's
resolution floor (1/1001) sits above the Bonferroni threshold and would
otherwise make a real effect look like a measurement artifact.

**Key findings**: 68/75 sessions (91%) clear p<0.05 uncorrected; 59/75
(79%) survive Bonferroni correction. 7 sessions show no detectable
population-wide structure at all. The highest-z session (M29D23, z=128.8)
and lowest-z session (M20D25, z=-0.9) look exactly as different by eye as
their statistics suggest — a clean coherent block structure vs. pure
salt-and-pepper noise, once both are sorted by PC1. Not a session-size
artifact: z correlates with n_cells (r=0.51, expected — more power) but not
n_trials (r=-0.34), and significant/non-significant sessions don't differ
significantly in median cell count (p=0.25).

**Output**: `all_sessions_rasters/session_validity_summary.csv` — per-
session `valid_p05` and `valid_bonferroni` boolean columns other notebooks
can filter on directly.

**Status**: validated; the recommended default subset (`valid_p05`, 68/75
sessions) should be applied as a robustness check to notebook 5's
cross-region agreement estimates, particularly the low-session-count pairs.

---

## 7. `pettit_hippocampal_anchoring.ipynb` — does the classifier recover real biological ground truth in a different region and modality?

**Hypothesis**: the trial-clustering classifier, unchanged, should recover
a real experimenter-controlled cue-visibility manipulation in hippocampal
place cells — a stronger validation than notebook 1's simulated (RNN)
ground truth, in a completely different brain region and recording
modality (calcium imaging) from every other notebook in this collection
(MEC, electrophysiology).

**Method**: uses the Pettit2022 dataset (calcium imaging from hippocampal
CA1, Pettit, Yuan & Harvey 2022 Nat Neurosci — "Hippocampal place codes are
gated by behavioral engagement"; downloaded from Dryad and converted into
this project's `beh.nwb`/`clusters.npz` convention). Only two things needed
adapting, not the classification logic itself: position-binning uses
`nap.compute_1d_tuning_curves_continuous` (continuous calcium activity, not
spike times) instead of the spike-based version, and the whole population's
`TsdFrame` is passed to it in one call rather than looping per cell (~1000x
faster — the function vectorizes over columns internally). Some trials in
this dataset are genuinely cue-blind (`fraction_visible` in the trials
table, a real behavioural manipulation, not inferred or simulated) — the
classifier's population state is compared directly against this.

**Key findings**: in the best example session (ny144, 2019-08-29), the
classifier's population anchoring state correlates with real cue-visibility
at Spearman r=0.78 (p=4.2e-39); blind trials average 0.31 fraction-anchored
vs. 0.58 for visible trials (Mann-Whitney p=8.2e-26), forming two nearly
non-overlapping clusters. Of the 39 sessions, 6 have enough blind trials
(≥5) to test — **all 6 are positive and significant** (r=0.31-0.78, median
0.58, spanning 3 mice). This is the strongest validation of the classifier
anywhere in this collection. PC1 explained variance across all 39 sessions:
mean=9%, range=[5%,20%] — a real but consistently smaller population-wide
factor than MEC's 15% (notebook 5).

**Bug caught before trusting any of this**: the ground-truth lookup
initially assumed session trial numbering starts at 1; it doesn't (ny144
starts at trial 3, ny182/ny188 both start at trial 60), silently misaligning
every trial's label for any session not starting at 1. This produced two
spurious *negative* correlations before being caught, fixed (index on
`first_trial + i`, not `i+1`), and verified directly against source data at
multiple points per session — not just by checking whether the correlation
"looked better." A general lesson for anyone working with this dataset's
trial numbering.

**Status**: validated — the one notebook in this collection with real,
not inferred or simulated, ground truth.

---

## 8. `multi_context_vr_learning_anchoring.ipynb` — does population anchoring track behavioural learning of a novel context?

**Hypothesis**: a new dataset (`harry_multi_context_vr`, 6 mice, 30 usable
sessions) has mice alternating in ~20-trial blocks, every session, between
a long-familiar VR context (`rz1`) and a novel one with a different reward
location (`rz2`), learned — by eye, during the experiment — over the
following few days. Expectation going in: population anchoring should be
weak in `rz2` early on and strengthen as the context is mastered, mirroring
the behavioural learning curve.

**Method**: single combined ecephys+behaviour NWB per session (a different
file layout from COHORT12, but `travel` plays exactly the same role as
`compute_vr_tcs`'s `dt`, so the same single-call-per-session tuning-curve
trick applies). Cell curation via the same `curate_clusters` thresholds
used everywhere else. No open-field session exists for this dataset, so
there is no grid/non-grid identity — anatomical region (Allen CCF, already
in the units table) is the only cell-type information available, and isn't
used for stratification yet. Anchoring classification (`trial_cluster_labels`),
PC1 explained variance, and circular-shift significance testing are all
copied verbatim from notebooks 2 and 6 — no re-derivation, so results are
directly comparable to the rest of the collection. Reward-zone identity
confirmed independently via first-lick position (DLC-tracked licking, not
available in the COHORT12 dataset).

**Key findings**: `rz1`/`rz2` are confirmed physically distinct reward
locations (tight, separate first-lick clusters). Behavioural learning is
**not** a group-level effect — corr(day, hit rate): rz1 r=-0.00 p=0.999,
rz2 r=-0.09 p=0.642 (n=31 mouse-day rows) — though individual mice range
from clearly improving (M26) to clearly never learning it (M27). Population
anchoring is real and robust (29/30 sessions significant vs. the
circular-shift null — a higher hit-rate than notebook 6's 68/75), with mean
PC1 explained variance (9.6%) sitting between this collection's MEC (15%,
notebook 5) and hippocampus (9%, notebook 7) numbers — a useful independent
check that the method reproduces sane numbers on a third, unrelated
dataset. It does **not** correlate with learning day, whole-session or
per-context (all |r|<0.1, all p>0.6, n=30 sessions) — anchoring is already
strong from the first recorded session in a context, not something that
visibly builds as the context is mastered.

**Extended to trial type, speed profiles, and within-day blocks**: since
`b`/`nb` trials (randomly interleaved within context blocks, not
sub-blocked) are meaningfully different task conditions — and any learning
signal, if present, would most plausibly be concentrated in the hardest
one (`rz2`-`nb`) — every cross-day test above was repeated at full context
× type resolution. The null held: behaviour stayed flat in three of four
conditions (rz1-b showed the only hint of a trend, r=0.31 p=0.089 —
notably on the *easiest* condition, the opposite of what "still learning
`rz2`" predicts) and anchoring stayed flat in all four (|r|≤0.17, p>0.37).
A complementary, non-hit-rate measure of task performance — a global PCA
on every trial's position-binned running-speed profile (6521 trials
pooled, computed without excluding stationary epochs since the approach-
to-stop is the signal itself; PC1 explains 18.2% of variance and is
essentially an "overall running vigour" axis, r=0.79 with mean speed) —
also shows no meaningful efficiency gain across days: PC1 vigour and trial
duration are both essentially flat, and where duration reaches
significance (rz2-b, r=0.07, p=0.003) it moves the *wrong* direction for a
learning story (trials get marginally longer, not shorter).

**A direct, independent behavioural readout beyond hit rate**: where on
the track the animal actually stops, per trial, computed from the pooled
speed profiles as the longest contiguous run of low-speed bins. A real bug
was caught and fixed here: the first version assumed a real stop lasts
until the end of the trial's binned range and detected 0% of stops,
because this track continues past the reward zone to a fixed per-trial
extent, so a genuine stop is a localized mid-trial dip, not a sustained
stop to the end. Fixed, it validates cleanly against the independent
lick-based check: median stop position on hit trials is 92cm (`rz1`) /
122cm (`rz2`), matching the lick-based medians almost exactly, with
detection rate graded sensibly by performance label (hit 82%, slow 97%,
try 42%, run 34%). A general lesson for reusing this notebook's
speed-profile machinery: don't assume the "settle to a stop" signature
looks the same as it does in a track that ends at the reward zone.

**The one place the data pushed back**: trials are delivered in ~20-trial
blocks per context, which opens a within-day timescale distinct from the
across-day one above. On each mouse's *first* recorded day in a context,
hit rate doesn't rise across blocks — it consistently **falls**, in all
four context × type conditions, sharpest in the single hardest one (rz1-b
r=-0.36 p=0.029; rz1-nb r=-0.49 p=0.003; rz2-b r=-0.29 p=0.101; rz2-nb
r=-0.58 p<0.001). Population anchoring (reusing already-cached label
matrices, no new session loading) shows the same within-day decline on day
1 (rz2 r=-0.46 p=0.008; rz1 r=-0.34 p=0.063). So on this faster timescale
— unlike across days — behaviour and anchoring *do* move together, just
both downward, which reads more like within-session satiety/fatigue than
a learning curve. Following the user's suggestion to look at this per
mouse as a day × block hit-rate matrix (not just pooled) revealed
something the pooled day-1-only test couldn't show on its own: the decline
is **not unique to day 1**. Pooled across mice/conditions per day_rank, it
is significant on days 1–3 (r=-0.42, -0.49, -0.28; all p≤0.001) and
**fades to non-significance by days 4–5** (r=-0.12, -0.12; p=0.18, 0.16).
The within-day decline itself is an early-training phenomenon that
disappears with general task experience — arguably the clearest *learning*
signal in this whole dataset, just not the one being tested for: not
learning where the reward is, but learning to sustain engagement across a
full session.

**Per-animal detail**: an example PC1-sorted anchoring raster for each
mouse's single most significant session, plus each mouse's own rz2
anchoring (PC1) vs. rz2 hit-rate trajectory across its own days. **M26**
— the one mouse with significant *within-mouse* behavioural improvement on
rz2 (r=0.90, p=0.037, n=5 days) — also has the most positive anchoring
trend of any mouse (r=0.60), though it doesn't reach significance alone
(p=0.28, n=5 days is simply underpowered). No other mouse pairs the two
consistently (M25: r_anchor=-0.14/r_hit=-0.31; M27: r_anchor=0.60/r_hit=
0.00; M28: r_anchor=0.40/r_hit=-0.80, opposite signs; M29: r_anchor=
-0.40/r_hit=-0.30) — the group-level null isn't hiding a general
within-mouse effect, but M26 is a genuine, suggestive exception.

**The user's own suggestion, checked directly**: is either context better
anchored than the other, visible block-by-block? A PC1-sorted anchoring
raster for every recorded day of every mouse (one figure per animal, all
days stacked, context strip + block-boundary lines) makes the question
visual; a paired test makes it exact. Within the same session, `rz1` PC1
is higher than `rz2` PC1 in 13/30 sessions and lower in 17/30 (mean
difference -0.007, Wilcoxon signed-rank p=0.393) — indistinguishable from
chance. No context is consistently better anchored, confirming the
context-split null from earlier, now checked cell-by-cell rather than only
in the summary statistic. (One implementation bug caught along the way:
the first version of this raster tried translucent background shading to
mark context blocks, which was completely invisible under the opaque
raster — fixed with a proper thin context-color strip above each row.)

**Two zone-specific trial-execution-quality metrics, and a real
methodological lesson**: beyond binary hit/miss, a **zone-specificity
ratio** (speed at the true reward zone ÷ a control-window speed) and
**stop-position error** (cm off the true reward-zone location) both
validate cleanly against the `performance` labels (hit best, run worst).
At the correct mouse-day-aggregated resolution, neither shows a reliable
cross-day trend in any context × type condition (all p>0.06) — but a
trial-pooled version of this exact test first produced a spurious
"significant" rz2-b improvement (r=-0.12, p<0.001) that evaporated, and
partly reversed, once checked per mouse (3 mice improving, 3 getting
worse, individually significant in *both* directions — a textbook
Simpson's-paradox pooling artifact from unequal per-mouse trial counts).
General lesson for this notebook's convention: any cross-day trial-level
correlation must aggregate to mouse-day level first, not pool raw trials
across mice, exactly like every other day-trend test here already does.

**How, and when, is `rz2` actually learned — re-indexed by exposure, not
by day**: `rz2` recurs ~5-6 times within day 1 alone (blocks alternate
context), so re-indexing every `rz2` trial by "the Nth time this mouse has
ever seen it" (continuous across the whole recording) turns the day-level
null above into a proper exposure-indexed learning curve, and separates
two distinct, both fast, processes. **Task-level success is zero-shot**:
hit rate in the first 10 exposures ever is statistically indistinguishable
from that mouse's own long-run `rz1` rate (Wilcoxon paired, n=6 mice,
p=1.0) — though the literal first-ever trial is a coin flip across mice
(3/6 hit it immediately). **Stopping precision is one-shot, not
gradual**: stop-position error drops sharply from the first block of
exposures (1-10) to the second (11-20) in 5/6 mice (p=0.063, the strongest
signal n=6 permits) and then **plateaus** with no further change through
exposures 21-60 (p=0.44, only 1/6 mice still improving). Distribution
plots of stop position (context × day × type) confirm this visually: no
progressive narrowing across days and no second mode near the *other*
context's location (i.e. no visible habit-driven misfires onto the
familiar `rz1` position while in `rz2`). Neither process operates on the
multi-day timescale the notebook was originally built to test — which
reframes the day-level anchoring-vs-learning null above: there mostly
isn't a slow multi-day behavioural learning curve in this dataset for
anchoring to fail to track in the first place.

**But that one-shot result is trial-type-specific, not general**:
repeating the same test with a **type-specific exposure counter** (the Nth
`rz2-b` trial ever, separately the Nth `rz2-nb` trial ever — `b`/`nb` are
randomly interleaved, not blocked) shows the calibration is clean only on
**cued** trials — **6/6 mice** improve from exposures 1-10 to 11-20
(p=0.031, the maximum n=6 permits), then plateau. On **uncued**
trials it's weak (4/6 mice, p=0.56) and precision often gets *worse*, not
better, from exposures 11-20 to 21-60 (5/6 mice, p=0.063) — the opposite
of a learning signal. Sensible mechanistically: a signaled reward zone
gives an immediate landmark to calibrate a stop against; without a
signal, the animal has no comparable fast error signal (only distance run
through the always-visible context wall pattern, plus memory), and may be
more exposed to the same satiety/engagement decline documented above. So
"one-shot precision calibration" is a property of the cued condition
specifically — the harder, unsignaled condition shows no clean learning
signal on any timescale this notebook tests, day-level or exposure-level.

**Not reframed after the fact**: this notebook was built expecting to find
the opposite of what it found, both initially (anchoring strengthening
with learning) and after the trial-type/speed-profile/stop-position
extensions were added specifically to check whether the null was hiding in
a pooled average or a hit-rate-specific blind spot. It wasn't. The real
effects that did appear (the within-day decline, and its fading across
days) weren't hypothesised going in either, and are reported as a probable
state/satiety/task-experience effect, not spun into evidence for the
original context-learning hypothesis. Put together, this is a genuine
boundary condition on the paper's premise (Figure 1: cognitive maps
require a task-anchored state to stay useful) — the neural state and
reward-location competence don't have to rise together on a multi-day
timescale — while also showing anchoring and behaviour *aren't* fully
independent: they co-vary within a session, and that within-session
coupling itself is learned, just not the thing originally hypothesised.

**Part E, new and separate from the anchoring-vs-behaviour question above**:
does the population's *spatial code itself* generalize across contexts?
A Bayesian (Poisson) position decoder trained only on `rz1` trials,
tested both within-context (held-out `rz1`) and cross-context (`rz2`).

**A real bug, caught before it became a false claim**: the track is
circular (the VR corridor teleports from 230cm back to 0cm), so position
0 and 230 are the same seam, not opposite ends of a line. The first
version of this analysis used plain linear `|decoded - true|` error,
which doesn't know that, and it massively inflated errors for any window
whose true or decoded position fell near either edge (e.g. true=5cm
scored against decoded=225cm as a ~220cm error instead of the correct
~10cm). That bug produced a strikingly clean-looking "compressive" bias —
decoded position over-shooting early on the track and under-shooting
late, crossing zero near `rz2`'s reward zone — that mostly *disappeared*
once circular (wraparound-aware) distance was used instead. That result
was written up once before the bug was caught; it is not being carried
forward.

**Now run across the full session batch, not just one example** — the
single most direct way to check whether the single-session result
generalizes, and it does, strongly. Across 30/31 sessions: median
within-context error 9.1cm, cross-context 28.1cm, shuffle null 57.6cm
(median fold-change 2.4x); cross-context error is worse than
within-context in **29/30 sessions** (Wilcoxon paired, p=3.5e-8). This is
one of the most consistent effects in the whole notebook — real,
above-chance transfer, but a genuine and highly reliable degradation
crossing context. The illustrative single-session detail (position-binned
error magnitude and signed bias) now uses whichever session the batch
itself identifies as best (lowest within-context error among sessions
with >200 cells and a >3x cross-context fold-change, computed
programmatically, not hand-picked) -- currently `M29` D29 (n_cells=288,
within-context error 5.1cm, ~4.2x fold-change to cross-context) -- rather
than the notebook's usual running-example session (`M22` D42, which turns
out to be one of the weaker sessions for this specific analysis, 26th of
30 by decoding quality). Picking dynamically from the `best` variable
computed in the batch cell, not a hardcoded mouse/day, means this panel
can't silently drift out of sync with what the batch actually found. In that session: signed error
by true-position is modest and only mildly structured (mostly within
about ±25cm) — real degradation, but no defensible claim yet about a
*specific* bias pattern, since that finer-grained analysis is still only
single-session, not batch-validated the way the headline error-magnitude
result now is.

**Status**: fully built, executed end-to-end (~16 min for the full batch,
67 cells total, including the pooled speed-profile PCA, per-trial
stop-position detection, per-mouse × condition block-level passes, a
full per-mouse all-days anchoring raster with a paired context test, two
zone-specific trial-execution-quality metrics, an exposure-indexed
learning-curve analysis split by context and trial type, an opening
task schematic drawn from the real environment structure — black start/
teleport boxes, `rz1` black walls with a white 20cm-spaced spot pattern
vs. `rz2` white walls with the same pattern in black. That wall pattern
is the context landmark and is visible on every trial regardless of type
(an earlier version of the schematic incorrectly toggled it with trial
type — corrected directly from the experimenter's description); what
actually differs by type is whether the reward zone itself is *signaled*,
shown as a bright marker on cued trials and a faint dashed outline on
uncued ones — and the new Part E cross-context position decoder, now run
across the full 30-session batch with one illustrative example (`M26`
D22) chosen from those results.
Feeds `FIGURES.md` Figure 7 (now 14 panels, A–N) for Parts A-D; Part E is
new and **not yet placed in the figure flow**, though the batch result is
now strong enough to be a serious candidate — see `FIGURES.md` for the
reasoning.
Not yet done: region-stratified anchoring (labels cached but unused);
distinguishing satiety (reward-count-driven) from generic time-on-task
fatigue as the cause of the within-day decline — the current trials table
doesn't obviously carry the reward-volume data needed to separate these;
distinguishing *why* Panel L's zero/one-shot pattern happens — pure
generalization from `rz1`'s already-mastered task structure (same track,
same rules, just a different position) vs. genuinely fast associative
learning that just happens to saturate within one exposure block — which
would need a session with no prior `rz1` experience at all to tell apart,
and this dataset doesn't have one; and running the Part E decoder across
the full session batch (not just one example) plus checking whether
cross-context decoding accuracy itself improves across `rz2` exposures —
a direct neural analogue of Part D's behavioural one-shot/zero-shot
result.

---

## Cross-cutting open questions

See `FIGURES.md` for the full paper structure and the concrete,
prioritized next steps this list feeds into — kept here at the
per-notebook-finding level; `FIGURES.md` is the up-to-date plan of record
for how these get resolved and slot into the 7-figure flow.

- Does the eye-tracking effect (notebook 3) replicate across sessions, and
  does its strength correlate with the population PC1 anchoring strength
  from notebook 2's all-sessions batch? (Notebook 5 shows PC1 strength
  itself relates to cross-region agreement — a natural next step is
  relating it to the eye-tracking effect size too, session by session.)
  → `FIGURES.md` Figure 5E.
- ~~Do the MEC+PARA / MEC+HPF multi-region sessions show different
  task-anchoring dynamics or cross-region coupling than single-region
  sessions?~~ Answered by notebook 5: yes, and cerebellum (a region with no
  obvious task relevance) breaks the pattern as expected.
- Is the MEC-VIS agreement (notebook 5) actually mediated by shared
  engagement/arousal rather than a direct visual-landmark pathway? Now
  concretely scoped as two tests rather than one open question: does
  cross-region *functional coupling* (not just label agreement) strengthen
  in the population-anchored state (`FIGURES.md` Figure 4 Part B), and does
  hippocampal `engagement_cluster` — an independent, licking-derived
  measure with no pupil/eye-tracking equivalent in that dataset — also
  predict anchoring (`FIGURES.md` Figure 5C)? See `FIGURES.md`'s
  "Hypothetical mechanisms of anchoring" section for the full hypothesis
  space (sensory correction / global arousal / attractor dynamics / task
  engagement) these two tests are designed to distinguish between.
- ~~Is subsetting sessions by PC1 explained variance statistically
  justified, or could "high vs. low PC1" sessions be indistinguishable from
  chance?~~ Answered by notebook 6: most sessions (68/75) are statistically
  real, but 7 are not distinguishable from independent per-cell noise and
  should be excluded from population-level pooling.
- Notebook 5's cross-region agreement analysis was run on all 75 sessions
  without filtering by notebook 6's validity test — does restricting to
  `valid_p05` sessions change the pooled kappa estimates, especially for
  the low-n pairs (CERE-MEC, HPF-MEC)? → `FIGURES.md` Figure 4 Part A.
- ~~Does the classifier recover real (not simulated) ground truth, in a
  different region and modality?~~ Answered by notebook 7: yes, robustly —
  6/6 testable sessions positive and significant, r up to 0.78.
- Do hippocampal sessions (notebook 7) clear notebook 6's permutation
  significance bar at a similar rate to MEC's 68/75, given hippocampus's
  consistently lower raw PC1 explained variance? → `FIGURES.md` Figure 6D.
- ~~Does hippocampal pupil/arousal data exist for any Pettit2022 sessions,
  to test whether the same shared-arousal account proposed for MEC-VIS
  has an analog here?~~ No — Pettit2022 has no eye-tracking channels. But
  it has something arguably better for this specific question: an
  independent, non-physiological engagement measure (`engagement_cluster`,
  licking-derived), which sidesteps whether pupil/licking measure the same
  thing and instead tests convergence across genuinely different
  operationalizations of "engagement."
- Notebook 1's validated ground-truth framework has only been used to
  validate notebook 2's classifier on net7; it hasn't yet been used to
  stress-test the classifier's robustness in other simulated conditions
  (e.g. different grid spacings, noise levels, or population sizes).
- ~~Notebook 8's group-level null (population anchoring doesn't correlate
  with learning day) doesn't rule out a real effect in the individual mice
  that do show behavioural change — does that mouse's own PC1 trajectory
  track its own learning curve?~~ Checked: M26 (the one mouse with
  significant within-mouse rz2 improvement, p=0.037) also has the most
  positive within-mouse anchoring-vs-day trend (r=0.60), though
  underpowered to call alone (p=0.28, n=5 days) — suggestive, not
  resolved. → `FIGURES.md` Figure 7 Panel I.
- Notebook 8's within-day block-level decline (hit rate and anchoring both
  fall across ~20-trial blocks, sharpest in the hardest condition) reads
  as satiety/fatigue rather than learning — and itself fades to
  non-significance by days 4-5, which is consistent with a task-experience
  account but doesn't distinguish satiety (cumulative reward count) from
  generic fatigue (time-on-task/trial count). Needs reward-volume data
  this dataset's trials table doesn't obviously expose yet. → `FIGURES.md`
  Figure 7 Panel G.
