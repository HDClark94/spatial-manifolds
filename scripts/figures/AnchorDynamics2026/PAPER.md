# Spatial coding across the medial–lateral axis of the MEC

*Working draft — assembled 2026-09-08. Prose sections are the author's text preserved verbatim. Editorial notes are marked `> **NOTE:**` / `> **TODO:**` and are not part of the manuscript.*

> **The manuscript now lives in [`MANUSCRIPT.md`](MANUSCRIPT.md)** (written 2026-10-03), which is this paper's Summary, Introduction, Results and Discussion rewritten in the flow and style of Clark & Nolan (2024, *eLife*) and Tennant et al. (2022, *Current Biology*): declarative section headings, alternatives stated before the data that test them, explicit validation paragraphs, "Together, these data indicate…" closers, statistics kept light in the prose, and no bold, tables or editorial boxes in the manuscript text.
>
> **This document remains the working draft and the record.** It keeps the point-by-point argument, the figure captions, the Methods, the consolidated open items, and — most importantly — the corrections, caveats and warning blocks explaining *why* each analysis is specified the way it is. Those are deliberately not carried into the manuscript, and should not be deleted from here: several of them record errors that were made and fixed, and the reasoning that prevents them recurring. When a number changes, change it in both.

---

## Title options

The argument is no longer about the mediolateral axis; it is about a global
engagement-linked population state read out by particular cell types. Titles are
grouped by what they commit to. **The options below still say "arousal"; see the
2026-10-03 note at the foot of this section for why that word should not appear
in the title, and for the reworded versions.**

**State the main result (preferred).**

1. **An arousal-linked population state gates spatial anchoring in entorhinal grid cells and interneurons**
2. **Grid cells and local interneurons track a global arousal state that their own connectivity does not carry**
3. **Arousal gates the anchoring of entorhinal spatial coding to a task reference frame**
4. **Entorhinal spatial anchoring is set by a brain-wide arousal state, not by local circuit coupling**

**Lead with the identity result, hedge on mechanism.**

5. Grid cells and putative interneurons, but not spatial coding in general, follow a population anchoring state
6. Cell identity, not synaptic partnership, determines which entorhinal neurons share an anchoring state

**Lead with the behavioural consequence.**

7. A population anchoring state in entorhinal cortex predicts success on a path-integration task
8. When entorhinal spatial coding anchors to the task, mice navigate better — and the pupil says why

> **NOTE on choosing, revised 2026-10-02.** (1), (2) and (4) are the ones that
> state the paper's distinctive claim: the dissociation between cell identity and
> wiring, and the global state that follows from it. (2) is the boldest and most
> falsifiable — it foregrounds the null in Figure 4, which is what makes an
> external-state account necessary rather than merely compatible.
>
> **(4) is now better supported than when these were written, not worse.** The
> earlier note cautioned against "brain-wide" because the cerebellar control did
> not sit at zero. That reading has reversed: asked without privileging the
> entorhinal axis, every structure recorded has a coherent anchoring state of its
> own and, corrected for reliability, they are the same state (Figure 6). The
> cerebellum not sitting at zero was never a failed control — it was the result.
> "Global" and "brain-wide" are now claims the data support; what the data do
> *not* support is that the state is parahippocampal or entorhinal.
>
> A title in this family is therefore worth adding:
>
> 9. **A global arousal state, read out by entorhinal grid cells and local inhibition rather than built by them**
>
> Avoid any title that leads with the mediolateral axis. That result survives
> (Figure 3) but it is a supporting observation about where the responsive cells
> sit, not the organising claim. Avoid equally any title implying the state is
> entorhinal, parahippocampal, or specific to MEC: Figure 6 excludes it.
>
> **"Arousal" is the wrong word in a title, revised 2026-10-03.** Options 1–4 and
> 9 all use it, and it points the reader the wrong way. Pupil radius is our
> measurement and it *is* the standard arousal index, but the anchored state is
> the **constricted-pupil** state and also the state in which the animal performs
> — hit rate improves monotonically as the pupil narrows, with no inverted U in
> the observed range (§5). A title saying "arousal state" will be read as
> "the aroused state is the anchored one", which is the opposite of the data, or
> as "anchoring happens when the animal is drowsy", which the behaviour excludes.
> Prefer **attentional engagement**, **task engagement** or **focus**, and let
> the pupil appear as the measure rather than as the construct:
>
> 10. **A global engagement state, read out by entorhinal grid cells and local inhibition rather than built by them** *(option 9, reworded)*
> 11. **Attentional engagement gates the anchoring of entorhinal spatial coding to a task reference frame** *(option 3, reworded)*

---

## The argument, point by point

The figures make one argument in seven steps. Each step is stated as the claim
it supports, with the figure that carries it and the main caveat attached.

**1 — There is a population anchoring state, and it matters for behaviour.** *(Figure 1)*
On a virtual linear-track memory task, a cell's firing is anchored to the task
reference frame on some trials and not others. The labels are correlated across
simultaneously recorded cells: PC1 of the cell × trial label matrix beats a
circular-shift null by roughly twofold, and blocks of tens of trials share a
state. The state predicts performance — the hit rate is higher on anchored
trials, and markedly so on uncued trials, where the animal must path-integrate
(ADI = +0.094, *p* = 0.012).

**2 — The state is entered by particular cell types, not by spatial coding in general.** *(Figure 2)*
Chance here is near +0.15, not zero, because a cell anchored on most trials
correlates with a population that is also anchored on most trials. Against each
cell's own circular-shift null, only grid cells (+0.119) and putative
interneurons (+0.080) exceed chance; non-grid spatial cells sit exactly on their
null and non-spatial cells below it. Grid cells and interneurons also agree with
*each other* (+0.109) at nearly their within-group level, so they form a
coordinated subnetwork rather than two populations independently tracking one
signal. Cell identity is taken from the **open field**, never from the session
being analysed.

**3 — That subnetwork has an anatomical address: medial, and superficial.** *(Figure 3)*
Mediolateral position predicts both what a cell is (grid enrichment, ρ = −0.078,
*p* = 0.025) and whether it follows the state (ρ = −0.109, *p* = 0.034), tested
within session on the 27 sessions whose probes span at least two shank pitches.
Layer predicts identity (superficial 8.9% grid against deep 3.9%, *p* = 0.002).
Dorsoventral position, anterior-posterior position and probe depth predict
neither. *Caveat:* because mediolateral position predicts identity and identity
predicts following, the two accounts of the following gradient are not separable
at this sample size (non-grid cells alone: same effect size, *p* = 0.111).

**4 — The subnetwork is real in the wiring — but the wiring does not carry the state.** *(Figure 4)*
Putative monosynaptic connections confirm the grid–interneuron pairing in
millisecond spike timing: `interneuron → grid` is the strongest pathway in the
dataset (3.93% of testable pairs, 5.1× over identity shuffling, *p* = 0.0005).
Detecting it required extending the standard criteria, which find a causal peak
and therefore excitation only, onto the lower Poisson tail. **Yet connected
pairs share the anchoring state no more than unconnected pairs matched for
session, identity and probe distance** (grid–interneuron *p* = 0.46;
interneuron–interneuron *p* = 0.83), and both nulls resolve effects a third the
size of the identity effect. Two grid cells agree whether or not a synapse is
detectable between them.

**5 — So the state is imposed from outside the local circuit, and it is attentional engagement.** *(Figure 5)*
Step 4 is the pivot: a state that travels with cell identity but not with
synaptic partnership is not propagated cell-to-cell, which points at a common
input. Pupil radius — the standard non-invasive index of the arousal/engagement
axis — tracks the state directly, falling on entry to the anchored state and
rising on exit (−0.41 and +0.49 z, *p* < 10⁻⁶), and performance improves
monotonically as the pupil narrows (hit rate 0.83 → 0.39 across pupil deciles),
so the constricted-pupil anchored state is the focused one, not a drowsy one.

**6 — The state is global; the specificity is cellular, not anatomical.** *(Figure 6)*
Asked without making entorhinal cortex the frame — each region building its own
axis from its own cells — every structure recorded has a coherent anchoring state
of its own, visual cortex included (split-half +0.330, *p* = 2.4 × 10⁻⁵). Corrected
for how reliably each region estimates it, those states are indistinguishable:
MEC–subicular +0.920, MEC–visual +0.863, neither different from 1. At matched cell
count MEC's state is only marginally more coherent than visual cortex's
(*p* = 0.065). What does differ is the **cell type**: grid cells exceed visual
cortex by +0.149 per cell (*p* = 9.9 × 10⁻⁶) and interneurons by +0.144, against
+0.050 for other entorhinal cells — three times the regional contrast. One state,
everywhere, read out most strongly by grid cells and local inhibition. *Caveats:*
disattenuation amplifies noise, so "the same state" excludes differences above
~0.3 rather than demonstrating identity; and the behavioural effect is not
region-specific at all — a visual-cortex state carries a larger Anchoring
Dependence Index (+0.148) than the entorhinal one (+0.085).

**7 — The field potential changes with the state in a frequency-specific way, and
theta decouples from running speed.** *(Figure 7)*
The spectrum is restructured rather than rescaled. Across the 26 sessions that
switch often enough to test, theta (−0.094, *p* = 0.020) and mid gamma (−0.108,
*p* = 0.006) fall in the anchored state while slow gamma **rises** (+0.105,
*p* = 0.0013); because the difference reverses sign across the gamma range —
crossing zero near 67 Hz in 25 of 26 sessions — no uniform change in amplitude,
referencing or spiking contribution can produce it, which is what retires the
earlier "the decrease is broadband" worry. Theta's nesting of gamma changes for
one band only: theta–fast-gamma coupling falls 15.9% (*p* = 5 × 10⁻⁶, 23 of 26)
while theta–slow-gamma coupling does not (*p* = 0.71). Separately, all four
speed–theta coupling measures fall when the population is anchored — the
amplitude slope from 0.374 to 0.218 (*p* = 0.045) and the frequency slope from
0.0126 to 0.0070 Hz per cm/s (*p* = 0.012) — and the gain changes rather than
only the level, so this is not a state in which the animal merely runs
differently. *Caveats:* slow and fast gamma are two separate band
effects pointing in opposite directions, not a coupled shift — their magnitudes
are uncorrelated across sessions (ρ = −0.08) and the quadrant count is what the
marginals already imply (Fisher *p* = 0.59); the accompanying theta amplitude fall is
marginal (−1.66%, *p* = 0.032); the hypothesis that gamma cycles are countable
computational slots was tested and **not** supported, event counts tracking power
and running rather than cue condition; and the regional and laminar tests of the
speed decoupling are still pre-gate and must be re-run before they are quoted.

**The through-line.** A global, engagement-linked state is read out by grid cells
together with local inhibition, at a specific mediolateral and laminar address,
rather than by spatial tuning as such or by entorhinal cortex as a structure.
Steps 4–6 are what turn this from a description into a claim about mechanism:
the state is not built by the local circuit that expresses it (4), it tracks a
global behavioural variable (5), and it is present in every structure recorded
with only the *readout* differing by cell type (6). What the entorhinal circuit
contributes is not the state but a cell type tuned to it.

## Abstract

*(v3 — written against the current Figures 1–3 and the eye-tracking result. v2 below is the earlier mediolateral-organisation framing and is kept for comparison; the two make different claims and only one can survive.)*

Spatial firing in the medial entorhinal cortex (MEC) is not fixed to the external world. On a virtual linear-track memory task the same cell's firing is anchored to the task reference frame on some trials and not on others, and we asked what governs that switching, using 4-shank Neuropixel recordings from 8 mice (61 sessions, 6,455 MEC cells) with open-field recordings on the same days so that cell identity is assigned independently of the task.

Anchoring is a property of the population rather than of single cells. Trial-by-trial labels are correlated across simultaneously recorded cells, and the first principal component of the cell × trial label matrix exceeds a circular-shift null preserving each cell's anchored fraction by roughly twofold. Blocks of tens of trials share a state, and the population crosses between states over about three trials.

The state is not entered by spatial coding in general, but by particular cell types. Because a cell anchored on most trials correlates with a population that is also anchored on most trials, chance lies near +0.15 rather than zero; measured against each cell's own null, only grid cells (+0.119, *p* = 0.002) and putative interneurons (+0.080, *p* = 0.0003) exceed it, while non-grid spatial cells sit exactly on their null (−0.000) and non-spatial cells below it (−0.024). Grid cells and interneurons agree with each other (+0.109) at nearly their within-group level, forming a coordinated subnetwork that is also visible in spike timing, where interneuron→grid is the strongest monosynaptic pathway. The coordination is nonetheless not synaptic: connected pairs share the state no more than unconnected pairs matched for identity and distance.

Transitions coincide with a change in attentional engagement, pupil radius falling on entry to the anchored state and rising on exit (−0.41 and +0.49 z, *p* < 10⁻⁶). Performance improves monotonically as the pupil narrows (hit rate 0.83 to 0.39 across pupil deciles, *p* < 10⁻²⁷⁰, unchanged by controlling running speed), so the constricted-pupil anchored state is a focused one rather than a drowsy one.

The subnetwork has an anatomical address. Tested within session on the 27 sessions whose probes span at least two shank pitches, mediolateral position predicts both what a cell is (grid enrichment, ρ = −0.078, *p* = 0.025) and whether it follows the state (ρ = −0.109, *p* = 0.034), and layer predicts identity (8.9% of superficial spatial cells against 3.9% of deep, *p* = 0.002), while dorsoventral position, anterior-posterior position and probe depth predict neither. The state itself is global rather than entorhinal. Every structure recorded builds a coherent anchoring axis from its own cells — visual cortex included (split-half +0.330, *p* = 2.4 × 10⁻⁵) — and corrected for how reliably each estimates it, those axes are indistinguishable from identical (MEC–subicular +0.920, MEC–visual +0.863). What is specific is the cell type rather than the structure: grid cells express the state +0.149 more strongly than visual cortex per cell (*p* = 9.9 × 10⁻⁶) and interneurons +0.144, against +0.050 for other entorhinal cells, while the region-level difference does not reach significance (*p* = 0.065).

The field potential changes with the state in a frequency-specific way rather than as a gain change: across the 26 sessions that switch often enough to test, theta (−0.094, *p* = 0.020) and mid gamma (−0.108, *p* = 0.006) fall in the anchored state while slow gamma **rises** (+0.105, *p* = 0.001), the difference crossing zero at a median of 67 Hz in 25 of 26 sessions, and theta–fast-gamma phase–amplitude coupling falls 16% (*p* = 5 × 10⁻⁶) while theta–slow-gamma coupling does not change. Theta also decouples from running speed when anchored, with its speed gain roughly halved, while the cellular speed code and spike–theta phase locking are unchanged.

These results describe a global, engagement-linked state that the entorhinal circuit reads rather than builds — expressed by grid cells together with local inhibition, at a specific mediolateral and laminar address, rather than by spatial tuning as such or by entorhinal cortex as a region.

> **CORRECTED 2026-10-01, amended 2026-10-02.** This paragraph previously stated that following "shows no mediolateral or dorsoventral gradient once computed within session (*p* = 1.00)" and that subicular (+0.265) and visual (+0.141) cells follow the MEC axis at least as strongly as MEC cells do. The first came from the `nunique > 3` session filter that excluded the multi-shank sessions (see the CORRECTION in Results §3); the mediolateral gradient is real. The second was then "corrected" to *visual cortex sits below its own null*, which was itself wrong — see the correction in Results §6. Visual cortex **does** follow the state; entorhinal cells simply follow it about twice as strongly, which is a graded claim and is tested as such.
>
> **RESOLVED (2026-10-02) — the framing is cellular, not anatomical.** Asking the question without privileging the entorhinal axis settles it: every structure has its own coherent state, those states are indistinguishable once corrected for reliability, and the region-level difference between MEC and visual cortex does not reach significance (*p* = 0.065) while the grid-cell contrast does by four orders of magnitude (*p* = 9.9 × 10⁻⁶). The paper should not claim that the state is parahippocampal, nor that the behavioural effect is entorhinal — a visual-cortex state carries a larger ADI (+0.148) than the entorhinal one (+0.085). The defensible claim is a global state with a cell-type-specific readout, which is also what Figures 2–5 describe.

---

*(v2)*

The medial entorhinal cortex (MEC) contributes to navigation and cognitive map formation through activity of neurons with spatial firing fields. Grid cells are the most studied spatial neurons in the MEC, while other spatial cells are assumed to support the grid code. However, the anatomical and functional relationships between these grid and non-grid spatial cells are largely unclear. We address this using high density multi-shank recordings from neurons in the MEC of mice during freely moving and head fixed behaviours. We find that grid cells are enriched in medial MEC, whereas non-grid spatial cells are more evenly distributed. While nearby non-grid spatial cells have activity coordinated with the grid network, the activity of distal non-grid spatial cells can be dissociated, especially when anchoring of grid activity to task reference frames is disrupted. Our results suggest a revised model for functional organisation of the MEC in which it generates multiple spatial representations in parallel, with the grid network and supporting cells enriched medially, and networks of non-grid spatial cells predominant more laterally.

> **NOTE:** The abstract makes no claim about future-location coding, consistent with that analysis having been dropped.

---

## Aim

Quantify differential spatial computation across the medial–lateral axis of the MEC.

---

## Introduction

*(~1000 words in last publication)*

The grid cell system located in the medial entorhinal cortex (MEC) provides a neural substrate for spatial coding and path integration ^1,2^, and forms a key component of cognitive map theories for cognition ^3–7^. However, grid cells are a minority among spatial cells in the MEC, with other spatial cells having border or object-related firing ^8–10^, or stable spatial firing without a simple field-like structure ^8,11^. During learned navigation behaviours the spatial firing of MEC neurons is further diversified, with many neurons generating ramping representations that track spatial or task variables ^12–14^. It's unclear whether this representational diversity reflects distinct elements of a single integrated network or functionally distinct neural pathways that can operate in parallel (Figure 1a–b).

Under models in which neurons in the MEC form a single fully integrated network, non-grid cells could either be part of a continuous attractor that supports grid dynamics ^2,15,16^, or they could provide spatial anchoring signals that prevent drift in the relationship between attractor activity and an animal's spatial location ^17,18^. This view is supported by evidence that grid and non-grid spatial cells have coherent dynamics when recorded from animals exploring open arenas, even when the population representation as a whole spontaneously remaps ^19^. Nevertheless, it's unclear whether coherent population activity extends to other task conditions. For example, on linear tracks the activity of grid cells can become uncoupled from the environment, appearing to drift in a fashion that is distinct from spatial remapping ^12,20^, but whether population coordination of grid and non-grid spatial cells is maintained under these conditions is unclear. A key issue is that the apparent coherence between MEC neurons may simply reflect shared spatial or other sensory input rather than an integrated population-level computation.

Knowing whether grid and non-grid spatial cells have a common anatomical organisation within the MEC may also help elucidate the nature of their underlying neural circuits. Anatomical-functional relationships have been demonstrated for the dorsoventral axis of the MEC, with grid cells at more ventral locations having larger inter-field spacing ^21^, and head direction cells becoming less well tuned at more ventral locations ^22^. These differences correlate with dorsoventral organisation of neural intrinsic properties and connectivity ^23–25^, suggesting structure-function relationships that may be critical for computations to emerge ^26^. Along the mediolateral axis, while firing properties of co-modular grid cells are similar across distances of several millimetres ^27^, imaging of grid cell activity suggests they may be enriched in the more medial MEC towards the border with the parasubiculum ^28^. Neural intrinsic properties appear more uniform along the medio-lateral axis ^29,30^, although anatomically medial and lateral parts of the MEC can be distinguished through differences in the width of layers 1 and 2. Whether this organisation is reflective of different functional properties of grid and non-grid spatial cell types is unclear.

We set out to distinguish hypotheses under which grid and non-grid cells are anatomically co-located and functionally coordinated, from hypotheses under which they are anatomically separated or their functional coordination breaks down (Figure 1b). To address anatomical organisation, we targeted the MEC using high-density multi-shank Neuropixel 2.0 probes to record from many grid and non-grid cells simultaneously (Figure 1c). To investigate functional coordination we recorded their activity while mice engaged in open field exploration (Figure 1d) as well as a head-fixed path integration task (Figure 1e), with the intention that different behavioural conditions may reveal dissociations between sub-networks within the MEC ^12,31,32^.

We find that grid cells are enriched in medial parts of the MEC, whereas non-grid spatial cells are similarly distributed across the medial-lateral axis. Grid and non-grid spatial cells could nevertheless not be dissociated by their spatial coding properties alone, which are shared across the mediolateral extent of the MEC. Instead, the two populations could be dissociated by the extent to which their activity predicts activity of other neurons in the MEC. First, when predicting single neuron activity using position and the activity of co-recorded cells as co-variates, co-recorded grid cells improved predictions of the activity of grid compared with non-grid spatial cells. Second, medially located non-grid spatial cells substantially improved predictions of the activity of medial grid cells, whereas laterally located non-grid spatial cells did not. Third, in behavioural conditions in which grid cell activity becomes independent from a task reference frame, prediction of grid cell activity from the activity of non-grid spatial cells at the same mediolateral position is maintained, whereas predictions using activity of non-grid spatial cells at more distal locations is reduced.

Our results demonstrate that the MEC does not function as a single coherent network. Instead they support a model in which medial and lateral parts of the MEC constitute distinct functional domains. Activity in these domains is often correlated because both encode representations of space, but can be dissociated under behavioural conditions that cause grid representations to lose their anchoring to task reference frames.

> **TODO — Introduction:**
> - Paragraph 5 has been rewritten to remove the future-location finding; the populations are now dissociated by *coordination* rather than by *coding properties* from the outset. The replacement sentence ("could nevertheless not be dissociated by their spatial coding properties alone") is a placeholder claim — decide what, if anything, now supports it, or cut it and go straight to the coordination result.
> - Numeric citations ^1–41^ are placeholders inherited from the source document — no bibliography exists yet. Needs a reference list before this is a manuscript. Note ^33^ (the predictive-grid-cell reference) is now uncited and can come out of the list.
> - Word count of current text: ~900. On target.

---

## Results

*(~3000 words in last publication)*

### Claim skeleton

> Bullet summary of what each Results subsection asserts, its figure, and its current evidential status.

| § | Claim | Figure | Argument step | Status |
|---|---|---|---|---|
| 1 | Recordings span the ML extent of MEC in OF + VR; a population anchoring state exists, beats a circular-shift null ~2×, and predicts performance on uncued trials (ADI = +0.094, *p* = 0.012) | Fig 1 | 1 | Built; **cell counts pending** classification v2 |
| 2 | Against each cell's own null, only grid cells (+0.119) and putative interneurons (+0.080) follow the state; they also follow each other (+0.109), so they form a coordinated subnetwork | Fig 2 | 2 | Built |
| 3 | That subnetwork is medial, within the superficial grid population: ML predicts identity (ρ = −0.078, *p* = 0.025) and following (ρ = −0.109, *p* = 0.034) within session on 27 multi-shank sessions; DV, AP and depth predict neither; layer predicts identity (*p* = 0.002) but not following within session (ρ = +0.028, *p* = 0.154) | Fig 3 | 3 | Built; identity-vs-following not separable at this *n* |
| 4 | Monosynaptic connections confirm the pairing (`int → grid` strongest, 3.93%, 5.1×, *p* = 0.0005) — **but connected pairs share the state no more than matched unconnected pairs**, so the wiring does not carry it | Fig 4 | 4 | Built; informative nulls except grid–grid (underpowered) |
| 5 | Pupil radius tracks the state, falling on entry to anchored (−0.41 z) and rising on exit (+0.49 z), *p* < 10⁻⁶; and performance falls monotonically with pupil size across the whole range (0.826 → 0.391 by decile), so the constricted-pupil anchored state is the **focused, better-performing** one rather than a drowsy one | Fig 5 | 5 | Built |
| 6 | Each region has its own coherent anchoring state and, corrected for reliability, they are the same state; the specificity is **cellular** (grid/interneuron, +0.149 over VIS) not anatomical (region-level *p* = 0.065) | Fig 6 | 6 | Rebuilt 2026-10-02 around region-own axes; behavioural effect **not** region-specific |
| 7 | The LFP change is spectrally structured, not a gain change: theta −0.094 and mid gamma −0.108 fall while slow gamma **rises** +0.105; theta–fast-gamma coupling falls 16% and theta–slow-gamma does not; and theta decouples from running speed | Fig 7 | 7 | Rebuilt 2026-10-03 post-gate (26 switching sessions); speed-decoupling paragraphs post-gate, the mechanistic controls below them still pre-gate |

> **Dropped from Results.** The xgBoost coupling sections (co-recorded cells improving prediction; medial vs lateral non-grid cells predicting grid cells; proximal vs distal non-grid cells under task-independent firing) and the aspirational mediolateral context-decoding section are no longer part of the argument and have been removed, along with their legacy figure captions. The analyses and figures still exist on disk (`fig5_cross_region_coupling.pdf`, `fig6_*`, `mcvr_*`); nothing has been deleted, but none of it is cited by the current seven-step argument. If any of it returns, it returns as a new step with its own claim.

---

### 1. Recordings

We recorded from 4-shank Neuropixels 2.0 probes, targeted to maximally sample the medial-lateral extent of the MEC, as mice explored an open arena and then while they carried out a location-memory task in a virtual track environment (Figure 1c). The task was configured so that mice learned to stop in a reward zone that on some trials was indicated by a visual cue and on other trials could only be efficiently localised using a path integration strategy ^31,32^. We analysed data from mice that had 26.6 ± 6.3 days of training (range 19 to 41 days) (Figure 1d). Neurons with grid fields (n **?** / **?** neurons) in the open arena typically had repeating localised firing on the linear track, whereas neurons that in the arena had non-grid spatial firing patterns (n = **?**/**?** neurons) often had ramping firing patterns on the track (Figure 1e).

**Firing is anchored to the task reference frame on some trials and not others, and the switching is a population property.** Each cell's trial-by-trial rate map was classified as anchored or non-anchored against the cell's own circularly shifted null (Methods, *Classification of anchored and non-anchored trials*). The resulting labels are not independent across cells: the first principal component of the cell × trial label matrix exceeds a circular-shift null that preserves each cell's anchored fraction by roughly twofold, blocks of tens of trials share a state, and the population crosses between states over about three trials (Figure 1f, g). We refer to the per-trial fraction of anchored cells as the **population anchoring state**.

**The state predicts behaviour.** Mice were more likely to stop in the reward zone on anchored trials, and the advantage was markedly larger on uncued trials, where the zone can only be localised by path integration, than on cued trials, where a visual landmark is available. Quantified per session as the difference between the uncued and cued anchoring benefit (the Anchoring Dependence Index; Methods), the effect is positive (ADI = **+0.094**, *p* = 0.012). Anchoring therefore matters most when the animal has to rely on an internal estimate of position — which is what makes the state worth explaining.

> **TODO:** Both `n` values are blocked on the classification v2 sweep (running; 100 shuffles/cell, good+mua units, ≥500 spikes, GC = grid ∧ SI significant, NGS = SI significant). Report per-session OF1 and OF2 separately as well as the union/intersection.

---

### 2. The population state is followed by grid cells and interneurons, not by spatial coding generally

The population anchoring state is an average over cells, so it does not by itself say which cells enter it. **Cell identity is taken entirely from the open field**, never from the virtual-reality session being analysed: grid, non-grid spatial and non-spatial classes come from open-field rate maps, and the speed score used below from open-field running. This is deliberate. Firing on the linear track changes with engagement, motivation and task demand — which is the phenomenon under study — so classifying cells on the same session would let those dynamics define the categories and then rediscover them. Identity assigned in a different environment, on a different day's behaviour, cannot be shaped by the anchoring state it is used to explain. Across 6,455 medial entorhinal cells (61 sessions, 8 mice) each cell's label sequence was correlated with the population axis, recomputed leave-one-out, and compared against its own circular-shift null. **The null is not near zero** — a cell anchored on most trials correlates with a population that is also anchored on most trials — and it is what the comparison must be made against:

| identity | mean *r* | chance | excess | *p* vs chance | mice above chance |
|---|---|---|---|---|---|
| grid | 0.276 | 0.157 | **+0.119** | **0.002** | 6/7 |
| putative interneuron | 0.228 | 0.149 | **+0.080** | **0.0003** | **7/7** |
| non-grid spatial | 0.147 | 0.147 | −0.000 | 0.33 | 5/8 |
| non-spatial | 0.118 | 0.141 | −0.024 | 0.59 | 3/8 |

*p* is a mixed model of the excess against zero with random intercepts for mouse and for session within mouse; cell-level rank tests give values four to eight orders of magnitude smaller and are not reported.

Only grid cells and putative interneurons exceed chance; non-grid spatial cells sit exactly on their null and non-spatial cells below it. Grid cells and interneurons are statistically indistinguishable from one another (mixed model with mouse and session random effects, *p* = 0.84) and both exceed the two other classes (*p* ≤ 8 × 10⁻⁷). The state is therefore tracked by grid cells together with local inhibition, rather than by spatial coding in general.

The two measures are reconcilable: 19–24% of non-grid and non-spatial cells individually beat the 5% false-positive rate, so those classes contain genuine followers, but their group average sits at chance — a minority of followers inside a majority at the null.

**Grid cells and interneurons follow each other, not merely the same thing.** That both exceed chance does not establish that they are coordinated: two populations could track a common signal independently. Correlating every pair of cells within session and averaging by identity pair — as excess over an independent circular-shift null, so a pair that is simply anchored most of the time scores nothing for it — gives a 4 × 4 matrix in which grid–interneuron agreement reaches **+0.109** (29 sessions, *p* = 1.4 × 10⁻⁶) against **+0.058** for the other off-diagonal pairs, and is indistinguishable from either within-group value (grid–grid +0.115, interneuron–interneuron +0.115). Grid cells and putative interneurons therefore form a coordinated subnetwork with respect to the anchoring state rather than two populations independently reflecting it. Aligned to a population state transition, all four identities switch together over roughly three trials, and identity does not predict switch timing (*p* = 0.15).

**Cells locked into one state.** Under the absolute gate a cell whose two clusters both fall on the same side of its null is labelled anchored on every trial, or on none. Its correlation with the population axis is then undefined, so it appears as "no agreement" while never having had the opportunity to agree. These are 24.8% of cells in the 35 sessions where the population actually switches, and the split is strongly identity-dependent and runs in opposite directions: putative interneurons lock **on** (109 always-anchored against 2 never, unanimous across mice), non-spatial cells lock **off** (39 against 293), while grid and non-grid spatial cells are balanced. Non-grid spatial cells dominate both locked groups only because they are ~70% of the population; they lock at their base rate.

> **The locked-ON interneurons are the speed cells (added 2026-10-03, Figure 2I).** The asymmetry above — interneurons lock on 108:2, non-spatial cells lock off 39:293 — has a mechanism. A speed-tuned cell repeats its profile on every trial, because speed along the track is stereotyped, so the classifier's trial-against-template comparison returns *anchored* every time and the cell never varies. Entorhinal speed cells are predominantly fast-spiking interneurons, which is the class that locks on. Measured on the **open-field** speed score, so the task cannot have shaped it: interneurons that lock on score **+0.141** against **+0.065** for those that vary (*p* = 0.005, n = 103 and 223), and the same gradient appears in principal cells (+0.042 against +0.029, *p* = 7.6 × 10⁻⁵) and across all MEC cells (51% speed cells among always-anchored against 41% among varying). *This is a statement about what the classifier labels, not a claim that speed cells are uninteresting: it says the locked group is largely the speed-coding group, which is why it should not be read as a separate population with its own anchoring biology.*

**What a non-anchored trial is.** A non-anchored trial could be a field that has moved or a field that has gone. Correlating each trial's map against the cell's anchored template at every circular offset, the correlation at zero offset collapses (+0.33 → +0.07) while the best over offsets barely falls (+0.56 → +0.53) — which reads as displacement and is not. Two **unrelated** cells, template against trial, reach +0.500 under the same best-of-offsets search; non-anchored trials sit 0.029 above that floor, and their mean best offset is 47 cm against the 50 cm expected from a uniform pick on a 200 cm track. Non-anchored trials therefore show loss of spatial correspondence, with no evidence of a coherent field elsewhere.

**Anchoring is about where cells fire, not how much.** Firing rate differences between states are small and their sign depends on the measure: −0.090 in log₂ when rate is averaged over position bins, −0.028 when computed as spikes per running second, and **+0.036** when trials are additionally speed-matched (all *p* < 0.005; *n* = 6,517). A rate read off a rate map divides by occupancy, which running speed sets. A ~3% effect that reverses under a change of denominator and a behavioural control is not a rate effect.

Where these cells sit, and whether their position predicts any of this, is the subject of Section 3.

**Negative results.** Cells that stay anchored while the population does not are not a distinct type (their composition matches the population) and are not the open-field speed-modulated cells (+0.046 against +0.044, *p* = 0.99; per-mouse inconsistent in sign). On switch timing, note that the apparent finding that ~half of cells "lead" the population is definitional rather than empirical: the population state crosses its threshold when half the cells have switched, so roughly half must lead whatever the biology.

---

### 3. The subnetwork has an anatomical address: medial, and superficial

If grid cells and interneurons form a subnetwork, it should have a location. We localised neurons by reconstructing the position of each of the four shanks of every probe and registering recorded cells to a standardised atlas, fitting the best-fit recording plane to **device contacts** rather than to cell locations, since per-cell mediolateral position is a per-shank nominal value plus jitter and fitting to it would fit the approximation rather than the probe geometry (Figure 3B, C).

**Identity is organised mediolaterally.** Grid cells are enriched medially within session (median ρ = −0.078 against grid membership, 18 of 27 multi-shank sessions, *p* = 0.025) and superficially (8.9% of superficial spatial cells against 3.9% of deep, higher in 21 of 31 sessions, *p* = 0.002), while dorsoventral and anterior-posterior position predict identity no better than chance (*p* = 0.90 and 0.10). The gradient is large enough to see without a statistic: open-field rate maps from four M25 sessions whose probes sat at ≈3000, 3270, 3540 and 3720 µm, with grid and non-grid cells drawn in the ratio actually recorded, yield 10 grid cells of 24 at the medial end and 1 of 24 at the lateral end (Figure 3A).

**Following is organised on the same axis, in the same direction.** Pooling cells across sessions produces a large mediolateral gradient — medial 0.205 against lateral 0.120 at the 3400 µm boundary, ρ = −0.15, *p* = 2 × 10⁻²⁵ — but pooling is not a safe test here, because probe placement differs markedly between animals (16.5% medial in one, 63.1% in another), so a cell's mediolateral coordinate carries information about which session it came from. The test must be made **within session**, where every cell shares one probe placement, and on sessions that can answer it: mediolateral position is assigned per shank, so a single-shank session has no mediolateral variation at all and contributes only noise, and 33 of 61 sessions span under 50 µm. Restricted to the **27 sessions spanning at least 200 µm**, i.e. two shank pitches or more, the gradient survives: median ρ = **−0.109**, negative in 19 of 27 sessions, Wilcoxon *p* = **0.034**. Dorsoventral position (*p* = 0.99, 59 sessions) and probe depth (*p* = 0.92) show nothing, and for those two axes the session-selection criterion makes no difference, since every session varies along the shank — which is what indicates the mediolateral result is specific to the axis rather than produced by the restriction.

One multi-shank session shows the gradient on its own. In M25 D25 (213 cells, 4 shanks, 720 µm span) agreement with the population axis falls monotonically across the shanks — 0.31, 0.53, 0.36 and 0.04 from medial to lateral — giving ρ = −0.255, *p* = 0.0016, more than twice the dataset median effect (Figure 3G). The session is anchored on most of its trials, so its raster is largely one colour; it is shown for the mediolateral sampling and the agreement profile beside it, not as an illustration of state alternation.

**The two accounts are not separable at this sample size.** Because mediolateral position predicts identity and identity predicts following, the following gradient could be nothing more than the identity gradient. Restricting to non-grid cells leaves the effect size almost unchanged (median ρ = −0.100) but the significance does not hold at 26 sessions (*p* = 0.111), so we report both and claim neither over the other.

Given the curved shape of the MEC and the flat plane of a 4-shank probe, the mediolateral organisation could in principle be a laminar sampling artifact — different shanks reaching deep rather than superficial layers. There is no bias of recording site for superficial or deep layers across shanks (Extended Figure 2B), and layer and mediolateral position organise identity independently.

> **CORRECTION, recorded deliberately.** An earlier version of this section reported that the mediolateral gradient in following "vanishes within session" (median ρ = −0.003, *p* = 1.00). That analysis selected sessions by the number of distinct mediolateral values, `nunique > 3`, which **excludes** the cleanly sampled 2-, 3- and 4-shank sessions — whose cells take only a few discrete per-shank values despite spanning 180–720 µm — while **including** sessions whose coordinates differ only by sub-micron numerical jitter around a single shank. It therefore ran the test almost entirely on sessions with no mediolateral variation. Selecting on span rather than on distinct-value count reverses the conclusion.

> **TODO:**
> - ~~The 3400 µm medial/lateral boundary is drawn in Figure 3B, C but no criterion for it is stated.~~ **RESOLVED 2026-10-03: the boundary line has been removed from the slice panels.** It was presentational only — every within-session test in the figure runs on the continuous coordinate — so dropping it costs nothing and removes an unjustified threshold from the figure.
> - Re-run distributions on classification v2. Add the OF1-vs-OF2 stability panel (grid score OF1 vs OF2; ML distributions for cells classified as GC in one vs both sessions).

---

### 4. The grid–interneuron pairing is present in the wiring, but the wiring does not carry the state

That grid cells and interneurons agree is a trial-level statistic, and could describe two populations independently tracking one signal. Putative monosynaptic connections give the same 4 × 4 question an independent answer in millisecond spike timing (Figure 4): `interneuron → grid` is the strongest pathway in the dataset (3.93% of testable ordered pairs, 5.1× over within-session identity shuffling, *p* = 0.0005), and `grid → grid` the strongest excitatory one (0.217%, 2.6×, *p* = 0.004). Detecting the inhibitory half required extending the standard criteria, which find a causal peak and therefore excitation only, onto the lower tail of the same Poisson test; the resulting connections are slower than excitatory ones (median 3.0 against 1.0 ms) and originate from waveform-classified interneurons at 3.1× their population share, neither of which the detector can produce by construction. **The coordination is nonetheless not synaptic.** Compared against unconnected pairs from the same session, of the same identity pair and at a similar probe distance (±40 µm), connected pairs share the anchoring state no more strongly — grid–interneuron +0.176 against +0.187 (*n* = 193, *p* = 0.39), interneuron–interneuron +0.090 against +0.090 (*n* = 116, *p* = 0.90). Both comparisons resolve effects of 0.045 and 0.043 at 80% power, a third the size of the identity effect they are testing against, so the nulls are informative; the grid–grid comparison (*n* = 27, minimum detectable effect 0.157) is not, and is reported as underpowered rather than as a null. Anchoring therefore travels with what a cell is, rather than with what it is wired to — as expected of a state imposed on the population by a common input to which grid cells and local inhibition are more sensitive, rather than one propagated cell-to-cell.

---

### 5. The state is a reflection of attentional engagement

Section 4 is the pivot of the paper. A state that travels with what a cell *is*, but not with what it is wired to, is not being propagated cell-to-cell through the local circuit that expresses it. Two grid cells agree at *r* ≈ 0.51 whether or not a synapse is detectable between them, and the grid–interneuron comparison would have caught a wiring effect a third the size of the identity effect. The coordination therefore has to be imposed on the population by something outside it — a common input to which grid cells and local inhibition are simply more sensitive than other cells are.

That reframes the question from *which cells talk to each other* to *what the whole population is listening to*, and makes a global behavioural-state variable the natural candidate. Pupil diameter is the standard non-invasive index of arousal, and it is available on 80 of 81 virtual-reality sessions.

**It tracks the state.** Pupil radius differs between the two states by ≈ 0.24 SD, and changes at transitions in a consistent direction: entering the anchored state it falls by **0.414** z, and entering the non-anchored state it rises by **0.493** z (*p* = 1.7 × 10⁻⁷ and 4.4 × 10⁻⁸; Figure 5). The anchored state is therefore the constricted-pupil state and the non-anchored state the dilated one — and, as the next paragraph shows, the constricted state is the one in which the animal performs.

**The constricted-pupil state is the better-performing one, monotonically.** Pupil size predicts performance across its entire range rather than at an optimum: binning trials by within-session pupil *z* into deciles, the hit rate falls at every step from **0.826** in the smallest decile to **0.391** in the largest (14,011 trials, 58 sessions), and the anchored fraction falls with it. A mixed model gives −0.123 per *z* (*p* = 8 × 10⁻²⁷¹), and adding a quadratic term does not produce an inverted U: the quadratic coefficient is **positive** (+0.018, *p* = 3 × 10⁻¹⁵), so the fitted curve turns upward with a minimum at *z* = +3.4, beyond the observed data. Within the range the animal actually occupies, the relationship is simply downward. It is also not a running-speed artifact — mean speed is flat across pupil deciles (37–40 cm/s) and controlling for within-session speed leaves the coefficient unchanged (−0.128, quadratic +0.011).

**The shift is tonic, not localised — and that is the take-home.** Compared between states in each of 50 position bins, the pupil difference has **the same sign in all 50**, reaches significance in **48**, and barely varies along the track (mean −0.266 z, range −0.432 to −0.136, SD across bins **0.075 z**). It shows no trend with position (ρ = −0.264, *p* = 0.064), and the reward zone is marginally *less* affected than the rest (−0.244 against −0.269). This is the distinction that matters for interpretation: attention paid selectively at task-relevant locations would concentrate the difference there, whereas a change in tonic engagement shifts the entire traversal. The data show the latter.

**What a smaller pupil means here: focus, not drowsiness.** A constricted pupil is read as low arousal only when it accompanies disengagement, and that is not this state. The anchored, constricted-pupil trials are the ones on which the animal localises the reward zone correctly, most strongly when it must do so by path integration (Figure 1), and performance improves monotonically as the pupil narrows. A drowsy or disengaged animal does not show that profile. The interpretation the data support is that pupil constriction here indexes **focused task engagement** — a selective, exploitative mode — and dilation the distractible, exploratory alternative. This is the pairing the adaptive-gain account of locus coeruleus function predicts (Aston-Jones & Cohen): low *tonic* noradrenergic tone with a small baseline pupil in the engaged, exploitative mode, high tonic tone with a large pupil in the exploratory one. Pupil size remains our measurement; the state it marks is one of attentional focus, and the paper should not call the anchored state "low arousal", which carries the wrong connotation for a state defined by better performance.

Pupil radius does not change instantaneously: the single trial spanning a transition accounts for only 16–31% of the total change. Because the classifier's median filter smooths each cell's label sequence over ±2 trials, the data cannot distinguish a genuinely graded change from an abrupt one whose onset is imprecisely timed, and we make no claim about which it is.

**The engagement account makes a prediction about other regions.** If the state is a global behavioural variable rather than an entorhinal computation, it should appear outside the entorhinal cortex. Section 6 tests that, and the answer is not a simple yes.

---

### 6. One state, shared across structures, expressed most strongly by one cell type

A global behavioural state should be readable outside MEC. Asking that question properly means not making entorhinal cortex the frame: scoring other regions against the MEC axis can only ask how much they resemble MEC, and cannot ask whether they have a state of their own. Each region therefore builds its **own** first principal component of its own cell × trial label matrix, and the axes are compared with one another (Figure 6). The states themselves are shown first (Figure 6 supplement), for the two sessions that bracket the range among those recording all three structures: the block structure is shared and visible in the rasters and in single-cell rate maps without any statistic, while the entorhinal–visual agreement runs from +0.82 down to +0.15, which is the context in which the averages below should be read.

**Every region has a coherent state of its own.** Split-half reliability — the axis built twice from disjoint halves of a region's own cells, size-matched to 8 cells per half so that MEC's hundreds of cells carry no advantage over visual cortex's tens — is +0.447 in MEC (27 sessions, *p* = 1.5 × 10⁻⁸), +0.474 in the subicular complex (10 sessions, *p* = 0.0039) and **+0.330 in visual cortex** (21 sessions, *p* = 2.4 × 10⁻⁵).

**And one axis is an adequate description of it.** That a first principal component exists says nothing about whether the rest of the label matrix is noise or a second state — PC1 is by construction the largest component of whatever is there. The obvious statistic, the fraction of variance PC1 explains, cannot answer this across regions, because variance explained by the top component of an *n* × *T* matrix **rises as *n* falls**: with few cells the leading eigenvalue absorbs more of the total by chance alone. Plotted by region that statistic would largely report how many cells each region contributed, in the direction that makes the *worst*-sampled region look the most one-dimensional. We therefore scored out of sample — build the axis from one half of a region's cells, measure the variance it explains in the other half — which a direction fitted to noise cannot do, at any *n* (`region_state_dimensionality.py`; Figure 6 supplement 3).

Across 60 sessions, PC1 estimated from 8 cells explained **0.052** of held-out entorhinal variance against a circular-shift null of 0.023, with **+0.061** over null in the subicular complex and **+0.020** in visual cortex. Scored by component at the same cell count, the excess over null was **+0.0314** for PC1 and **+0.0006** for PC2 — a ratio of **53×** — with PC3 to PC5 at null (*p* = 0.076, 0.30, 0.25). A population carrying two independent states would have both generalise, since the components are orthogonal by construction. The label matrix is therefore effectively one-dimensional, and "the population anchoring state" is a well-posed object rather than a convenient summary of something richer.

> **Three things this panel should not be read as saying.** (i) PC2 is **not** zero: at *p* = 4.3 × 10⁻⁴ it clears its null, and the claim rests on the 53-fold ratio, not on a dichotomy. The honest phrasing is *effectively* one-dimensional. (ii) The absolute values are small — 5% of held-out variance in MEC — because this is a *single direction* in a space with as many dimensions as trials, where chance is 2.3%; the quantity to read is the ratio to null (≈ 2.3×), not the percentage. (iii) The null **also rises with cell count** (0.020 at *n* = 4 to 0.046 at *n* = 64), because circularly shifted cells keep their own block structure and their top component still captures shared low-frequency structure, estimated better as *n* grows. This is why the comparison is against a shuffle null at matched *n* rather than against the 1/*T* expected for a random direction, which would have been far too permissive.

> **Two observations worth following up.** The subicular state is *more* one-dimensional than the entorhinal one at every matched cell count (0.084 against 0.052 at *n* = 8, and the gap holds from *n* = 4 to 16), which is the opposite of what a "MEC is the centre" account predicts and is consistent with §6's finding that the specificity is cellular rather than anatomical. And MEC has **not plateaued at *n* = 64**, so the measured one-dimensionality is a lower bound on the true value rather than an estimate of it.

**And it is the same state.** A raw correlation between two regions' axes cannot distinguish *different state* from *no state*, since an axis estimated from few noisy cells is itself noisy. Dividing by the geometric mean of the two reliabilities (Spearman's correction for attenuation) separates them. Corrected, MEC and the subicular complex agree at **+0.920** and MEC and visual cortex at **+0.863**; neither differs from 1 (*p* = 0.098 and 0.515), and the two do not differ from each other (*p* = 0.66). The raw gradient — 0.46 against 0.31 — is therefore a difference in how noisily each region estimates the state, not in which state it is.

**What differs is the cell type, not the region.** At matched cell count MEC's own state is only marginally more coherent than visual cortex's (+0.417 against +0.334, paired within session, *p* = 0.065). Per cell, with every cell scored out-of-sample against an axis built from half the entorhinal population (Methods), the picture is quite different: grid cells exceed visual cortex by **+0.149** (*p* = 9.9 × 10⁻⁶, 26 of 32 sessions) and interneurons by **+0.144** (*p* = 1.3 × 10⁻⁶, 29 of 33), while other entorhinal cells exceed it by only +0.050. The cell-type contrast is three times the regional one, and the regional one does not reach significance.

The state is therefore **global, and the specificity is cellular**: one state, present in every structure recorded, read out most strongly by grid cells together with local inhibition — which is what Sections 2 and 3 found from inside MEC, now with the regional alternative excluded.

> **This also explains the behavioural result, which had looked anomalous.** A population state built from visual cortex carries an Anchoring Dependence Index of +0.148 (*p* = 0.0014, 26 sessions), *larger* than the entorhinal state's +0.085 (*p* = 0.0015, 60 sessions), and the ADI is unchanged when the state is computed from MEC cells alone (*p* = 0.81). If the state is one global variable legible from any sufficiently sampled structure, that is expected rather than puzzling. **The paper should claim no behavioural specificity for the entorhinal readout.**
>
> **Reconciled with Figure S3 (2026-10-03).** The speed-matched supplement computes the same index and finds MEC significant (+0.120, *p* = 0.010, 21 sessions) while visual cortex is not (+0.133, *p* = 0.16, 10 sessions) — which reads as entorhinal specificity and is not. **The filters differ, and S3's is a strict subset of this one.** Here a region needs ≥ 10 cells and ≥ 10 trials per cue type. S3 additionally needs ≥ 20 trials per cue type, ≥ 10 trials in *each state* within each type, and ≥ 20 trials surviving the speed match with ≥ 5 per state and both outcomes present — which cuts 60 MEC / 26 VIS to 21 / 10. The VIS mean stays **higher in both** (+0.148 vs +0.085; +0.133 vs +0.120); only the *p* value moves, and MEC clears its own 80%-power threshold by 0.003 while VIS and SUB sit below theirs. Direct between-region tests on the matched data find nothing (MEC vs VIS *p* = 0.82; MEC vs SUB *p* = 0.71). State both session counts wherever either number is quoted.

> **CAVEAT — what "the same state" is and is not licensed to mean.** Disattenuation divides by √(r_ii·r_jj) ≈ 0.38 for the MEC–visual pair, so it amplifies noise by ≈ 2.6×: the bootstrap CI is [0.49, 1.23] and 8 of 19 sessions exceed 1. "Not different from 1" therefore excludes differences larger than about 0.3 and no more; it is not a demonstration of identity. SUB–VIS (4 sessions) and cerebellum (1 session at this cell-count threshold) are not testable.

> **CORRECTION, recorded deliberately (2026-10-02).** Two earlier versions of this section were wrong in opposite directions. The first reported raw correlations and concluded the state is followed as strongly outside MEC as within it. The second replaced them with "excess over each cell's own circular-shift null" and concluded that **only** entorhinal groups exceed chance and that **visual cortex sits below its own null** (−0.039). That statistic subtracted the mean of an **absolute** null from a **signed** observed correlation. The null is centred at zero — measured signed mean −0.008 over 12,722 cells — so mean|shifted *r*| is simply E|x| = 0.798 × SD of a zero-centred distribution, and subtracting it biases every cell down by ≈ 0.8 SD of its own null, differentially by region because null widths differ (0.130 to 0.157). The stated motivation, that a cell anchored on most trials correlates with a population that is also anchored on most trials, does not survive measurement either: the anchored fraction is unrelated to the signed null mean (ρ = −0.05, *p* = 0.08) and only weakly related to its spread (ρ = +0.14). Chance for a signed correlation is **zero**. Visual cortex was never below it: 67% of its cells correlate positively (*p* = 3 × 10⁻²⁵) and 18.5% individually beat their own null against a 5% false-positive rate. The −0.039 was not significantly below zero in any case (*p* = 0.143).


### 7. The field potential changes with the state in a frequency-specific way, and theta decouples from running speed

If task anchoring is a network state rather than a property of individual cells, it should be visible in the local field potential. It is, and the change is not a single effect on theta: the spectrum moves in opposite directions in neighbouring bands, which is the observation that constrains what the state can be doing.

**The spectrum is restructured, not rescaled.** Welch spectra were computed per trial from the entorhinal LFP (1 kHz, `nperseg` = 1024 → 0.98 Hz bins, mains bands notched), whitened by multiplying power by frequency so that peaks are visible against the 1/*f* background, and *z*-scored within session so that no session's absolute amplitude dominates. Across the **26 sessions with at least 8 trials in each state** (Figure 7E, F), the anchored − non-anchored difference changes sign across the gamma range:

| band | anchored − non (z) | *p* | sessions in direction |
|---|---|---|---|
| delta 2–5 Hz | +0.160 | 0.036 | 17/26 up |
| theta 6–10 Hz | **−0.094** | **0.020** | 20/26 down |
| beta 12–20 Hz | +0.063 | 0.28 | — |
| slow gamma 30–48 Hz | **+0.105** | **0.0013** | 21/26 up |
| mid gamma 60–100 Hz | **−0.108** | **0.0056** | 19/26 down |
| high 100–200 Hz | −0.088 | 0.056 | 16/26 down |

Wilcoxon signed-rank on per-session band means, paired across states. Because the difference **reverses sign** between 30–48 and 60–100 Hz, no uniform change in amplitude, electrode referencing or spiking contribution to the field can produce it: those act in one direction at all frequencies. The pooled difference profile is positive from ~20 Hz to ~60 Hz and negative above it, changing sign at 63.7 Hz, and 25 of 26 sessions have such a crossing with a median at **67 Hz** (Figure 7J). The anchored state therefore shifts power *down the gamma range* rather than raising or lowering gamma as a whole.

> **This supersedes the earlier "the decrease is broadband" concern**, which was raised when only theta and an 80–200 Hz band had been measured post-gate and both fell. With the full spectrum, 30–48 Hz moves the other way, so the effect is band-specific and the broadband reading is excluded. The slow-gamma band contains the 50 Hz mains notch; removing the 48–52 Hz bins *strengthens* the effect (+0.109, *p* = 3.6 × 10⁻⁴) and the 48–52 band on its own is not significant (*p* = 0.063), so mains contamination is not producing it.

**Theta's nesting of gamma changes for fast gamma only.** Within-theta modulation was quantified as Tort's modulation index over 18 phase bins, with a phase-shuffle null subtracted per session. Theta–**fast**-gamma coupling falls in the anchored state by **15.9%** (0.00240 → 0.00202 over null, *p* = 5 × 10⁻⁶, lower in 23 of 26 sessions), while theta–**slow**-gamma coupling does not change (−7.9%, *p* = 0.71, 11 of 26) (Figure 7G). So the band whose power falls is also the band that uncouples from theta, and the band whose power rises stays nested as it was. *The slow-gamma null is the weaker of the two — its absolute MI is five times smaller and its band contains the mains notch — so "slow gamma does not uncouple" should be read as an unsupported difference rather than a demonstrated invariance, and the planned re-run on a clean 30–48 Hz band is still worth doing.*

**Both bands move, in opposite directions. They are not coupled to each other.** Slow gamma rises in 21 of 26 sessions and mid gamma falls in 19 of 26, and the difference index (slow − fast) is positive in 21 of 26 (+0.213, *p* = 4.4 × 10⁻⁵) (Figure 7H, I). Those are the two band effects restated, and they are the claim. Two stronger readings both fail. The sizes of the two changes are **uncorrelated** across sessions (ρ = −0.08, *p* = 0.70), where a redistribution of a fixed amount of power predicts a negative correlation. And the sessions falling in the slow-up/fast-down quadrant, **16 of 26**, are no more than the two marginals already imply: given 21 of 26 up and 19 of 26 down, independence predicts 15.3, and the 2 × 2 table gives Fisher *p* = 0.59. The pattern is therefore two separate band effects that happen to point in opposite directions, not one coupled shift, and should be described that way.

> **CORRECTION, recorded deliberately (2026-10-03).** An earlier draft of this paragraph tested the quadrant count against a binomial with *p* = 0.25 — 16 of 26 against 6.5 expected, *p* = 8.6 × 10⁻⁵ — and read it as evidence that the two bands move together. That null assumes each band is equally likely to go up or down, which is exactly what the band effects above have already rejected: it tests the marginals a second time and reports them as a conjunction. Against the **observed** marginals the expectation is 15.3 and the result is null (*p* = 0.59). `fig7_theta_gamma.py` still prints and plots the binomial *p*, and panel H's title still counts the quadrant; both should be replaced by the Fisher test on the 2 × 2 table before the figure is used. This is the second null-specification error corrected in this paper — the first being the signed-observed-against-absolute-null statistic in §6 — and the shared lesson is worth stating once: when a statistic conditions on something the data have already established, its null must condition on it too. (The §3 correction is a different fault, a session-selection criterion that excluded the sessions able to answer the question.)

> **How the gamma result should be read, given what the pupil says.** The direction of the slow-gamma effect depends on what the anchored state is taken to be. Read as low arousal it is awkward: cortical arousal generally *raises* gamma power, so an aroused-to-unaroused transition predicts gamma falling at all gamma frequencies, which is not what happens. Read as **focused task engagement** — which is what Section 5 establishes, since performance improves monotonically as the pupil narrows — it is the expected direction. Slow and fast gamma in the entorhinal–hippocampal system are associated with different input streams and, in Colgin's framework, with internally-guided retrieval against externally-driven encoding respectively; an animal localising a reward zone by path integration is in the internally-guided case, and that is also the condition in which anchoring matters most behaviourally (Figure 1). The slow-gamma increase, the fast-gamma decrease and the loss of theta–fast-gamma coupling are then one coherent shift toward internally-guided processing in the focused state. *This is an interpretation offered for the pattern, not a result: the band assignments come from CA1 work, our recordings are entorhinal, and we have not shown that the two bands here carry the inputs the framework assigns them.*

**Discrete gamma events: the demand prediction fails, the position result holds.** If gamma cycles were discrete computational slots, uncued trials — which require path integration rather than reading a landmark — should contain more of them. Detecting supra-threshold envelope excursions lasting at least one gamma cycle, and counting them per theta cycle across 10,802 trials from 44 sessions (occupancy 2.2% slow, 2.4% fast), the raw comparison looks supportive: fast-gamma events are more frequent on uncued trials (+0.0057 per theta cycle, *p* = 0.016, 26 of 43 sessions). The effect does not survive control. Adding trial speed and theta frequency to the model **reverses its sign** (−0.0054, *p* = 4 × 10⁻⁵), and adding band power removes it entirely (+0.0003, *p* = 0.68). Event counts here track power and running, not task demand, and **we report the slot prediction as tested and not supported.**

The same analysis gives a positive result about *where* on the track gamma occurs. Slow-gamma event rate peaks at **92 cm**, inside the reward zone, in the pooled profile and at a median of 92 cm per session, with **40 of 44 sessions** peaking between 70 and 130 cm; fast-gamma events show no such peak (median 150 cm). The reward-zone elevation survives control for speed and band power (+0.356 Hz, *p* = 5 × 10⁻⁸ for 80–110 cm against the rest of the track) but does not interact with either cue condition (*p* = 0.93) or anchoring state (*p* = 0.77). Slow gamma is therefore organised by track position — concentrated where the animal must decide to stop — but that organisation is not itself modulated by the state or by whether the zone is cued, so it stands beside the state result rather than explaining it.

**Theta also weakens, marginally.** Across the same 44 sessions theta amplitude is lower on anchored trials: Δ z = **−0.224** (*p* = 0.033, paired across the 31 sessions with speed-matched trials in both states), an amplitude fall of **1.66%** (*p* = 0.032). The effect is not a shift of the rhythm out of the analysis window — instantaneous frequency is unchanged (−0.016 Hz, 7.799 vs 7.815 Hz, *p* = 0.54) and the spectral peak moves by −0.110 Hz (*p* = 0.016), far too small to move power out of a 4 Hz band. Theta therefore weakens rather than shifts. At this effect size the amplitude result should not lead the section, and it does not; it is reported because the band-resolved spectrum independently confirms its sign and significance (−0.094, *p* = 0.020) on a different estimator and a different session set.

**The relationship between theta and behaviour also changed, and this is the more robust of the effects.** Theta amplitude and frequency both track running speed, and that coupling was roughly halved during anchored epochs: on the 19 sessions with at least 15 trials in each state, the within-session regression slope of theta amplitude on speed fell from 0.374 to 0.218 (Wilcoxon *p* = 0.045) and of theta frequency on speed from 0.0126 to 0.0070 Hz per cm/s (*p* = 0.012). The corresponding correlations fell from *r* = 0.424 to 0.244 for amplitude (*p* = 0.0015, 15 of 19 sessions) and 0.398 to 0.185 for frequency (*p* = 3.4 × 10⁻⁴, 16 of 19 sessions). Mice do run more uniformly when anchored (speed SD 6.07 vs 8.58 cm/s, *p* = 2.2 × 10⁻⁵), which attenuates a correlation on its own, but the result is not that artifact: the regression slope is insensitive to range restriction and falls by the same factor; the per-session change in correlation does not track the per-session change in speed variability (ρ = −0.01, *p* = 0.92); restricting both states to their common speed range leaves the slope difference intact (0.12 vs 0.34, *p* = 0.008); and a circular-shift null on the state vector, which preserves run structure and within-session drift, gives *p* = 0.005.

> **⚠ THE NUMBERS BELOW THIS POINT ARE PRE-GATE AND MUST BE RE-RUN.** Everything above this line is post-gate: the spectrum, the modulation indices and the gamma events come from `spectrum_by_anchoring_w*.csv`, `pac_by_anchoring_w*.csv`, `gamma_events.csv` and `gamma_by_position.csv`, all built 2026-10-02/03 against the gated labels in `data/population_state/anchoring_trials.csv`, and the amplitude and speed-coupling paragraphs from the rebuilt `theta_frequency.csv` and `speed_theta_coupling.csv`. Every statistic **below** still comes from `data/lfp/PREGATE/`, computed when the classifier reported ~52% of trials anchored on data with no anchoring; the per-cell null gate cut the sessions with ≥15 trials in both states from 55 to 19. **Directions survive but magnitudes and *n* do not**, and three of the controls change character:
>
> | quantity | pre-gate | post-gate | status |
> |---|---|---|---|
> | sessions / trials | 58 / 14,022 | 44 / 10,802 | |
> | sessions, both states ≥15 trials | 55 | **19** | the power loss |
> | theta amplitude Δ | −3.95%, *p* = 2.7 × 10⁻⁶ | **−1.66%, *p* = 0.032** | halved, now marginal |
> | Δ z amplitude | −0.446, *p* = 1.5 × 10⁻⁶ | **−0.224, *p* = 0.033** | halved, now marginal |
> | instantaneous frequency | −0.039 Hz, *p* = 0.032 | **−0.016 Hz, *p* = 0.54** | now a clean null — *helps* the "weakens not shifts" claim |
> | peak frequency | −0.042 Hz, *p* = 0.017 | **−0.110 Hz, *p* = 0.016** | 2.6× larger; still negligible against a 4 Hz band, but no longer quotable as "two orders of magnitude" without checking |
> | speed→amplitude slope | 0.250 → 0.091, *p* = 0.0025 | **0.374 → 0.218, *p* = 0.045** | ratio 0.36 → 0.58; marginal |
> | speed→frequency slope | 0.0113 → 0.0064, *p* = 1.7 × 10⁻⁴ | **0.0126 → 0.0070, *p* = 0.012** | holds |
> | *r*(speed, amplitude) | 0.297 → 0.145, *p* = 6.9 × 10⁻⁴ | **0.424 → 0.244, *p* = 0.0015** | holds |
> | *r*(speed, frequency) | 0.356 → 0.186, *p* = 1.6 × 10⁻⁷ | **0.398 → 0.185, *p* = 3.4 × 10⁻⁴** | holds |
>
> **What this changes for the argument.** The *decoupling* result survives the gate — all four coupling measures still fall, and the frequency ones comfortably. The *weakening* result does not survive at anything like its former strength: at −1.66% and *p* = 0.032 on 31 sessions it is a marginal effect, and the section's opening sentence can no longer lead with it. Reorder so the decoupling is the claim and the amplitude fall is secondary, or drop the amplitude claim.
>
> **Files still to re-run** (all PREGATE-only): `phase_locking.csv`, `region_speed_slope.csv`, `residual_speed.csv`, `theta_residual_speed.csv`, `speed_tuning_cells.csv`, `speed_tuning_decode.csv`, `speed_poprate.csv`, `speed_from_position.csv`, `speed_theta_timecourse.csv`, `speed_deltapr2.csv`. The scripts are `phase_locking.py`, `region_slope.py`, `residual_speed.py`, `theta_residual.py`, `speed_tuning.py`, `poprate.py`, `speed_theta_time.py`. They already read `anchoring_trials.csv`, so they need running, not editing. **`lfp_power_by_anchoring.csv` has been dropped from this list rather than re-run:** `lfp_spectrum_pac.py` supersedes `lfp_batch.py` for band power, measures the whole spectrum instead of five fixed bands, and is what showed that the change reverses sign inside the old 30–80 Hz gamma band. **Expect the regional and laminar tests to lose their power first** — `region_speed_slope` was reported on 37 MEC / 17 subicular / 32 visual sessions, and the subicular result (*p* = 0.0079, 17 sessions) is the one most likely to evaporate.

That the *gain* changes and not only the level distinguishes two accounts. A state in which the animal simply runs differently would move theta along its existing speed relationship; instead the relationship itself flattens, so during anchored epochs entorhinal theta is less driven by self-motion.

The obvious mechanistic reading — that the velocity signal driving the entorhinal network is down-weighted when position is pinned by landmarks — is **not supported by the spikes**. Speed tuning of individual entorhinal cells is unchanged between states: among 3,719 positively speed-modulated cells (40.3% of 9,256 recorded cells), the rate-normalised tuning gain is identical (p = 0.28 across 57 sessions), only 49% of cells are flatter when anchored, and on a common speed support the gain is marginally *steeper* (p = 0.021). Population decoding of running speed is likewise not degraded — with bins matched for count and speed distribution it is slightly better when anchored (r = 0.761 vs 0.729, p = 2.6 × 10⁻⁴).

Because anchored trials have both more reliable position coding and a stereotyped speed profile, position can substitute for speed, so these measures were repeated with position partialled out. Speed proved to be 4.5 times more predictable from position when anchored (R² = 0.220 vs 0.048, p = 4.6 × 10⁻¹⁰, 51/56 sessions) — the behaviour itself is more stereotyped. Partialling position out of both firing rate and speed, entorhinal sensitivity to *unexpected* speed is essentially preserved: 0.0356 vs 0.0374 Hz per cm/s (−4.7%, p = 0.046, 33/56 sessions), and nothing once normalised by firing rate (p = 0.36). What falls is the amount of unexpected speed there is to encode (SD 13.9 vs 15.8 cm/s), which is enough to account for the drop in the partial correlation (−25%, p = 1.2 × 10⁻⁴) and in the per-SD slope (−14%, p = 9 × 10⁻⁵).

The same residual analysis applied to theta shows that the field potential does not behave this way. Theta's sensitivity to unexpected speed falls by 15.6% when anchored (0.00610 vs 0.00722 per cm/s, p = 1.7 × 10⁻³, 36/56 sessions) — essentially the same proportional fall as for total speed — while its relationship to the *expected*, position-predictable component of speed is unchanged or slightly stronger (+5.5%, p = 0.055). The decoupling is therefore not a consequence of stereotyped running leaving less unpredicted self-motion to respond to; theta responds less per unit of unpredicted self-motion.

Comparing like with like — 250 ms bins, position partialled out of both sides — theta's speed sensitivity falls about three times as much as the cells' does (−15.6%, p = 1.7 × 10⁻³ against −4.7%, p = 0.046 and null once rate-normalised).

Spike timing relative to theta is also unchanged. Across 7,542 cells in 59 sessions, with spike counts and speed distributions matched between states, the mean resultant length of spike–theta phase locking is identical (0.1414 anchored vs 0.1425, p = 0.79, 30/59 sessions lower) and the preferred phase shifts by 1.1°. The speed dependence of locking does differ, but in the opposite direction to the one a loss of entrainment would predict: locking discriminates fast from slow running *more* strongly when anchored (fast − slow tercile +0.0386 vs +0.0286, p = 1.3 × 10⁻⁵), which also serves as a positive control that the measurement detects state differences at all.

Three properties of the spiking are therefore unchanged between states — rate coding of speed, sensitivity to unexpected speed, and spike–theta timing — while the amplitude of the field potential and its scaling with speed both fall. Because LFP theta amplitude reflects summed synaptic and dendritic currents rather than local spiking, the most likely locus is the magnitude of the speed-scaled synaptic drive reaching MEC, upstream of a local circuit whose relationship to the rhythm is preserved.

A localised change in drive to entorhinal cortex is also not supported. Computing the same speed→theta slope for every channel group, entorhinal or not, with the anchoring state still defined by entorhinal cells, the decoupling is not specific to MEC: the per-session median change is −0.00114 in MEC (p = 0.051, 37 sessions), −0.00507 in the pre/para/postsubicular complex (p = 0.0079, 17 sessions) and −0.00128 in visual cortex (p = 0.111, 32 sessions). The subicular complex shows a larger and more reliable change than MEC itself. Nor does the change have a laminar profile within MEC: across entorhinal layers, session-median changes do not differ (Kruskal–Wallis p = 0.71; ENTm2 is nominally p = 0.0083 but does not survive correction across five layers).

> **Four mechanistic readings have now been tested and failed**, and the text should be correspondingly cautious about proposing a fifth. (i) Reduced velocity gain predicts flatter speed tuning and worse speed decoding: neither occurs. (ii) Theta tracking prediction error predicts that sensitivity to *unexpected* speed is preserved: it falls by the same proportion as for total speed. (iii) Entrainment predicts weaker phase locking: locking is unchanged and its speed dependence strengthens. (iv) A localised change in synaptic drive to MEC predicts regional and laminar specificity: the subicular complex changes more than MEC, visual cortex changes comparably, and there is no laminar profile. The empirical result — theta amplitude decouples from running speed while the cellular speed code and spike–theta timing do not — is robust and survives every control applied to it. The interpretation is not settled, and the Discussion should say so rather than presenting any mechanism as established.
>
> **Caveat on the regional test.** The non-entorhinal groups sit on the same probe as the entorhinal ones, so volume conduction cannot be excluded, and a *shared* effect was always going to be ambiguous where a divergent one would have been decisive. Two observations argue against pure conduction: visual cortex has a speed→theta slope of the opposite sign to MEC in both states, so its signal is not simply entorhinal theta appearing elsewhere; and cerebellum, the one distant structure recorded, shows no change. Cerebellum is available in only 5 sessions with ≥4 channel groups, however, so it does not function as the negative control it was intended to be.
>
> This result sits with the finding that the anchoring axis itself is not MEC-specific (see *Per-cell agreement with the population axis*): both say the phenomenon is broader than entorhinal cortex.

> **Medial septal drive as a candidate, and the sign problem it has to solve.** Robinson, Ying, Hasselmo & Brandon (2024, *Cell Reports* 43:114590, doi:10.1016/j.celrep.2024.114590) optogenetically inhibited medial septal GABAergic neurons and found that grid cell spatial periodicity was disrupted during inhibition and during short inter-stimulus intervals, recovering over longer ones, and that **grid cell phase precession was disrupted as well**. That is the most direct demonstration available that the septal input which sustains entorhinal theta also sustains the spatial and temporal structure of the grid code, and it makes reduced septal drive a natural candidate for a state in which theta changes and the grid code becomes less anchored.
>
> **The direction does not yet fit, and the text must not paper over this.** Reduced septal drive produces weak theta *together with* a disrupted grid code. In our data the two come apart: theta amplitude is **lower** in the anchored state (−4.1%), which is the state in which the code is *best* locked to the track, and higher in the non-anchored state. A simple "non-anchored = less septal drive" account therefore predicts the opposite of what we observe on the LFP.
>
> **A cell-type distinction may resolve it, and phase precession is the experiment that decides.** The septal contribution is not one signal: GABAergic neurons act as the pacemaker that Robinson et al. silenced, whereas the speed signal reaching MEC is carried largely by glutamatergic septal and supramammillary projections (Fuhrmann et al., 2015; Justus et al., 2017). Our results — a halved speed→theta gain, unchanged spike–theta phase locking, and unchanged rate coding of speed — look like a change in the *speed* pathway with the pacemaker intact, not like the pacemaker silencing Robinson et al. performed. The two accounts make opposite predictions for phase precession: pacemaker-like disruption predicts precession **degraded** in the non-anchored state, whereas a speed-pathway change with an intact pacemaker predicts precession **preserved**. This is testable in the existing data (see *Phase precession by anchoring state* in Methods) and should be run before either account appears in the Discussion.

> **BOTH SEPTAL ACCOUNTS NOW HAVE EVIDENCE AGAINST THEM (2026-10-03), and the note above is superseded.** The speed-pathway half was tested directly, on the population the account requires: entorhinal speed cells are predominantly fast-spiking interneurons (here 58.7% of putative interneurons are open-field speed cells against 41.5% of principal cells, OR 2.01, *p* = 1.1 × 10⁻¹³, with the relationship continuous in waveform duration), and their open-field labels transfer to the task with a **twelvefold** separation in VR speed slope (0.138 against 0.012/0.010/0.0008 Hz per cm/s for the other three groups). Within those cells **speed coding does not change with state**: slope +11.9% (*p* = 0.121) and gain +6.8% (*p* = 0.565) on common speed support, 29 sessions — if anything *steeper* when anchored, the wrong direction. The null is informative, resolving 17% in slope and 18% in gain at 80% power against observed 12% and 7%, so a halving of speed drive of the size the LFP shows is excluded in the cells that carry it. `speed_interneurons.py` → `speed_interneurons.csv`.
>
> **And the experiment that was supposed to decide between them does not.** `phase_precession.py` ran on 3,347 cells in 44 sessions with both a naive and a field-matched comparison (the latter restricted to trials on which the cell actually has a field, which is the control the classifier's definition makes mandatory). Both are null — naive −0.0016, *p* = 0.30; matched +0.0015, *p* = 0.60 — and **neither null is informative**. Precession is present in direction (slopes negative in 617/1045 cells, *p* = 3 × 10⁻⁹; 20/27 sessions) but not in magnitude: −0.035 cycles per field traversal where the phenomenon should give half to one cycle, and a circular–linear correlation of 0.042 against a chance level of 0.037 for the median spike count, so the signal above chance is ~0.005 while the minimum detectable state difference is 0.018–0.027. Restricting to grid cells does not rescue it (0.040 against chance 0.034). The probable cause is pooling spikes across field traversals rather than fitting each pass separately — close to worst-case here, because field position varies trial to trial by construction. **Do not cite phase precession in either direction until a per-traversal version has been run.**

> **Levels of analysis do not agree, and the text must not mix them.** The halving of the speed→theta slope (0.250 → 0.091) is a *trial-level* statistic: it relates a trial's mean speed to its mean theta across trials. Within trials, at 250 ms resolution, the same comparison gives −15% and does not reach significance on total speed (p = 0.42); only the position-partialled version does (p = 1.7 × 10⁻³). Both are real and they answer different questions — trial-to-trial covariation versus moment-to-moment covariation — but an earlier draft of this section quoted the trial-level LFP effect against the bin-level spiking effect and reported a thirteen-fold dissociation. The matched comparison is threefold.

> **TODO:** the one spiking measure that moves with the LFP is summed population rate, whose normalised speed slope is shallower when anchored (0.00202 vs 0.00237 on common speed support, p = 1.3 × 10⁻³). This sits between the per-cell null and the LFP effect and deserves a direct test — whether it reflects the 22.5% of cells that are *negatively* speed-modulated becoming more strongly so, which per-cell medians over positively-tuned cells would miss.
>
> **Do not re-run the ΔpR² test.** Speed's unique contribution beyond position does fall when anchored (0.0180 vs 0.0271, p = 3.0 × 10⁻⁵, cyclic position, 1,105 cells), and the positive control behaves (position beyond speed rises, p = 2.1 × 10⁻⁵), but the measure is uninterpretable here: speed is 4.5× more redundant with position when anchored, so a covariate carrying less independent information must contribute less regardless of the neurons. The residual-speed analysis above replaces it. Note also that at the pipeline's default 200 rounds / depth 5 every single-covariate pR² was negative; 50 rounds at depth 3 fixes that and preserves the direction.

This is a statement about the population of sessions, not about the moment-to-moment time course within one. Computing the speed–theta correlation in a sliding 25-trial window and comparing it to the anchored fraction in the same window, the coupling is lower in predominantly anchored windows (r = +0.147 vs +0.265, p = 5.2 × 10⁻⁴ across 51 sessions, holding for 15- and 40-trial windows), but no individual session resolves the effect: the median per-session correlation between window coupling and window anchoring is −0.148 (34/55 negative, Wilcoxon p = 0.012) and only 2 of 55 sessions beat their own circular-shift null. This is expected rather than contradictory — a 25-trial window correlation carries a standard error near 0.2, and states occur in long blocks (median 19 per session), so a single session contains few effectively independent transitions. The coupling change should therefore be reported as a property of the anchored state across sessions, and single-session traces used only as illustration.

> **TODO:**
> - ~~The decrease is **broadband**: 80–200 Hz power falls more than theta does, so the effect could be a general amplitude change — candidates being a change in spiking contribution to the LFP, or an electrode-referencing effect.~~ **RESOLVED 2026-10-03, and it was the opposite.** That concern was raised when only theta and an 80–200 Hz band had been measured and both fell. Measuring the whole spectrum shows 30–48 Hz moving the *other way* (+0.105, *p* = 0.0013) while 60–100 Hz falls (−0.108, *p* = 0.006): the difference reverses sign across the gamma range, which no uniform amplitude, referencing or spiking-contribution change can produce, since all of those act in one direction at every frequency. The effect is band-specific. This is now the opening result of §7 rather than a caveat on it.
> - Laminar profiles disagree between measures. Cell-level PC1 following peaks in ENTm2 (+0.196, Kruskal–Wallis p = 2.3 × 10⁻⁷) whereas the LFP theta effect peaks in ENTm1 (−0.017 vs −0.002 to −0.004). The LFP layer effect therefore **cannot** be presented as confirming the cell-level laminar organisation. The defensible link between the two is a per-session test — does a session's theta effect predict its cell-following strength? — not a laminar comparison.
> - Drop the mediolateral split for this analysis: three independent tests are negative (LFP p = 0.17; cell PC1 within-session median ρ = −0.051, p = 0.087; DV within-session median ρ = −0.015, p = 0.57). Pooled DV and ML correlations look overwhelming (p = 1 × 10⁻⁴³ and 8 × 10⁻²⁵) but are driven entirely by between-session differences.

---

## Discussion

*(~1500 words in last publication)*

> **⚠ THE WHOLE DISCUSSION IS THE SUPERSEDED ARGUMENT AND HAS TO BE REWRITTEN.**
> Every paragraph below argues the mediolateral two-populations story: that MEC
> contains two functionally independent populations of non-grid spatial cells,
> one coordinated with the medial grid network and one dissociable from it,
> separated by xgBoost predictivity and exposed when grid firing decouples from
> the task reference frame. **The Results no longer contain that claim** — the
> xgBoost sections were removed on 2026-10-01 because they are not in the
> argument, and the dissociation was never shown with the controls it needs
> (cell-count, firing-rate and distance matching; the one medial/lateral effect
> tested did not survive rate matching, *p* = 0.75).
>
> What the Discussion now has to discuss, following the seven steps:
> 1. A population anchoring state exists in MEC and predicts path-integration performance.
> 2. It is read out by grid cells together with local inhibition, not by spatial tuning as such — and the two agree with *each other*, so they are a subnetwork.
> 3. That subnetwork is medial, and sits within the superficial grid population. **This is where the mediolateral story survives**, and it is now about grid identity and state-following rather than about two populations of non-grid cells. The superficial half of this belongs to identity: layer predicts what a cell is, but not whether it follows, once tested within session.
> 4. The subnetwork is real in the wiring but the wiring does not carry the state — the pivot, and the paper's main mechanistic claim.
> 5. The state is a reflection of attentional engagement.
> 6. The state itself is global — every structure recorded expresses the same one — and the specificity is cellular, not anatomical.
> 7. Theta decouples from running speed in the anchored state, and four mechanistic readings of that have been tested and failed.
>
> Material below worth keeping: the **multi-domain organisation** section (medial
> grid enrichment, layer 1/2 width, molecular and intrinsic-property gradients)
> transfers almost intact to step 3; the **parallel systems** framing does not,
> because it rests on the dropped dissociation. The named citations in the
> *Alternative / earlier Discussion material* block (Brun 2008; Sun 2015;
> Rowland 2018; Tennant 2018) should be inherited by the rewrite — the main text
> still has no bibliography.
>
> Nothing below has been edited, so the old argument is recoverable if any of it
> returns.

### Claim skeleton

- Two functionally independent populations of non-grid spatial cells exist in MEC: one co-localised with (and coordinated with) the medial grid network, one distal and dissociable.
- They are hard to tell apart by coding properties — both are spatial — and are separated only by their *coordination* with the grid network, exposed when grid firing decouples from the task reference frame.
- Implication: MEC is not one coherent network. It implements multiple spatial computations in parallel.
- Situates against the classic MEC/LEC division, which is already known to be more graded than the textbook version.
- Open question: does the functional organisation correspond to cellular/molecular gradients? (Layer 1/2 width differences; some intercellular signalling gradients; minor ML intrinsic-property differences relative to the strong DV gradients.)
- Broader frame: spatial cognition is not unitary; multiple systems operating independently but coordinating. Grid system = CAN-based, available on first exposure. Second system = ramp-like dynamics, possibly learned, possibly tracking task-specific reference frames.
- Take-homes: power of integrated population analysis across multiple task conditions; parallel computation in cognitive circuits.
- Future experiment: region-specific (medial vs lateral) lidocaine inactivation for dissociable deficits — ties back to Sarah's original paper.

---

*Full draft text:*

Our results identify two functionally independent populations of non-grid spatial cells in the MEC. The first population co-localizes with grid cells and is enriched in the medial MEC. The second population is located more distally from the high-density grid cell area. Distinguishing these two populations is challenging, as both encode spatial localization. We differentiated the two populations when the grid representation and the associated non-grid cell population decoupled from the task reference frame. Our results suggest a new view of the MEC as implementing multiple systems for spatial computation, and suggest how spatial cognition could take advantage of parallel and independently generated neural representations of location.

### Multi-domain organisation of the MEC

The classic view of the entorhinal cortex divides it into two functionally distinct domains: the MEC and the LEC. Traditionally, the MEC is thought to contain spatial cells, including grid cells and to provide a cohesive, coordinated representation of location. By contrast, in the classic view the LEC contains non-spatial cells, encoding content information that becomes integrated with spatial signals within the hippocampus. Recent experimental evidence suggests a more complex organization. Spatial neurons are also present in the LEC, although they are located relatively far from the border with the MEC, while in the MEC, two-photon imaging studies have identified enriched populations of grid cells near the medial border adjacent to the parasubiculum. Our results align with these findings, showing that grid cells are enriched medially, but also that the non-spatial grid cells form multiple functionally independent networks. Non-grid spatial cells found near the enriched grid cell area coordinate with grid cells, whereas those located more laterally in the MEC can be functionally dissociated from the grid cell network. Therefore, similar to the LEC, which exhibits a topographic organization of representations, our results model for the MEC in which it generates multiple spatially organised representations.

A foundational question raised by our results is whether this functional organization corresponds to cellular or molecular properties of neurons in the medial entorhinal cortex (MEC). Although the borders of the MEC are well-defined and its layered organization is consistent along the medial-lateral axis, there are differences in the architecture of its cell layers. Specifically, layer 2 is wider and layer 1 narrower near the medial border with the parasubiculum. At the molecular level, many genes exhibit constant expression along the medial-lateral axis of the MEC; however, some intercellular signaling molecules display gradients between medial and lateral regions. At the level of cell physiology, mediolateral differences in intrinsic properties have been reported, though these differences are relatively minor compared to the well-established dorsal-ventral gradients. A critical future experimental goal is to relate molecular, electrophysiological, and circuit properties of functionally identified neurons to their functional identity.

### Parallel computational systems for spatial cognition

The existence of multiple systems in the medial entorhinal cortex (MEC) related to spatial cognition aligns with the broader perspective that spatial cognition and memory are not unitary processes. Instead, they are supported by multiple systems that operate independently yet coordinate with one another. In the case of the MEC, it is plausible that multiple computational strategies are advantageous for estimating location. The grid system computes location using a neural circuit organized around a continuous attractor. Within this system, non-spatial cells support the operation of the attractor or provide readouts that facilitate downstream decoding. The second system identified here exhibits distinct dynamics, characterized by neurons with ramp-like firing fields, which have also been observed in mental navigation tasks. A critical unresolved question is whether these ramping dynamics exclusively encode space or represent more general information. The primary function of these systems may be to track location within task-specific reference frames. It is possible that this complementary system generates learned representations, whereas the grid system provides representations available upon first exposure to an environment. Together, these complementary systems may enhance robustness and support different mechanisms for generalization.

### Big-picture take-home *(to write)*

- Power of integrated population analysis across multiple task conditions.
- Parallel computation in cognitive circuits.

> **TODO — Discussion:**
> - "the non-spatial grid cells form multiple functionally independent networks" — garbled; should read "the non-grid spatial cells".
> - "our results model for the MEC" — missing a verb ("our results **support a** model for the MEC").
> - Discussion is silent on the **limits** of the claim. Needed: the medial/lateral split is operational (probe geometry), not a defined anatomical border; the dissociation is demonstrated in one task under one manipulation (anchoring loss); no causal evidence.
> - The lidocaine future-experiment paragraph is in the figure plan but not yet written into the Discussion.

---

### Alternative / earlier Discussion material

> **NOTE:** The block below overlaps heavily with the Results and reads as an earlier draft of the Discussion or of a Results-summary. Retained here so it isn't lost; not currently placed in the manuscript. It contains named citations (Brun 2008; Sun 2015; Rowland 2018; Tennant 2018) that the main text lacks and should inherit.

Previous studies have established that grid cells are organized along the dorsoventral axis of the MEC, with grid scale increasing from dorsal to ventral regions (Brun et al., 2008). In contrast, the organization of grid cells along the medial-lateral axis has often been assumed to be homogeneous. Recent anatomical evidence, however, suggests that grid cells may be preferentially located near the border between the MEC and the parasubiculum, challenging the notion of medial-lateral uniformity (Sun et al., 2015; Rowland et al., 2018). By precisely registering the anatomical locations of recorded neurons to standardized brain atlases, we find a distinct spatial segregation: grid cells predominantly reside in the medial MEC, whereas non-grid cells are evenly distributed. Given this anatomical distinction, we then asked whether the two populations could be functionally dissociated.

To further investigate the relationship between the firing activity of each neuronal population, we investigated the extent to which one neuron's firing activity could be predicted based on the firing activity of another while mice engaged with a linear track navigation task that requires the MEC (Tennant et al., 2018). Using xgBoost, a state-of-the-art machine learning method for predicting time-series using gradient-boosted trees, we asked how well firing activity could be predicted from a range of covariates including position alone, the activity of other neurons or a combination of position and other neurons firing. We found that both grid cells and non-grid cells firing could be predicted by position and this prediction was improved with the addition of either cell group. Expectedly, when using the activity of grid cells as covariate in combination with position, firing of comodular grid cells was best predicted, followed by non-comodular grid cells and non-grid spatial cells. On the other hand, when using the activity of non-grid spatial cells as covariate in combination with position, predicted firing of grid cells outperformed predicted firing of non-grid spatial cells. This was observed with as few as a single neuron in the covariate up to the maximum number of corecorded spatial cells. To determine what factors influence the predictability of neural firing based on that of another, we considered if anatomical location or other spatial coding properties were correlated with improved predicted firing. We found euclidean distances between any two target and covariate cells were well correlated and that… **[incomplete]** Furthermore, spatial coding properties such as grid score, theta index and firing rate **[incomplete]**

Taking advantage of epochs of task disengagement which correlates with non-anchored grid representations (task independent firing), we asked whether non-grid spatial cells' firing could still predict grid firing. **[incomplete]**

---

## Figures

The scheme is now single and consistent. The legacy captions — written when the paper was organised around the mediolateral axis and the xgBoost coupling analysis, and carrying their own duplicate numbers — have been removed along with the Results sections that cited them.

| # | figure | **builder (authoritative)** | PDF | exploratory notebook |
|---|---|---|---|---|
| 1 | the population anchoring state and behaviour | `fig1A_stack.py` | **`fig1_v2_M26D18.pdf`** (the figure proper); `fig1A_stack.pdf` (schematic) | `figure1_schematics.ipynb` |
| 2 | which single units follow it | `fig2_single_units.py` | `fig2_single_units.pdf` | `figure2_single_units.ipynb` |
| 3 | where those cells sit | `fig3_anatomy.py` | `fig3_anatomy.pdf` | `figure3_anatomy_identity.ipynb` |
| 4 | the local circuit, and the null that pivots the paper | `fig4_monosynaptic.py` | `fig4_monosynaptic.pdf` | `figure4_monosynaptic.ipynb` |
| 5 | engagement: pupil constriction tracks the state | `fig5_pupil_arousal.py` | `fig5_pupil_arousal_M21D19.pdf` | `figure5_pupil_arousal.ipynb` |
| 6 | one state, everywhere; the specificity is cellular | `fig6_cross_region.py` | `fig6_cross_region.pdf` | — |
| 7 | band-specific LFP change; theta decouples from speed | `fig7_theta_gamma.py` | `fig7_theta_gamma.pdf` | — |

> **The `.py` script is the authoritative builder for every main figure, not the notebook.** This column was previously headed "notebook" and named `figure3_anatomy_identity`, `figure4_monosynaptic` and `figure5_pupil_arousal` for Figures 3–5, which is wrong: those notebooks are where the analyses were developed, but the PDFs the paper uses are written by the `fig*.py` scripts, and the scripts are what is version-controlled as the figure source. They were re-run from a clean interpreter on 2026-10-03 to verify this (see *Reproducibility* below for which passed). Where a notebook is still listed it is for provenance only — do not rebuild a figure from it and expect the paper's numbers.

> **Figure 7 is now `fig7_theta_gamma.py` → `fig7_theta_gamma.pdf`** (post-gate, rebuilt 2026-10-03), with `fig7_supp_band_examples.py` as its supplement. The old `fig6_theta_panel.py` → `fig6_theta_anchoring.pdf` is pre-gate and holds the speed-decoupling panels that the new figure does not yet carry; it must be rebuilt, not cited. Figure 6 was rebuilt from scratch on 2026-10-02 as `fig6_cross_region.py`; the older `fig1_supp_nonmec.py` is superseded by it and should be retired once nothing else reads its outputs.

### Reproducibility: build order, and what was verified

Every main figure and supplement is built by a standalone `.py` script that can be run from a clean interpreter — `python scripts/figures/AnchorDynamics2026/<script>.py`, no notebook kernel and no manual cell execution. Several scripts bootstrap shared helpers by extracting marked cells out of `lick_raster_by_trial_type.ipynb` and `exec`-ing them, so that notebook is a **build dependency** of the figures that do so (`fig1_supp_rasters.py`, `fig1_supp_nonmec.py`, `fig5_pupil_arousal.py`, `fig6_region_examples.py` and `region_row.py`, hence `fig6_supp_region_examples.py`). That coupling is the most fragile thing in the pipeline: editing a helper cell in that notebook silently changes several figures, and it has caused one real error already — a constant named `MIN_CELLS` in the notebook overwrote a differently-intended `MIN_CELLS` in a figure script, so session selection ran at 25 cells rather than 10 (fixed by renaming to `MIN_REGION_CELLS` and declaring it *after* the exec).

**Build order.** Nothing downstream rebuilds its own inputs, so the stages must be run in order. Stages 1–3 are fast; stage 4 needs the external LFP volume mounted.

| stage | run | produces |
|---|---|---|
| 0 | `scripts/figures/figure_0_classification/create_classification_datasheet_v2.ipynb` | `data/cell_classifications_v2.csv` — open-field cell identity, the root of everything |
| 1 | `build_anchoring_labels.py` | `anchoring_cells*.csv`, `anchoring_trials*.csv` — the gated classifier labels |
| 2 | `build_trial_table.py`, `build_trial_maps.py` | `trial_table.csv`, `trial_map_stats_w*.csv`, `trial_map_null.csv`, `trial_rate_speed_w*.csv` |
| 3 | `build_per_cell_pc1.py`, then `build_unit_table.py`; `build_heldout_pc1.py`, `region_axis_coherence.py` | `per_cell_pc1.csv` + `pc1_by_region.csv`, `unit_table.csv`, `pc1_heldout.csv`, `region_axis_coherence.csv` |
| 4 | `lfp_spectrum_pac.py` (worker/n\_workers args), `gamma_events.py` | `spectrum_by_anchoring_w*.csv`, `pac_by_anchoring_w*.csv`, `gamma_events.csv`, `gamma_by_position.csv` |
| 5 | `cache_rnn_of_ratemaps.py` | the RNN 2D rate-map caches used by the Figure 1 supplement |
| 6 | the `fig*.py` scripts | the PDFs |

`build_per_cell_pc1.py` must precede `build_unit_table.py`, which reads `per_cell_pc1.csv`. Stage 0 is the one the paper's outstanding `n = ?` cell counts are waiting on.

**Verified 2026-10-03.** Re-run from a clean interpreter, in a single pass, with the logs checked rather than only the exit codes:

| script | result |
|---|---|
| `fig1A_stack.py`, `fig2_single_units.py`, `fig3_anatomy.py`, `fig4_monosynaptic.py`, `fig5_pupil_arousal.py`, `fig6_cross_region.py`, `fig7_theta_gamma.py` | all rebuilt their PDFs |
| `fig1_supp_classifier.py`, `fig1_supp_rasters.py`, `fig1_supp_speedmatch.py`, `fig2_supp_nonanchored.py`, `fig2_supp_rate.py`, `fig2_supp_reliability.py`, `fig6_region_examples.py`, `fig6_supp_region_examples.py`, `fig7_supp_band_examples.py` | all rebuilt their PDFs |

No tracebacks in any log. **Exit codes were not treated as sufficient**, deliberately: two scripts return 0 in under three seconds, and this pipeline has already produced one case of a script logging success on empty results (the LFP batch, when the external volume unmounted mid-run, reported "wrote …" for all four workers). Each log was read for a `saved`/`wrote` line naming the PDF.

> **Two things the audit found, both now fixed.**
>
> **`fig7_supp_band_examples.py` read a different file from its parent figure.** The supplement read the merged `data/lfp/spectrum_by_anchoring.csv` while `fig7_theta_gamma.py` reads the worker shards `spectrum_by_anchoring_w*.csv`, and **there is no merge step connecting them** — `lfp_spectrum_pac.py` writes shards when given worker arguments and the un-suffixed file when run single-threaded, so the merged file is whatever a previous run happened to leave behind. The two were identical (28 sessions, 14,280 rows each), which is why nothing looked wrong, but a re-run of the shards without a matching single-worker run would have left the supplement quietly disagreeing with the figure it supports. The supplement now reads the shards.
>
> **The figure index named notebooks as the source for Figures 3–5.** `figure3_anatomy_identity.ipynb`, `figure4_monosynaptic.ipynb` and `figure5_pupil_arousal.ipynb` are where those analyses were developed; the PDFs are written by `fig3_anatomy.py`, `fig4_monosynaptic.py` and `fig5_pupil_arousal.py`. The index now names the scripts and lists the notebooks as provenance only.

> **One stale configuration, not a failure.** `fig5_pupil_arousal.py` builds two example sessions, M21 D19 and M29 D17. M29 D17 is no longer a valid two-state example under the gated classifier — it has no non-anchored trials at all (anchored fraction 0.67 across 182 trials) — and the script declines it with that message rather than drawing a one-state panel. Nothing in the paper cites an M29 D17 panel, so no figure is affected, but the second example should either be replaced with a session that still switches or removed.

> **Not reproducible from scratch within this repository:** the `data/lfp/PREGATE/` analyses listed in Results §7 (they need re-running, which is the point of that warning), and stage 0, which lives outside this figure directory. The LFP stages additionally require the external volume `clark2025_lfp`; `fig7_theta_gamma.py` and `fig7_supp_band_examples.py` read raw LFP directly and take minutes rather than seconds for that reason.

### Figure 1. Recording grid and non-grid cells in the MEC

- **(A)** Recording and experimental paradigm.
- **(B)** Simultaneously recorded grid and non-grid cells in open field and virtual reality linear track environments.
- **(C)** Mice were implanted with Neuropixel 2.0 probes loaded on a reusable APOLLO drive assembly. Recording sites were recorded from at least twice and mice were exposed to an open field arena and a virtual reality location memory task to record neural correlates of behaviour.
- **(D)** Behaviour from open field and virtual reality location memory task.
- **(E)** Firing rate maps in open field and virtual reality location memory task.

> Versions v2 and v3 exist in the source but are unannotated. The figure-plan version of Figure 1 additionally calls for: a **schematic of the competing hypotheses** (unimodal vs multimodal anatomical arrangement) and a panel on spatial representation differences across the dorsal–lateral axis. The hypothesis schematic is referenced by the Introduction as *Figure 1a–b* but is not in the caption above — **missing panel**.

### Extended Figure 1. High density Neuropixel 2.0 recordings in open field and virtual linear track environments

> Referenced from Results §2 as Extended Fig. 1A, B (distribution of spatial vs non-spatial cells). Caption needs writing.

### Figure 1 supplement. The anchoring classifier, step by step, against RNN ground truth

`fig1_supp_classifier.py` → `fig1_supp_classifier.pdf` (2D maps cached by `cache_rnn_of_ratemaps.py`)

Every anchoring result in the paper rests on the trial-correlation classifier being right, and on recorded data there is no ground truth — a cell's firing either does or does not match its own template, and nothing independent says which answer is correct. A path-integrating RNN is the only place accuracy can be **measured** rather than inferred, because there the anchoring state is something we impose. The classifier is imported from `spatial_manifolds.anchoring` rather than reimplemented, so the figure cannot drift from the pipeline.

**A–D, how the task is recreated.** net7 (Ng = 1024, trained purely by path integration in a 2.2 × 2.2 m box) is run along a fixed straight 200 cm chord through that box — chosen to cross about three firing fields of its best grid units — and teleported back to the start at the end of every trial: the linear-track paradigm and its trial structure, in a network that never saw a track (**A**, with one trial's trajectory; **C**, four trials with the teleport resets dotted). **B** is the point of the construction: a grid unit's periodic firing along the track *is* a slice through its hexagonal field arrangement, not a separately specified 1D property.

The anchoring manipulation is the teleport. In condition **A (cued)** the hidden state is re-anchored at each teleport by a place-cell initial activation, exactly as at the start of every training sequence — the network is told where it is. In condition **C (blind)** the teleport still happens, position really does reset, but the network is never told: no re-anchoring, and the teleport step itself is fed as zero velocity, so there is no self-motion information at the jump. The hidden state carries over uncorrected and the representation drifts off the track reference frame. One continuous session runs A (35 trials) → C (30) → A (35), so `phase_of_trial` is a true per-trial anchored/non-anchored label and the classifier never sees it. **D** shows the manipulation worked, in decode error, before any classifier is applied.

**E–H, the classifier, one step per panel, on one cell.** (1) the trial × position rate map the classifier starts from; (2) the trial × trial correlation matrix, in which the blind block appears as a decorrelated 30 × 30 block; (3) each trial's mean correlation with the others, against the **2-means centres** and the cell's **own circular-shift null**, which is the gate; (4) the label sequence against truth, before and after the median filter. The NaN-aware interpolation over the 22% of unvisited position bins is not given a panel: it is a prerequisite of having a rate map at all, rather than a decision the classifier makes.

> **The walkthrough cell is chosen to stay active through the blind block, and that constraint is not cosmetic.** The highest grid-score units go nearly silent during C, so their blind-block trials fall below the classifier's activity floor and are force-labelled non-anchored *before* the correlation step is reached. They score 1.000, but the correlation matrix, the 2-means split and the gate never decide anything, which makes them useless for illustrating steps 2–4. Unit 413 keeps firing throughout and is decided by **decorrelation**, which is the mechanism the paper relies on.

**I, J, does the gate earn its place?** **The A → C → A session cannot answer this**, and the figure says so rather than implying otherwise. That session is a 70/30 split, which is exactly where plain 2-means already works: there really are two groups of trials and the split it is forced to make is the right one. Ground-truth accuracy is 0.990 with the gate and 0.990 without it, so that number carries no information about the gate. The gate exists for the cases the session does not contain — a cell locked in **one** mode, and lopsided splits — because 2-means always returns two clusters, so a cell anchored on every trial is still cut roughly in half, and the median filter then launders the salt-and-pepper into runs that look like state structure.

Sessions of 30 trials were therefore spliced from the switch session's own pools (70 cued, 30 blind) as *m* anchored trials followed by 30 − *m* non-anchored ones, kept as two contiguous blocks rather than interleaved, because real states occur in blocks and the median filter interacts with run length. **I** plots reported against true anchored fraction: ungated, a session that is **0% anchored is reported as 50.6% anchored**, and the ungated curve is flat near 0.5 across the whole lower range; gated, it tracks the identity line. **J** shows per-trial accuracy over the same range, with the band marking where the A → C → A session sits — the two curves are indistinguishable there and separate only below ~0.25. Across the sweep, mean |reported − true| fraction error — the quantity the paper actually reads, since the population state is a per-trial fraction — falls from **0.181 without the gate to 0.049 with it**.

**K, what the median filter costs.** The filter exists to remove isolated false-positive *anchored* labels inside runs of genuinely non-anchored trials, which arise when a drifting periodic cell happens to align in phase with other non-anchored trials. It works: median per-unit accuracy rises 0.950 → 0.990, bought entirely in **specificity** (0.809 → 0.885) at no cost to sensitivity (0.993 → 0.993), so it corrects the error it was built for rather than trading one error for another. Those numbers are reported rather than plotted; the panel shows the price instead, because the price is what constrains the paper.

**K is the panel that constrains what the paper may claim.** Sessions of varying non-anchored epoch length are spliced by placing the **first** *k* trials of the C block between the two A blocks — the principled choice, since the hidden state drifts progressively during C, so the first *k* C trials are exactly what a *k*-trial cue loss would have produced; splicing later C trials would sample a more-drifted state than a short epoch ever reaches and would flatter the classifier. A size-5 median filter erases any run shorter than 3 trials **by construction**, and recall on the non-anchored epoch collapses accordingly (0.00 at 1 and 2 trials, 0.55 at 3, 0.79 at 5). The paper therefore makes no claim about single-trial or two-trial anchoring dynamics, and **the ~3-trial transition width reported in Figure 1 is at the floor of what this classifier can resolve rather than a measured biological time constant.**

> **Headline numbers.** 1,024 of 1,024 units classified; median accuracy **0.990** (IQR 0.950–1.000); 911 units (89.0%) at ≥ 0.90; sensitivity 0.993, specificity 0.885, so misses are almost entirely non-anchored trials called anchored rather than the reverse.
>
> **One caveat that should be stated rather than buried.** Accuracy is not independent of gridness: Spearman ρ = **+0.189** (*p* = 1.0 × 10⁻⁹) across classified units. The effect is small against a median accuracy of 0.990 and every quintile sits far above chance, but the classifier was adopted partly on the claim that it is general rather than grid-tuned, and on this test that claim needs softening to *works across the gridness range, slightly better on grid-like cells*. Low-gridness RNN units are also not literally ramping cells, so this is a weaker test of generality than recorded non-grid spatial cells would give — it is simply the only one where ground truth exists.


### Figure 2. Which single units follow the population anchoring state

**A, B** Two example sessions (M26 D18, M28 D25). For each: the full MEC anchoring raster (cells in PC1-loading order on *x*, trials on *y*), PC1 of that raster, then one example unit from each identity — grid, non-grid spatial, non-spatial, putative interneuron — and two **locked** units, one always anchored and one never anchored, labelled with their own identity. Rate maps are trial × position, NaN-aware smoothed along position only (σ = 2 bins), peak rate annotated. Every panel in a row shares the trial axis. Dashed lines mark the major population transitions, taken as the thresholded population state after a 9-trial median filter, as in Figure 1; the filter prevents short flickers from covering the raster in lines.

**C** How the per-cell null is built, worked through on one real cell: its label sequence, a circularly shifted copy, and the resulting null distribution of |*r*| with chance and the observed value marked.

**D** Agreement with the population axis as excess over that null, by identity, with mixed-model tests against chance (mouse and session random effects) and the count of mice above chance.

**E** Fraction of cells individually beating their own null, per mouse, against the 5% false-positive rate of the same test.

**F** What each identity does in the trials around a population switch (−5 to +5), each cell timed against a population state recomputed without it.

**G** Agreement *between* identities as a 4 × 4 matrix, reported as excess over an independent circular-shift null. Grid–interneuron reaches +0.109 against +0.058 for the other off-diagonal pairs — essentially the within-group level (+0.115).

**H** Locked cells, in the 35 sessions where the population switches: their identity composition, the always:never ratio per mouse, each locked group against the base rate, and open-field spatial information by lock state.

*Supplements:* **S1** what a non-anchored trial is, with the best-offset null; **S2** firing rate by state across three measures; **S3** split-half reliability of the agreement score; **S4** anatomical organisation of following. The interneuron classification and the monosynaptic evidence that the group is real are now **Figure 4**.

### Figure 3. Grid cells are enriched in the medial MEC, and so is following

`figure3_anatomy_identity.ipynb` → `fig3_anatomy.pdf`

**A** The gradient shown rather than summarised, with the recording sequence keyed at left (OF 1 → VR → OF 2, OF 1 highlighted): every map comes from OF 1, the arena session *preceding* the task, which is what keeps cell identity independent of the behaviour it is later used to explain. Open-field rate maps from four M25 sessions whose probes sat at different mediolateral positions (≈3000, 3270, 3540 and 3720 µm), 24 cells each, sampled **proportionally** so that the grid:non-grid ratio drawn is the ratio actually recorded in that session rather than a hand-picked illustration. Grid cells are taken in grid-score order (red borders), non-grid spatial cells spread evenly across the dorsoventral extent (blue borders) so the block is not all one depth. The medial session yields 10 grid cells of 24; the lateral session yields 1.

**B, C, D** Best-fit slices through the recorded cells — M25, M29 and M28, three mice that each sample in to the medial end — each under a 3D view of that animal's probe in MEC (teal), with the contacts in orange. The plane is fitted to **device contacts**, not to cell locations, because per-cell mediolateral position is a per-shank nominal plus jitter and fitting to it would fit the approximation rather than the probe geometry. Both axes are micrometres and the panels are drawn to **equal aspect**, so the slice has the proportions of the tissue. Dashed line: the 3400 µm medial/lateral boundary.

**E, F** Fraction of spatial cells classified as grid, against mediolateral and dorsoventral position, binned by octile with SEM. Titles give the **within-session** Spearman correlation, which is the test that matters.

**G** Superficial (L1–3) against deep (L5–6), paired within session; grey lines are sessions.

**H** The gradient in one session. The M25 D25 anchoring raster (213 cells, 4 shanks, 720 µm span) with cells ordered by mediolateral position rather than by PC1 loading, so the columns read as shanks, beside each shank's mean agreement with the population axis: 0.31, 0.53, 0.36 and 0.04 from medial to lateral, giving ρ = −0.255, *p* = 0.0016 — more than twice the dataset median. This is the single-session version of panel H. **Read the anchored fraction in the title before reading the raster:** the session is anchored on most of its trials, so most of the raster is one colour and the panel shows how cells are sampled across shanks and how agreement falls laterally, not a strong state alternation.

**I, J** Mediolateral and dorsoventral position against agreement with the population anchoring state, pooled and within session. The pooled mediolateral correlation is large (ρ = −0.15, *p* = 2 × 10⁻²⁵) and is **not** the test: probe placement differs markedly between animals, so a cell's mediolateral coordinate carries information about which session it came from. The within-session median printed beside it is the test.

**K** The within-session correlations for all three axes side by side — mediolateral, dorsoventral and probe depth. Only mediolateral differs from zero.

**Session selection for the mediolateral tests.** Mediolateral position is assigned per shank, so a single-shank session has no mediolateral variation and contributes only noise; 33 of 61 sessions span under 50 µm. Panels D, H and the M–L entry of J therefore use the **27 sessions spanning at least 200 µm**. For dorsoventral position and probe depth the criterion changes nothing (*p* = 0.99 and 0.92 either way), because every session varies along the shank — which is what shows the mediolateral result is specific to the axis rather than produced by the restriction.

> The regional comparison that was panel **J** of this figure — per-session agreement by region, scored as excess over each cell's own circular-shift null — is now **Figure 6**, where it belongs with the cross-region argument.

### Figure 4. The local circuit: wired together, but the wiring does not carry the state

`figure4_monosynaptic.ipynb` → `fig4_monosynaptic.pdf`

**A** The classification: peak-to-trough duration against mean firing rate for all 8,300 cells with a waveform, with the marginal distributions and both thresholds. Putative interneurons (orange) are the cells that are narrow **and** fast — 830 cells, 10% of the population. The waveform boundary is a fixed **0.4 ms**, a convention rather than a refitted quantity: an antimode refitted per dataset moves with whichever cells are supplied, so sessions, mice and reanalyses would each get a slightly different boundary and cells would change class for reasons unrelated to the cells. The unsupervised two-component fit is kept as corroboration and drawn beside it — components 0.25 and 0.69 ms separated by 4.0 SD, antimode at 0.423 ms, within one histogram bin of 0.4 — and the empirical density minimum lies at 0.38–0.40 ms, so 0.4 is if anything the better cut as well as the conventional one. The choice is near-inconsequential in any case: 0.423 against 0.400 moves 12 cells of 842. The rate threshold is the top quartile (10.2 Hz), likewise a convention, because log firing rate separates at only 1.8 SD and is not usefully bimodal; it is conservative, rejecting 610 of 1,440 narrow-spiking cells, which revert to their open-field class.

**B, C** What each kind of cell does to its targets, against the hollow-Gaussian predictor with the 0.7–4.7 ms causal window shaded: a cell classified as an interneuron in A produces a causal **trough**, a broad-spiking cell a causal **peak**. Insets magnify ±5 ms, the region the test is computed on; the wide view is kept alongside because it is what shows the predictor tracking the slow structure, and therefore that the deviation is local.

**G** The two connection types are kinetically distinct — inhibitory latencies are slower than excitatory ones (median 3 against 1 ms), shown as paired bars at each latency, each sign normalised within itself because inhibitory connections outnumber excitatory ones seven to one and the comparison is of shape rather than count.

**D** **The validation.** Presynaptic enrichment by identity, relative to each group's share of the classified population. Inhibitory connections originate from the cells panel A called interneurons at 3.1× their share; excitatory connections do not. This is an independent test: the detector operates on spike-train cross-correlograms and has no access to waveform shape, firing rate, or the class labels, so there is no route by which an arbitrary threshold in A could produce this.

**E, F** Connection probability between identity groups for each sign — the independent, spike-timing counterpart to the trial-level agreement in Figure 2G, as a percentage of the ordered pairs testable in each session: inhibitory, where `int → grid` is the strongest pathway in the dataset (3.93%, 5.1× over within-session identity shuffling, *p* = 0.0005), and excitatory, where `grid → grid` leads at 0.217% (2.6×, *p* = 0.004). Each matrix sits directly below the example correlogram of its own sign, so columns 2 and 3 of the figure each read as one sign from a single pair to the whole population. Both are shown because neither is interpretable alone — the excitatory matrix is an order of magnitude sparser throughout, and only the comparison shows this to be a property of the sign rather than of the identities. Each is scaled to its own maximum, since a shared scale renders the excitatory matrix uniformly black.

**H** Detection validated against a ±10 ms spike-jitter null, per session and per sign: FDR 0.09 excitatory, 0.10 inhibitory.

**I** Whether the wiring carries the anchoring state. Anchoring-label correlation for connected pairs (filled) against unconnected pairs matched for session, identity pair and probe distance (open, ±40 µm), with the minimum detectable effect at 80% power printed beside each. Grid–interneuron and interneuron–interneuron are informative nulls (MDE 0.045 and 0.043 against an identity effect of ~0.11); grid–grid (*n* = 27, MDE 0.157) is underpowered and is greyed and labelled as such rather than counted as a null.

### Figure 5. The anchoring state is a reflection of attentional engagement

`figure5_pupil_arousal.ipynb` → `fig5_pupil_arousal_M21D19.pdf`

> **CAPTION TO WRITE.** The figure exists and is built from the script; the panel-by-panel caption has not been written. Content, in the order the figure presents it: an anchored and a non-anchored trial shown as seven moments of a single traversal with the DeepLabCut landmarks drawn on, beside mean pupil radius against track position; the MEC population anchoring raster with PC1 and the pupil-radius heatmap sharing one trial axis; pupil dilation by state; dilation around state transitions (−5 to +5 trials); and z-scored dilation against track position for the two states.

Key values: pupil radius differs between states by ≈ 0.24 SD, and changes at transitions by −0.414 entering the anchored state and +0.493 entering the non-anchored state (*p* = 1.7 × 10⁻⁷ and 4.4 × 10⁻⁸).

> **A panel is missing and the figure needs it, because it is what fixes the interpretation.** Hit rate by decile of within-session pupil *z* — 0.826, 0.712, 0.659, 0.582, 0.552, 0.534, 0.472, 0.481, 0.440, 0.391 across 14,011 trials in 58 sessions — with the anchored fraction on a second axis, falling with it. Without this panel the figure says only that the pupil constricts in the anchored state, which a reader will take as "the anchored state is less aroused"; with it, the figure says that performance improves monotonically as the pupil narrows, so the constricted state is the **focused** one. Add the quadratic fit to the same panel: the coefficient is positive with a minimum at *z* = +3.4, outside the data, so there is no inverted U in the observed range and the figure should not leave room for one to be assumed. Mean running speed is flat across the deciles (37–40 cm/s) and is worth printing on the panel for the same reason.

### Figure 6. One state, shared across structures, expressed most strongly by one cell type

`fig6_cross_region.py` → `fig6_cross_region.pdf`
(inputs: `build_heldout_pc1.py`, `region_axis_coherence.py`)

**A** One session recording all three structures (M25 D25; 72 entorhinal, 71 subicular, 68 visual cells), with the per-trial anchored fraction computed separately from each. The state is visible in all three and they move together — MEC–subicular *r* = 0.93, MEC–visual *r* = 0.65.

**B** **Does each region have a state of its own?** Split-half reliability of each region's own first principal component: the axis built twice from disjoint halves of that region's cells and correlated. Size-matched to 8 cells per half, since reliability rises with cell count and MEC contributes hundreds of cells where visual cortex contributes tens. Every region has one — MEC +0.447, subicular +0.474, **visual cortex +0.330** (*p* = 2.4 × 10⁻⁵). This question could not be asked while every region was scored against the entorhinal axis.

**C** **Is it the same state?** A raw correlation between two axes confounds *different state* with *no state*, because an axis from few noisy cells is itself noisy. Dividing by the geometric mean of the two reliabilities separates them. Corrected, MEC–subicular +0.920 and MEC–visual +0.863, neither different from 1 (*p* = 0.098, 0.515), and not different from each other (*p* = 0.66). *The correction amplifies noise by ≈ 2.6× for the visual pair — CI [0.49, 1.23], 8 of 19 sessions above 1 — so this excludes differences above ~0.3, and is not a demonstration of identity.* SUB–VIS (4 sessions) is plotted and not tested.

**D** Per-cell expression of the state, by group, with **every cell scored out-of-sample**: the axis is built by SVD from a random half of the entorhinal cells and the other half, together with all non-entorhinal cells, scored against it, over 8 splits. Leave-one-out would not do — it removes a cell's own contribution but still measures an entorhinal cell against an axis built from its correlated network, while subicular and visual cells were never in the axis at all. Chance is zero.

**E** **The panel the figure turns on.** Three advantages over visual cortex, each paired within session: grid cells **+0.149** (*p* = 9.9 × 10⁻⁶), other entorhinal cells +0.050 (*p* = 1.6 × 10⁻⁴), and the region-level axis reliability of MEC over visual cortex +0.083 (*p* = 0.065). The cell-type contrast is three times the regional one, and the regional one does not reach significance. The specificity is cellular, not anatomical.

**F** The Anchoring Dependence Index — AUC(uncued) − AUC(cued) for the state predicting a hit — by region. A state read from **visual cortex** carries +0.148 (*p* = 0.0014), larger than the entorhinal state's +0.085 (*p* = 0.0015). Consistent with the ADI being unchanged under an MEC-only population (*p* = 0.81), and expected if the state is one global variable legible from any sufficiently sampled structure.

> **Chance is zero, and two earlier versions of this figure had it elsewhere.** See the correction in Results §6: subtracting the mean of an *absolute* null from a *signed* correlation biases every cell down by ≈ 0.8 SD of its own null, differentially by region, which produced the retired claim that visual cortex sat below its own null. The null's width sets significance, not a baseline.


### Figure 6 supplement. The same state, described separately by each structure

`fig6_region_examples.py` → `fig6_region_examples.pdf`

Figure 6's statistics are abstract — split-half reliabilities, correlations between principal components corrected for attenuation — and are easier to believe, and easier to doubt in the right places, after seeing the states themselves and the single cells underneath them.

**One session per row**, read left to right as three structures — entorhinal, subicular, visual — each contributing a raster and two cells, with the three components overlaid at the right. Everything within a structure's group is built from that structure's own cells: the raster's cells are ordered by **that region's** PC1 loading, and the component is signed to **that region's** own anchored fraction. Nothing is scored against the entorhinal axis, so agreement between the three groups is agreement between independent descriptions of one session rather than a consequence of a shared frame. Magenta = anchored, teal = non-anchored; trials on the vertical axis throughout, as in Figures 1 and 2, and the fraction of label-matrix variance each component explains is printed above its raster.

**The two cells per structure** are those with the largest PC1 loading in it, shown as trial × position maps, each with that cell's own anchored/non-anchored labels as a strip immediately to its left. They show what the raster summarises: firing in register across anchored trials and out of register across non-anchored ones.

**The rightmost panel** overlays the three *z*-scored components on the shared trial axis, coloured as the structure titles in the same row.

> **The two sessions bracket the range; they do not represent it.** Of the four sessions recording all three structures with ≥ 20 label-varying cells each, these are the strongest (**M26 D19**, entorhinal–visual +0.82) and the weakest (**M28 D25**, +0.15) agreement. Showing the two best would misrepresent Figure 6C, whose disattenuated mean of +0.863 carries a confidence interval of [0.49, 1.23] precisely because sessions differ this much. The threshold is 20 rather than 10 deliberately: at 10 the worst session reaches −0.49, but it has 10 subicular and 21 visual cells, so its axes are barely estimated and bracketing on extremes is the selection most vulnerable to that. The population statistics are unaffected — they use every qualifying session and correct for reliability explicitly, which is what a single session cannot do.

> **A visual cell's map is not a place field, and the figure should not be read as claiming it is.** The classifier operates on the trial-to-trial similarity of a cell's position-binned profile, whatever produces that profile; in visual cortex on a virtual track that is largely the position-locked visual scene rather than an allocentric field. The claim is that those profiles are consistent on the *same trials* in visual cortex as in MEC — not that visual cells carry a spatial code of the entorhinal kind.

> Two further things are visible and both matter for how Figure 6 is read. The **visual rasters are noisier**, with fewer cells carrying the block structure cleanly, which is the reliability difference (+0.330 against +0.447) that panel C divides out — seeing it is the best guard against reading "the same state" as "indistinguishable data". And **PC1 explains only 8–20% of the variance** in these label matrices, so the component is a dominant shared mode rather than a complete description of the population.


### Figure 6 supplement 3. One axis is an adequate description of the state

`region_state_dimensionality.py` → `fig6_supp_dimensionality.pdf`
(`--replot` rebuilds the figure from the saved CSV without repeating the resampling)

Figure 6 establishes that each region has a coherent axis and that the axes agree. It does not establish that a *single* axis describes a region's label matrix, which is what the phrase "the population anchoring state" assumes. Both panels score **out of sample** — the axis is built from one half of a region's cells and evaluated on the other half — because in-sample variance explained rises as cell count falls and would rank the regions by how few cells they contributed.

**A** Out-of-sample variance explained by PC1 against the number of cells used to build the axis, per region, mean ± s.e.m. across sessions, with the circular-shift null. The curve is shown rather than a single matched number so that the cell-count dependence is visible rather than hidden, and so that any two regions can be compared at a common *n*. Curves are solid where at least 5 sessions contribute and dotted where 3 or 4 do; the axis is scaled to the solid points, which clips one 3-session subicular value at *n* = 24 that would otherwise compress every other curve into the lower third of the panel. *Note that the null rises too*: circularly shifted cells retain their own block structure, so their leading component still captures shared low-frequency structure and estimates it better as *n* grows. The quantity that matters is the separation between a curve and the null, not the height of the curve.

**B** Out-of-sample variance explained over null, by component, with the axis built from 8 cells. PC1 stands at +0.0314 and PC2 at +0.0006, a ratio of 53×, with PC3–PC5 at null. Because components are orthogonal by construction, a population carrying two independent states would have both generalise. Cerebellum reaches this cell count in only one session and is excluded here, as the panel notes; a one-session bar beside a sixty-session bar invites a comparison it cannot support.

> **The title of panel B states a ratio, not a dichotomy, and that is deliberate.** PC2 does clear its null (*p* = 4.3 × 10⁻⁴). An earlier draft of this panel was titled "only PC1 generalises", which the data do not support. What they support is that PC1 carries roughly fifty times the generalising structure of PC2, so the matrix is *effectively* rather than *strictly* one-dimensional.

### Figure 6 supplement 2. The state outside MEC, where MEC does not dominate the sample

`fig6_supp_region_examples.py` → `fig6_supp_region_examples.pdf` (same row layout as supplement 1, via `region_row.draw_row`)

The main region examples come from sessions where entorhinal cortex supplies most of the cells, which is the usual case and the weakest place to argue from: a state MEC expresses strongly will look shared with anything sampled beside it. These are the other sessions, chosen for their **sampling** rather than their result.

**A — M21 D25**, visual cortex contributing *more* label-varying cells than MEC (66 against 64). M21 is the animal whose probe traversed visual cortex properly rather than clipping its ventral edge, so its visual cells sit 0.7–1.1 mm from the nearest entorhinal cell. **The two components are anti-correlated here (−0.50).** **B — M28 D20**, visual-rich, visual cells ~720 µm from MEC, +0.65. Between them these two bracket what visual cortex does.

**C — M25 D20**, subicular complex contributing more cells than MEC (44 against 38): M–S +0.56, S–V +0.47, M–V +0.29.

**D — M27 D23** and **E — M26 D13**, the only sessions with enough label-varying cerebellar cells to build an axis from. **Cerebellum carries a matching state in both — M–C +0.60 and +0.58** — and its first component explains as much of its label matrix as MEC's does of its own (15% and 26% against 13% and 8%).

> **The cerebellum was intended as a negative control and is not one.** This was already visible in the pooled numbers, where it failed to sit at zero; here it is visible directly, in two sessions, as a coherent state that tracks the entorhinal one. It should be reported as a structure that shares the state, not as a control that failed — and the claim that the state is global is the better for it, since cerebellum is the one recorded structure with no parahippocampal connectivity to explain it away.

> **What these are and are not.** Examples, selected on sampling, with single-session correlations. The cerebellar rows rest on 15 and 18 cells, where an axis is barely estimable; they are shown so that "every structure recorded has a state" can be inspected at its weakest point, not because two sessions establish it. Panel A is the standing counterexample and is included for that reason: whatever the pooled disattenuated agreement of +0.863 means, it is an average over sessions that include one running the other way.


### Figure 7. The field potential changes with the state in a frequency-specific way

`fig7_theta_gamma.py` → `fig7_theta_gamma.pdf` (post-gate, rebuilt 2026-10-03)
(inputs: `lfp_spectrum_pac.py` → `spectrum_by_anchoring_w*.csv`, `pac_by_anchoring_w*.csv`)

> **This is one figure, deliberately: the example and the population statistics together.** `fig7_example_spectrogram.py` and `fig7_spectrum_pac.py` are the earlier split drafts — the example alone and the population panels alone — and are **superseded**. Nothing in this document cites them; they should be deleted or moved aside so that Figure 7 has exactly one source. The combination is the point of the figure: the population spectrum (E) is a mean of session-normalised curves and carries no time axis, so on its own it cannot show that the band changes *track the state transitions* within a session. Panels A–D supply that, and panels E–J supply the evidence that the single session is not a lucky one.

The figure is **portrait**, sized to A4 with margins (7 × 10 in), in four rows: the example session full width (A–D), then the six population panels two to a row (E F / G H / I J). Reading order is therefore A–D down, then left-to-right and down. Both states are shaded behind every trial-axis panel, in magenta and teal as in Figures 1 and 2, so that a reader tracks the state rather than remembering which colour means "something"; dashed vertical lines mark major transitions (sign changes of the state after a 9-trial median filter).

**A** One session (M26 D18) as the population axis — PC1 of the cell × trial label matrix, *z*-scored, signed to the anchored fraction — with mean running speed overlaid in gold on the right-hand axis. The axis is shown rather than the anchored fraction because it is the quantity the rest of the paper uses, and because the major transitions are defined on it.

**B** The spectrogram for the same session: per-trial Welch power, 1.9–150 Hz, rebinned into 56 log-spaced bands, *z*-scored within frequency, and smoothed along the **trial** axis only (σ = 1.8 trials) — not across frequency, which would blur the band structure the panel exists to show. Horizontal lines mark the three bands quantified below. Log-spaced rebinning matters: linear Welch bins oversample the top of a log frequency axis and render as speckle.

**C** The three band traces for the same session, *z*-scored: theta 6–10, slow gamma 30–48, fast gamma 60–100 Hz.

**D** Theta–gamma modulation index for the same session in a 9-trial sliding window, **both bands**: fast 60–100 Hz on the left axis (red) and slow 30–55 Hz on the right (blue). The two need separate axes because slow-gamma MI is roughly five times smaller than fast; plotted on a shared axis it flattens against the baseline, and the reader cannot see that it does *not* fall — which is the comparison panel G makes at the population level and the reason the example has to show both. An earlier version plotted fast gamma alone and so illustrated only half the claim.

A–D are one session, chosen because its band changes track its state transitions; the population panels are what the claim rests on, and `fig7_supp_band_examples.pdf` shows three further sessions plus a counterexample.

**E** **The two states as spectra.** Session-normalised log(power × frequency) against frequency for each state, mean ± s.e.m. over the **26 sessions with ≥ 8 trials in each state**. Multiplying power by frequency compensates the 1/*f* background so that spectral peaks are visible; subtracting each session's mean across frequencies removes differences in absolute amplitude between recordings. Theta and slow-gamma bands are shaded.

**F** **The panel that excludes a gain change.** Anchored − non-anchored difference by band, paired within session, with Wilcoxon *p* printed. Theta −0.094 (*p* = 0.020) and mid gamma −0.108 (*p* = 0.006) fall; **slow gamma +0.105 (*p* = 0.0013) rises**. Because the difference reverses sign between 30–48 and 60–100 Hz, no uniform change in amplitude, referencing or spiking contribution can account for it.

**G** **Theta–gamma coupling, by band.** Tort modulation index over 18 theta phase bins, with a phase-shuffle null subtracted, for each state. Fast gamma uncouples (−15.9%, *p* = 5 × 10⁻⁶, 23 of 26 sessions); slow gamma does not change (*p* = 0.71). The slow-gamma band contains the 50 Hz mains notch and its absolute MI is five times smaller, so that null is the weaker of the two and is labelled as such on the panel.

**H** Per-session slow-gamma change against fast-gamma change. **This is the panel that declines the trade-off reading**, and both of its statistics are negative: the two changes are uncorrelated in magnitude (ρ = −0.08, *p* = 0.70, annotated on the panel), where a redistribution of a fixed amount of power predicts a negative correlation; and the 16 of 26 sessions in the slow-up/fast-down quadrant are what the marginals already imply (15.3 expected given 21/26 up and 19/26 down, Fisher *p* = 0.59), which the title now states rather than advertising the count alone. *An earlier version titled this panel with the count and tested it against a binomial at p = 0.25 — see the correction in Results §7.*

**I** The difference index (slow − fast change) per session, sorted: positive in 21 of 26 (+0.213, *p* = 4.4 × 10⁻⁵).

**J** The difference profile across 20–120 Hz: positive to about 60 Hz and negative above it, with 25 of 26 sessions showing a crossing in that range. The shaded band is the **SEM across sessions** at each frequency — deliberately across sessions and not across frequency bins, because the question the panel answers is whether the sessions *agree* on where the difference changes sign. Without it a reader cannot tell a resolved crossover from one noisy curve that happens to cross zero.

> **FIXED 2026-10-03, recorded because the figure circulated with the wrong number.** Panel J's title was a hardcoded string reading "crossover at a median 43 Hz", which does not reproduce from `spectrum_by_anchoring_w*.csv`: the per-session median crossing is **67 Hz** (the pooled mean profile changes sign at 35.9 and 63.7 Hz), in 25 of 26 sessions. The "25/26" was right and the 43 Hz was not. The title is now computed from the data by `crossover()`, which smooths over 5 bins — single 0.98 Hz bins cross zero repeatedly on noise — and takes the last upward-to-downward crossing in 25–90 Hz so that a wobble near 25 Hz is not reported as the gamma crossover. Separately, `PBANDS` specified delta as 1–4 Hz while the spectrum starts at `FLO` = 1.9 Hz, so that bar was computed over 1.9–4 Hz; it is now 2–5 Hz, matching the Results table.

> **The speed-decoupling material is not in this figure.** Results §7 reports it (theta's speed gain roughly halved, with four coupling measures falling) from `theta_frequency.csv` and `speed_theta_coupling.csv`, and it still needs its own panels or a second supplement. The old pre-gate `figure7_theta_anchoring.ipynb` → `fig6_theta_anchoring.pdf` built those panels on `data/lfp/PREGATE/` and must not be used as it stands.

### Figure 7 supplement. Band power tracking the population state in four further sessions

`fig7_supp_band_examples.py` → `fig7_supp_band_examples.pdf`

Figure 7A–D is one session, and a single session chosen because its bands track its state is the weakest form of evidence in the paper. Four sessions are shown in two columns, each as the population axis, the log-spaced spectrogram and the three band traces on a shared trial axis, with both states shaded and major transitions marked, exactly as in Figure 7A–C.

**A — M26 D19**, **B — M25 D23** and **C — M21 D20** show the population pattern in single sessions: slow gamma rising and fast gamma falling across anchored blocks. **D — M27 D19** is drawn in red as a **counterexample** — a session with clear state transitions whose band traces do not follow them. It is included because the population effect is 16 of 26 sessions in the predicted quadrant, not 26 of 26, and a supplement showing only the sessions that work would misrepresent Figure 7H.

### Figure 7 supplement 2. Discrete gamma events

`gamma_events.py` → `gamma_events.csv`, `gamma_by_position.csv` *(analysis complete; figure not yet built)*

The two results to plot, from 10,802 trials in 44 sessions: (i) **the demand test, negative** — fast-gamma events per theta cycle are raw-higher on uncued trials (+0.0057, *p* = 0.016) but the effect reverses with speed and theta-frequency control (−0.0054) and vanishes with band power (*p* = 0.68), so event counts track power and running rather than task demand; and (ii) **the position result, positive** — slow-gamma event rate peaks at 92 cm, inside the reward zone, in 40 of 44 sessions between 70 and 130 cm, surviving control for speed and power (+0.356 Hz for 80–110 cm against the rest of the track, *p* = 5 × 10⁻⁸) but not interacting with cue condition (*p* = 0.93) or state (*p* = 0.77). Fast gamma shows no such peak.

### Figure X. Spatial information is conserved across task environments

> Unnumbered in source. Candidate for Extended Data alongside Figure 2.

### Extended Figure 2. Probe trajectories and contact coverage in the MEC

- **(A)** Probe trajectories for all mice. Note that the x and y axes are not to scale for the left plot in order to visualise the probe spread more easily.
- **(B)** For each anatomical axis, we quantified the estimated MEC layer for all Neuropixel 2.0 electrode contacts. For the AP and DV axes, we found a clear trend that superficial layer contacts were more abundant than deep layer contacts in the posterior and ventral directions. This was not observed for the ML axis.
- **(C)** *(empty in source)*


## Methods

> **⚠ INCOMPLETE.** The source text was truncated mid-sentence during the location-memory task description ("*We projected a virtual trac…*"). Everything from that point onward is missing and needs to be supplied. The sections below marked **[MISSING]** are inferred from what the paper requires, not from the source.

### Animal details

All experiments were carried out under a UK Home Office project licence, were approved by the Animal Welfare and Ethical Review Board of the University of Edinburgh College of Medicine and Veterinary Medicine, and conformed with the UK Animals (Scientific Procedures) Act 1986 and the European Directive 86/609/EEC. 7 male and 1 female C57BL/6J mice were used in this study (age range **[TODO]**). Mice were housed with littermates in a holding room with a 12 hour light-dark cycle, temperature range of 20–24 °C and humidity of 45 to 65%. Cages included bedding and nesting material, a half-dome cardboard shelter, a colourful toy and a running wheel. Mice had ad libitum access to food and water. Before headpost/Neuropixel implants, mice were single housed to avoid damage to the recording device.

### Head-fixed behaviour in virtual reality

Mice were head-fixed similar to previous experiments (**[TODO: refs]**). Mice with surgically attached low-profile RIVETS headposts (Form3 resin printer, in Grey Pro Resin) were connected to a custom-milled stainless steel RIVETS clamp (Ronal Tool Company, Inc) situated above a cylindrical polystyrene treadmill. We projected a virtual track generated using Blender3D (blender.com) onto a torus-shaped screen (Talbot Designs Limited) that surrounded the animal. A rotary encoder (Pewatron, E6-2500-472-IE) relayed treadmill rotation to Blender via an Arduino Due to update the virtual projection in correspondence to the animal's movements. A lickport (Harvard Apparatus) was placed in front of the animal which could dispense sucrose water rewards (10%; 5–10 μl per reward). A custom Arduino Due received input from Blender3D to signal the opening of a solenoid valve (NResearch, 225PNC1-21) to allow the reward to flow.

> **TODO — treadmill circumference and the rotary encoder artifact.** The treadmill circumference (~58 cm nominal) must be stated here, because a periodic speed artifact locked to treadmill rotation is present in **every** session examined (31/31 MCVR sessions; 85/85 clark2025 VR sessions), with a mouse-specific spatial period — M22 at 61.1 cm in both datasets. Its amplitude (±~8 cm/s) is comparable to the 10 cm/s stop-detection threshold, so it is not merely cosmetic: it can influence stop detection and therefore behavioural scoring. Methods must state (i) the measured per-animal period, (ii) whether speed was notch-filtered before analysis, and (iii) whether stop detection was performed on raw or filtered speed. This is a reviewer-facing issue — better declared than found.

### Handling

After headpost attachment, mice were handled twice a day for 7 days before training began. Littermates were placed via tube-transfer in a playground arena containing tubes, running wheels, climbing ropes, and colourful toys to promote exploration. The experimenter placed their hands in the playground on the first day of handling without actively touching the mice. From day 2 onwards, once mice began to climb onto the experimenter's hand within the arena, mice were cupped and removed from the arena for 10–20 seconds before being placed back into the arena. This continued with increased duration of handling over time until mice showed no signs of distress when being handled. Once achieved, mice were weighed daily to record a baseline weight prior to water restriction. When mice had at least 3 successive days of recorded weights, water was removed from their cage for 24 hours. From this point, mice were given 0.75–1.5 mL water per day to maintain a bodyweight of 85% the baseline weight, which was normalised to the expected daily growth for the animal's age.

### Habituation

Mice were then slowly introduced to the virtual reality setup. On habituation days 1 and 2, the mouse was set in a perched position within a holding cup for 5 minutes, with the lick port positioned between the VR display and the mouse. The experimenter manually moved the treadmill to simulate movement through the VR environment. On habituation days 3 and 4, the mouse was placed non-head-fixed on the running wheel for 10 minutes with the lickport positioned in front of the running wheel between the VR display and the mouse. Again, the running wheel was manually controlled by the experimenter to facilitate movement through the VR environment. On habituation days 5 and 6, the mouse was head-fixed for 10 and 15 minutes respectively from its surgically attached headpost to the RIVETS clamp. To encourage the mouse to move forward through the VR environment, the experimenter coaxed the mouse by slowly controlling the running wheel until the mouse took over with its own movements. After each habituation session, mice were returned to the playground for 15 minutes of free play.

### Location memory task

On training days, mice were collected from the holding room 30–60 minutes before recording, handled briefly and placed in the playground. Mice were head-fixed in the virtual reality setup for 30 minute sessions. We projected a virtual track that the mouse could navigate through. The track had a length corresponding to 200 cm of running wheel travel distance, with a 60 cm track zone, a 20 cm reward zone, a second 60 cm track zone and a 60 cm black box to separate successive trials (30 cm at the beginning and end of each trial). Once the mouse reached the end of a trial, it was teleported back to the start. The distance visible ahead of the mouse was 50 cm. The reward zone was either marked by distinct vertical green and black bars on cued trials, or was not marked by a visual cue at all on uncued trials. Trials were delivered in a pseudo random fashion such that all mice on a given session experienced the same trial sequence. The probability of each trial type was equal on any given trial, e.g. cued (p = 0.5) or uncued (p = 0.5). Compared to habituation, mice were required to stop within the reward zone (register a speed < 3 cm/s, calculated on a rolling average of 6 frames at 60 frames per second) to receive a reward. As mice initially moved slowly on the first couple of training days, rewards were typically dispensed every trial. As mice sped up across training days, the mice needed to develop a selective stopping strategy to maximise their rewards within the 30 minute session for both trial types. Once mice demonstrated a clear selective stopping strategy for the reward zone, mice were chronically implanted with multi-shank Neuropixels 2.0 silicon probes to record single unit activity while the expert mouse performed the task.

### Dual-context location memory task

Once all recording sites in the brain were recorded from, mice advanced to a similar but more demanding dual-context location memory task. Mice were again required to stop at a reward zone either marked or not by a visible cue, but now the location of the reward zone was conditioned on the visual context of the trial. The track length was extended to 230 cm to accommodate an additional reward zone. Context 1 remained similar to the original task with a reward zone 60 cm away from the black box and the same visual pattern on the track walls of white circles on a black background. Context 2 however required the mouse to stop at a reward zone 90 cm from the black box. Within this context the mouse experienced a different wall pattern of black circles on white walls. Both contexts conveyed the same amount of optic flow information about distance travelled. Contexts were delivered in blocks of 20 trials, alternating between context 1 and context 2. Within a block, the first trial was always a cued trial while all successive trials had an equal probability of the next trial being cued (p = 0.5) or uncued (non-beaconed; p = 0.5). Mice were trained on the dual-context task for 5–6 days while single unit activity was recorded simultaneously.

> **NOTE — verified against the analysis code.** Track lengths (200 cm / 230 cm), the 20-trial alternating block structure and the ~30 cm separation between the two reward zones all match the values hard-coded in the notebooks. Two things to confirm:
> - **Reward-zone coordinate convention.** With a 30 cm black box at trial start and the zone beginning 60 cm later, context 1's 20 cm zone spans 90–110 cm (centre 100). The analysis notebooks mark the zones at **92 cm and 122 cm**. The 30 cm offset between contexts is consistent, but the absolute origin is not — state whether positions are measured from the start of the black box or from its end.
> - **"the first trial of a block was always cued"** is a testable claim and worth verifying in the data, because it bears on any analysis that pools early-block trials.

### Surgeries

**Headpost attachment.** Inhalation anaesthesia was induced using 4–5% isoflurane / 95–96% oxygen, and sustained at 1–2% isoflurane / 98–99% oxygen with a flow rate of 1 L/minute. Carprofen and buprenorphine were given subcutaneously at the recommended dosage (20 mg/kg and 0.1 mg/kg respectively) for analgesia, in addition to 1 mL saline to maintain hydration. An oval incision was made to expose the top of the skull. The exposed skull was scored with a scalpel blade to increase the bonding surface area for the future Neuropixel 2.0 implant. The edges of the oval incision were glued in place on the skull (Vetbond tissue adhesive, 3M) and a low-profile RIVETS headpost (3D printed in Grey Pro Resin on a Form3 3D printer) was attached along the incision perimeter and held in place with UV curing resin cement (RelyX™ Unicem, 3M). We aimed to achieve a window through the headpost with both bregma and lambda landmarks clearly visible, with at least 5 mm clearance anterior and posterior for the craniotomy and stereotaxic head straightening in the future surgery. Silicon was also applied to the headpost window to cover the exposed skull until the future surgery. After the surgery, mice recovered for ~20 minutes on a heat mat and had unlimited access to Vetergesic jelly (0.5 mg/kg of body weight buprenorphine in raspberry jelly) for 48 hours.

**Chronic Neuropixels 2.0 implantation.** Prior to surgery, we assembled the NP2 reusable Apollo drive (https://doi.org/10.7554/eLife.98522.2; https://github.com/Coen-Lab/chronic-neuropixels) using the recommended 3D printer materials. The reference and ground were shorted together with a fine silver wire attached to a grounding screw (M1 × 4 mm screw; AccuGroup SFE-M1-4-A2). Probes reused across animals were cleaned and stored in preparation for surgery (Neuropixel 1.0 user manual; https://www.neuropixels.org/support). Anaesthesia induction, analgesia and hydration were all applied as in the headpost surgery. The silicon over the headpost window was removed to expose the skull. An indent was bored on the left hemisphere in the centre of the intraparietal skull plate to accommodate a single grounding screw. A craniotomy was made from 3 mm lateral to the midline along the lambdoid suture, ending at 4 mm lateral to the midline. The dura was removed and DuraGel (Cambridge Neurotech) inserted to seal the craniotomy. The probe was loaded onto a stereotaxic handle oriented rostrally 15° from vertical and lowered over the craniotomy site. The most medial shank of the multi-shank probe was positioned 3 mm lateral of the midline, with 250 µm spacing between adjacent shanks up to roughly 3.8 mm. The ground screw was secured into the bored indent over the left hemisphere. The probe was brushed with a pipette tip 10 times with DiI (V22889, Thermo Fisher). The probes were manually advanced at < 5 µm s⁻¹ to an angled depth of 3 mm or until resistance was met. If resistance was met, probes were retracted by 10 µm. The probes were then secured by cementing the sacrificial Apollo drive docking module to the skull and headpost. Exposed electrical connections were tucked into the payload lids and taped over with electrical tape until recording. Mice were given the same recovery regime as previously described.

### Chronic electrophysiological data collection

Electrophysiological signals were received onto a headstage for filtering, amplifying, multiplexing and digitizing. A 3 metre interface cable connected the probe to a National Instruments acquisition system, which relayed signals to a data acquisition card on a desktop computer running OpenEphys. The Neuropixel acquisition plugin recorded voltage traces at 30 kHz from 384 simultaneous recording sites, in addition to an auxiliary input channel receiving aperiodic transistor-transistor logic (TTL) pulse information from an Arduino Due for data synchronisation with behavioural data recorded in Blender. The same TTL pulse information was relayed to Blender and recorded at 60 Hz, as well as an infrared LED visible to a camera capturing the side profile of the head-fixed mouse.

> **TODO:** three citations are unresolved placeholders in the source — the National Instruments acquisition system, OpenEphys, and the Neuropixel acquisition plugin.

### Offline spike sorting

The electrophysiological data was processed using SpikeInterface (Buccino et al. 2020) and Kilosort (Pachitariu et al. 2024). The data and probe geometry were loaded using SpikeInterface. Bad and noisy electrode channels were then removed and a phase shift correction was applied, accounting for mismatched sampling times across Neuropixel probes. Both steps were originally developed by the IBL (Banga et al. 2022). The preprocessed recording was passed to Kilosort v4.0.16 with default parameters and motion correction turned off. Kilosort applies a bandpass filter and local whitening, then detects and sorts individual spikes into clusters using a template matching algorithm. After sorting we use SpikeInterface to compute cluster location by monopolar triangulation (Boussard et al. 2021); locations are combined with the registered anatomy to label the brain region of each cluster.

Clusters were curated with **bombcell** (Fabre et al. 2023), which classifies each unit from its waveform and spike train into `good`, `mua`, `noise` or `non_soma`. We keep units labelled `good` or `mua` and discard `noise` and `non_soma`; `mua` is retained because the population measures used here are built from trial-level label sequences rather than from single-unit isolation, and excluding multi-unit clusters would discard a substantial fraction of the recorded population for no gain in the quantity being measured. Where single-unit isolation does matter — the putative monosynaptic connections of Figure 4, whose cross-correlogram test is sensitive to the spike-sorting errors that split or merge units — the restriction is stated with that analysis. Bombcell is the only curation applied; no additional quality-metric thresholds are imposed.

> **Resolved (2026-10-01).** This section previously also specified SpikeInterface quality-metric thresholds (rate > 0.5 Hz, presence ratio > 0.9, ISI violation < 0.5, SNR > 1) alongside the bombcell labels, which are different curations that do not select the same units. Curation is bombcell only, as the pipeline does, and the threshold sentence has been removed.
>
> The source carried a bracketed note — *"We also selected several recordings for manual curation and compared the two curations. We found that the thresholding approach was more conservative than our manual curation, giving confidence that the thresholded units were of high quality"* — which referred to the thresholds, not to bombcell. If a bombcell-against-manual comparison was run on any recordings it is worth a sentence here, since it answers the obvious reviewer question; if not, the note should stay out.

### Explantation, histology and probe localisation

Following experiments, inhalation anaesthesia was induced using 4–5% isoflurane / 95–96% oxygen, and sustained at 1–2% isoflurane / 98–99% oxygen with a flow rate of 1 L/minute. The Apollo drive payload was disconnected from the docking module and the probes slowly removed from the brain. Mice were then given a lethal dose of sodium pentobarbital (Euthatal, Meridal Animal Health, UK), and transcardially perfused with PBS followed by 4% paraformaldehyde (PFA). Mouse heads were removed and left in 4% PFA solution for at least 2 days. Brains were then extracted and placed in 4% PFA solution for another 2 days, then transferred to 30% sucrose solution for at least 24 hours. Brains were frozen and sliced into 60 µm sagittal sections using a freezing microtome. Sections were stained with DAPI, mounted and imaged using widefield microscopy (Zeiss Axioscanner), using a blue channel for DAPI and an orange-yellow channel for DiI fluorescence. Sections were aligned and registered to the Allen Mouse Brain Atlas using the SHARP-Track package in MATLAB (https://github.com/cortex-lab/allenCCF/tree/master/SHARP-Track). This involved marking the DiI fluorescence to reconstruct the trajectory that each shank took in the brain. In cases where the probes passed through the cortex, we used additional information recorded across the shank to calculate the shank trajectory. We evaluated noise profiles, theta power (6–12 Hz) and curated cluster density to calculate likely exit points of the probe through the cortical surface.

> **TODO:** the source carries an unresolved "(Figure reference?)" at the end of this paragraph — point it at the panel showing the noise/theta/density exit-point estimation.
>
> **Still needed here:** how unit coordinates are assigned from the reconstructed trajectories, and the definition and origin of the two coordinate systems the analysis uses (`coord_CCFs_*` and `coord_SCs_*`) — see the note in the encoding-model section.

### Reconstruction of probe locations

By measuring 250 µm between adjacent tips in 8 animals, we targeted the medial entorhinal cortex across the medial-lateral axis, starting medially (around 3 mm medial of the midline) at the MEC-parasubiculum border, which has been shown to display a sharp anatomical division of grid cells (Obenhaus et al. 2022), out to lateral regions of the MEC (around 3.75 mm lateral of the midline), close to the borderlands of the LEC. We targeted dorsally to afford adequate sampling of grid fields in a 1 m × 1 m open field and accurate classification of grid cells.

> **⚠ Animal counts disagree across the manuscript.** *Animal details* says **7 male and 1 female** (8 mice); this section says "in 8 animals" but then "We recorded from 9 mice (8 males and 1 female)". Pick one and make it consistent everywhere, including the Results.
>
> **TODO:** "around 3 mm **medial** of the midline" in the source should read *lateral* of the midline (as in the surgical description); corrected above. Ages are still missing from *Animal details*.

### Anatomical coordinates and the mediolateral axis

Each cluster carries two sets of coordinates, both shipped with the dataset, and both are used:

- `coord_CCFs_x/y/z` — Allen CCF coordinates, with `coord_CCFs_x` the mediolateral axis. Used for the anatomical distributions in Figure 2.
- `coord_SCs_x/y/z` — the same locations expressed relative to the midline. Used for the medial/lateral split, and via `coord_SCs_x/y` as the distance metric for ordering covariate cells in the encoding models.

The two are related by an exact constant offset, verified across all 49,747 registered clusters and every clark2025 session:

> `coord_CCFs_x − coord_SCs_x = 5700` µm

so `coord_SCs_x` is signed distance from the midline (negative in the left hemisphere, where all probes were implanted) and the CCF midline lies at *x* = 5700 µm. Analyses that threshold on mediolateral position negate `coord_SCs_x` first, so the working variable throughout is **absolute distance from the midline**. Across entorhinal clusters this spans 3000–3759 µm (median 3510), consistent with the surgical targeting of 3.0–3.8 mm lateral.

**Medial/lateral boundary.** Cells are classified medial at < 3400 µm from the midline and lateral at ≥ 3400 µm, dividing the entorhinal sample 43.2% / 56.8% (11,300 / 14,876 clusters). The boundary separates the layer-defined subdivisions in the expected order — ENTm1 90.9% medial, ENTm2 56.5%, ENTm3 30.8%, ENTm5 18.3%, ENTm6 16.1%, ENTl6 0% — which is a consistency check on the registration rather than an independent justification of the value.

> **TODO:** 3400 µm is a round number near the middle of the sampled range. Either derive it from the Figure 2 grid-cell distribution or show the medial/lateral results are stable across a sweep of boundary positions.
>
> **⚠ Coverage is very uneven across animals**: M21 is 16.5% medial, M28 63.1%, M29 60.7%, and M22/M27 are 100% medial on tiny samples (35 and 807 clusters). Medial-vs-lateral comparisons must therefore be made *within session*; pooling cells across mice would confound the contrast with which animal contributed them.

### Open field recordings

The open arena consisted of a metal box with a square floor area, removable metal walls and a metal frame (frame parts from Kanya UK: C01-1, C20-10, A33-12, B49-75, B48-75, A39-31, ALU3), with an A4-sized cue card in the middle of one of the metal walls. For open-field exploration sessions, mice were placed in the arena while tethered via a Neuropixel cable and were prompted to explore with small crumbled pieces of chocolate Weetos, for a maximum of 20 minutes per session.

> **TODO — remaining gaps in this section.** (i) **Floor dimensions**: the analysis assumes a 1 m × 1 m arena (`box_width = 1.0` throughout, and the open-field rate maps and grid scores are computed on that assumption), so the actual floor area must be stated and must match. (ii) **Tracking**: method and sampling rate. (iii) **Lighting**. (iv) **The OF1/OF2 protocol**: that two open-field sessions were recorded, the interval between them, and — stated explicitly, because the classification depends on it — that both were acquired on the same recording day as the virtual-reality session whose cells they classify. (v) Whether the cue card was in a fixed allocentric orientation across sessions and mice.
>
> Note that the chocolate-Weeto prompting is worth keeping in the text rather than trimming: it means open-field coverage is partly experimenter-driven rather than spontaneous, which is relevant to occupancy-normalised measures (spatial information, grid score) and is the kind of detail a reviewer asks about when rate maps are the basis of cell identity.

### Analysis of behaviour during the location memory task

To quantify trial performance, we used a number of metrics based on previous work (Tennant et al. 2022; Pettit et al. 2024). We classified trials into discrete behaviours that describe the speed profile and engagement for any given trial. Trials were first classified into rewarded and unrewarded based on whether the mouse stopped (speed < 3 cm/s) within the reward zone, initiating a reward to be dispensed. To discriminate between rewarded trials in which mice ran and slowed to a stop (demonstrating stop selectivity for the reward zone) and rewarded trials in which mice maintained a slow speed throughout the track but still initiated rewards, trials in which the mouse's average speed outside of the reward zone was > 10 cm/s were classified as **hits**, while the remainder were removed from behavioural analysis. Unrewarded trials were split into **try** and **run** trials by comparing their average speeds in the reward zone to hit trials: the 95th percentile of speeds in the reward zone for hit trials was used to discriminate between try trials (< 95th percentile speed) and run trials (> 95th percentile speed).

**Session-level stopping strategy.** Trial-level performance describes single trials; sessions were additionally summarised by the stopping *strategy* the animal used, which distinguishes solving the task by sight from solving it by path integration.

For each session and each trial type, stops were binned by track position into 20 bins and the resulting profile **normalised within trial** — each trial's stops sum to one before averaging across trials — so that a trial with twenty stops contributes no more than a trial with one. Pooling raw stops instead allows the heaviest-stopping trials to dominate the profile and can manufacture an apparent reward-zone preference out of dwell time rather than choice. Stops inside the 30 cm start box were excluded, as they precede any decision about the zone.

Stop **selectivity** is the proportion of stopping falling inside the reward zone divided by the proportion of the track the zone occupies, so that 1.0 denotes stopping spread uniformly along the track. Sessions were then assigned to one of three strategies:

- **cue-dependent** — uncued selectivity below 2.0, or uncued hit rate below 0.25;
- **expert** — uncued hit rate at or above 0.70;
- **partial** — intermediate.

Across 79 sessions this gave 26 cue-dependent, 25 partial and 28 expert. The two extremes are behaviourally distinct in the way the task intends: cue-dependent sessions show selectivity of 5.9 when the zone is cued but 2.0 when it is not, with an uncued hit rate of 0.11; expert sessions show 5.2 and 4.8 with an uncued hit rate of 0.86. Both stop precisely at the zone when shown it; only one does so unaided.

**Stopping rate is reported separately, not as a strategy.** Six sessions exceed 15 stops per trial (median 19.5 against 3–8 elsewhere). These are not less selective once profiles are normalised within trial, and they span all three strategies (3 partial, 2 expert, 1 cue-dependent), so excessive stopping is an independent descriptor of a session rather than a stage of the ladder. It is nonetheless visible in the first stop of each trial, which for these sessions falls at 0.2–0.5 of chance — systematically away from the reward zone.

> **NOTE on using the first stop.** The first stop of a trial is the intuitive index of choice, and it is diagnostic for the over-stopping sessions above. It is not, however, a good general measure in this task: expert animals stop several times per trial and their first stop typically falls early on the track, so first-stop selectivity collapses to near chance (1.84 against 5.27 for the within-trial normalised measure) in animals whose stopping is in fact highly selective. Within-trial normalisation addresses the same problem — heavy stoppers dominating a pooled profile — without discarding the later stops that carry the behaviour.

**Independent confirmation.** The strategies were validated by a model that was never told them. Stop counts were modelled per trial and position bin with three regressors — a constant, the reward zone on cued trials, and the reward zone on uncued trials — and the coefficient of partial determination computed for each. The uncued-zone regressor accounts for 6.7% of the variance in expert sessions and **0.09%** in cue-dependent sessions, a difference of nearly two orders of magnitude, while the cued-zone regressor is substantial in both (11.0% and 8.9%).

> **NOTE:** animals were trained to expert on the cued task before implantation, so recording session index is not a learning axis. Strategy here is close to a fixed property of the animal — M20 is cue-dependent in all 13 of its sessions, M29 expert in all 9 — rather than a stage animals pass through. Any figure plotting strategy against session should say so, or it will be read as a learning curve.
>
> The thresholds (selectivity 2.0, uncued hit rate 0.25 and 0.70) were chosen from the observed distributions and are a judgement call; the CPD result above does not depend on them.

> **NOTE:** this accounts for the four values of the `performance` field in the data (`hit`, `try`, `run`, `slow`) — `slow` being the rewarded-but-not-selective category excluded from behavioural analysis. Say so explicitly, and state whether `slow` trials are also excluded from the *neural* analyses or only the behavioural ones.
>
> **TODO (from source):** stop and lick selectivity metrics exist but are not currently used. The source carries a to-do for a supplementary panel showing stop and lick selectivity across hit, try and run trials. Either build it or drop the metrics.
>
> **⚠ The rotary encoder artifact belongs here as well as in the VR apparatus section.** A periodic, treadmill-locked oscillation of roughly ±8 cm/s is present in every session examined (31/31 MCVR, 85/85 clark2025 VR), with a mouse-specific spatial period (M22: 61.1 cm in both datasets). Both thresholds defined in this paragraph sit inside that amplitude — the 3 cm/s stop criterion and the 10 cm/s hit criterion — so the artifact can flip trial classifications, not merely add cosmetic ripple. State the measured per-animal period, whether speed was filtered before thresholding, and whether trial classification used raw or filtered speed.

### Local field potentials and theta phase

We compute the LFP per recording channel by applying a bandpass filter between 1 and 300 Hz to the raw 30 kHz recordings. To compute the theta phase of the LFP, we downsample the LFP to 60 Hz and then further bandpass filter between 5 and 10 Hz. We subsequently apply the Hilbert transform to obtain the phase. When linked to individual clusters, we use the phase of the channel where the template amplitude of the cluster was maximised.

> **NOTE:** this defines the LFP covariate used in the Figure 4 v2 encoding models. Downsampling to 60 Hz before extracting 5–10 Hz theta is above Nyquist and therefore fine, but the order of operations (downsample, then bandpass) should be stated as deliberate.

### Theta and the population anchoring state

*Implementation: `scripts/figures/AnchorDynamics2026/figure7_theta_anchoring.ipynb`, with the batch scripts `lfp_batch.py`, `theta_freq.py`, `speed_theta.py`, `phase_locking.py`, `region_slope.py`, `residual_speed.py`, `theta_residual.py`, `speed_tuning.py`.*

> **⚠ SESSION AND TRIAL COUNTS IN THIS SECTION ARE PRE-GATE.** The counts below (58 sessions, 14,022 trials; 55 of 58 sessions with ≥15 trials in both states) were computed before the anchoring classifier's per-cell null gate. Post-gate the same pipeline gives **44 sessions and 10,802 trials**, of which **19** hold at least 15 trials in each state — the speed–theta coupling analysis in particular loses two thirds of its sessions, because the gate removes the sessions whose apparent state alternation was classifier noise. The *methods* described here are unchanged and correct; only the *n*s move. See the warning table in Results §7. The following subsection (*The spectrum, theta–gamma coupling and gamma events*) is post-gate throughout and its counts stand as written.

**Signal and channel selection.** Band power was computed from the band-limited (1–600 Hz) LFP sampled at 1 kHz, stored as 48 channel groups per probe, each the average of 8 adjacent contacts. Each session ships an anatomical assignment of every contact (CH1–CH384) to a brain region; a channel group was treated as entorhinal when the majority of its contacts were assigned to ENTm. Groups are contiguous blocks of the shank and therefore almost always fall within a single layer, making the majority vote unambiguous. A median of 26 of 48 groups per session were entorhinal; sessions with none (all of M22, consistent with their absence of entorhinal spatial cells) were excluded, leaving 58 sessions and 14,022 trials.

**Depth was retained rather than averaged.** Theta amplitude varies steeply through the entorhinal layers, so averaging across groups mixes signals of very different magnitude. Power was computed per trial and per channel group and, where a single value per trial was required, groups were combined only after within-group standardisation.

**Band power.** Power spectral density was estimated per trial with Welch's method and integrated over delta (1–4 Hz), theta (6–10 Hz), beta (10–20 Hz), gamma (30–80 Hz) and high (80–200 Hz) bands. Both relative and absolute power are reported. Relative power is compositional — the five bands sum to one, so a fall in theta mechanically raises the others and one effect can appear as four. Absolute power breaks that coupling and is the primary measure; it was z-scored within session and channel group before pooling, because absolute amplitude varies by an order of magnitude across probes and depths.

**Separating a frequency shift from a weakening.** Power in a fixed 6–10 Hz window falls if the rhythm weakens *or* if it moves out of the window, and these are different claims about the circuit. The two were separated using the stored theta phase: instantaneous frequency was taken as the derivative of unwrapped phase (computed as a wrapped per-sample difference, because a small fraction of NaN samples propagates through `np.unwrap` to the whole trace), and peak frequency as the spectral maximum within 4–12 Hz. Theta amplitude was taken as the Hilbert envelope of the 6–10 Hz signal.

**Speed matching.** Theta amplitude and frequency both track running speed, so every comparison between anchored and non-anchored trials was repeated on a speed-matched subset, formed by matching trials within deciles of trial mean speed. Comparisons were paired across sessions, so no session dominates.

**Speed–theta coupling.** Beyond a difference in level, we asked whether the *coupling* between speed and theta differs by state. Per session we fitted `theta ~ speed × state` by ordinary least squares and took the interaction term as the change in gain, and separately computed the within-state Pearson correlation. Because speed is genuinely less variable inside anchored epochs (SD 6.4 vs 8.4 cm/s, p = 1 × 10⁻⁸, 48/55 sessions), which attenuates a correlation on its own, the **regression slope is the primary measure** — it is unaffected by range restriction — with correlations reported alongside. Sessions contributed only when both states held at least 15 trials (55 of 58). Three controls were applied: (i) both states were trimmed to their common 5–95% speed support and the slopes refitted; (ii) the per-session change in correlation was tested against the per-session change in speed SD, to check for a dose–response with the confound; and (iii) a circular-shift null in which the state vector was rolled within session (200 shifts), preserving run structure and any within-session drift in theta.

**Speed tuning of entorhinal cells.** Firing rate and running speed were binned at 250 ms and restricted to running bins (≥ 3 cm/s), each bin taking the anchoring state of its trial. All curated entorhinal cells were used (grid, non-grid spatial and narrow-spiking). Speed cells were defined pooled across states, so that the definition could not itself be state-biased, as cells whose rate–speed correlation exceeded a circular-shift null of the binned rate (200 shifts, p < 0.05, positive); this null preserves the autocorrelation of firing, which makes a parametric p meaningless at this bin size. Per cell and state we computed the regression slope of rate on speed (Hz per cm/s), the gain (slope divided by mean rate, which separates a change in speed sensitivity from a change in overall firing), and the correlation, each on all running bins and again after trimming both states to their common 5–95% speed support. Sessions, not cells, are the unit of analysis: cells within a session share one behavioural state, so a cell-level test would count the same state change hundreds of times.

**Population speed decoding.** Ridge regression from population binned counts to running speed, five-fold cross-validated, fitted separately within each state after matching bins across speed deciles and subsampling to equal count, so neither state received more training data or a wider speed range.

**Regional and laminar specificity of the coupling change.** To ask whether the decoupling is confined to entorhinal cortex, the speed→theta slope was computed for every channel group on the probe rather than only the entorhinal ones, with each group assigned to a region family by majority vote over its eight contacts, and with the anchoring state still defined by entorhinal cells. Theta amplitude per trial was taken as the RMS of the 6–10 Hz signal over running samples, regressed on trial mean speed within each state on the states' common 5–95% speed support. Slopes were normalised by mean amplitude within state, because absolute LFP amplitude varies by an order of magnitude across regions and depths and would otherwise dominate any cross-region comparison; raw slopes are reported alongside. Sessions contributed a region only when it held at least four channel groups, and only when both states retained at least 15 trials after trimming, which excluded seven otherwise usable sessions whose non-anchored blocks were too short (13–23 trials). Laminar comparisons use one value per session per layer: channel groups within a session share a behavioural state and a probe, and a per-group test counts the same session dozens of times.

**Spike–theta phase locking.** Each cluster was assigned the theta phase of the channel group containing its extremum channel, rather than a session average, because theta phase rotates through the entorhinal layers. Spikes were restricted to running samples in labelled trials, then matched across speed deciles between states and subsampled to equal count; locking strength is the mean resultant length averaged over 20 such draws. Both matching steps are necessary and run in the same direction: the mean resultant length is biased upward at small samples, and anchored firing rates are about 7% lower, so unmatched counts would make anchored cells appear more strongly locked. The speed dependence of locking was taken as the difference between the top and bottom speed terciles of the matched set, with counts equalised within each tercile.

**Unexpected speed.** Because anchored trials have both more reliable position coding and a more stereotyped speed profile, position can stand in for speed and inflate any apparent speed coding. Position was therefore partialled out of both sides: within each state, cyclic position (sin and cos of 2π·position/L) was used to predict speed and, separately, each cell's spike count, both cross-validated (three folds; xgboost, 50 rounds at depth 3, `count:poisson` for rate and squared error for speed). The residuals — unexpected speed and unexpected rate — were then regressed on one another. The speed residual is orthogonal to position by construction, so a difference between states cannot arise from position substituting for speed. Sensitivity is reported both per cm/s (physical gain) and per SD of residual speed, because the two answer different questions when the states differ in how much unexpected speed they contain.

**Within-session time course.** To ask whether the coupling changes *while* the state changes rather than only between pools of trials, the speed–theta correlation was recomputed in a sliding window of 25 consecutive trials (windows requiring ≥60% valid trials) and compared with the anchored fraction of the same window. Neighbouring windows share trials, so the window-level correlation is autocorrelated and cannot be assessed parametrically; per-session significance was taken from a circular shift of the state vector alone (500–2000 shifts), which leaves the speed and theta series and the state run structure intact and breaks only their alignment. The analysis was repeated at window widths of 15 and 40 trials.

### The spectrum, theta–gamma coupling and gamma events by anchoring state

*Implementation: `lfp_spectrum_pac.py` → `spectrum_by_anchoring_w*.csv`, `pac_by_anchoring_w*.csv`; `gamma_events.py` → `gamma_events.csv`, `gamma_by_position.csv`; figures `fig7_theta_gamma.py` and `fig7_supp_band_examples.py`. All of these read the **gated** labels in `data/population_state/anchoring_trials.csv`.*

**The full spectrum rather than named bands.** The band-power analysis above integrates over five bands fixed in advance, which cannot detect a change that reverses sign inside one of them — and 30–80 Hz "gamma" is exactly where that happens. Power spectral density was therefore estimated per trial and per entorhinal channel group with Welch's method at `nperseg` = 1024 on the 1 kHz signal, giving 0.98 Hz resolution over 1.9–150 Hz. The window length matters for the low end: `nperseg` = 512 gives 1.95 Hz bins whose first bin above 2 Hz is centred at 3.9 Hz, which leaves 2–4 Hz empty.

**Mains.** Bands at 48–52, 98–102 and 148–152 Hz were excluded from every band average. The 50 Hz line sits inside the slow-gamma band, so this is not cosmetic: removing it *strengthens* the slow-gamma effect (+0.109, *p* = 3.6 × 10⁻⁴ against +0.105), and the 48–52 Hz band tested on its own does not reach significance (*p* = 0.063), which establishes that the effect is not line noise tracking the state.

**Whitening and normalisation.** Spectra are reported as log₁₀(power) + log₁₀(frequency) — power multiplied by frequency — so that spectral peaks are visible against the 1/*f* background instead of being buried under it, and then mean-centred within session across frequencies so that no recording's absolute amplitude dominates the average. Per-band differences are computed on within-session *z*-scores and tested with a Wilcoxon signed-rank on per-session means, paired across states, with the session as the unit of analysis throughout.

**Session inclusion.** A session contributed when it held at least 8 trials in each state, giving 26 sessions. This threshold is lower than the 15 used for the speed-coupling analyses because a band-power mean is estimated from far fewer trials than a within-state regression slope; at 15 the same analysis gives 21 sessions and every reported effect holds (theta −0.099, *p* = 0.029; slow gamma +0.079, *p* = 0.018; mid gamma −0.125, *p* = 0.0025).

**Theta–gamma phase–amplitude coupling.** Coupling was quantified with Tort's modulation index: the gamma envelope was averaged into 18 bins of theta phase, normalised to a distribution, and the index taken as its Kullback–Leibler divergence from uniform, divided by log(18). A per-session null was built by shuffling the phase series in blocks and the reported quantity is the index minus its null mean, so that differences in the amount of data between states cannot produce a difference in coupling. *Theta phase is stored in these files on 0–2π, not −π to π; wrapping it into the symmetric range is required before binning, and omitting that step returns NaN for every session.*

**Gamma events.** To ask whether discrete gamma events, rather than continuous power, carry the state or the task demand, the envelope of each gamma band was *z*-scored within session and thresholded at 2 SD; a supra-threshold excursion lasting at least one cycle of the band's centre frequency was counted as an event. Events were expressed per theta cycle, with the number of available theta cycles per trial as the denominator, so that a trial is not credited with more events simply for being longer or faster. Models are mixed models with a session random intercept, and the covariates added in sequence are trial mean speed, instantaneous theta frequency and band power — power last, because it is the quantity the event count is meant to be an alternative to, and an event effect that disappears on adding it is an effect on power.

**Gamma events by track position.** Event rate was additionally binned into 5 cm position bins, weighted by occupancy, separately by state and cue condition. *Position binning here is a place where an identity comparison silently empties the result: comparing a `numpy.bool_` state label to a Python `bool` with `is` is always false, so an earlier version returned empty position profiles for every session while reporting success. The binning is vectorised over per-trial arrays indexed by trial number, which also removes an apparent hang caused by a Python-level comprehension over ~1.8 million samples per session.*

> **Mount checking.** The LFP lives on an external volume (`clark2025_lfp`). When it unmounted mid-run, an earlier version of the batch script logged "wrote …" for all four workers on empty results. The scripts now check the mount and exit non-zero rather than writing an empty file that looks like a finished analysis.

### Cell classification — spatial metrics

*Implementation: `scripts/figures/figure_0_classification/create_classification_datasheet_v2.ipynb`. All spatial metrics are computed at the animal's true position; no temporal or spatial shift of the trajectory is applied.*

**Rate maps.** Open-field firing rate maps were constructed on a 40 × 40 bin grid spanning the 1 m × 1 m arena (2.5 cm bins), computed as the ratio of binned spike counts to binned occupancy, and smoothed with a Gaussian kernel of σ = 1.5 bins. Bins with zero occupancy were treated as undefined and excluded from all subsequent calculations.

**Spatial autocorrelogram.** The spatial autocorrelogram (SAC) of each rate map was computed as the Pearson product-moment correlation between the map and a copy of itself displaced by every possible two-dimensional lag, evaluated over the region of overlap. For efficiency this was implemented with FFT-based convolution. Lags at which fewer than 3 bins overlapped, or at which either overlapping region had near-zero variance (< 10⁻¹²), were set to zero rather than being allowed to produce singular values, and the result was clipped to [−1, 1]. This implementation was verified against direct spatial-domain convolution (agreement to 1.8 × 10⁻¹³).

**Grid score.** Gridness was computed from the SAC using the annulus-sweep procedure of Banino et al. (2018), as implemented in the `GridScorer` class shared with the recurrent network model used elsewhere in this work, so that scores are directly comparable between recorded and simulated populations.

For a given annular mask *M* covering the SAC, let *S* be the SAC and *S*(θ) the SAC rotated by θ about its centre (cubic spline interpolation, output shape preserved). Writing *A* = Σ*M* for the ring area and

> *m* = Σ(*S*·*M*) / *A*,  *C* = (*S* − *m*)·*M*,  *V* = Σ*C*² / *A* + 10⁻⁵

the rotational correlation at angle θ is

> *r*(θ) = [ Σ *C* · ((*S*(θ) − *m*)·*M*) / *A* ] / *V*

and gridness for that mask is

> gridness = (*r*₆₀ + *r*₁₂₀)/2 − (*r*₃₀ + *r*₉₀ + *r*₁₅₀)/3

Two properties of this estimator should be stated rather than left implicit. First, the rotated map is centred by the mean and normalised by the variance of the **unrotated** masked SAC, not by its own; *r*(θ) is therefore a normalised cross-product rather than a symmetric Pearson correlation. It coincides with Pearson only insofar as rotation preserves the ring mean and variance, which holds approximately by the symmetry of the annulus but not exactly. Second, the 10⁻⁵ added to *V* regularises rings that are close to flat, and bounds *r* for cells with negligible structure inside the annulus.

Because the appropriate annulus depends on grid spacing, which is not known in advance, gridness was evaluated over a sweep of 10 annuli and the **maximum** taken as the cell's grid score. All ten share an inner radius of 0.2 × 40 = 8 bins (20 cm), excluding the central peak; outer radii are linearly spaced from 0.4 to 1.0 in the same units, i.e. 16 to 40 bins (40 to 100 cm) in ten steps. The SAC extends to a maximum lag of 39 bins, so the widest annuli span the entire defined region of the autocorrelogram.

Unlike field-detection approaches, this procedure returns a grid score for every cell: no cell is excluded for failing to yield a minimum number of autocorrelogram peaks. Significance is established by the shuffle procedure below rather than by a structural precondition.

> **NOTE — this supersedes an earlier description.** A previous draft of the Methods specified a different estimator: min(*r*₆₀, *r*₁₂₀) − max(*r*₃₀, *r*₉₀, *r*₁₅₀), with the annulus derived per-cell from detected fields (SAC binarised at 20% of maximum, requiring more than 7 local maxima, ring set from the mean distance of the 6 maxima nearest the centre). That version is **not** what produced any number in this draft, and it differs in consequence as well as in form: the min/max statistic is systematically more conservative, and the field-detection ring leaves gridness undefined for any cell failing the 7-peak test, whereas the sweep scores every cell. The description above is the one to report. *(The earlier text's ring definition was also internally inconsistent — "middle border = 1.25 × average distance, outer border = 0.25 × average distance" places the outer border inside the inner one.)*

**Spatial information.** Spatial information was quantified as the Skaggs information rate in bits per spike,

> SI = Σᵢ pᵢ (λᵢ / λ̄) log₂(λᵢ / λ̄)

where pᵢ is the occupancy probability of bin i, λᵢ the firing rate in that bin, and λ̄ the occupancy-weighted mean rate. This quantity is non-negative for all rate maps by Jensen's inequality; negative values cannot arise from data and indicate an implementation fault.

**Border score.** Firing fields were defined as connected components of the rate map exceeding 30% of the map maximum and covering more than 200 cm². The border score is (cM − dm)/(cM + dm), where cM is the largest proportion of any single wall covered by a single field, and dm is the mean distance of field firing from the nearest wall, weighted by firing rate and normalised by half the arena width.

**Head direction score.** Head-direction tuning curves were computed as occupancy-normalised firing rate as a function of head direction, and tuning strength was quantified as the mean resultant vector length.

**Significance testing.** Significance of each metric was assessed against a within-cell null distribution generated by circularly shifting the spike train relative to the behavioural trajectory by a random offset of at least 20 s, wrapping spikes past the end of the session to the beginning, and recomputing the metric. 100 such shuffles were generated per cell. A cell was scored as significant for a given metric if its observed value exceeded the 95th percentile of its own null distribution. Shuffle offsets were drawn from a generator seeded deterministically on (mouse, day, session, cluster, shuffle index), so that null distributions are exactly reproducible and can be extended with additional shuffles without recomputing existing ones.

**Inclusion criteria.** Only units labelled `good` or `mua` by bombcell were analysed; units labelled `noise` or `non_soma` were excluded. Units were required to have at least 500 spikes in the session to be classified.

**Classification.** Cells were classified from open-field activity as:

- **Grid cell (GC)** — significant grid score *and* significant spatial information;
- **Non-grid spatial cell (NGS)** — significant spatial information but not classified as a grid cell;
- **Non-spatial (NS)** — neither.

Classification was performed independently for the two open-field sessions (OF1 and OF2) recorded on each day, and both are reported.

To increase confidence that grid-like dynamics were genuinely captured and classified as grid cells, we analysed two open-field sessions acquired on the same recording day (OF1 and OF2). The motivation for using both sessions was to reduce the chance that cells with grid-like activity would be missed in one recording and subsequently misassigned to the non-grid spatial (NGS) population.

> **NOTE:** that rationale implies the operative rule is **grid cell in *either* session**, since the stated aim is to avoid false negatives leaking into the NGS population. Stating it explicitly matters, because it is also the more permissive rule and it directly inflates the GC count relative to requiring both.
>
> **This must be reported alongside it:** GC classification currently replicates between OF1 and OF2 in only **~24%** of cases. Under an either-session rule that low replication is precisely what generates the extra grid cells, so the two numbers have to appear together — otherwise the rule looks like a free gain rather than a bias/variance trade. Part of the instability is threshold noise at finite shuffle count (why the count was raised to 100), but the residual is a real property of the classifier.
>
> A useful control, cheap given both sessions exist: report the anatomical result (Figure 2) under both the either-session and both-sessions rules. If medial enrichment survives the strict rule, the permissive rule is a presentational choice rather than a load-bearing one.

### Grid module identification

Grid modules were identified **within each open-field session**, from the geometry of each cell's spatial autocorrelogram. Modules are defined over simultaneously recorded cells, so no analysis pools cells across sessions: a cluster formed across sessions could at best reflect a spacing band common to different animals, not a module.

**Feature space.** For every curated cell in a session — grid, non-grid spatial and non-spatial alike — the central disc of the spatial autocorrelogram was extracted, flattened and z-scored, giving a description of autocorrelogram *shape* independent of firing magnitude. All cells were embedded, not only grid cells: the question is where grid cells sit within the structure of the whole population, which cannot be asked if only grid cells are present.

**Embedding and clustering.** Cells were embedded in two dimensions with UMAP (Manhattan metric, spectral initialisation, `min_dist` = 0.05, `n_neighbors` scaled with session size) and clustered with DBSCAN (`min_samples` = 3). The DBSCAN neighbourhood radius was set per session by the **k-distance knee** — the point of the sorted k-nearest-neighbour distance curve lying furthest from the chord joining its endpoints — rather than by maximising a silhouette. Silhouette rewards many tight clusters and therefore collapses to the smallest radius available: it selected the sweep floor in 12 of 28 sessions, producing a median of 10 clusters (up to 25) and discarding 34% of cells as noise. On identical embeddings the knee criterion gave a median of 5 clusters and 2.5% noise.

**A cluster is called a module only if it is both grid-enriched and geometrically coherent**, requiring all of:

- at least 5 grid cells;
- a grid-cell fraction at least **2×** that of the session as a whole;
- an interquartile range of grid spacing ≤ 10 cm across its grid cells;
- a circular standard deviation of grid orientation ≤ 10°.

Enrichment alone is insufficient: a cluster can be grid-dominated while containing cells from several modules. The coherence criteria are what distinguish a module from a bin of grid cells, and in practice they reject clusters that an enrichment-only rule would have accepted.

**Grid cells outside every qualifying cluster are left unlabelled.** They remain classified as grid cells and simply carry no module assignment. Coverage — the proportion of grid cells receiving a label — is reported rather than maximised; a method that assigns modules confidently to some cells and abstains on the rest is preferable to one that forces every grid cell into a module.

**Parameter choice.** `min_samples` and the enrichment threshold were selected by sweeping `min_samples` ∈ {3,4,5,6} and enrichment ∈ {1.5, 1.75, 2.0, 2.5} on fixed embeddings, scoring each setting on module yield, coverage, cross-session reproducibility and module coherence. The 2× enrichment threshold is not a convention: cross-session reproducibility is 1.000 at 2.0 for every `min_samples` tested, falling to 0.897 at 1.75 and 0.768 at 1.5.

**Validation.** Because module identity is a property of a cell rather than of a recording, the two open-field sessions of each day provide an independent test. Module labels were computed separately in each session, and agreement assessed on the cells labelled in both, pairwise and label-invariantly (module indices are arbitrary per session): for every pair of such cells, whether the two sessions agree that they belong to the same module or to different modules. Agreement was **1.000 in 8 of 9 testable days** (0.78 in the ninth). Module geometry also recurred across sessions without being constrained to — for example M25 D24 yielded modules at 51.5 cm and 37.2 cm spacing in the first session and 51.5 cm and 34.0 cm in the second.

> **NOTE — what this method does and does not deliver.** It identifies a dominant, well-sampled module per session with high reliability; it does not resolve the full module structure. Roughly half of grid cells receive a label, and only a minority of sessions yield two or more modules, so module counts here are a lower bound. Sessions yielding no module are predominantly those with fewest grid cells, so absence of a module reflects sampling rather than absence of modular organisation.
>
**Yield.** Across 28 open-field sessions this identified **31 modules in 24 sessions**; 6 sessions yielded two or more. **480 of 962 grid cells (49.9%)** received a module label, the remaining 482 being grid cells without a module assignment. Modules were geometrically tight — median interquartile spacing range 3.6 cm and median circular orientation SD under 4°, against admission thresholds of 10 cm and 10°.

### Classification of anchored and non-anchored trials

Firing on the linear track was classified, trial by trial and cell by cell, as **anchored** or **non-anchored** to the task reference frame.

**This is not the spectral classifier used previously.** Anchoring on the linear track has previously been assessed by testing whether firing is spatially periodic at a frequency commensurate with the track (Clark & Nolan). That method was built for, tuned on and validated against grid cells, and its published form computed each label over a rolling window of five trials either side — eleven trials of support. Neither suits this paper, which classifies grid and non-grid cells with the same instrument and evaluates labels on individual trials. The classifier used here asks only whether a trial's spatial firing profile resembles the same cell's other trials, so cells are judged by self-consistency rather than by periodicity and no firing-field shape is advantaged.

**Procedure.** For each cell:

1. Assemble trial-by-trial firing rate maps into an *n*<sub>trials</sub> × *n*<sub>bins</sub> matrix (2 cm bins over the 200 cm track).
2. Smooth each trial's profile (Gaussian, σ = 1.5 bins) **NaN-aware**: unvisited position bins are missing data, not measured silence, so the smoothed map is normalised by a correspondingly smoothed validity mask. Zero-filling would assert a measurement never made, and because the unvisited pattern follows the animal's occupancy it is shared across simultaneously recorded cells — which would make trials correlate through the trajectory rather than through firing.
3. Mark trials whose summed smoothed activity falls at or below 10⁻⁶ as invalid; classify no cell with fewer than 6 valid trials. Invalid trials are **assigned to the non-anchored class**, not excluded. This operates on a substantial fraction of data — a median 12.3% of trials per cell — and because silence is correlated across simultaneously recorded cells it injects a shared component into the label matrix. Whether that component explains the population structure is tested directly below (*Coordinated silence*); it does not.
4. Correlate every valid trial's profile against every other (Pearson), giving a trial × trial correlation matrix, and summarise each trial by the **mean of its correlations with all other valid trials**, excluding the self-correlation on the diagonal. Note that this is a mean of *pairwise* correlations, **not** a correlation against the cell's average firing rate profile: the two are related but not identical, and the pairwise form avoids each trial being scored partly against itself through its own contribution to the mean profile.
5. Partition those one-dimensional values by 2-means clustering, and label the cluster with the higher centre — trials that are mutually self-consistent — as **anchored**. There is therefore **no fixed correlation threshold**; the boundary is set per cell by the data, which is what allows cells with very different field structure and signal-to-noise to be classified on the same footing. Empirically the two centres sit near −0.01 and +0.08, and the implied boundary falls at a median mean-correlation of **0.033** (IQR 0.015–0.063, 5th–95th percentile 0.003–0.126; 221 cells from three sessions), with a median silhouette of 0.60 for the two-cluster partition.
6. Median-filter the label *sequence* (size 5, not the firing data) to remove phase-aliasing false positives, where a non-anchored trial of a periodic cell aligns in phase with other non-anchored trials by chance. This raises specificity from 0.808 to 0.884 at negligible cost to sensitivity (0.990 to 0.993).

Because trials are compared by Pearson correlation, the classifier is invariant to the scale and offset of the input: scaling activity by 10⁻³ or 10³ returns bit-identical labels, so the same code applies unchanged to spike counts, calcium fluorescence and simulated network activations. The only scale-dependent parameter is the validity floor in step 3. The median filter in step 6 erases any run shorter than three trials, which sets the **temporal resolution at three trials** — against eleven for the spectral classifier. Recall of a non-anchored epoch is 0.02 at one trial and 0.07 at two, rising to 0.64 at three and above 0.80 from five onward. Analyses concerned with briefer events can disable the filter and accept the lower specificity.

**Validation.** Accuracy was measured against a simulated recurrent-network session with known anchoring ground truth that the classifier never saw (`anchoring_classifier_validation.ipynb`): the position cue was present for 35 trials, withdrawn for 30 so that the network's position estimate drifted off the task reference frame, then restored for 35. Across 1,024 simulated units, median accuracy was **0.990**, sensitivity **0.993** and specificity **0.884**. Accuracy depended only weakly on how grid-like a unit was (Spearman *r* = +0.219, *p* = 1.5 × 10⁻¹²; least grid-like quintile 0.970 against 0.990 for the most), supporting its use on grid and non-grid populations alike. The same classifier, unmodified, recovers experimenter-controlled cue visibility in two-photon calcium imaging of hippocampus (Pettit et al.), so agreement across three signal types is evidence about the phenomenon rather than about a shared measurement pipeline. On recorded data, where no ground truth exists, cluster separation was quantified by the silhouette score of the 2-means partition and reported per cell.

**Which cells define the population state.** Population-level anchoring — the per-trial fraction of cells labelled anchored, from which every population measure in this paper is derived — is computed over **medial entorhinal spatial cells only**, defined as cells classified as grid or non-grid spatial whose reconstructed position falls within ENTm.

This restriction is necessary rather than cosmetic. The recordings are not confined to MEC: across 11,611 spatial cells, 66.1% are in ENTm, with the remainder in visual cortex (14.9%), parasubiculum, presubiculum and subiculum (11.8%), and other structures (5.8%). The per-session proportion varies far more than that average suggests, from 0 to 100%, and 23 of 81 sessions are less than half ENTm. In 19 sessions there are no ENTm spatial cells at all: those probes sat in subiculum, presubiculum, parasubiculum, visual cortex or, in a few cases, cerebellum. Anatomical reconstruction is available for every one of these cells, so this reflects genuine electrode placement rather than failed registration. A population state computed over all recorded cells would therefore not be a measure of the medial entorhinal cortex in a substantial minority of sessions.

Restricting to ENTm removes 22 of 81 sessions at the 25-cell threshold, including all sessions from one animal (M22) whose probe did not reach MEC, and leaves a median of 112 ENTm spatial cells per retained session. The region filter is applied in one place (`spatial_cell_ids`, with `REGION_PREFIX = 'ENTm'`) and inherited by every downstream analysis.

**Implementation.** The classifier is implemented once, in `src/spatial_manifolds/anchoring.py`, and imported by every notebook that uses it; the module also defines the anchored/non-anchored colour convention used throughout the figures. It was previously duplicated inline across notebooks, which had allowed the copies to diverge in their handling of unvisited bins.

**Population anchoring state and its structure.** The per-trial fraction of cells labelled anchored summarises the population state, and its major axis of variation is the first principal component of the cell × trial label matrix, taken as a per-trial score. PC1's sign is arbitrary and is fixed by orienting it to correlate positively with the anchored fraction.

Whether this reflects a genuinely shared state, rather than the arithmetic of averaging, requires a null: cells anchored on most trials agree with one another most of the time by construction. Each cell's label sequence was therefore shifted **circularly and independently**, preserving every cell's anchored fraction and its run structure while destroying only the alignment between cells. Two statistics were compared against this null — the mean pairwise correlation between cells, and PC1's share of the label variance. Across 59 qualifying sessions PC1 exceeds its null in the large majority, typically by roughly a factor of two.

**Major population transitions.** Several figures mark the trials at which the population changes state. The raw thresholded state (anchored fraction > 0.5) changes too often to mark directly: these sessions carry ten or more runs, half of them only a few trials long, and drawing every change buries the raster under lines. A transition is therefore defined as a sign change of the thresholded population state **after a 9-trial median filter** (`scipy.ndimage.median_filter`, `mode='nearest'`), which removes runs shorter than about five trials while leaving the block structure intact. The same definition and the same filter width are used in every figure that marks transitions, so the lines are comparable across them.

Two alternatives were rejected. Requiring both runs adjacent to a boundary to exceed a length silently drops a boundary whenever a long block is interrupted by a brief one, so clear block edges went unmarked; filtering the state first does not have this failure. Marking the raw changes is faithful but unreadable.

Note that this filter is **separate from, and applied after, the size-3 median filter inside the per-cell classifier** (step 6 above). The classifier's filter smooths each *cell's* label sequence and sets the method's temporal resolution; this one smooths the *population* summary and exists only to decide which transitions are worth drawing. No statistic in the paper is computed on the filtered population state — it is used for annotation, and the switch-timing analysis locates transitions on the unfiltered state recomputed without the cell in question.

**Coordinated silence: tested, and not the explanation.** The classifier assigns trials on which a cell emits no spikes to the non-anchored class, and silence is correlated across simultaneously recorded cells because it largely reflects trials on which the animal did not run. This injects a shared component into the label matrix that has nothing to do with anchoring, and the circular-shift null cannot remove it, since shifting preserves each cell's marginal and therefore the same proportion of forced labels. Every population measure was therefore recomputed with silent trials **held as missing and dropped pairwise** rather than forced to zero.

The structure is unaffected. Across 9,464 cells the mean correlation with the population axis is 0.164 with silent trials forced and 0.165 with them excluded; by region the largest change in either direction is 0.003. Cerebellar cells, which have the lowest silent-trial fraction of any group (2.4% against 6.7% for entorhinal cells), change by −0.003. The population measures in this paper are therefore not measuring coordinated silence, and the control previously listed as outstanding is resolved.

**Per-cell agreement with the population axis.** Each cell's label sequence was correlated against the MEC PC1 score. For cells that are themselves part of the MEC population the axis was **recomputed by SVD with that cell removed**, since a cell otherwise helps define the axis it is scored against; cells outside MEC never enter the decomposition and need no such correction, and applying one to them as well would bias the two groups differently. Each cell was additionally compared against its own circular-shift null, and results are reported as the correlation minus that null, because a cell anchored on most trials correlates with a population that is also anchored on most trials irrespective of any coordination. Roughly a quarter of MEC spatial cells are individually locked to the axis (median 27% per session, against a 5% false-positive rate), so the state is carried by a minority subpopulation rather than entered by the whole of MEC together.

**Mediolateral position.** Within MEC, cells were split at 3400 µm. Scoring both halves against the joint MEC axis is not a fair comparison, because each half helps define the axis it is measured against and the larger half dominates it. The axis was therefore built from each half **separately**, with cell counts matched by subsampling before the decomposition, and used to score both halves — giving a medial→medial, medial→lateral, lateral→lateral, lateral→medial matrix. Because grid cells constitute 10.9% of medial but only 3.3% of lateral cells and follow the axis more strongly, every comparison was repeated on non-grid cells alone.

> **RESOLVED, then re-corrected (2026-10-02).** This previously read: *cells outside MEC follow the MEC axis at least as strongly as cells within it (non-MEC +0.178 against MEC +0.166, p = 0.034), pre-, para- and postsubiculum highest of any region (+0.265), visual cortex +0.141.* Those were raw correlations. The first correction replaced them with "excess over each cell's own circular-shift null", under which only entorhinal groups exceeded chance and **visual cortex sat below its own null** (−0.039). **That correction was itself wrong**: it subtracted the mean of an *absolute* null from a *signed* correlation, and since the null is centred at zero (signed mean −0.008 over 12,722 cells) the subtraction biases every cell down by ≈ 0.8 SD of its own null — differentially by region, because null widths differ. See Results §6. The settled position: chance for the signed correlation is **zero**; every region recorded follows the axis above it, visual cortex included (+0.099, *p* = 2.2 × 10⁻⁷, 18.5% of cells individually significant against 5% expected); and the regional difference is graded, established by within-session paired contrasts with every cell scored out-of-sample (grid over visual +0.149, *p* = 9.9 × 10⁻⁶). The subicular-vs-visual contrast is underpowered at 14 sessions. The cerebellar "negative control" was never a control — it follows the axis too (+0.163 on 5 sessions) — and the behavioural claim is not region-specific at all, a visual-cortex state carrying a larger ADI (+0.148) than the entorhinal one (+0.085).

### Single-unit identity and its relation to the population state

**Identity comes from the open field, by design.** Every cell-type label used in this figure — grid, non-grid spatial, non-spatial — is taken from the open-field recordings (`cell_class_of1`), and the speed score likewise from open-field running, never from the virtual-reality session under analysis. The linear-track task is the setting whose dynamics are in question: firing there varies with engagement, motivation and task demand across a session, and those are precisely the variables the anchoring state tracks. Classifying cells on the same data would let the effect define the categories and then recover itself. Assigning identity in a different environment makes the categories independent of the state they are used to explain, at the cost of any cell whose tuning differs between environments being mislabelled for the track — a cost we accept as the conservative direction, since it can only blur differences between classes, not manufacture them. The one exception is the interneuron call below, which is derived from the spike waveform and is therefore environment-independent.

**Putative interneurons.** No interneuron label exists for this dataset, so one was derived from the spike waveform. A two-component Gaussian mixture was fitted to log peak-to-trough duration and the threshold taken as the antimode, rather than set by convention. The distribution is convincingly bimodal — component means 0.246 and 0.690 ms, separated by 4.03 SD — giving a threshold of 0.423 ms. Firing rate was fitted the same way and is **not** comparably bimodal (components 2.3 and 8.8 Hz, 1.81 SD apart), so rate cannot define the group and is used only as a second requirement: putative fast-spiking cells are narrow-waveform **and** in the top quartile of firing rate (≥ 10.2 Hz). This yields 842 of 1,495 narrow cells; the 653 narrow-but-slow cells excluded are an acknowledged ambiguity. As an independent check on a property they were not selected for, putative interneurons are the most speed-modulated group in the open field (*p* = 7 × 10⁻³¹).

**Open-field speed score.** Each session ships a time-shifted speed GLM (`time_shifted_S_glms.parquet`), and it is **not used**: its median score is negative for nearly every cell and only ~1% of entorhinal cells reach FDR significance, so a null result against it could not distinguish absent speed modulation from an insensitive statistic. Speed modulation was instead computed directly, in the standard form: the Pearson correlation between a cell's instantaneous firing rate and running speed in 200 ms bins (boxcar smoothed over 2 bins) restricted to moving periods (≥ 2 cm/s), tested against a circular-shift null of the full rate trace (200 shifts). This reaches significance in ~51% of cells and replicates across OF1 and OF2 at *r* = 0.75, which is what licenses the negative results that rest on it. OF1 is used where both sessions exist, with OF2 retained as the reliability estimate rather than averaged in.

**Dissent and locking.** A cell can disagree with the population in two ways that the correlation cannot separate — it can be anchored when the population is not, or non-anchored when the population is. Both were computed separately: `stay` = P(cell anchored | population non-anchored) and `drop` = P(cell non-anchored | population anchored), each requiring at least 5 trials in the conditioning state. Cells whose labels do not vary at all (zero label SD, i.e. the absolute gate assigned every trial to one class) have an undefined correlation and are analysed separately as **locked**. Analyses of locking are restricted to sessions where the population itself switches (≥ 10 trials in each state, 35 of 62 sessions), because in a single-state session every cell is locked by construction; the restriction strengthens the reported asymmetries rather than producing them.

**What a non-anchored trial contains.** Each trial's smoothed map was cross-correlated against the cell's **anchored template** — the mean map over its anchored trials only, so the template is not contaminated by the trials scored against it — at every circular offset, recording the correlation at zero offset, the best correlation over offsets, and its offset. Cells with fewer than 8 anchored trials were not given a template. The best-over-offsets statistic is a maximum over 100 offsets and is therefore large by construction, so two nulls were computed on a 10-session subset: a bin-shuffled trial map, which destroys spatial structure while preserving the rate distribution (+0.189), and a **different cell's template from the same session**, which preserves real spatial structure on both sides (+0.500). The second is the operative floor, and reporting the best-offset correlation without it inverts the conclusion.

**Firing rate by state.** Rate was computed two ways, and the distinction matters. A rate averaged over the position bins of a rate map divides by **occupancy**, which running speed sets; a rate computed as spike count divided by running duration does not. The two correlate at *r* = 0.96 across cells but differ in mean (6.84 against 9.03 Hz), which is enough for a ~3% state difference to change sign between them. Rates are reported time-averaged, over running samples only, with trial duration also taken as running duration so that a trial on which the animal stopped often is not scored as low-rate for that reason. Speed matching within session, in deciles of trial mean speed with equal counts per decile, is reported alongside. **Every split uses the population state, never the cell's own label**: a trial with more spikes yields a better-estimated map, which is more likely to match the others, which earns the anchored label — scored that way the effect reverses.

**Agreement between identity groups.** Whether two identities are coordinated with each other, rather than each independently tracking the population, was tested by correlating every pair of cells' label sequences within session and averaging by identity pair. The diagonal is the mean within-group pairwise correlation excluding self-pairs. Because two cells that are each anchored on most trials correlate for that reason alone, every entry is reported as excess over a null in which each cell's sequence is circularly shifted **independently** (20 draws per session), which preserves each cell's anchored fraction and run structure and destroys only the alignment between cells. Restricted to sessions in which the population switches and to cells whose labels vary; locked cells have no correlation to compute. The matrix is symmetric by construction and is cached to `identity_agreement.csv`.

**Switch timing.** Implemented in `transition_profiles.py`. Each cell's label sequence was cut in a ±6-trial window around every population state transition and oriented so that positive denotes entering the anchored state; the cell's switch point is the steepest rise of its averaged profile and the lag is that point minus the population's. Transitions were found on the population state recomputed **without the cell in question**. Note that "do cells lead the population" is not a testable question as posed: the population state is the fraction of cells anchored thresholded at one half, so it crosses when half the cells have switched and ~50% of cells must lead whatever the biology. Only the *difference between identities* is informative, and it is tested by likelihood ratio between mixed models with and without an identity term; it is not significant (*p* = 0.15). Figure 2F shows the aligned profiles over ±5 trials.

**Reliability of the agreement score.** Trials were split into odd and even — which keeps the halves matched for slow within-session drift in a way a first/second split does not — and the agreement score recomputed within each half using the leave-one-out population state of that half. The split-half correlation across cells is 0.939, or 0.969 after the Spearman–Brown correction for full-session length, and is uniform across identities (0.952–0.985). Attenuation of the identity comparisons is therefore negligible, and no identity is measured appreciably less reliably than another.

**Anatomical axes.** `coord_SCs_x` is the mediolateral coordinate (stored negative; magnitude 2610–3774 µm, and the axis on which the 3400 µm medial/lateral boundary is defined), `coord_SCs_y` is dorsoventral and `coord_SCs_z` is anterior-posterior. Positions were z-scored within mouse before modelling, since animals differ in probe placement and raw coordinates would let between-animal placement appear as a within-animal gradient. Each axis was tested alone and with cell identity added, since identity is itself anatomically non-uniform.

**Hierarchical statistics.** Cells are nested in sessions in mice, so comparisons over cells treat correlated observations as independent. All identity and anatomy comparisons in this figure are linear mixed models with random intercepts for mouse and for session within mouse, with pairwise contrasts Holm-corrected. The cell-level rank tests were four to eight orders of magnitude smaller, and one conclusion changes: grid cells and putative interneurons differ at *p* = 0.045 over cells and *p* = 0.84 once the hierarchy is modelled.

### Putative monosynaptic connections

Reported in Figure 4.

Implemented in `monosyn.py`. Spike trains were cross-correlated in 1 ms bins over ±50 ms and compared against a hollow-Gaussian predictor (σ = 5 ms, 60% hollow) that preserves slow co-modulation while removing the millisecond structure a synapse produces. Detection is a Poisson test with continuity correction on the causal window 0.7–4.7 ms, Šidák-corrected across the bins examined, with the zero-lag bin excluded and minimum spike counts enforced on both cells.

This is the Stark & Abeles / English et al. convolution-predictor family, and differs from the common variant that convolves with a plain Gaussian and thresholds each bin at a fixed percentile in three respects. The predictor is **hollow** — 60% of the kernel mass at the centre is removed — so a genuine peak does not contribute to the baseline it is tested against; with a plain kernel a strong peak inflates its own predictor and the test becomes conservative in exactly the cases of interest. Multiplicity is handled by **Šidák correction** across the causal bins rather than by requiring consecutive significant bins; at 1 ms binning the two are close to equivalent, since the usual "two consecutive 0.5 ms bins" rule asks the deviation to span 1 ms, which one of our bins already does. And the per-bin α is **3 × 10⁻⁴** rather than the conventional 10⁻³, set from a jitter null rather than by convention (below).

**Rejection rules and their correction.** The published criteria reject a pair if any non-peak bin exceeds 2.5 SD, or if any anticausal bin reaches *p* < 0.01. Applied per-bin and uncorrected across the 101 bins of the window, these fire by chance on ~45% and ~39% of pairs respectively and together reject roughly two thirds of all pairs regardless of content — on one test session they returned zero connections in 17,366 pairs, discarding pairs with *z* = 10.4 causal peaks. Both rules are given the same Šidák correction the detection receives, across the number of bins each actually examines. Nothing else about the criteria was changed.

**Setting the threshold from a null, not from convention.** Relaxing a rejection can only increase the count, so a higher count is not evidence the correction is right. The operating point was set from a **jitter null**: the same spikes displaced by ±10 ms, which destroys millisecond timing while preserving firing rate, slow co-modulation and the coarse correlogram shape. Sweeping the peak threshold on one session (20,052 pairs), α = 3 × 10⁻⁴ is where the null is essentially empty while the real count is still near its maximum. Across the full dataset that threshold gives a false-discovery rate of 0.09 for excitatory and 0.10 for inhibitory connections, estimated from the jitter null run in every fourth session (15 of 60).

**Inhibitory connections.** The criteria as published detect a causal *peak* and therefore excitatory connections only; an inhibitory connection appears as a *trough* and cannot be found by them at any threshold. `detect(kind='inh')` mirrors every criterion onto the lower tail of the same Poisson test, taking the causal extremum as a minimum, with the rejection rules, corrections, window and minimum counts unchanged, so the two signs are held to identical standards and each carries its own jitter null. Two properties external to the detector validate the result: inhibitory latencies are slower than excitatory ones (median 3.0 against 1.0 ms), and inhibitory connections originate from waveform-classified interneurons at 3.1× their population share while excitatory connections do not — the waveform classification played no part in detection.

A third check addresses the specific reason inhibitory detections are often discarded, namely that they are the far side of an excitatory connection in the *opposite* direction, produced by a short-timescale oscillatory process rather than by a synapse. That pattern is essentially absent here: of 7,775 inhibitory connections, **19 (0.24%)** have a detected excitatory connection in the reverse direction and 7 (0.09%) in the same direction. The ±10 ms jitter null speaks to the same concern from the other side, since it preserves the coarse correlogram shape and any slow co-modulation while destroying millisecond timing.

**Connection probability by identity.** Reported as a percentage of the *ordered pairs actually testable*, counted per session from the number of cells of each identity present, rather than as a share of detected connections, so that a group's abundance does not inflate its apparent connectivity. Enrichment was tested by permuting identity labels among the cells of each session (2,000 shuffles), which holds cell counts, connection counts and probe geometry fixed and moves only the labelling.

**Do connected pairs share the anchoring state?** Every within-session pair of cells whose labels vary was scored by the Pearson correlation of their label sequences (≥ 20 shared trials). Connected pairs were compared against unconnected pairs drawn from the **same session**, of the **same identity pair**, at a similar probe distance (±40 µm, up to 20 controls per connected pair). All three constraints are necessary: a monosynaptic peak is only detectable between cells the probe records well, which are usually neighbours; neighbouring cells share anchoring through common input, shared local field and spike-sorting bleed, which manufactures the connection and the correlation together; and identity alone predicts agreement, so an unmatched comparison would recover the identity effect of §2 and report it as a wiring effect. Cells with constant labels are excluded throughout — under the absolute gate about a third of cells are all-anchored or all-non-anchored, and a constant vector has no correlation with anything. Comparisons are Mann–Whitney on connected against pooled matched controls. Cached to `data/monosyn/pairs_by_identity.csv`.

**Power, and which nulls are reportable.** These comparisons are the paper's only substantive null, so each is accompanied by the minimum effect it could detect (two-sided α = 0.05, 80% power, pooled SD). The benchmark is the effect the null is denying — the identity effect of §2, ~0.11. Grid–interneuron (*n* = 193, MDE 0.045) and interneuron–interneuron (*n* = 116, MDE 0.043) resolve effects roughly a third that size, so their nulls carry information; the grid–interneuron result holds even under the conservative assumption that the matched controls contribute no more information than the connected pairs themselves (MDE 0.060). Grid–grid does not: 27 connected pairs give MDE 0.157, larger than the effect being tested for, so its *p* = 0.31 is reported as underpowered and is excluded from the claim. The remaining heterogeneous pool reaches significance (*p* = 0.032) on an effect of +0.006, an order of magnitude below the identity differences in the same panel and resolvable only because it contains 3,238 pairs; it is not interpreted as a wiring effect.

### Eye tracking: pupil size

Reported in Figure 4 (`figure5_pupil_arousal.ipynb` → `fig5_pupil_arousal_M21D19.pdf`).

The eye contralateral to the probe was filmed at 30 Hz during virtual-reality sessions and the pupil tracked with DeepLabCut, giving eight perimeter landmarks (n, ne, e, se, s, sw, w, nw), each with a likelihood, plus a derived centroid. Eye tracking is available for 80 of 81 recorded virtual-reality sessions.

**DeepLabCut training.** A single network was trained across all animals rather than one per mouse: 200 frames were labelled from one session per animal, and the pooled set used to train one model. Training on every animal together means the same network, with the same learned notion of where the pupil boundary lies, is applied to every session, so a between-animal difference in pupil size cannot arise from a between-animal difference in the labeller or the model. It also avoids a per-animal model overfitting the handful of sessions it was trained on. Each frame carries the eight pupil-perimeter landmarks described below, and the per-frame likelihoods the network returns are used as the tracking-quality criterion.

**Pupil size.** Pupil size is the mean Euclidean distance from the eight perimeter landmarks to their centroid. The centroid was verified to be exactly the centroid channel supplied in the file (*r* = 1.00000, maximum absolute difference 0.0000 px). We use this landmark-derived radius rather than the `eye_dilation` channel supplied alongside it, because that channel is not reproducible from the landmark geometry — it correlates with the landmark-derived mean diameter at only *r* = 0.56–0.86 across sessions and is approximately twice its magnitude, so its definition cannot be recovered from the data and it cannot be audited.

**Blink and tracking-loss rejection.** Frames whose mean landmark likelihood falls below 0.5 are discarded (typically 1–3% of frames; sessions exceeding 20% are excluded entirely). The centroid itself carries no likelihood, being derived, so the mean over the landmarks it is computed from is used instead.

**Head-movement artifacts.** Although the animal is head-fixed, the head can shift within the headplate, and tracking can fail transiently. Frames in which the landmark centroid departs from its local baseline (median over a ±2 s window) by more than 0.8 pupil radii and returns within a few frames are discarded: these are tracking failures and partial occlusions by the lid or a whisker, and they corrupt the landmark geometry from which the radius is computed. Thresholds are expressed in units of each session's own median pupil radius rather than in pixels, because camera distance and zoom differ between sessions (median radius 7.8–12.7 px), so a fixed pixel threshold would correspond to a different physical displacement in each.

The pipeline additionally detects sustained steps, in which the head slips to a new position within the headplate, and subtracts the cumulative offset. **That correction does not affect any result reported here**, because pupil radius is the mean distance from the perimeter landmarks to their own centroid and is therefore invariant to translation of the eye within the camera frame. It is retained in the code, and described here only so that the processing applied to the recordings is on record.

**Per-trial measures.** Pupil radius is binned onto the same trial × position-bin grid as the neural rate maps (2 cm bins, `moving` epochs only), so that each column corresponds to the same stretch of track in both. Two per-trial quantities are then formed:

- **Pupil radius**: the mean binned radius over the trial.
Values are **z-scored within session** before pooling across sessions, because the raw pixel scale depends on camera placement and is not comparable between sessions.

**Session inclusion.** A session contributes if fewer than 20% of its frames fail the likelihood threshold, it retains at least 40 trials with position-bin coverage of 0.5 or better, and — the operative criterion — it has at least **30 trials in each of the anchored and non-anchored classes**. Fifty-seven of eighty sessions qualify.

The per-class minimum was calibrated rather than assumed. We took the 24 sessions with at least 90 trials in both classes, subsampled them to *n* trials per class, and measured the spread of the resulting estimate around that session's own full-data value. The estimator is unbiased at every *n* (|bias| ≤ 0.01 z), so the criterion concerns precision alone:

| *n* per class | 5 | 10 | 15 | 20 | 25 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|
| SD of estimate (z) | 0.61 | 0.42 | 0.34 | 0.28 | 0.25 | 0.22 | 0.18 | 0.16 |

Against an effect of approximately 0.55 z, a session with 20 trials per class has a 95% error band of ±0.55 — as wide as the effect itself, so such a session cannot establish even the sign of its own estimate. At 30 per class the SD is about half the effect, which is where a per-session value becomes individually interpretable; 40 per class buys little further precision at the cost of five more sessions.

The pooled result does not depend on this choice: the median session effect runs from −0.561 with no per-class minimum to −0.507 at a minimum of 50, with *p* ≤ 3 × 10⁻⁹ throughout. The threshold determines which individual session values may be read, not whether the effect exists. It does, however, matter for the per-session picture, because the excluded sessions carry the largest magnitudes (down to −2.2 z) — the expected signature of noise inflating |effect| in small samples, and the reason the unfiltered per-session distribution has a heavier tail than the effect warrants.

**Track profiles.** To locate where along the track the states differ, the binned quantities were averaged across trials within each state and within session, and then across sessions, retaining the position axis. Profiles are reported both in absolute terms — alongside the all-trial mean, computed across every trial rather than as a weighted average of the two states, since the two classes do not share the same valid bins — and as each state's departure from that all-trial mean. The second form is necessary because the position profile is much larger than the state effect: pupil radius varies by approximately 2.5 z across the track, against a between-state difference of 0.24, so on an absolute axis the two states are visually indistinguishable.

**Video frames and their alignment.** Example figures showing the eye require individual video frames, and these align to the behavioural data exactly, with no synchronisation step: for each session the number of video frames, the number of rows in the camera capture log, and the number of eye samples in the NWB file are identical (58,925 for M21 D19, 62,634 for M28 D18), all at 30.000 fps. Video frame *i* is therefore eye sample *i*, and the NWB timestamps that sample on the ephys clock.

Two details cause silent errors if overlooked. First, trial numbering: the anchoring analysis uses trials **clipped to the ephys window and renumbered from 1**, whereas the NWB trials table retains the raw numbering (M21 D19's clipped trials are raw 10–116), so an unmapped lookup is displaced by nine trials. Second, coordinate frames: the DeepLabCut landmark coordinates in the NWB are relative to a **cropped** region, whose origin is given per session in `all_eye_crops.csv`. Files named `*_eye_zone.avi` are already cropped, but more tightly than that table records, so their landmarks require a further fixed offset. That offset was solved from the data — matching the centroid of the dark pupil to the NWB centroid across sampled frames — and verified visually before use: (−30.4, −31.4) px for M28 D18 and (−31.0, −31.7) px for M29 D17, with a between-frame SD of under 2 px in each case.

**Error bars on position profiles.** Pupil radius binned by track position is summarised as the mean ± SEM **across trials**, not across camera frames. Frames within a trial are highly correlated — the pupil changes little in 33 ms — so treating the several hundred frames contributing to a position bin as independent understates the standard error by a factor of two to three (0.03 px against 0.11 px in a representative session). The trial is the independent unit.

**Bin-occupancy correction.** The tuning-curve routine used for binning computes bin occupancy from the dense (~60 Hz) position trace but averages the sparse (~30 Hz) eye-camera samples, and initialises empty bins to zero rather than to NaN. A position bin the animal visited but which contains no eye-camera sample therefore retains a placeholder value of exactly 0. Since a true pixel coordinate of 0 is impossible here (raw values never fall below ~90), such bins are identified by exact equality with zero and set to NaN. Approximately 3–4% of bins are affected.

**Peri-transition analysis.** To test whether pupil radius tracks the anchoring state itself rather than some session-long covariate of it, trials were aligned to the first trial of a new population state, separately for entries into the anchored and into the non-anchored state.

Windows were required to be **state-pure**: the population state must be constant across the whole of the four trials preceding the transition and constant across the four trials following it. This is stricter than requiring a minimum run length either side, and the distinction is consequential. With a three-trial run criterion the state outside that run flickers substantially — eight trials before a transition, 46% of trials are already in the state being entered, and only 15% of windows are free of flicker across ±8 trials. Because that contamination pulls the pre-transition plateau toward the post-transition value, it both understates the total change and overstates the fraction of it occurring in a single trial: entry into the non-anchored state appears as a 95% one-trial step under the run criterion, but as 31% of a change twice as large on state-pure windows. A window of ±4 trials retains 196 and 198 transitions (entries into the anchored and non-anchored state respectively) across 56 sessions; ±6 trials would retain fewer than half as many.

Segments were averaged within session before averaging across sessions, because switch rate varies several-fold between sessions and pooling raw transitions would weight sessions by how often they switched. Pre- and post-transition means were compared by Wilcoxon signed-rank test across sessions.

On state-pure windows pupil radius changes in the expected direction at the transition — −0.414 entering the anchored state and +0.493 entering the non-anchored state (*p* = 1.7 × 10⁻⁷ and 4.4 × 10⁻⁸) — and does not change instantaneously: the single trial spanning the transition accounts for only 16–31% of the total change. The classifier's size-5 median filter smooths the label sequence over ±2 trials, so the data cannot distinguish a genuinely graded change from an abrupt one whose detected onset is imprecisely timed, and we make no claim about which it is.

**Pupil size against performance, and the test for an inverted U.** Whether pupil size indexes drowsiness at one end and distraction at the other — the classical inverted-U — or varies monotonically with engagement over the range an animal actually occupies is an empirical question, and it decides how the anchored state should be described. Trials from `eye_anchoring_trials.csv` were merged with the behavioural trial table (14,011 trials, 58 sessions) and pupil radius *z*-scored within session, since absolute radius differs with camera placement. Hit rate was then computed by decile of within-session pupil *z*, and modelled as a mixed model with a session random intercept, with and without a quadratic term in pupil *z*. A genuine inverted U requires the quadratic coefficient to be **negative** with a turning point inside the data; here it is positive (+0.018, *p* = 3 × 10⁻¹⁵) with a minimum at *z* = +3.4, so within the observed range the relationship is monotonically decreasing (−0.123 per *z*). Because pupil size and running speed covary in general, the model was refitted with within-session *z*-scored trial mean speed as a covariate; mean speed is flat across pupil deciles (37–40 cm/s) and the pupil coefficient is unchanged (−0.128, quadratic +0.011), so the relationship is not a behavioural artifact. The same analysis restricted to uncued trials gives the same signs.

### Anchoring Dependence Index

The **Anchoring Dependence Index (ADI)** quantifies how much more the population anchoring state matters to behavioural success when the reward zone is *not* cued — that is, when the animal must locate it by path integration rather than by sight.

**Definition.** For each session, the per-trial **population anchored fraction** (the proportion of simultaneously recorded spatial cells labelled anchored on that trial, from the classifier above) was used to discriminate rewarded from unrewarded trials, separately within each trial type, and the discriminability quantified as the area under the ROC curve. ADI is the difference

> ADI = AUC(uncued) − AUC(cued)

Positive values mean the anchoring state carries more information about success when the reward zone is not cued. Sessions entered the analysis if they had at least 25 classified spatial cells and at least 25 trials of each type with both outcomes represented; 78 sessions qualified.

**Why AUC rather than correlation.** The obvious metric — the correlation between anchored fraction and trial outcome — is the wrong one here, because the two trial types have very different base rates (median hit rate 0.85 cued, 0.55 uncued). With little outcome variance to explain, *r* on cued trials is attenuated for reasons that have nothing to do with anchoring. AUC is formally invariant to class prevalence and does not have this defect. Consistent with that, the correlation-based index is roughly 50% larger than the AUC-based one (+0.148 vs +0.100) and correlates twice as strongly with the cued hit rate (ρ = +0.449 vs +0.221), which is the signature of the artifact rather than of a larger effect.

**Result.** Across all 78 sessions the anchored fraction discriminated outcome better on uncued than on cued trials: median AUC 0.733 uncued against 0.593 cued, **ADI = +0.100**, Wilcoxon signed-rank *p* = 1.0 × 10⁻⁴, positive in 52 of 78 sessions.

**Caveat — the measure is unreliable near ceiling and floor.** Prevalence invariance is a property of AUC as an estimand, not a guarantee about its estimate. When performance on a trial type approaches ceiling or floor, the AUC for that type rests on a handful of trials of the minority outcome, and those trials are a *selected* sample: the few failures in a session with an 0.85 hit rate are unusual events — a disengaged trial, a missed stop on an otherwise correct approach — not representative ones. The quantity being estimated is then no longer comparable between the two trial types, and since cued trials sit closer to ceiling than uncued trials in almost every session, this asymmetry runs in the same direction as the effect. The all-session ADI is positively correlated with the cued hit rate (ρ = +0.221, *p* = 0.052), which is what this concern predicts.

**Control — restricting to sessions with power on both trial types.** We therefore repeated the analysis on sessions where **both** trial types had hit rates within [0.2, 0.8], so that both AUCs are estimated from a substantial minority class. This is a power criterion, not a data-quality one: no session is excluded for being badly recorded. Fifteen sessions met it, and the effect was unchanged in size and remained significant — cued AUC 0.626, uncued AUC 0.701, **ADI = +0.103**, *p* = 0.0012, positive in 13 of 15. The result is not an artifact of asymmetric ceiling effects. It is also not sensitive to where the bounds are placed: [0.15, 0.85] gives ADI +0.110 (*n* = 24, *p* = 3.2 × 10⁻⁴) and [0.1, 0.9] gives +0.110 (*n* = 39, *p* = 3.5 × 10⁻⁵).

The criterion must be applied to **both** trial types. Restricting only on the cued hit rate (< 0.8), which is the intuitive move since cued trials are the ones near ceiling, admits sessions where the *uncued* hit rate is near floor and gives ADI = +0.035 (*n* = 31, *p* = 0.23) — the effect appears to vanish. This is the same failure at the other end of the scale: a session in which the animal almost never succeeds on uncued trials supplies an uncued AUC estimated from a few atypical successes.

**Prevalence matching was considered and rejected.** The obvious alternative to excluding sessions is to subsample the majority outcome within each session so that the two trial types have equal hit rates. This does not work, and we state it because it is the first thing a reader will propose. Matching leaves ADI essentially unchanged (+0.103 matched vs +0.100 raw, *p* = 8.5 × 10⁻⁵, positive in 41 of 60 sessions) and leaves its correlation with the cued hit rate slightly *worse* (ρ = +0.298 matched vs +0.221 raw, *p* = 0.021). Equating the rate does not equate what a miss *is*: a miss drawn from a session at 0.85 remains an atypical event after subsampling, because subsampling discards hits and cannot manufacture representative misses. The inclusion criterion addresses the problem at its source; matching only rescales it.

**Regional specificity.** The index is not specific to the medial entorhinal cortex. Recomputing the population anchored fraction from MEC cells alone, in the 59 sessions where both an all-cell and an MEC-only population state could be formed, gives ADI = +0.101 (*p* = 0.001, 37/59) against +0.099 for all cells in the same sessions; the paired difference is null (*p* = 0.81, MEC larger in 23 of 59). This is reported as a negative result, not a control that passed.

### xgBoost encoding models

> **⚠ ORPHANED — no Results section now cites this.** The three xgBoost Results sections (co-recorded cells improving prediction above position; medial vs lateral non-grid cells predicting grid cells; proximal vs distal non-grid cells under task-independent firing) were removed on 2026-10-01 because none of them appears in the seven-step argument. The method is kept here because the analyses and figures still exist on disk and the description is accurate, but it supports no current claim. Delete it if the analyses do not return; if they do, the `[Controls]` gap — cell-count matching, firing-rate matching and a distance control — has to be closed first, since the one medial/lateral effect that was tested did not survive rate-matched stratification (*p* = 0.75).

Neuronal firing was modelled with gradient-boosted regression trees (XGBoost) configured for Poisson output, using the `MLencoding` framework of Benjamin et al. (2018).

**Binning and covariates.** Spike trains and behavioural variables were binned at 10 ms. The base covariate was track position (the animal's cumulative travel modulo the 200 cm track length). Neural covariates were the binned spike counts of simultaneously recorded cells. All covariates were supplied with temporal history: each covariate was projected onto a basis of raised-cosine temporal filters spanning the preceding 1000 ms (100 filters at 10 ms resolution). Spike history of the target cell itself was not included, so that predictions cannot be trivially satisfied by autocorrelation.

**Model parameters.** Objective `count:poisson`; learning rate 0.05; maximum depth 5; 580 estimators; subsample 0.6; minimum child weight 2; gamma 0.4.

**Cross-validation.** Models were fit with 10-fold cross-validation using *contiguous* folds: the recording was divided into 10 consecutive blocks, each serving once as the test set with the remainder as training data. Contiguous rather than randomly interleaved folds are required here because the temporal history filters span 1000 ms, and random bin assignment would leak training data into the test set.

**Scoring.** Model performance was quantified as the Poisson pseudo-R²,

> pR² = 1 − (L_S − L_1)/(L_S − L_0)

where L₁ is the log-likelihood of the model, L₀ the log-likelihood of a homogeneous-rate null model (mean firing rate of the training data) and L_S the log-likelihood of the saturated model. A value of 0 indicates performance equal to the mean-rate null and 1 indicates a perfect fit. Where performance is reported for a subset of trials or for a behavioural condition, pR² was recomputed on the time bins belonging to that subset using the same fitted predictions and the session-wide mean rate as the null, so that values remain comparable across conditions.

**Covariate populations.** For the analyses of coordination, covariate cells were drawn from a defined population (co-modular grid cells, non-co-modular grid cells, or non-grid spatial cells) and added incrementally, ordered by physical distance from the target cell (nearest first, farthest first, or in random order as stated for each analysis). Distance was computed in the probe-plane coordinates (`coord_SCs_x`, `coord_SCs_y`) of the reconstructed recording sites. Reported comparisons are made at matched numbers of covariate cells.

**Medial and lateral covariate sets.** For the mediolateral analyses, non-grid spatial cells were divided into medial (`coord_SCs_x` < 3400 µm) and lateral (`coord_SCs_x` ≥ 3400 µm) populations, and each was used as the covariate set in otherwise identical models.

> **TODO:**
> - **Coordinate systems** are defined in *Anatomical coordinates and the mediolateral axis* above; Figure 2 and Figure 5 use the same plane, related by a constant 5700 µm offset.
> - **Justify 3400 µm.** State whether this was chosen from the Figure 2 distributions, and show that the result is not knife-edge dependent on it (a sweep of boundary position is the natural control).
> - **Matching.** State explicitly how medial and lateral covariate sets were matched for cell count and for firing rate. This is the `[Controls]` gap in Results §4.
> - Specify whether the LFP covariate (Figure 4 v2) is a raw band-limited signal, theta phase, or theta power, and its band.

### Statistics

> **[MISSING] — to write.** Required:
> - GLMM specification for the anatomical distribution comparisons in Figure 2 (response, link function, fixed effects, random effects for mouse and session), and the software used.
> - How non-independence of simultaneously recorded cells is handled throughout. Pooling cells across sessions as independent observations inflates significance; comparisons should be made at the session level, or with session as a random effect, wherever the claim is about a population rather than a cell.
> - Multiple-comparison correction across the anatomical axes (AP, DV, ML) and across cell types.
> - Definition of the error bars and centre points in every figure.
> - Exact tests, test statistics, degrees of freedom, and n (cells, sessions, mice) for each reported comparison.

### [MISSING] Data and code availability

---

## Open items — consolidated

**Blocking:**
1. ~~**Curation conflict** — supplied Methods specify SpikeInterface quality thresholds; the classification pipeline uses bombcell `good`+`mua`.~~ **Resolved 2026-10-01:** curation is bombcell only, matching the pipeline; the quality-metric thresholds have been removed from Methods.
2. **Cell counts** everywhere (`n = ?`) — waiting on the classification sweep.
3. **§7's mechanistic controls are still pre-gate.** Updated 2026-10-03: the section now *leads* with post-gate results — the full spectrum, theta–gamma coupling and gamma events (26 switching sessions) — and the amplitude and speed-coupling paragraphs are post-gate too. What remains pre-gate is everything from the warning table onward: the speed-tuning, phase-locking, residual-speed, regional and laminar controls, in ten CSVs in `data/lfp/PREGATE/`. Nothing below that table should be quoted until re-run. The band-power file is **superseded** rather than pending — `spectrum_by_anchoring_w*.csv` replaces `lfp_power_by_anchoring.csv` and answers the broadband question that the old file raised.
3a. **Re-run the slow-gamma modulation index on a clean 30–48 Hz band.** The current slow-gamma PAC band contains the 50 Hz mains notch and its absolute MI is a fifth of fast gamma's, so "slow gamma does not uncouple from theta" (*p* = 0.71) rests on a weak null and is currently reported as an unsupported difference rather than an invariance.
3b. **Settle the slow/fast gamma question within session.** The two changes co-occur in 16 of 26 sessions but are uncorrelated in magnitude across sessions (ρ = −0.08), so the paper says "both move in opposite directions" and not "redistribution". A trial-by-trial slow-against-fast correlation within session, with speed and state partialled out, is the test that would decide it.
4. **Discussion is the superseded argument in its entirety** and must be rewritten against the seven steps. See the warning at the head of the Discussion.

**Structural:**
5. ~~Figure numbering collisions (two Fig 6, two Ext Fig 4, no Fig 7).~~ **Resolved 2026-10-01:** the legacy captions were removed with the Results sections that cited them; Figures 1–7 are now a single consistent scheme, listed at the head of the Figures section.
6. Figure 7 is now `fig7_theta_gamma.py` → `fig7_theta_gamma.pdf` (post-gate). What remains is the **speed-decoupling panels**, which exist only in the pre-gate `fig6_theta_panel.py` and need rebuilding as a second Figure 7 supplement. `fig1_supp_nonmec.py` is superseded by `fig6_cross_region.py` and still computes the retired signed-minus-absolute excess — retire it rather than leaving two regional analyses that disagree.
7. Figure 1 hypothesis schematic referenced by the Introduction but absent from the caption.
8. Figure 3 caption describes a panel **J** (the regional comparison) that has moved to Figure 6, and was written before panel **G** (the M25 D25 mediolateral raster) was added. Both need fixing, and `figure3_anatomy_identity.ipynb` markdown still describes panel J.
9. Introduction paragraph 5 still previews the dropped xgBoost dissociation and must be rewritten against the seven steps.

**Reporting:**
10. No bibliography; citations are unresolved superscript numbers. Unresolved named placeholders remain for the NI acquisition system, OpenEphys and the Neuropixel acquisition plugin.
11. ~~The 3400 µm medial/lateral boundary still needs justifying, or dropping.~~ **Done 2026-10-03 — dropped from Figure 3.** *Coordinate correspondence remains documented: `coord_CCFs_x − coord_SCs_x = 5700` µm exactly.*
12. **Animal counts disagree**: *Animal details* says 8 mice (7M/1F); *Reconstruction of probe locations* says both "8 animals" and "9 mice (8 males and 1 female)". Ages missing throughout.
13. **Reward-zone coordinate origin**: task description implies context 1's zone spans 90–110 cm; the analysis code marks 92 cm and 122 cm. The 30 cm inter-context offset is consistent, the absolute origin is not.
14. Rotary encoder artifact (~±8 cm/s, every session) must be declared — it straddles *both* behavioural thresholds, the 3 cm/s stop criterion and the 10 cm/s hit criterion.
15. Anchoring classifier passband: the implementation applies 1/250–1/10 cm⁻¹, not the 1/500–1/20 cm⁻¹ the code names.
16. OF1↔OF2 grid classification replication (~24%) must be reported alongside the either-session rule.
17. The MCVR mediolateral context result (null on 7 of 7 tests) is no longer a Results section. If it is reported at all it is as a negative result in Extended Data; decide, since the Discussion currently asserts lateral specialisation for contextual embedding.
18. Discussion needs a limitations paragraph and the lidocaine future-experiment paragraph — part of the full rewrite in item 4.
19. Statistics section still unwritten; `slow` trials — state whether excluded from neural as well as behavioural analyses.
20. Stop/lick selectivity metrics exist but are unused; either build the supplementary panel or drop them.
21. **Figure 5 needs the pupil-against-performance panel** (hit rate by pupil decile, with the anchored fraction and the quadratic fit). The analysis is done and written into §5 and Methods; without the panel the figure invites the reading that the anchored state is drowsy, which the behaviour excludes.
22. **Build Figure 7 supplement 2 from `gamma_events.csv` / `gamma_by_position.csv`.** Both results are computed and written into §7 — the negative demand test and the positive reward-zone position result — but neither is plotted.
23. **"Arousal" should leave the title, the abstract and the section headings.** Done for §5, §6, the abstract and the argument summary on 2026-10-03; the title options list still carries it in options 1–4 and 9, with reworded alternatives 10 and 11 appended. Check the Introduction and Discussion when they are rewritten (items 4 and 9).
24. **Figure scripts depend on helper cells inside `lick_raster_by_trial_type.ipynb`**, which they `exec` at import. Five figures break together if a helper cell changes, and a name collision in that notebook has already silently altered session selection once. Move the shared helpers into a plain module (`region_row.py` is the model — it pulls what it needs *by name* rather than doing `globals().update()`) and leave the notebook for exploration.
25. **`lfp_spectrum_pac.py` has no merge step.** It writes worker shards with `_w*` suffixes or an un-suffixed single-threaded file, and nothing reconciles the two. The merged `spectrum_by_anchoring.csv` and `pac_by_anchoring.csv` currently in `data/lfp/` are leftovers of an earlier run; every figure now reads the shards, so the merged files should be deleted to stop a third reader picking the stale path.
26. **Retire the superseded Figure 7 drafts** `fig7_example_spectrogram.py` and `fig7_spectrum_pac.py` (the example and the population panels as separate figures). Nothing cites them and Figure 7 combines both.
27a. **Figures `fig2_supp_anatomy` and `fig2_supp_interneurons` were retired 2026-10-03** at the author's request and their PDFs deleted. `fig2_supp_anatomy.py` is kept so the figure can be regenerated; `fig2_supp_interneurons.pdf` had **no generating script in this directory**, so that one is gone for good unless it exists elsewhere. The interneuron waveform classification it showed is still described in Methods and is still the basis of the `putative_int` label used in Figures 2 and 5.
27. **Replace or drop Figure 5's second example session.** `fig5_pupil_arousal.py` still attempts M29 D17, which has no non-anchored trials under the gated classifier; it declines cleanly, but the configuration is stale.


**Remaining `[MISSING]`:** Data and code availability. *Open field recordings was written on 2026-10-01 (arena, tethering, Weeto prompting, 20 min maximum); floor dimensions, tracking method and sampling rate, lighting, and the OF1/OF2 same-day protocol are still outstanding within it.*
