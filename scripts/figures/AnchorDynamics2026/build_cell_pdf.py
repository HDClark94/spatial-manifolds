"""Compile MANUSCRIPT.md + figures + captions into a Cell-style PDF mock-up.

No LaTeX, pandoc or HTML-to-PDF engine is available on this machine, so the
document is composed directly with PyMuPDF: a small line-breaking layout engine
over the base-14 fonts, with the existing figure PDFs embedded as vector content
via show_pdf_page (so figures stay vector, not rasterised).

Cell Press conventions approximated here: running title block, author line and
affiliations, Summary, Highlights, two-column body, figures placed in-flow with
captions set beneath them in small type, and STAR Methods as a trailing section.

WHAT IS AND IS NOT REAL. The text and every number come from MANUSCRIPT.md
(and the STAR Methods from PAPER.md). Citations are author-year in the text;
the reference list is REFS, keyed by first author and year, and holds exactly
the works cited in the text, Methods and legends (check_citations fails the
build otherwise), set alphabetically as Cell does.

Writes AnchorDynamics2026_Cell_mockup.pdf
"""
import os
import re
import sys

import pymupdf

FIG = os.path.dirname(os.path.abspath(__file__))
SRC = f'{FIG}/MANUSCRIPT.md'
OUT = f'{FIG}/AnchorDynamics2026_Cell_mockup.pdf'

PW, PH = 595.0, 842.0          # A4 points
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 48.0, 56.0, 50.0
GUTTER = 18.0
COLW = (PW - 2 * MARGIN_X - GUTTER) / 2

BODY, BODY_LEAD = 8.6, 11.0
CAP, CAP_LEAD = 7.2, 9.0
H1, H2 = 11.5, 9.2
# AVENIR NEXT for every display element -- title, authors, section heads, figure
# caption titles, highlights, running head. Cell Press set display matter in a
# humanist geometric sans, which Helvetica/Arial is not; Avenir Next is much
# closer. Faces are split out of the macOS TrueType Collection by
# extract_avenir.py, because PyMuPDF loads ONE face from a file and a .ttc gives
# it only face 0 -- ask it for Regular and you silently get Bold.
#
# BODY STAYS SERIF. Cell research articles set running text in a serif; an
# all-Avenir body would read as a different template, not a closer one.
_FD = os.path.join(FIG, 'fonts')
AVR, AVD, AVB, AVI = 'avr', 'avd', 'avb', 'avi'
_FONTFILES = {AVR: os.path.join(_FD, 'AvenirNext-Regular.ttf'),
              AVD: os.path.join(_FD, 'AvenirNext-DemiBold.ttf'),
              AVB: os.path.join(_FD, 'AvenirNext-Bold.ttf'),
              AVI: os.path.join(_FD, 'AvenirNext-Italic.ttf')}
_HAVE_AV = all(os.path.exists(f) for f in _FONTFILES.values())

# EVERYTHING is Avenir Next -- body included, not just display. (An earlier
# version kept the body in Times on the reasoning that Cell sets running text in
# a serif; the brief here is to use the one face throughout the document, with
# the figure PDFs deliberately left in their own Arial.) The SERIF names are
# retained so existing call sites keep working.
SERIF, SERIF_I, SERIF_B = ((AVR, AVI, AVD) if _HAVE_AV
                           else ('tiro', 'tiit', 'tibo'))
SANS, SANS_B = (AVR, AVD) if _HAVE_AV else ('helv', 'hebo')
AR, ARB = SANS, SANS_B

_MEASURE = {}
if _HAVE_AV:
    for _n, _f in _FONTFILES.items():
        _MEASURE[_n] = pymupdf.Font(fontfile=_f)


def text_len(t, font, size):
    """Metrics for embedded faces; get_text_length only knows the base-14."""
    f = _MEASURE.get(font)
    return f.text_length(t, size) if f else pymupdf.get_text_length(t, font, size)


RULE = (0.70, 0.12, 0.12)      # Cell Press red

TITLE = 'Entorhinal grid codes are gated by attention'
AUTHORS = ('Harry Clark,^1 Wolf de Wulf,^1 Chris Halcrow,^1 '
           'and Matthew F. Nolan^1,2,*')
AFFIL = ['^1Centre for Discovery Brain Sciences, University of Edinburgh, Edinburgh, UK',
         '^2Simons Initiative for the Developing Brain, University of Edinburgh, Edinburgh, UK',
         '*Correspondence: mattnolanSed.ac.uk']

JOURNAL = 'Cell'
CELL_BLUE = (0.00, 0.42, 0.71)
GREY = (0.58, 0.58, 0.58)
# "Grid cells and fast-spiking interneurons express it" read as exclusive,
# which Figure 3D no longer supports: every class exceeds its own null and
# those two lead by about a factor of two.
IN_BRIEF = (
    'Clark et al. show that the anchoring of entorhinal spatial firing to a '
    'task reference frame is a property of the population rather than of single '
    'cells. Every cell class follows this state, grid cells and fast-spiking '
    'interneurons most strongly, yet synaptically connected pairs share it no '
    'more than matched unconnected pairs. Among the structures recorded, MEC '
    'stands out: elsewhere the labels largely record behaviour. It tracks '
    'engagement, indexed by pupil '
    'constriction, and predicts success when the reward zone must be found by '
    'path integration. The grid attractor keeps running when the state is lost: '
    'what the state sets is whether it is corrected to the task.')
CITATION = [
    'Clark et al., 2026, manuscript draft',
    'Style mock-up compiled from MANUSCRIPT.md - not a published article',
]

# Cell allows four bullets of at most 85 characters. The previous list ran to
# five, two of them over the limit, and one -- "grid cells and fast-spiking
# interneurons follow it; spatial coding in general does not" -- stated the
# claim the signed-null correction retired. Dropped to make room: the
# connectivity null and the cerebellum. Both are in the In brief and the
# abstract; neither fits in 85 characters without losing its qualifier, and a
# highlight that drops the qualifier on a null is worse than no highlight.
HIGHLIGHTS = [
    'Anchoring of entorhinal firing to the task frame is a population state',
    'Grid cells and fast-spiking interneurons follow it most strongly',
    'Pupil constriction tracks the state, identifying it as focused engagement',
    'The grid attractor survives the loss of anchoring; it is left uncorrected',
]
# A fifth candidate, over Cell's four, for the regional result:
# 'Other structures' labels mostly record behaviour; MEC carries the state'

# Figure order, source file, and the caption shown beneath it.
FIGURES = [
    ('Figure 1', 'fig1_composite.pdf',
     'A population anchoring state in medial entorhinal cortex improved path integration-based navigation',
     '(A) The task: head-fixed mice run a 200 cm virtual linear track and stop in a reward zone, '
     'visually cued on half the trials and uncued on the rest, where it can be localised only by '
     'path integration. (B, C) Stops against track position for one session, cued and uncued '
     'trials separately. (D) Three example units as trial x position rate maps: firing anchors to '
     'the track on some trials and not others. (E) The MEC anchoring raster for the same session, '
     'cells against trials, with the first principal component of that matrix beside it. PC1 '
     "exceeds a circular-shift null that preserves each cell's anchored fraction by roughly "
     'twofold, and blocks of tens of trials share a state. (F) The anchored fraction per session, '
     'with the median. (G) Hit rate by state, cued and uncued trials separately. The dashed line is chance, derived as '
     'follows. A stop is the onset of a period below the task\'s own 3 cm/s reward '
     'criterion, and a trial is a hit if any stop falls in the 20 cm reward zone. For each '
     'trial the whole set of its stops is then shifted rigidly by a distance drawn '
     'uniformly from 0 to 200 cm and wrapped around the track, and the trial is scored a hit '
     'if any shifted stop lands in the zone; the chance level is that score averaged over 500 '
     'shifts per trial and over trials. The shift keeps how many times the animal stopped and '
     'how those stops were spaced, and destroys only where they sat relative to the reward '
     'zone, so it asks how often this trial would have hit had the zone been elsewhere. It '
     'comes to 0.24 on cued and 0.21 on uncued trials over 62 sessions, and it is not 20/200 '
     'because most trials carry several stops, which come in clusters that a rigid shift '
     'moves together. Trials with no stop are kept, as misses under both the data and the '
     'null. Chance varies widely between sessions (0.02 to 0.66) with how much the animal '
     'stops, so the line is a mean; within a session it does not differ between the two '
     'states, and subtracting each state\'s own chance widens the gap in G rather than '
     'narrowing it. (H) The Anchoring '
     'Dependence Index, the uncued minus cued anchoring benefit: +0.094, p = 0.012.'),
    ('Figure 2', 'fig1_supp_neighbour_bars.pdf',
     'The anchoring state predicts performance from where hits and misses run alike, most clearly in MEC',
     '(A) Top, the track on the two trial types, as in Figure 1: black boxes at both ends, and the '
     'reward zone (90-110 cm, shaded) marked by green bars on cued trials and unmarked on uncued '
     'trials. Below, speed by position, mean and SEM across sessions, on cued (green) and uncued (pink) trials '
     'as in Figure 1, hits solid and misses dashed, and the stretch of track each anchoring '
     'classifier is trained on (shaded): the whole track; the task region (50-110 cm, from where hits start '
     'slowing to the end of the reward zone, dotted); and the black box (-30 to 30 cm: the last 30 '
     'cm of the previous trial and the first 30 cm of this one, which run on through the black box, '
     'with no visual cues). Each is the gated classifier of Figure S1 without its median filter, so '
     'that no trial\'s label carries its neighbours\' firing. (B) Miss minus hit mean speed per '
     'session on the two stretches (bars, mean and SEM; points, sessions). In the task region misses '
     'run faster than hits, because hits slow to stop (uncued +12.7 cm/s, p = 1e-11; cued +5.8, p = '
     '2e-8), and more so on uncued trials (cued against uncued, paired Wilcoxon, 53 sessions, p = '
     '2e-8). In the black box the difference is about a quarter of that and of the opposite sign '
     '(uncued -2.9, p = 3e-4; cued -3.3, p = 1e-5), the same for both trial types (p = 0.4): mice '
     'run slightly slower before a miss, a sign of engagement rather than of the stopping that '
     'defines a hit. (C) The two hypotheses for what the labels carry. Top, in the abstract: a map '
     '(lower ring) tied point for point to the world (upper ring; arrows) when anchored. If '
     'behaviour follows the map (H1), the map slips against a world that has not changed, so the '
     'labels change even where hits and misses run alike, in the black box, and the outcome follows. '
     'If the map follows behaviour (H2), what the mouse does changes and the map moves with it, the '
     'mapping intact, so the labels change only where behaviour differs. Below, the same two cases as simulated cells over '
     'one simulated session run through the same classifiers. The state runs in epochs of trials; '
     'cued and uncued trials are drawn half and half, cued trials are mostly hits throughout (90%), '
     'and an uncued hit is likelier when anchored (85% against 20%). Left (H1), a cell that carries the state: grid-like fields '
     'that hold their place along the track on anchored trials and shift on non-anchored ones; its '
     'black-box and task-region labels both follow the state. Right (H2), a cell that follows '
     'behaviour: a speed cell driven by the measured speed profile of its trial type and outcome (A), with no '
     'state; its black-box labels do not vary and its task-region labels follow the outcome trial '
     'by trial. Each shows the rate map (trials x position, the two stretches marked above), the '
     'state, the outcome of each cued (green) and uncued (pink) trial as in D-F, each classifier\'s labels per trial, and the AUC over null '
     'that hypothesis predicts in G. (D-F) One example session of each structure: '
     '(D) MEC in M25 D23, (E) the subicular complex in M26 D19 and (F) visual cortex in M28 D20. Examples were chosen by eye '
     'from the three sessions of each structure nearest its median on both the black-box r '
     '(Figure S5E) and the black-box AUC of G, among sessions that show the state (at least 40 '
     'MEC or 20 subicular or visual cells, 80% of them labelled on 90% of trials, both states on '
     'at least 20% of trials, and a state at least as persistent as the structure\'s median). '
     'The cells are three of the twelve that load most on the '
     'whole-track state, the three with the most reliable anchored-trial maps (split-half r) among those firing at least 1 Hz. '
     'Above, their rate maps, trial x position from -30 to 200 cm (the last 30 cm of the previous trial '
     'first), smoothed as the classifier smooths them, the two stretches marked above. Below, the '
     'whole-track labels of every cell the classifier could label (anchored purple, non-anchored teal), '
     'as in Figure 1: cells that change label in order of their PC1 loading, then, past a gap, the locked cells (one label on every trial), anchored first; then each trial\'s outcome, one column '
     'for cued (green) and one for uncued (pink) trials (filled, hit; white, miss; grey, the other '
     'type); then the state on each trial, PC1 (z) (Figure S2), from the whole-track, task-region and black-box labels (purple above zero, teal below, as in Figure 1). '
     'The black-box state recovers the whole-track state only partly in any structure (r = 0.43, 0.60 and 0.44 here; '
     'medians 0.25, 0.23 and 0.06 over all sessions, Figure S5E), so single sessions do not separate the structures; '
     'the difference lies '
     'in the distributions across sessions, in G and H, and in how the black-box r follows the '
     'state\'s persistence (Figure S5E). (G) The trial\'s own state against its outcome, uncued trials. Above, what enters: the state '
     'each classifier gives trial n, from the whole track, the task region or the black box (the '
     'last 30 cm of trial n-1 and the first 30 cm of trial n), against whether trial n was a hit. '
     'Below, for each structure (at least 15 cells, 10 in the cerebellum; sessions below each; 63 '
     'sessions, 8 mice), on the uncued trials that have a cued trial within three before them, as in H: '
     'AUC over its circular-shift null (200 shifts), bars the mean with SEM, points sessions, '
     'Wilcoxon signed-rank across sessions. From the whole track and the task region every '
     'structure predicts the outcome (task region +0.18 to +0.30). From the black box only MEC '
     '(+0.059, p = 0.004) and the subicular complex (+0.080, p = 0.048) do; visual cortex (+0.000, '
     'p = 0.85) and the cerebellum (+0.010, p = 0.45) do not. (H) The state on the nearest '
     'preceding cued trial against the uncued outcome. Above, what enters: the nearest cued trial '
     'within three trials before uncued trial n, read from its task region or its black box; '
     'nothing from trial n or any later trial is used. On cued trials the cue sets where the mouse '
     'stops, so a label that only records stopping barely varies there. Hatched (res.): that '
     'trial\'s own behaviour (hit, and the first five principal components of its '
     'speed-by-position profile) regressed out of its state, since engagement drifts over a '
     'session (Figure S5B, D). From the task region MEC, visual cortex and the subicular complex '
     'predict (+0.12 to +0.17) and the cerebellum does not (+0.050, p = 0.064); with the cued '
     'trial\'s behaviour removed, MEC (+0.039, p = 0.009), visual cortex (+0.046, p = 0.011) and '
     'the subicular complex (+0.086, p = 0.006) still do. From the black box only MEC predicts '
     '(+0.062, p = 0.002), and only MEC still does with the behaviour removed (+0.042, p = 0.002; '
     'visual cortex +0.006, p = 0.71; subicular complex +0.027, p = 0.34; cerebellum +0.033, '
     'p = 0.25). (I) Which hypothesis of '
     'C each structure\'s results fit. Top, what each predicts for each test (filled, above the '
     'null; open, at it): if behaviour follows the map (H1), every test is above its null; if the '
     'map follows behaviour (H2), only the trial\'s own task region is, because only there do hits '
     'and misses run differently. Below, each structure (filled, above its null, p < 0.05; open, '
     'not; mean AUC over null beneath). The three black-box tests decide (dashed): H1 if all are '
     'above their nulls, H1 in part if some are, and H2 suffices if none is. No black-box effect '
     'is what H2 predicts but does not establish H2, since a small effect could be missed. The '
     'residual task-region test (grey) is shown but does not decide, since regressing behaviour '
     'out removes only the behaviour measured. MEC meets all three black-box predictions of H1, '
     'the subicular complex one (its own black box), and visual cortex and the cerebellum none. '
     'Throughout, a trial\'s state is its z-scored score on the first principal component of that '
     'classifier\'s own cell x trial labels, as in every figure (Figure S2); the fraction of cells '
     'anchored leaves every verdict unchanged (MEC black box: G +0.049, p = 0.004; H +0.050, '
     'p = 0.006; H res. +0.035, p = 0.030; subicular complex G +0.083, p = 0.010; visual cortex and '
     'cerebellum, no black-box test below p = 0.17).'),
    ('Figure 3', 'fig2_single_units.pdf',
     'Grid cells and putative interneurons follow the population state most strongly',
     '(A, B) Two example sessions: the MEC anchoring raster with PC1, and example single cells of '
     'each identity, including cells locked into one mode. (C) The null for one cell, its labels '
     'circularly shifted 200 times against the population axis, with the observed value marked. '
     'Chance lies near +0.15 rather than at zero, because a cell anchored on most trials '
     'correlates with a population that is also anchored on most trials. (D) Agreement with the '
     "population axis by identity, as excess of |r| over each cell's own null. Every class exceeds "
     'chance and they differ in degree: grid +0.190 (p = 5e-12) and putative interneuron +0.137 (p '
     '= 8e-18) against non-grid spatial +0.087 (p = 1e-22) and non-spatial +0.056 (p = 2e-17). '
     'Grid and interneuron do not differ (p = 0.71) and both exceed the other two (Holm p <= '
     '3e-6). (E) The fraction of cells individually beating their own null, per mouse: 47% of grid '
     'cells and 41% of interneurons against 32% and 20%, all above the 5% false-positive line. (F) '
     'Agreement in the trials around a population transition. All four identities switch together '
     'over about three trials, and identity does not predict switch timing (p = 0.70). (G) '
     'Agreement between identities as a 4 x 4 matrix over its own null; grid-interneuron reaches '
     '+0.101 (30 sessions, p = 1e-6), close to either within-group value (grid-grid '
     '+0.110, interneuron-interneuron +0.126). (H) Which identities never leave '
     'one mode: interneurons lock anchored (132:7) and non-spatial cells lock non-anchored '
     '(36:275). (I) Open-field speed score for interneurons that lock anchored against those that '
     'vary: +0.132 against +0.072, p = 0.003. (J) The same as a count across entorhinal cells: '
     'speed-modulated cells lock anchored and non-anchored about equally (0.96:1, p = 0.64) while '
     'cells with no open-field speed tuning lock non-anchored at 0.57:1 (p = 3e-12), interaction '
     'odds ratio 1.70 (p = 1e-6), so speed tuning shifts locking towards anchored. (K) Open-field theta index against agreement '
     'with the population axis, pooled over all cells. Points are deciles of the index with their '
     'standard error and the line is the mixed-model fixed effect with a band of one standard '
     'error. The slope is +0.209 per unit index alone (p = 1e-21) and +0.177 with identity in the '
     'model (p = 1e-13), and holds within every class: grid (+0.333, p = 0.013), non-grid spatial '
     '(+0.196, p = 4e-10), non-spatial (+0.186, p = 0.008) and interneurons (+0.103, p = 0.029), so theta rhythmicity predicts following '
     'in its own right rather than restating cell type. (L) Whether each '
     "class's own state predicts behaviour, as the Anchoring Dependence "
     'Index: AUC(uncued) minus AUC(cued) for a state built from five '
     'randomly drawn cells of that class, so a class with 92 cells in a '
     'session is not compared against one with five. Every class is '
     'positive -- grid +0.093, non-grid spatial +0.038, putative '
     'interneuron +0.035, non-spatial +0.021 -- but only the grid and non-grid spatial states clear '
     'Holm correction (p = 0.033 and 0.033; interneuron 0.16, non-spatial 0.16), and no pair '
     'of classes differs (smallest Holm p = 0.20 across the 28 sessions '
     'carrying all four). The ordering follows panel D, but this figure '
     'cannot show that one class carries the behavioural index more than '
     'another.'),
    ('Figure 4', 'fig4_subnetwork_anatomy.pdf',
     'The responsive subnetwork is medial, where grid cells are enriched',
     'Grid cells are enriched medially and superficially in MEC, and following the population '
     'state is organised along the same mediolateral axis. (A) Open-field rate maps from four M25 '
     'sessions whose probes sat at different mediolateral positions, sampled so the grid to '
     'non-grid ratio drawn is the ratio recorded. Left, the recording day: the 1 m x 1 m open-field '
     'arena (OF1), the virtual-reality task with the mouse head-fixed on a cylinder before the screens, '
     'and the arena again (OF2). Every map comes from OF1, the arena session preceding the task. (B, C) Two multi-shank sessions, among the strongest of the 26 by within-session '
     'rho on MEC cells, against a median of -0.094, so they show the gradient clearly rather than '
     'typically; two more, one of which lacks it, are in Figure S7. Left, the trial-by-cell '
     'anchoring raster split by shank from medial to lateral, cells within a shank ordered dorsal '
     "to ventral, under a strip giving each cell's region, beside z-scored PC1, with dashed lines "
     'where PC1 changes sign and holds the new sign for at least three trials. Right, the '
     '4-shank Neuropixels 2.0 probe, shanks S1 to S4 from medial to lateral as in the raster, the '
     "session's 384 recorded contacts dark grey and the inactive ones light grey, with each cell at "
     'its localised position (spread across the shank for display) coloured by its agreement with '
     "the population axis above chance: |r| minus the mean |r| of the cell's own circular-shift "
     'null, as in Figure 3D, so a cell locked to the opposite state also counts as following. The '
     'scale diverges from white at chance, green above and purple below; cells with no agreement '
     '(locked on every trial) are grey. Grid cells are triangles. Above the probe, '
     'agreement above chance for each shank as a boxplot. Cells from every region on these shanks '
     'are shown, and the per-shank means and rho quoted below include the parasubicular and '
     'visual cells. Regions are shaded in greys: MEC darkest, then pre/parasubiculum and visual cortex. (B) M25 D25, 204 '
     'cells (58 grid) over four shanks spanning 720 um and anchored on most trials. Agreement is '
     '0.33, 0.53, 0.36 and 0.05 from medial to lateral (rho = -0.255, p = 0.002), and the grid '
     'cells sit on the three medial shanks. (C) M26 D19, 404 cells (33 grid) over four shanks '
     'spanning 720 um, in which PC1 changes sign 13 times. Following is concentrated on the two '
     'medial shanks (mean agreement 0.32 and 0.44) and nearly absent from the two lateral ones '
     '(0.09 and 0.09; rho = -0.464, p = 1e-19). (D) The four identities of Figure 3 as fractions of all classified MEC cells (9,583) '
     'against mediolateral position: lines, the within-session slope of the mixed model below, drawn through '
     'the fitted value at the mean session position and spanning the most medial to the most lateral shank, '
     'shaded 95% CI of the fixed effects; points, '
     'single shanks, each shank of a multi-shank session separately (124 shanks with at least 10 cells). '
     'Titles, the within-session slope from a linear mixed model with mouse as a random effect '
     '(random intercept and within-session slope per mouse, sessions nested in mouse; 8 mice, 64 sessions). '
     'Grid cells fall laterally (-1.2 percentage points per 100 um, p = 0.04; median within-session '
     'rho = -0.093, p = 0.011, 25 sessions); non-spatial cells rise across sessions (rho +0.085, p = 0.009) '
     'but not reliably once mouse is a random effect (p = 0.10); non-grid spatial cells (p = 0.94) and '
     'putative interneurons (p = 0.69) show no gradient. Grid membership among spatial cells alone gives median '
     'rho = -0.078 (p = 0.028); grid fraction does not vary with dorsoventral position (p = 0.99) and is higher in superficial than '
     'deep layers (8.6% against 4.0% of spatial cells, p = 0.004). (E) Mediolateral position against agreement above '
     'chance. Line, the within-session slope of the mixed model, drawn as in D (pooled over sessions, '
     'rho = -0.20, p = 8e-59); '
     'points, the mean of each shank (117 shanks with at least 10 cells), the shanks of each of the 24 multi-shank '
     'sessions joined. Within session the gradient holds with mouse as a random effect (mixed model as in D: '
     '-0.015 per 100 um, p = 0.010) and over the 26 sessions spanning at least two shank pitches (median '
     'rho = -0.062, p = 0.002); within session neither dorsoventral position (p = 0.75) nor probe depth (p = 0.88) shows a '
     'gradient, and the gradient survives an axis built without each cell\'s own shank (median rho = -0.081, p = 0.001).'),
    ('Figure 5', 'fig4_monosynaptic.pdf',
     'Monosynaptic connectivity does not account for shared anchoring',
     '(A) The interneuron call: waveform peak-to-trough duration against firing rate, with the '
     'marginal distributions and the fixed 0.4 ms boundary. (B) What a classified interneuron does '
     'to its target, a causal trough in the cross-correlogram. (C) What a broad-spiking cell does, '
     'a causal peak on the same scale. (D) Presynaptic enrichment by identity: inhibitory '
     'connections originate from the cells panel A called interneurons at 4.9 times their '
     'population share, which the detector cannot produce by construction since it never sees the '
     'waveform. (E, F) The inhibitory and excitatory connection matrices, each as a percentage of '
     'the ordered pairs an identity contributes. Interneuron-to-grid is the strongest pathway in '
     'the dataset (3.20%, 4.8x over identity shuffling, p = 0.0005); excitatory connections are an '
     'order of magnitude sparser, and grid-to-grid (0.28%) is no more frequent than identity '
     'shuffling predicts (1.2x, p = 0.23). (G) Latency by connection type: inhibition is the '
     'slower, median 3.0 against 1.0 ms. (H) Detection against a jitter null, for both signs. (I) '
     'Whether connected pairs share the anchoring state more than unconnected pairs matched for '
     'session, identity pair and probe distance. They do not (grid-interneuron p = 0.60, '
     'interneuron-interneuron p = 0.99, grid-grid p = 0.10). The first two resolve effects a third '
     'the size of the identity effect; connected pairs of the other identities show a small excess '
     '(+0.008, p = 6e-5), a tenth of it. All narrow-spiking cells, '
     'as a continuum of firing rate, are shown in Figure S19.'),
    ('Figure 6', 'fig5_pupil_arousal_M21D19.pdf',
     'The state tracks pupil-linked engagement',
     '(A) The recording arrangement, with infrared illumination and the side camera. (B) The eye '
     'with the DeepLabCut perimeter landmarks from which pupil radius is computed. (C) One '
     'anchored and one non-anchored trial as seven moments of a single traversal, with pupil '
     'radius printed beneath each frame. (D) Mean pupil radius against track position for the two '
     'state blocks of this session, with the two example trials overlaid. (E) The MEC anchoring '
     'raster for the same session, cells that vary ordered by PC1 loading and cells locked into '
     'one mode past a gap. (F) PC1 of the same label matrix. (G) Pupil radius for every trial, on '
     'the trial axis of E and F, so a state block can be read across. (H) Pupil radius by state, '
     'paired within session over 29 sessions: -0.593, p = 1e-08. (I) Pupil radius against track '
     'position for the two states. The difference has the same sign in all 50 position bins and '
     'reaches significance in 49, with no trend along the track (rho = -0.079, p = 0.58) and the '
     'reward zone if anything less affected than the rest, so this is a tonic shift rather than '
     'attention paid at particular places.'),
    ('Figure 7', 'fig7_supp_vis_examples.pdf',
     'Visual cortex shares the entorhinal anchoring state, more weakly than MEC carries it',
     '(A, C) Two sessions whose cells reach far into both structures. Left, one anchoring '
     'raster split into a visual and an entorhinal (MEC) block, cells within each block '
     "ordered from strongest to weakest agreement with that structure's own PC1, then, past a "
     'gap, the locked cells bundled, always anchored before always non-anchored; each block '
     "sits under a strip giving every cell's region in the grey key of Figure 4, with the "
     "structure's cell count and locked share above. Middle, each structure's own PC1 "
     '(z-scored; their correlation above). Dashed lines are the trials on which the session '
     'PC1 changes sign for at least three trials. Right, the 4-shank Neuropixels 2.0 probe (probeinterface '
     'NP2014) with the session\'s 384 recorded contacts dark, every curated cell drawn at its own '
     'probe position and coloured by its agreement above chance with the MEC population\'s '
     'PC1 on a dark grey-white-orange scale (white at chance; locked cells, which have no '
     'agreement, drawn at chance). Far right, a check of the depth axis: the shank\'s LFP '
     'spectra (log power z-scored across channels per frequency, as Figure S14) on the same '
     'axis, with the physiological VIS / MEC boundary (orange; onset of the 40-300 Hz block) '
     'and exit from cortex (purple; deepest 10-44 Hz peak) of Figure S14 drawn across spectra '
     'and probe, against the atlas border (dashed); and the histological section holding the '
     'shank\'s DiI track (red, DiI, over the blue DAPI counterstain), the whole section. (B, D) Trial x position rate maps of six switching (not locked) cells '
     'from each session, three of them visual, numbered dorsal to ventral and ringed '
     'on the probe: the most active of the three most dorsal visual cells, further visual cells at '
     'evenly spaced depths through the visual cells and the '
     'rest at evenly spaced depths through MEC, at each depth the most active on the track of the three '
     'nearest switching cells, whatever its agreement (titles: agreement with its own PC1, '
     'depth below cell 1). In B, MEC cells 202 (the first example cell of Figure 1) and 175 '
     'are chosen as clear followers, and visual cells 230, the clearest visual follower that '
     'fires readably (7.8 Hz), 262 and 268, the most active of the visual cells above it (the most dorsal visual cells all fire below 0.4 Hz on the track). '
     "Beside each map, the cell's own label on every "
     "trial; dashed lines, the session's transitions. (E) Agreement above chance of every cell"
     " in the 15 sessions recording both structures with its own structure's PC1 (rebuilt without it) and "
     "with the other structure's. Visual cells follow the MEC axis as well as their "
     'own (median 0.027 against 0.026, within-session p = 0.72); MEC cells agree more with their own '
     'axis than the visual one, not reliably (0.063 against 0.049, p = 0.17). '
     "(F) The same agreement against each cell's 3D atlas distance to the other "
     'structure (visual cells left, MEC cells right): lines are the mean and SEM in 200 um '
     "bins holding at least 5 cells, with each session's mean removed. Agreement with the MEC "
     'axis is flat on both sides (within-session rho +0.065, p = 0.39; +0.005, p = 0.98), so '
     'visual cells far from MEC follow it as well as those at the border; MEC cells nearer visual '
     'cortex agree slightly more with the visual axis, not reliably (rho -0.038, p = '
     '0.083). (G-I) Whether the two structures share one state, in the 10 sessions whose cells'
     ' reach at least 500 um from the VIS-MEC border on both sides (90th percentile of each '
     "cell's 3D atlas distance to the other structure), and within them only cells at least "
     '100 um from the other structure. Near the border '
     'an error in assigning cells to structures would pass for sharing; the sessions left '
     'out include M27 D23, an earlier example whose visual cells all lie within 190 um of MEC.'
     " (G) Switch-triggered averages: the change in the fraction of each structure's varying "
     'cells in the anchored mode, signed so that the new state is up and referred to the five '
     'trials before the switch, averaged per session and then across sessions (mean and SEM), '
     'around 55 MEC switches (left) and 82 visual switches (right); stippled, the middle 95% '
     'when the other structure is shifted in time 200 times. Visual cortex steps 56% as far as'
     ' MEC over the first five trials at MEC switches, beyond the shifts (p = 0.005), on the same trial; '
     'at visual switches MEC steps 44% as far as visual cortex, not beyond them (p = 0.055). (H) '
     'The correlation of the two PC1s against the ceiling set by their own split-half '
     'reliability: filled, beyond a null shifting one axis in time; letters, the sessions of A'
     ' and C. Where both axes are reliable (6 sessions) it is a median 0.99 of the ceiling. '
     '(I) Every cell against each axis: following it (r above the 95th percentile of its own '
     "circular-shift null), opposing it, neither, or locked, a cell's own structure's axis "
     'rebuilt without it. Visual cells follow the MEC axis as often as their own (12% against '
     '10% of 654), whereas MEC cells follow the visual axis less often than their own (17% '
     'against 22% of 1,620). Visual cortex has more locked cells (44 vs 12% in A, 55 vs 33% in C).'),
    ('Figure 8', 'fig7_theta_gamma.pdf',
     'A frequency-specific change in the entorhinal field potential',
     '(A-D) One session trial by trial, both states shaded behind every panel and dashed lines '
     'where PC1 changes sign for at least three trials. (A) The population axis. (B) The '
     'spectrogram, 1.9-150 Hz in 56 log-spaced bands, smoothed along the trial axis only. (C) The '
     'three band traces. (D) Theta-gamma modulation index for both bands on twinned axes, fast on '
     'the left and slow on the right, which they need because slow-gamma MI is several times '
     'smaller. (E) The two states as spectra across 25 switching sessions, the spectra above and '
     'the deviation between them below. The deviation crosses zero at a median of 62 Hz and does '
     'so in 24 of the 25 sessions. (F) The same difference band by band (bars, mean +/- SEM; points, sessions): slow gamma rises (+0.102, '
     'p = 0.0007) while high gamma falls (-0.120, p = 0.003), so the difference reverses sign across '
     'the spectrum; theta (-0.069, p = 0.22), delta and beta do not change reliably. (G) '
     'Theta-gamma coupling by band, both on one axis, one point per session with its two '
     'states joined (bars, mean +/- SEM): fast gamma (60-100 Hz) uncouples (-16.9%, p = 4e-5) and slow '
     "gamma (30-48 Hz, as in C-F) does not (+2.8%, p = 0.92). (H) The redistribution, tested pairwise within session. Left, each session's slow-gamma "
     'change against its fast-gamma change, with the equal-and-opposite line and the mean with its '
     'SEM. Right, the three contrasts drawn directly, one point per session. The difference '
     'between bands is large and reliable (slow minus fast +0.195, p = 2e-4) while the sum is not '
     'distinguishable from zero (+0.007, p = 0.79), nor is the change across 2-200 Hz (-0.033, p = '
     '0.30): power moves from fast to slow with no net change. The sum is zero on average rather '
     'than session by session, and the balance is per Hz -- weighting each band by its width moves '
     'the sum to -0.29 (p = 0.11) -- so what is supported is that power moves from fast to slow, '
     'not that the total is conserved. Where in the theta cycle each frequency peaks is shown in '
     'Figure S11.'),
    ('Figure 9', 'fig8_supp_attractor_sim.pdf',
     'The grid attractor survives the loss of anchoring, in the recordings and in a gated attractor model',
     'Each grid cell is given phase offsets on its module\'s torus by the spectral method, the population '
     'is projected onto them, and every result is referred to a shuffle of which offset belongs to which '
     'cell, which leaves all firing rates untouched and destroys only the lattice organisation. A simulated '
     'module of the gated attractor is put through the same code first. '
     '(A) The model (Figure 10) as simulated. A two-dimensional attractor sheet (36 x 36 units with hexagonal '
     'recurrent connectivity) always integrates the animal\'s movement along a 200 cm virtual track; a '
     'landmark correction reaches it only while the attentional gate is open. The simulated mouse decodes its '
     'position from the bump\'s phase against the phase the landmarks give along the track, slows over the 35 '
     'cm before the reward zone (90-110 cm), and stops for 1 s once the decoded position is in the zone\'s '
     'middle 10 cm. The simulation is made as noisy as the recordings: 36 of the 1,296 units are sampled at '
     'random, as a recording samples a module; each takes the mean rate and the over-Poisson variability of a '
     'recorded grid cell drawn at random from the 416 of these sessions (median 4.0 Hz and Fano factor 1.3 '
     'about the place-and-state expectation), with gamma-Poisson counts; recorded modules share almost no '
     'noise (median pairwise r of residuals 0.017), so none is added; and path integration is noisy (a random '
     'walk of about 0.6 rad per trial with the gate shut). The gate is opened and shut in blocks of trials and '
     'everything after the spikes is the recordings\' pipeline. '
     '(B) The simulated session, drawn as C: open for trials 1-40, shut for 41-70, open for 71-110 and shut for '
     '111-134 (black bars beside PC1). The state the classifier recovers follows the gate: PC1 is positive on '
     '94% of gate-open trials and none of the gate-shut ones. So does behaviour (left, each stop on each trial; zone '
     'shaded): the simulated mouse stops in the reward zone on 100% of anchored trials and 14% of non-anchored '
     'ones (100% and 6% by the gate itself), because the bump it reads its position from has slipped. The phase returns to the same sequence every '
     'gate-open trial and scatters from trial to trial with the gate shut. The simulated bump remains '
     'stronger than the recorded one (+0.282 anchored and +0.252 non-anchored above shuffle, against +0.056 '
     'and +0.040 in the recordings, E; +0.43 without the added noise), so spike-count noise and noisy path '
     'integration account for part of the difference but not most of it. On the trials drawn the bump decodes '
     'position within 20 cm on 78% of anchored and 50% of non-anchored bins. Over the whole session its speed '
     'tracks the animal\'s in both states but less closely with the gate open (r = 0.20 against 0.35 over 2.5 s, '
     'as in F), since the open gate pulls the bump back into register after the track\'s teleport to the start '
     'of each trial. The recordings show the opposite (F), which the model does not reproduce. '
     '(C) A recorded session, M29 D25, with 34 grid cells in one module. Top, from left: where the mouse '
     'stopped (speed below 3 cm/s) on uncued trials, the only ones on which it must find the zone unaided (in '
     'the zone on 78% of anchored and 64% of non-anchored uncued trials); the population anchoring state cell by cell and as PC1, with triangles marking the trials drawn below; the module\'s '
     'phase along and across the track at each position of each trial, regular and tied to position on '
     'anchored trials and losing that order on non-anchored ones (on a linear track the bump has one degree '
     'of freedom, so the axis labelled along is the one that advances most and the one labelled across the '
     'one that advances least; the torus above each map in B shows which cycle it runs round); the module\'s '
     'grid cells in the open field (OF1), ordered by phase along the track; and below them, where those cells '
     'sit on the unit cell of the sheet, coloured by phase along, which shows how unevenly 34 cells sample it. Bottom: '
     'the bump on the sheet over three consecutive uncued trials in each state, one row per trial and one '
     'frame every 10 cm (the module\'s position-binned activity laid out on the unit cell by the cells\' along '
     'and across phases, kernel-smoothed between cells and scaled frame by frame; frames spaced just over one '
     'lattice vector apart so that neighbouring unit cells nearly tile). The anchored trials are the run of '
     'three uncued trials with the largest bump above shuffle; the non-anchored trials are the run with the '
     'largest bump in the non-anchored epoch after trial 150. Under the frames, the animal\'s speed (grey) and '
     'the bump\'s (coloured; as in F) along the track for the same trials, each averaged over 2.5 s and '
     'z-scored. Right: position decoded from the bump on those trials (template matching against the '
     'module\'s mean phase in each 5 cm bin over anchored trials, the trials shown left out), within 20 cm on '
     '60% of anchored and 23% of non-anchored bins; and the two speeds against each other. Over three trials '
     'these illustrate rather than test. '
     '(D) The bump population vector per 100 ms in the simulation and in M29 D25, both states above the '
     'shuffled distribution in each; the simulated bump sits far higher. '
     '(E) Every session (22; hollow, modules of fewer than 15 grid cells). The bump exceeds its shuffle by '
     '+0.056 when anchored and +0.040 when not, both p = 5e-07, so the attractor does not collapse; the states '
     'differ by +0.016, p = 0.005, so it is modestly weaker, and equating the two states\' running speeds '
     'changes nothing (+0.016, p = 0.004). The 11 sessions with at least 15 grid cells, whose bump is measured '
     'most reliably, give the same result (both states above shuffle, p = 0.001, every session; anchored '
     'higher, p = 0.005). Aligned to the 130 anchored-to-non-anchored switches (not shown), the bump neither '
     'falls at the switch (p = 0.34) nor decays after it (p = 0.70). (F) The bump\'s '
     'speed against the animal\'s, every session, both averaged over 2.5 s; dashed, the same for the phase '
     'shuffle (r = 0.00). Over 100 ms the bump\'s speed is distorted by the readout itself: its phase is the '
     'rate-weighted circular mean of the cells\' offsets, a few dozen cells cover the torus unevenly, and the '
     'estimate lags over clusters of cells and jumps the gaps between them (a clean bump moved at a constant '
     '30 cm/s past the 36 offsets of the simulated module of B reads out at 1 to 199 cm/s). The distortion '
     'repeats once per lattice cycle and cancels over one, 2.5 s in the simulation (where, on non-anchored '
     'trials, r rises from 0.23 over 100 ms to 0.35); in the recordings a cycle along the track lasts 3 to 87 s, so the same 2.5 s reduces the '
     'distortion without removing it. The bump keeps pace with the animal in both states, above its shuffle '
     'in 21 of 22 sessions when anchored (median r = 0.186, p = 5e-6) and 19 of 22 when not (0.128, p = 1e-4), '
     'and less closely once anchoring is lost (higher when anchored in 15 of 22, p = 0.019; with mouse as a '
     'random effect p = 0.001, 6 mice); with the two states\' speed distributions matched r is 0.194 against '
     '0.134 (p = 0.007). '
     '(G) The bump\'s gain: the slope of its speed on the animal\'s, bump cm per cm run, over 100 ms bins, '
     'where the readout\'s distortion adds noise to the bump\'s speed (lowering r, hence the 2.5 s windows of '
     'F) but does not bias a slope on the animal\'s speed. Phase is converted to cm by its advance per cm on '
     'anchored trials, so a gain near 1 there is expected (median 0.93, p = 0.95 against 1); what is measured '
     'is the non-anchored shortfall, 0.60, below 1 (p = 5e-4) and below anchored in 16 of 22 sessions (p = '
     '0.003; mouse as a random effect p = 0.003). Over the 2.5 s windows of F, far fewer, the medians are 1.11 '
     'and 0.68, the difference not significant (p = 0.07; mouse as a random effect p = 0.26). The simulated '
     'gain is 0.85 with the gate shut and 0.57 with it open, again from the correction after each teleport. The bump under-integrates, and over a 200 cm trial the shortfall accumulates to most of a cycle, '
     'so the phase at a given place on the track is effectively re-drawn by the next trial, which is why the '
     'non-anchored rows of the phase maps in C lose their order. In E-G, Wilcoxon signed-rank tests over '
     'sessions unless stated.'),
    ('Figure 10', 'fig8_model.pdf',
     'Proposed engagement gate for position estimation in the entorhinal cortex',
     'A proposed mechanism, placed with Ideas and speculation rather than with the Results. All '
     'three panels are schematics; the values on A are measured. (A) The circuit. A self-motion drive is '
     'always present and a landmark drive carrying the task frame passes through an engagement '
     'gate; the gate is what the anchoring state is, open meaning the estimate is re-referenced to '
     'the task frame on every traversal and shut meaning it is not. In this task engagement goes '
     'with a constricted pupil, not a dilated one (Figure 6). The estimate is carried by grid cells '
     'and fast-spiking interneurons, the two populations that follow the state most strongly '
     "(Figure 3), and drives where the animal slows and stops. The slope on the self-motion arrow: "
     "theta's entrainment to running speed falls by about 40% when the gate opens (0.343 to 0.198, "
     'p = 0.018) while the open-field speed cells code speed no differently in the two states '
     '(+0.1%, p = 0.55), so the velocity signal looks intact and the rhythm is heeding it less. The '
     'circuit is drawn once rather than twice because the measurements say it is the same in both '
     'states: spike coupling to gamma does not move with the state (Holm p >= 0.10), nor does the '
     'laminar theta phase gradient, and connected pairs share the state no more than matched '
     'unconnected ones (Figure 5). (B) What the gate does to the map in one dimension, drawn as the world and map '
     'rings of Figure 2C. With the gate open the map sits on the world point for point. When the gate '
     'shuts the landmark drive no longer corrects the estimate: the bump under-integrates, so the map '
     'slips against a world that has not changed, and the error builds lap on lap (laps n to n + 5, each '
     'ring a step lower; dashed, where the anchored map sits). It grows by most of a cycle per lap, so '
     'the phase at a given place looks re-drawn from one lap to the next (Figure 9). When the gate '
     'opens again the map is re-anchored to the world. (C) The same in two dimensions. Grid-cell '
     'population activity lies on a torus, two periodic phases (Gardner et al., 2022), and grid cells and '
     'fast-spiking interneurons follow the state together (Figure 3), with interneuron-to-grid the '
     'strongest connection we detect (Figure 5), the recurrent inhibition a continuous attractor on the '
     'torus would rest on. The world is a flat sheet and the map the torus; one path through the world '
     'is drawn with its image on the torus, four points along it tied between the two (path faint where it '
     'runs round the back of the torus). With the gate open the path maps onto the torus point for point. '
     'With it shut both phases start from somewhere else and lag behind the movement, so the image ends '
     'away from where the anchored one would (dashed). The gate arrows between the columns apply to both '
     'B and C.'),
]

# Supplemental figures, in the order their main figure appears.
SUPPLEMENTS = [
    ('Figure S1', 'fig1_supp_classifier.pdf',
     'The anchoring classifier, step by step, against simulated ground truth, related to Figures 1 and 2',
     'The classifier validated on simulated grid cells, where each trial\'s state is imposed rather '
     'than inferred. 500 cells in four modules (spacings 42-116 cm, each cell\'s period 1.0-1.4 '
     'times its module\'s, as a 1D slice through a 2D lattice gives), Poisson spikes in 2 cm bins at '
     'about 30 cm/s, with trial-to-trial variation in gain and running speed. On an anchored trial '
     'every field is in place, to within 2.5 cm; on a non-anchored trial each module\'s offset is '
     're-drawn, the whole module moving together, as the recorded grid phase is on non-anchored '
     'trials. One session runs 35 anchored, 30 non-anchored and 35 anchored trials. (A) One cell\'s '
     'expected firing on three anchored (solid) and three non-anchored (dashed) trials. (B) The '
     'session for a cell from each of three modules (dashed lines, the switches). (C) The true '
     'module offsets per trial, in periods. (D) A population decoder, which matches each position\'s '
     'population vector against the anchored templates, shows the ground truth before any classifier '
     'is applied: 1.2 cm error on anchored trials, 40 cm on non-anchored ones. (E-H) The four steps '
     'of the classification on one cell, in order: the trial x position rate map, the trial x trial '
     'correlation matrix, the cell\'s mean correlation against its own circular-shift null, and the '
     'resulting label sequence against truth, with and without the median filter. Across cells, '
     'median accuracy is 0.94 (sensitivity 0.998, specificity 0.776); the filter raises specificity '
     'from 0.675 and accuracy from 0.89. Specificity is limited by the periodicity of the cells: a '
     're-drawn offset that lands near a whole period puts a non-anchored trial\'s fields where the '
     'anchored fields are, and no classifier of one cell\'s firing can tell. (I) Reported against true '
     'anchored fraction, with and without the gate, in sessions of 30 trials (10 sessions of 20 cells '
     'per point). Ungated, a session that is never anchored is reported as 69% anchored; the gate '
     'brings this to 31% and reports a fully anchored session correctly (99.8%), and tracks the truth '
     'from a third anchored upward. Mean error in the reported fraction falls from 0.27 to 0.09. (J) '
     'Per-trial accuracy over the same range; the band marks where the session of A-H sits, which is '
     'why that session cannot test the gate. (K) Recall against the length of a non-anchored epoch '
     '(8 sessions of 32 cells per point): the size-5 median filter erases runs shorter than three '
     'trials (recall 0.03 at two, 0.32 at three, 0.73 at ten). (L) Accuracy of the classifiers of '
     'Figure 2 on the same session, without the median filter as Figure 2 uses them: from the whole '
     'track 0.89, the task region (50-110 cm) 0.84 and the black box (-30 to 30 cm) 0.82, median '
     'over cells; sensitivity and specificity above each bar. Classifying a third of the track costs '
     'some accuracy, mostly sensitivity, but leaves both short-stretch classifiers accurate. The black '
     'box of a trial includes the end of the one before it, so at a switch it straddles two states; '
     'scored without the trials either side of a switch, its accuracy is 0.83.'),
    ('Figure S2', 'fig1_supp_population_state.pdf',
     'From cell labels to one population state per trial, related to Figures 1 and 2',
     'Two summaries of a cell x trial label matrix are in use: the fraction of cells anchored on '
     'each trial, and the trial\'s z-scored score on the first principal component of the matrix, '
     'signed to rise with the fraction. The fraction counts every cell once; PC1 weights each cell '
     'by how far its labels move with the others. They are compared here on the simulated grid '
     'cells of Figure S1, where the state is known, adding what Figure S1 lacks and recordings '
     'have: cells that do not follow the population. (A) One simulated session of 120 trials, the '
     'true state a two-state Markov chain (mean epochs about 14 anchored and 7 non-anchored trials). '
     'Whole-track labels of 30 cells of each of three kinds: followers, anchored when the '
     'population is; incoherent cells, each switching on its own with the same statistics; and '
     'competing cells, switching together on a second state independent of the first. (B) The '
     'two estimates for 30 cells, half followers and half incoherent (r with the true state, '
     'fraction 0.88, PC1 0.97). (C-E) Correlation of each estimate with the true state against '
     'the share of followers among 30 cells, the rest incoherent (solid) or competing (dashed), '
     'for each classifier of Figure 2, unfiltered (8 sessions, 25 draws each; mean and SEM). With '
     'incoherent non-followers PC1 is the better estimate, increasingly so as followers become '
     'fewer (black box, a quarter followers: PC1 0.82, fraction 0.58), because it down-weights '
     'cells that do not move with the rest. With a competing group PC1 is the worse: it follows '
     'whichever coherent group dominates, and with a quarter followers it tracks the competing '
     'state instead (r = -0.02, fraction 0.27). (F) The share of the true state\'s prediction of '
     'the uncued outcome (AUC over a circular-shift null, as Figure 2G) each estimate recovers, a '
     'quarter followers; solid, the rest incoherent; hatched, competing. (G) Recorded sessions: '
     'the correlation between the two estimates per session, structure and classifier. They agree '
     'from the whole track and the task region (medians 0.75-0.95) and much less from the black '
     'box (0.29 in visual cortex to 0.60 in MEC), where Figure 2 makes its case. (H) Which '
     'black-box estimate recovers the whole-track state of its own kind, the best estimate of the '
     'state the recordings offer: no better with PC1 in any structure, and worse in visual cortex '
     '(median r 0.05 against 0.17, paired Wilcoxon p = 0.008) and the cerebellum (0.03 against '
     '0.08, p = 0.055), as expected if those populations hold coherent structure other than the '
     'state. (I) The black-box tests of Figure 2I under each estimate (pale, p >= 0.05): the '
     'verdicts are the same, slightly stronger in MEC with PC1. The paper uses PC1 throughout, '
     'Figure 2 included, for consistency with the population axis of every other figure; we also '
     'tested every Figure 2 result with the fraction anchored, and it made no difference to any '
     'conclusion. The cost, from C-E and H, is that PC1 would follow a second coherent group if '
     'one dominated a structure, which is the likelier reading of its weaker black-box recovery '
     'in visual cortex and the cerebellum.'),
    ('Figure S3', 'fig1_supp_population_anchoring_examples.pdf',
     'Population anchoring states across sessions, related to Figure 1',
     'Six further sessions, ordered by the share of label variance the first principal component '
     'explains, from sessions where nearly all cells move together to sessions where they do not. '
     '(A-F) One session each, as five panels on a shared trial axis: stops on cued trials, stops '
     'on uncued trials, the anchoring raster with cells on the x axis, PC1, and hit rate by state '
     'and cue condition. Error bars are binomial standard errors, the unit here being the trial '
     'within one session.'),
    ('Figure S4', 'fig1_supp_regions.pdf',
     'The anchoring state in each recorded structure, related to Figure 1',
     'The anchoring state computed in turn from the cells of MEC, visual cortex, the subicular '
     'complex and the cerebellum, with and without matching running speed between states, and '
     'how far cells in different structures agree. (A) Hit '
     'rate by state and cue condition, all trials; sessions with at least 20 trials of each cue '
     'type and 10 in each state within each type, counts below each structure (Wilcoxon '
     'signed-rank across sessions). (B) The Anchoring Dependence Index, AUC(uncued) - AUC(cued), '
     'on the same sessions; black, mean and SEM. (C, D) The same on speed-matched trials: within '
     'each session and cue type, equal numbers of anchored and non-anchored trials drawn from '
     'each decile of trial mean running speed. Because speed and position covary on the track, '
     'a difference in speed between states could otherwise carry the behavioural effect. '
     'Sessions enter if each cue type keeps at least 20 trials, five in each state and both '
     'outcomes. (E) The anchored minus non-anchored difference in trial mean running speed per '
     'session and cue type (circles cued, squares uncued), before and after matching. (F) The '
     'share of trials matching keeps, cued (pale) and uncued (dark). (G) Agreement '
     'between structures, as Figure 3G does for identities: the mean correlation between the '
     'anchoring label sequences of cells in two structures, less the same under an independent '
     'circular shift of each cell, averaged over switching sessions (n, sessions with at least '
     'two cells in each; pairs with fewer than five sessions in grey). Units excluded by the ISI '
     'rule are dropped. (H) The same within sessions that record MEC alongside each structure: '
     'within MEC (left), between MEC and the structure (open) and within the structure (right); '
     'grey lines, sessions; mean and SEM.'),
    ('Figure S5', 'fig1_supp_neighbour_test.pdf',
     'Performance across the session, and how long the state lasts in each structure, related to Figure 2',
     'Labels are classified as in Figure 2, without the median filter, from the current curated '
     'cells of every structure (at least 15 cells, 10 in the cerebellum; 63 sessions, 8 mice). (A) '
     'The neighbour test of Figure 2H on one session: the outcome of each cued (green) and uncued '
     '(pink) trial (filled, hit; open, miss), and the state of MEC and the cerebellum from each '
     'classifier (z-scored PC1 score, as in Figure 2). An uncued trial (dark) is predicted from the state on the nearest cued '
     'trial before it (light, within three trials). (B) Left, hit rate over the same session, in a '
     'running window of 10 trials of each type (shaded, the trials in A). Middle, hit rate per '
     'session: cued trials, where the beacon marks the zone, 0.79; uncued trials 0.51 (paired '
     'Wilcoxon, p = 2e-11). Right, how much performance drifts over the session: the SD of hit rate '
     'across blocks of 10 trials of one type, minus its mean with that type\'s outcomes shuffled in '
     'order (500 shuffles), which removes the variability a hit rate carries by chance. Cued '
     'performance drifts as well as uncued (cued +0.050, p = 3e-8; uncued +0.071, p = 5e-10), and '
     'not detectably less (p = 0.069): the cue lifts performance but does not hold it steady. (C) '
     'Whether the state lasts from trial to trial: its autocorrelation over that of the same trials '
     'in permuted order (solid, task region; dotted, black box). From the black box the state '
     'persists in MEC (lag 1, +0.38), visual cortex (+0.27) and the subicular complex (+0.22), and '
     'barely in the cerebellum (+0.03). (D) What the preceding cued trial\'s own behaviour '
     'says about the uncued outcome, as AUC over a circular-shift null (200 shifts): cued trials that '
     'were hits (+0.049, p = 5e-5) and slowed in the zone (-0.090, p = 3e-5) go with uncued hits, '
     'so engagement drifts over a session, and that behaviour is regressed out in Figure 2H. '
     '(E) Per structure, across sessions: the correlation of the black-box state with the whole-track '
     'state against the whole-track state\'s persistence from one trial to the next (Spearman); '
     'ringed, the sessions of Figure 2D-F; shading in MEC, the session\'s grid cells. In MEC (rho = '
     '0.63, p = 4e-8; grid cells, rho = 0.49) and the subicular complex (rho = 0.77, p = 0.001) the '
     'black box recovers the state where the session has a lasting one; in visual cortex it does '
     'not (rho = 0.25, p = 0.2), however persistent the whole-track state, and in the cerebellum the '
     'relation runs the other way (rho = -0.56, p = 0.03, 15 sessions).'),
    ('Figure S6', 'fig2_supp_nonanchored_rate.pdf',
     'What a non-anchored trial is, related to Figure 3',
     'Two parts of one question: what is a non-anchored trial? (A-F) How a trial gets its '
     "label, step by step, on M26 D18, the session of Figure 1. (A) Two cells' trial x "
     'position rate maps (running only, smoothed, each scaled to its 99th percentile), each '
     "beside the label every trial receives. (B) For one cell, each trial's smoothed map "
     "correlated with every other trial's. (C) That matrix condensed to one number per trial, "
     'its mean correlation with all the other trials: anchored trials resemble one another, so'
     ' their row means are high. (D) The row means split into two groups by 2-means. Because '
     '2-means always returns two groups, both cluster centres (solid) are compared with the '
     "cell's own chance level (dashed: the 95th percentile of the same mean correlation after "
     "each trial's map is circularly shifted); here they straddle it and the split is kept. "
     "(E) The gate's other two outcomes: in cell 139 both centres lie above chance and every "
     'trial is anchored, in cell 116 both lie below and every trial is non-anchored, although '
     '2-means alone would have split each (dotted line). Bars in D and E are coloured by the '
     "final label. (F) Cell 109's labels after 2-means alone, after the gate, and after a "
     '5-trial median filter that removes isolated labels, against the population state (PC1 of'
     " all cells' labels > 0). Trials on which a cell fires no spikes have no map to compare "
     'and are labelled non-anchored. A non-anchored trial is therefore one whose map resembles'
     " the cell's other trials less than its anchored trials do. (G-J) Has the field moved, or"
     " has it gone? (G) Each trial's map correlated against the cell's anchored template at "
     'every circular offset. (H) Zero-offset correlation by state: +0.33 to +0.07. (I) Best-'
     'over-offsets correlation by state, which falls only from +0.56 to +0.53 and looks like '
     'displacement until the null is applied. (J) That null: a maximum over 100 offsets is '
     'large by construction, two unrelated cells reach +0.500 under the same search, and non-'
     'anchored trials sit 0.027 above that floor with a mean best offset of 47 cm against the '
     '50 cm a uniform choice would give. Non-anchored trials lose spatial correspondence with '
     'no evidence of a coherent field elsewhere.'),
    ('Figure S7', 'fig4_supp_anatomy_more.pdf',
     'Two more multi-shank sessions, related to Figure 4',
     'Drawn as Figure 4B, C: the trial-by-cell anchoring raster split by shank from medial to lateral, '
     "under a strip giving each cell's region, beside z-scored PC1, and the 4-shank probe with the "
     "session's recorded contacts dark grey, each cell coloured by agreement above chance and grid cells "
     'as triangles, under per-shank boxplots. (A) M28 D25, 257 cells over four shanks spanning 690 um and '
     'one of the most evenly balanced multi-shank sessions (61% of trials anchored, 6 PC1 transitions). '
     'Counted over every region its most medial shank looks unresponsive (mean agreement 0.14; rho = '
     '-0.105, p = 0.16), but the region strip shows that shank lies entirely in the parasubiculum. Across '
     'its three MEC shanks agreement falls from 0.37 to 0.12 and 0.16 (MEC only, rho = -0.291, p = 0.0003). '
     '(B) M29 D25, 344 cells over four shanks spanning 750 um, from a fourth animal, with no gradient: '
     'agreement across its three MEC shanks is 0.19, 0.29 and 0.18 from medial to lateral (MEC only, rho = '
     "-0.009, p = 0.94), and counted over every region the trend runs the other way (rho = +0.146, p = "
     "0.040). This animal's multi-shank sessions show the gradient weakly at best (strongest, rho = -0.19)."),
    ('Figure S8', 'fig4_supp_factors.pdf',
     'What predicts how closely an MEC cell follows the population state, related to Figure 4',
     'The 3,291 MEC cells on 175 shanks, in 26 sessions from 6 mice whose shanks span at least 200 um, with every '
     'factor measured; the outcome is agreement above chance, as in Figure 4E. Firing rate is held in every '
     'model as a covariate and not ranked: a cell\'s labels are built from its own spikes in the task, so rate '
     'partly sets how well its agreement is measured (alone it explains 4.1% of within-session variance; '
     'unique effect p < 1e-4). (A) Factors ranked by dominance analysis, each one\'s added variance averaged '
     'over every combination of the others (2,048 models, least squares with session fixed effects, firing '
     'rate in all), as a share of the 6.1% of within-session variance the eleven explain beyond rate. Several '
     'factors overlap by construction (putative interneurons are classed partly on waveform, grid cells on '
     'grid score), so a unique share would understate any that duplicate one another; dominance divides '
     'shared variance instead. Lines, 95% interval over 200 bootstraps of whole sessions; p, the unique '
     'effect (likelihood ratio in the mixed model of C, Holm). Mediolateral position ranks first (27%; p = 0.07) '
     'and spatial information second (22%; p = 0.0006); the two intervals overlap widely and position '
     'ranks first in 58% of bootstraps, spatial information in 32%, so their order is not settled. (B) The '
     'ranking three ways: dominance on every cell; dominance on the 2,898 cells firing at least 1 Hz in the '
     'task (the same order, rank rho = +1.00); and permutation importance from gradient-boosted trees with '
     'whole sessions held out, which assumes no linearity and agrees moderately (rho = +0.66): it also puts '
     'mediolateral position and spatial information at the top, but ranks grid score and waveform width '
     'higher and gives cell type and speed tuning almost nothing. (C) Every factor in one linear mixed model, mouse random intercept '
     'with session and shank nested as variance components; coefficients per SD (or against non-grid spatial '
     'cells), 95% CI; black, factor significant after Holm correction. (D) Where the variance in agreement '
     'lies in that model: mouse 10.2%, session within mouse 8.7%, shank within session 5.0%, all twelve '
     'cell properties together, firing rate included, 7.3%, and the remaining 68.8% between cells. (E) The '
     'mediolateral coefficient as factors are added: -0.017 per SD lateral beyond identity, layer and '
     'firing rate (p = 1e-4), -0.012 with every factor (p = 0.006); lateral cells are less theta-modulated '
     '(r = -0.40) and carry less spatial information (r = -0.13), so part of the gradient is shared with '
     'those. Firing rate predicts following within non-grid spatial and non-spatial cells separately '
     '(p = 2e-12, 8e-8; rate x type interaction p = 0.08), so it is not a proxy for interneurons.'),
    ('Figure S9', 'fig6_supp_dimensionality.pdf',
     'One axis is an adequate description of the state, and a second map would have shown, related to Figure 7',
     "Out-of-sample variance explained, with an axis built from one half of a population's cells and "
     'scored on the other, against a null that circularly shifts each cell\'s labels. (A, B, E) What a second '
     'stable map would look like. A population could hold more than one map, each expressed by a subset of '
     'cells and switching on its own schedule (Low et al., 2021). Simulated cells follow their population\'s '
     'block-structured state on 30% of trials and otherwise run their own blocky labels, the fraction set '
     'so that one simulated state matches MEC\'s first-component excess at 8 cells per half (0.034 against '
     '0.035). (A) One simulated session of two populations of 50 cells, each population driven by its own gate. '
     'For display, cells follow their population\'s gate on 90% of trials (so each population\'s labels move '
     'together) and the session is one in which the two gates mostly disagree (state r = -0.37); B and E use '
     'the calibrated 30%. Top, rate maps (trial by position) of four cells drawn at random from each '
     'population, numbered as columns of the raster below, with the cell\'s labels beside each; firing is added to the labels for display only: each cell is a 1D grid cell whose fields '
     'sit at its own phase on anchored trials, the phase re-drawn every non-anchored trial by an '
     'offset shared within its population. Below, the session\'s labels, each population\'s state '
     '(strips), and the session\'s first two components, z-scored, each tracking a mixture of the two states. (B) The '
     'by-component test of D on the simulations, 30 sessions each: one state gives one component; two '
     'states give two that clear their null. (C) Neuron-dropping curve per region: how much of the '
     'held-out half one axis explains as cells are added. (D) The same by component at 8 cells per half, '
     'per region. (E) The second component\'s excess as the axis is built from more cells. A second map '
     'grows with sampling (two populations 50/50, +0.017 at 32 cells per half; 75/25, +0.005); MEC\'s '
     'second component stays at its null at every sample size (-0.0008 at 32 cells per half, 45 sessions, '
     'p = 0.30; +0.0019 at 64, 20 sessions, p = 0.60), as one simulated state does. One axis is an adequate '
     'description of the state in every structure recorded.'),
    ('Figure S10', 'fig7_supp_band_examples.pdf',
     'Band power tracking the population state in individual sessions, related to Figure 8',
     'Four sessions as the population axis, the log-spaced spectrogram and the three band traces '
     'on a shared trial axis. (A-C) Sessions showing the population pattern, slow gamma rising and '
     'fast gamma falling across anchored blocks. (D) A counterexample, drawn in red: a session '
     'with clear state transitions whose band traces do not follow them. It is included because '
     'the population effect is a mean over sessions that individually scatter widely -- slow gamma '
     'rises in 20 of 25 and fast gamma falls in 19 of 25, not in all of them.'),
    ('Figure S11', 'fig7_supp_comodulogram.pdf',
     'Where in the theta cycle each frequency peaks, and how little it moves with the state,'
     'related to Figure 8',
     'Figure 8 fixes two gamma bands in advance and asks how strongly each is nested in theta; the '
     'modulation index discards phase, so it cannot ask where in the cycle they peak. This does, '
     'as a continuous function of frequency. (A, B) The phase-frequency map for each state over 30 '
     'sessions, each frequency row normalised to sum to one so colour is where a frequency peaks '
     "and not how much power it carries. Both states' preferred-phase ridges are drawn on both "
     'maps. (C) Their difference, which is where a state effect would live. The two maps share 94% '
     'of their pixel variance. (D) The same reduced to preferred phase against frequency. The '
     'relationship is a gradient, not a step between two bands: phase runs from about 130 degrees '
     'at 20-30 Hz to about 41 degrees at 60-100 Hz, and that separation holds in each state on its '
     'own (+42 degrees anchored, p = 4e-5; +53 degrees non-anchored, p = 4e-6). Paired within '
     'session the state moves it very little and not reliably: +3.1 degrees in the slow '
     'band (p = 0.36) and -3.1 degrees in the fast (p = 0.54). (E-G) The same question against '
     'depth: theta-triggered LFP and CSD down one shank for three example sessions, both states '
     'overlaid. (H) The laminar phase gradient over 27 sessions by layer. Theta advances about 100 '
     'degrees across the entorhinal layers and the gradient is identical in the two states: '
     "referenced to each session's own layer 2, every layer differs by less than 3 degrees with p "
     '>= 0.07.'),
    ('Figure S12', 'fig7_supp_gamma_locking.pdf',
     'Spike coupling to gamma, by cell identity and by anchoring state, related to Figure 8',
     'Whether fast-spiking interneurons are the cells coupled to gamma, and whether that coupling '
     'moves with the anchoring state. Coupling is pairwise phase consistency, which is unbiased by '
     "spike count. (A) The distance control. Coupling measured on a cell's own channel group is 2 "
     'to 4 times the value measured 120 um away, which is spike waveform leaking into the LFP, so '
     'every other panel uses the nearest group at least 120 um from the cell. (B) Coupling by cell '
     'identity and band. Grid cells are the most coupled class, not fast-spiking cells. (C) '
     'Preferred gamma phase by identity, on polar axes. In the slow band the interneurons have no '
     'consistent preferred phase at all (R = 0.05, against 0.22 to 0.28 for the three principal '
     'classes), so that band is one its putative generators are blind to. (D) The '
     'interneuron-minus-principal phase lag per session: a median +43 degrees in the fast band (p = 5e-6, later in '
     '26 of 31 sessions) and nothing in the slow band (-0.9 degrees, p = 0.93). The interneurons '
     'therefore FOLLOW the principal cells, the direction PING predicts, and do so in one band '
     'only. (E) Coupling by anchoring state. No band and no class moves '
     'with the state (Holm p >= 0.10). (F) Spike counts per cell and state, to show the null in E '
     'is not a power difference.'),
    ('Figure S13', 'fig7_supp_speed_tuning.pdf',
     'Speed interneurons keep their speed gain in both states; speed principal cells lose some of theirs, '
     'related to Figure 8',
     'Theta loses much of its entrainment to running speed in the anchored state, and the account in Figure 10 '
     'turns on whether the cells that carry the speed signal lose it too. Cells are grouped entirely outside '
     'the task, by open-field speed modulation and the waveform interneuron call, as in the Results; 4,133 MEC '
     'cells, 31 sessions, 7 mice. Each cell is fitted in one Poisson model over both states, on running bins '
     'within the two states\' common speed support: a position map and an intercept for each state (cyclic '
     'Fourier basis, four harmonics), speed, and speed x state. The speed coefficient is a gain, the '
     'proportional change in rate per 10 cm/s, which a change in overall firing cannot mimic; with position in '
     'the model it is identical to the gain on unexpected speed (speed minus its position-predicted value). '
     '(A) One open-field speed interneuron, rate against speed in each state (mean and SEM per 5 cm/s), the '
     'cell nearest its group\'s median change. (B) Speed gain in each state for the four groups, per session '
     '(inverse-variance weighted over the group\'s cells), sessions joined; above, the change when anchored '
     'and its p (mixed model, mouse random intercept). (C) The change with its 95% interval (hierarchical '
     'bootstrap, mice then sessions), with position held (filled) and left out (open); dashed, the 42% fall '
     'in theta\'s speed slope. Speed interneurons: -3.9% (-16 to +9%, p = 0.48; +3.0% with position left '
     'out), which excludes a change that size. Speed principal cells: -30% (-66 to -10%, p = 0.008, lower in '
     '22 of 27 sessions; -29% with eight harmonics, -27% above a 1 Hz rate floor), but -5% (p = 0.52) with '
     'position left out, since speed and position are collinear on the track, more so when anchored, and '
     'without position the speed coefficient carries spatial firing. (D) Population decoding of unexpected '
     'speed (position removed from speed and every cell\'s rate, ridge regression, bins matched between '
     'states for count and speed): 0.65 anchored against 0.72, p = 0.25. (E) There is less unexpected speed '
     'when anchored (SD 13.5 against 17.6 cm/s, p = 6e-8): mice run more stereotypically.'),
    ('Figure S14', 'fig_supp_lfp_spectra.pdf',
     'The visual cortex - MEC boundary in the local field potential, every mouse and shank',
     'One row per mouse and one column per shank, most medial on the left; where a shank was '
     'recorded on several days, the single-shank session with the most curated cells is shown.'
     " Each panel is one shank's LFP channels (each the mean of 8 neighbouring contacts, ~60 "
     'um) against dorsoventral position: left, the atlas region of each channel (green visual '
     'cortex, black MEC, grey other); middle, log Welch power (1-300 Hz, running only, mains '
     "frequencies removed) z-scored per frequency across the shank's channels; right, the "
     'curated cells by depth (G grid, N non-grid spatial, S non-spatial, I putative '
     'interneuron, each cell in one column), coloured by atlas region, with marker size '
     "proportional to the cell's firing rate in that session (to 100 Hz): cortical cells "
     'mostly fire below 20 Hz, while the cells below cortex include a tail of fast cells (90th'
     ' percentile 57-79 Hz in M26 D17 and M27), as cerebellar Purkinje cells would. The gross '
     'anatomy is then set by rules that use no atlas information. Every shank enters visual '
     'cortex and passes into MEC or parasubiculum, then either ends there or leaves cortex, '
     'some reaching cerebellum. The shank enters brain at its most dorsal curated cell. The '
     'visual cortex - MEC/parasubiculum boundary (orange) is the top of the block of high '
     'power from ~40 to 300 Hz, capped by a ~60 Hz component, that MEC carries and visual '
     'cortex lacks: channels belong to a block where power in any of 40-80, 80-150 or 150-300 '
     'Hz is above the shank average; of the blocks beginning at least 500 um above the exit '
     'from cortex (MEC is never thinner), the one with the most power is taken, and the '
     'boundary is its onset, where power has risen 30% of the way from the visual cortex level'
     " above to the block's peak. A shank with no such block shows no visual cortex crossing "
     '(all of M27): everything from its top to the exit is MEC/parasubiculum. The ventral edge'
     ' of cortex carries a band of strong 10-44 Hz power, found without choosing a band as one'
     " of four components of a non-negative factorisation of all single-shank sessions' "
     'spectra (peak 31 Hz). The shank leaves cortex (purple; grey below) at the deepest peak '
     'of 10-44 Hz power below the boundary when that peak is prominent: shanks with cells '
     'beyond a gap in the cells, all of them cerebellar, give prominences of 0.43-1.1 and '
     'shanks ending inside cortex at most 0.33. Without a prominent peak the shank ends inside'
     ' cortex, at its deepest cell. Where both are available the LFP edge and the first 300 um'
     ' gap in the curated cells agree to a median of 82 um (32 sessions). Dashed, the atlas '
     "border; the title gives the rule's offset from it. Over the 32 single-shank sessions "
     'crossing both structures in the atlas, the boundary lies within 250 um of the atlas '
     'border in 19 (median 194 um, none beyond 600 um), with offsets shared by every shank of '
     'a mouse (about -360 um in M21, -480 um in M28, +420 um in M29), as an error in '
     'registering that mouse to the atlas would produce; repeat recordings of one shank on '
     'different days agree to within 150 um in 9 of 10 cases (median 60 um). Multi-shank '
     "sessions take, for each cell's shank, the mean of that shank's single-shank estimates. "
     'Because grid cells are expected in MEC, cell labels then use a boundary corrected by '
     'cell type: where grid cells (all sessions of a shank pooled) run continuously up from MEC '
     'across the LFP boundary, it is raised to the top of that run, the run ending at the '
     'first gap of more than 100 um, and by at most 150 um; three shanks of M25 moved (51, '
     '125 and 125 um), relabelling 20 grid cells, and the panels show the uncorrected LFP '
     'estimate. '
     'Rule and atlas assign the same structure to 91% of the 13,748 cells the atlas places in '
     'visual cortex or MEC/parasubiculum.'),
    ('Figure S15', 'fig_supp_histology_example.pdf',
     'Estimating the gross anatomy of each shank: histology, local field potential and cells in M26',
     '(A) The sections of M26 (DAPI blue, DiI red; the shanks were coated in DiI before '
     'implantation), cut 60 um apart and shown lateral to medial at true aspect. (B) The DiI '
     'signal alone, red above the blue counterstain. (C) DiI summed per section. Each shank '
     'leaves a track that spreads into neighbouring sections; the peaks (orange) are assigned '
     'to the four shanks in mediolateral order, the most medial shank leaving only a weak, '
     'broad track. (D) For each shank, medial to lateral: above, its peak section, which shows'
     ' the track entering visual cortex and descending into the entorhinal region; below, that'
     " shank's single-shank recording, with the atlas region of each LFP channel (left strip),"
     ' the LFP spectra (log power z-scored per frequency across channels, as in Figure S14) '
     'and the curated cells by depth (G grid, N non-grid spatial, S non-spatial, I putative '
     'interneuron; coloured by atlas region; marker size grows with firing rate). The rules of'
     ' Figure S14, which use no atlas information, place the visual cortex - MEC/parasubiculum'
     ' boundary at the top of the block of 40-300 Hz power, capped by a ~60 Hz component '
     '(orange) and the exit from cortex at the deepest prominent peak of 10-44 Hz power '
     '(purple; grey below). On the three shanks the atlas places in both structures, the '
     'boundary lies 296 um above, exactly at, and 59 um above the atlas border; the most '
     'medial shank, which the atlas places in visual cortex and parasubiculum, leaves cortex '
     'at 2,799 um, 81 um below the last cell before a 400 um gap in the cells, and the 36 '
     'cells beyond it, many firing at 50-100 Hz, are taken to be cerebellum.'),
    ('Figure S15 (continued)', 'fig_supp_histology_example_M20.pdf',
     'The same estimate in a second mouse, M20',
     'Panels as on the previous page. M20\'s DiI leaves four discrete tracks, one per shank (sections 5, 7, '
     '10 and 13, 120-180 um apart, as the shanks\' 180 um mediolateral spacing in this animal predicts for '
     '60 um sections), so here each peak is a single shank without the spread of M26. The physiological '
     'boundary lies at the atlas border on the most lateral shank and 119, 180 and 537 um below it on the '
     'other three, medial to lateral; no shank leaves cortex before its last cell (exits at 4,031-4,316 um), '
     'and none records cells beyond it. The 537 um disagreement on shank 3 is the largest in this mouse and '
     'is where the atlas registration and the spectra part most.'),
    ('Figure S16', 'fig7_supp_more_examples.pdf',
     'More sessions in which visual cortex and MEC are recorded on one shank, related to Figure 7',
     'Five further single-shank sessions with clear switching in MEC and at least five '
     'switching visual cells, each drawn as a row of Figure 7 and on its agreement scale: '
     '(A, B) M25 D19, (C, D) M29 D19, (E, F) M28 D20, (G, H) M21 D26, (I, J) M26 D12. Left of '
     'each row, the anchoring raster split into visual and MEC blocks, each structure\'s own '
     'PC1, the probe with the recorded contacts and every cell coloured by its agreement with '
     'the MEC PC1, the shank\'s LFP spectra with the landmarks of Figure S14 and, where the '
     'DiI track of that shank was identified, its histological section. Inset, the spike '
     'waveform of the most dorsal curated unit (arrowhead on the probe): its mean on each of the 16 '
     'contacts nearest it, laid out as the contacts are (peak contact black, with its SD in grey). '
     'In every session it is a small visual cortex unit (11-17 uV peak to peak; within 15 um of '
     'the top recorded contact in three sessions, 90 and 180 um below it in the other two), none of '
     'them among the example cells. Right, five switching '
     'cells (two visual, three MEC) as trial x position rate maps beside their own labels. '
     'Across all 23 single-shank sessions with at least ten switching MEC and five switching '
     'visual cells, the visual PC1 correlates with the MEC PC1 above its circular-shift null '
     'in 15; in M21 D23, M21 D24 and M26 D11 it runs against it.'),
    ('Figure S17', 'fig6_supp_pupil_examples.pdf',
     'The pupil is smaller in the anchored state in further sessions, related to Figure 6',
     'Five further sessions, one per row, each drawn as Figure 6 C-G: (A) M26 D18, (B) M28 D19, '
     '(C) M27 D19, (D) M21 D22, (E) M28 D25. Left, the eye at seven moments of one anchored '
     '(purple) and one non-anchored (teal) traversal, with the DeepLabCut perimeter landmarks '
     'and the pupil radius (px) beneath each frame; each is the trial of its state\'s longest '
     'block whose radius profile is closest to that block\'s mean. Middle, mean pupil radius '
     'against track position for the two blocks (+- SEM across trials; trial counts in the '
     'legend), with the displayed traversals overlaid (thin lines). Right, the MEC anchoring '
     'raster and its PC1, the pupil radius on every trial on the same trial axis, and that map '
     'z-scored over trials at each 5 cm position bin (purple, smaller than usual for that '
     'position; orange, larger), which removes the position profile shared by every trial and '
     'leaves the change between blocks; arrowheads '
     'mark the displayed traversals, dashed lines the transitions. Above each row, the '
     'anchored minus non-anchored difference in z-scored radius over all of the session\'s '
     'trials, as in Figure 6H. Sessions were ranked by the difference between the two longest '
     'blocks relative to its standard error, which is what the middle panels show: the best '
     'session of each mouse in which the blocks differ clearly (M26, M28, M27, M21), then the '
     'next best overall. In M20, M22 and M25 the longest blocks differ little or not at all.'),
    ('Figure S18', 'fig5_supp_subnetwork.pdf',
     'Grid cells and interneurons follow the state together, but the wiring does not pick out the followers; direction of the connections along the probe, related to Figure 5',
     'MEC grid cells and putative interneurons tested for monosynaptic connections (1,191 cells, 63 '
     'sessions); a cell is wired if it has at least one detected connection, of either sign and '
     'direction, with another grid cell or interneuron (623 cells). Agreement is each switching '
     'cell\'s agreement above chance with the MEC PC1 (790 cells; locked cells have none). (A) One '
     'session\'s network on the 4-shank Neuropixels 2.0 probe (M29 D23; the session\'s 384 recorded '
     'contacts dark grey, the inactive ones light grey; black dots, the other cells): grid cells (triangles) and interneurons (circles) '
     'filled by agreement, locked cells hatched, joined by their inhibitory (purple) and '
     'excitatory (teal) connections, pre- to postsynaptic; other cells grey. (B) As identities '
     'they do share the state: the mean label correlation of grid-grid, grid-interneuron and '
     'interneuron-interneuron pairs exceeds that of the session\'s other pairs (per session, '
     'Wilcoxon). (C) Pooled over sessions, wired cells appear to follow more strongly than '
     'unwired cells of the same identity (cell counts above the bars). (D) Within a session they '
     'do not: wired minus unwired agreement per session, and the difference in a model with '
     'session fixed effects and log spike count (detection depends on spike count, and spike '
     'count predicts following, Figure S8): grid +0.036 (95% CI -0.010 to +0.082, p = 0.13), '
     'interneuron +0.012 (-0.021 to +0.046, p = 0.47). The pooled difference in C reflects '
     'sessions with more detected connections also following more strongly, not the wiring. '
     '(E) The number of grid and interneuron partners does not predict agreement (both '
     'residualised on session, identity and log spike count). (F) The wired cells\' labels are '
     'no more coherent with each other than sets of the same session\'s grid cells and '
     'interneurons matched for identity and spike-count tercile (1,000 sets; 34 sessions with at '
     'least four wired switching cells): +0.013, p = 0.11. This test has limited power, since in '
     'many sessions most of these cells are wired. Together with Figure 5I, which finds no extra '
     'state sharing between directly connected pairs, the shared state of grid cells and '
     'interneurons is a property of the identities, not of the detected connections. '
     '(G-I) Direction of all MEC connections along the probe, every identity (larger probe y is '
     'dorsal). Near either end of the probe most of a cell\'s candidate partners lie on one '
     'side, so each connection\'s expected chance of pointing dorsal is the fraction of the '
     'session\'s tested cells at the same distance (within 25 um) from its presynaptic cell that '
     'lie dorsal to it. (G) Direction of each connection '
     '(post- from presynaptic cell in the probe plane), as observed over expected connections '
     'in each direction, the expected count using the angles of the same candidate cells '
     '(dashed circle: chance; +-1 SE; faded: fewer than 30 expected, mostly across shanks). '
     'Inhibitory (7,382): 64.0% point dorsal where 57.9% are expected, so most of the '
     'ventral-to-dorsal majority is geometry and the excess is the dorsal bulge. Excitatory '
     '(1,431): 36.3% dorsal, 50.0% expected. (H) The excess by pre-post distance (pooled, 95% interval from the '
     'per-connection expectation): inhibitory connections point dorsal more than expected at '
     '100-400 um and not below 50 um or beyond 400 um; excitatory connections beyond 100 um point ventral. '
     '(I) Per session (at least 10 connections), both hold: inhibitory connections point dorsal more '
     'than expected (+0.043, p = 3e-4, 57 sessions; positive in 5 of 8 mice) and excitatory ones '
     'ventral (-0.142, p = 5e-8, 40 sessions; negative in 7 of 8 mice).'),
    ('Figure S19', 'fig5_supp_narrow_continuum.pdf',
     'Narrow-spiking cells form a continuum in firing rate, with no upward step at the interneuron rate cut, related to Figure 5',
     'Figure 5 calls a cell an interneuron if its waveform is narrow and its rate is at least '
     '8.8 Hz, the top quartile (dotted line in every panel). Here the rate cut is dropped: every measure is shown '
     'against firing rate for all narrow-spiking cells (orange) and, as the baseline for detection '
     'power, broad-spiking cells at the same rates (grey); bands are 95% intervals, bins with at '
     'least 15 cells. A two-class account predicts a jump at the cut, tested by local regression '
     'discontinuity (cells within 5.6-14 Hz, a separate line in log rate either side). (A) The '
     'narrow cells\' rates have one mode (2,133 cells, 50% below the cut; Silverman test p = 0.89) '
     'and no gap at the cut. (B, C) The share of narrow cells inhibiting a tested partner, and '
     'the share of partners they inhibit, rise steadily with rate, with no upward jump at the cut; '
     'the local fit puts a small downward step there (x0.58, 95% CI 0.39-0.85, 529 cells), the '
     'opposite of what a two-class account predicts. Below the cut narrow cells still inhibit 4.3 '
     'times as many partners as broad cells of the same rate (3.3-5.5, p = 8e-29), while broad cells '
     'stay flat over 1.5-10 Hz, so the rise is not detection power. (D) The same holds onto each '
     'postsynaptic class; narrow cells inhibit other narrow cells least. (E) Excitatory output '
     'falls with rate; below the cut narrow cells excite as many partners as broad cells (x0.88, '
     '0.63-1.22, p = 0.45), so the slow narrow cells include excitatory ones. (F) '
     'Excitatory input received falls with the cell\'s own rate in both groups; below the cut '
     'narrow cells receive 1.4 times as much as broad cells (1.09-1.83). (G) Inhibitory latency is '
     '2.5-2.9 ms at every rate (jump -0.23 ms, -0.46 to 0.00). (H) Agreement with the MEC '
     'population state rises with rate in narrow cells (+0.074 per 10-fold rate, p = 9e-14; '
     'mouse random effect, session nested), with no jump at the cut (-0.074, -0.164 to +0.016), '
     'and below the cut narrow cells follow as broad cells of the same rate do (-0.012, '
     'p = 0.18; 304 narrow, 3,180 broad cells). (I) Figure 5I along the continuum: pairs '
     'containing a narrow cell, split by its rate (tercile; for two narrow cells their geometric '
     'mean), against unconnected pairs matched for session, partner class and distance. '
     'Connected pairs share the state no more at any rate (p = 0.40, 0.46, 0.88; minimum '
     'detectable effects 0.012-0.022). The narrow cells are a graded mixture, inhibitory more '
     'often the faster they fire, and the 8.8 Hz cut selects the end where inhibitory cells '
     'dominate rather than a boundary between classes.'),
]

# Every work cited in the text, keyed by first author's surname + year, which
# is what cite_keys() reads out of an author-year citation. Each entry was
# checked against the published record (the PDF where the authors' library has
# it). check_citations() fails the build on a citation with no entry here, and
# warns on an entry nothing cites, so the list cannot drift from the text again.
REFS = {
    'Aston-Jones2005':
    'Aston-Jones, G., and Cohen, J.D. (2005). An integrative theory of locus '
    'coeruleus-norepinephrine function: adaptive gain and optimal performance. Annu. Rev. '
    'Neurosci. 28, 403-450.',
    'Banino2018':
    'Banino, A., Barry, C., Uria, B., Blundell, C., Lillicrap, T., Mirowski, P., Pritzel, A., '
    'Chadwick, M.J., Degris, T., Modayil, J., et al. (2018). Vector-based navigation using '
    'grid-like representations in artificial agents. Nature 557, 429-433.',
    'Benjamin2018':
    'Benjamin, A.S., Fernandes, H.L., Tomlinson, T., Ramkumar, P., VerSteeg, C., Chowdhury, '
    'R.H., Miller, L.E., and Kording, K.P. (2018). Modern machine learning as a benchmark for '
    'fitting neural responses. Front. Comput. Neurosci. 12, 56.',
    'Bieri2014':
    'Bieri, K.W., Bobbitt, K.N., and Colgin, L.L. (2014). Slow and fast gamma rhythms '
    'coordinate different spatial coding modes in hippocampal place cells. Neuron 82, 670-681.',
    'Bimbard2025':
    'Bimbard, C., Takács, F., Catarino, J.A., Fabre, J.M.J., Gupta, S., Lenzi, S.C., Melin, M.D., '
    'O\'Neill, N., Orsolic, I., Robacha, M., et al. (2025). An adaptable, reusable, and light '
    'implant for chronic Neuropixels probes. eLife 13, RP98522.',
    'Boussard2021':
    'Boussard, J., Varol, E., Lee, H.D., Dethe, N., and Paninski, L. (2021). Three-dimensional '
    'spike localization and improved motion correction for Neuropixels recordings. In Advances '
    'in Neural Information Processing Systems 34 (NeurIPS 2021).',
    'Buccino2020':
    'Buccino, A.P., Hurwitz, C.L., Garcia, S., Magland, J., Siegle, J.H., Hurwitz, R., and '
    'Hennig, M.H. (2020). SpikeInterface, a unified framework for spike sorting. eLife 9, '
    'e61834.',
    'Chen2016':
    'Chen, T., and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. In '
    'Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and '
    'Data Mining, pp. 785-794.',
    'Clark2024':
    'Clark, H., and Nolan, M.F. (2024). Task-anchored grid cell firing is selectively '
    'associated with successful path integration-dependent behaviour. eLife 12, RP89356.',
    'Colgin2009':
    'Colgin, L.L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., Moser, M.-B., '
    'and Moser, E.I. (2009). Frequency of gamma oscillations routes flow of information in the '
    'hippocampus. Nature 462, 353-357.',
    'Desikan2018':
    'Desikan, S., Koser, D.E., Neitz, A., and Monyer, H. (2018). Target selectivity of septal '
    'cholinergic neurons in the medial and lateral entorhinal cortex. Proc. Natl. Acad. Sci. '
    'USA 115, E2644-E2652.',
    'Fabre2023':
    'Fabre, J.M.J., van Beest, E.H., Peters, A.J., Carandini, M., and Harris, K.D. (2023). '
    'Bombcell: automated curation and cell classification of spike-sorted electrophysiology '
    'data. Zenodo. https://doi.org/10.5281/zenodo.8172821.',
    'Fuhrmann2015':
    'Fuhrmann, F., Justus, D., Sosulina, L., Kaneko, H., Beutel, T., Friedrichs, D., Schoch, S., '
    'Schwarz, M.K., Fuhrmann, M., and Remy, S. (2015). Locomotion, theta oscillations, and the '
    'speed-correlated firing of hippocampal neurons are controlled by a medial septal '
    'glutamatergic circuit. Neuron 86, 1253-1264.',
    'Gardner2022':
    'Gardner, R.J., Hermansen, E., Pachitariu, M., Burak, Y., Baas, N.A., Dunn, B.A., Moser, '
    'M.-B., and Moser, E.I. (2022). Toroidal topology of population activity in grid cells. '
    'Nature 602, 123-128.',
    'Gonzalez-Sulser2014':
    'Gonzalez-Sulser, A., Parthier, D., Candela, A., McClure, C., Pastoll, H., Garden, D., '
    'Sürmeli, G., and Nolan, M.F. (2014). GABAergic projections from the medial septum '
    'selectively inhibit interneurons in the medial entorhinal cortex. J. Neurosci. 34, '
    '16739-16743.',
    'Hardcastle2017':
    'Hardcastle, K., Maheswaranathan, N., Ganguli, S., and Giocomo, L.M. (2017). A '
    'multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal '
    'cortex. Neuron 94, 375-387.',
    'International2022':
    'International Brain Laboratory (2022). Spike sorting pipeline for the International '
    'Brain Laboratory. figshare. https://doi.org/10.6084/m9.figshare.19705522.',
    'Jacob2019':
    'Jacob, P.-Y., Gordillo-Salas, M., Facchini, J., Poucet, B., Save, E., and Sargolini, F. '
    '(2019). Path integration maintains spatial periodicity of grid cell firing in a 1D circular '
    'track. Nat. Commun. 10, 840.',
    'Justus2017':
    'Justus, D., Dalugge, D., Bothe, S., Fuhrmann, F., Hannes, C., Kaneko, H., Friedrichs, D., '
    'Sosulina, L., Schwarz, I., Elliott, D.A., et al. (2017). Glutamatergic synaptic integration '
    'of locomotion speed via septoentorhinal projections. Nat. Neurosci. 20, 16-19.',
    'Kempter2012':
    'Kempter, R., Leibold, C., Buzsaki, G., Diba, K., and Schmidt, R. (2012). Quantifying '
    'circular-linear associations: hippocampal phase precession. J. Neurosci. Methods 207, '
    '113-124.',
    'Kropff2015':
    'Kropff, E., Carmichael, J.E., Moser, M.-B., and Moser, E.I. (2015). Speed cells in the medial '
    'entorhinal cortex. Nature 523, 419-424.',
    'Low2021':
    'Low, I.I.C., Williams, A.H., Campbell, M.G., Linderman, S.W., and Giocomo, L.M. (2021). '
    'Dynamic and reversible remapping of network representations in an unchanging environment. '
    'Neuron 109, 2967-2980.',
    'Mathis2018':
    'Mathis, A., Mamidanna, P., Cury, K.M., Abe, T., Murthy, V.N., Mathis, M.W., and Bethge, M. '
    '(2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep '
    'learning. Nat. Neurosci. 21, 1281-1289.',
    'Obenhaus2022':
    'Obenhaus, H.A., Zong, W., Jacobsen, R.I., Rose, T., Donato, F., Chen, L., Cheng, H., '
    'Bonhoeffer, T., Moser, M.-B., and Moser, E.I. (2022). Functional network topography of the '
    'medial entorhinal cortex. Proc. Natl. Acad. Sci. USA 119, e2121655119.',
    'Pachitariu2024':
    'Pachitariu, M., Sridhar, S., Pennington, J., and Stringer, C. (2024). Spike sorting with '
    'Kilosort4. Nat. Methods 21, 914-921.',
    'Pettit2022':
    'Pettit, N.L., Yuan, X.C., and Harvey, C.D. (2022). Hippocampal place codes are gated by '
    'behavioral engagement. Nat. Neurosci. 25, 561-566.',
    'Reimer2016':
    'Reimer, J., McGinley, M.J., Liu, Y., Rodenkirch, C., Wang, Q., McCormick, D.A., and '
    'Tolias, A.S. (2016). Pupil fluctuations track rapid changes in adrenergic and cholinergic '
    'activity in cortex. Nat. Commun. 7, 13289.',
    'Robinson2024':
    'Robinson, J., Ying, J., Hasselmo, M.E., and Brandon, M.P. (2024). Optogenetic silencing of '
    'medial septal GABAergic neurons disrupts grid cell spatial and temporal coding in the '
    'medial entorhinal cortex. Cell Rep. 43, 114590.',
    'Steinmetz2021':
    'Steinmetz, N.A., Aydin, C., Lebedeva, A., Okun, M., Pachitariu, M., Bauza, M., Beau, M., '
    'Bhagat, J., Böhm, C., Broux, M., et al. (2021). Neuropixels 2.0: a miniaturized '
    'high-density probe for stable, long-term brain recordings. Science 372, eabf4588.',
    'Tennant2018':
    'Tennant, S.A., Fischer, L., Garden, D.L.F., Gerlei, K.Z., Martinez-Gonzalez, C., McClure, C., '
    'Wood, E.R., and Nolan, M.F. (2018). Stellate cells in the medial entorhinal cortex are '
    'required for spatial learning. Cell Rep. 22, 1313-1324.',
    'Tennant2022':
    'Tennant, S.A., Clark, H., Hawes, I., Tam, W.K., Hua, J., Yang, W., Gerlei, K.Z., '
    'Wood, E.R., and Nolan, M.F. (2022). Spatial representation by ramping activity of '
    'neurons in the retrohippocampal cortex. Curr. Biol. 32, 4451-4464.',
    'Tort2010':
    'Tort, A.B.L., Komorowski, R., Eichenbaum, H., and Kopell, N. (2010). Measuring '
    'phase-amplitude coupling between neuronal oscillations of different frequencies. J. '
    'Neurophysiol. 104, 1195-1210.',
    'Wang2020':
    'Wang, Q., Ding, S.-L., Li, Y., Royall, J., Feng, D., Lesnar, P., Graddis, N., Naeemi, M., '
    'Facer, B., Ho, A., et al. (2020). The Allen Mouse Brain Common Coordinate Framework: a 3D '
    'reference atlas. Cell 181, 936-953.',
    'Ye2018':
    'Ye, J., Witter, M.P., Moser, M.-B., and Moser, E.I. (2018). Entorhinal fast-spiking speed '
    'cells project to the hippocampus. Proc. Natl. Acad. Sci. USA 115, E1627-E1636.',
}

# one author-year citation: 'Kropff et al., 2015', 'Clark and Nolan, 2024',
# 'International Brain Laboratory, 2022'; and the narrative form
# 'Hardcastle et al. (2017)'
_AUTH = r"[A-Z][A-Za-zÀ-ſ'\-]+(?: et al\.| and [A-Z][A-Za-zÀ-ſ'\-]+| Brain Laboratory)?"
CITE_ONE = re.compile(rf'({_AUTH}),? ((?:19|20)\d\d)[a-z]?')
CITE_PAREN = re.compile(rf'\(((?:{_AUTH}),? (?:19|20)\d\d[a-z]?'
                        rf'(?:; (?:{_AUTH}),? (?:19|20)\d\d[a-z]?)*)\)')
CITE_NARR = re.compile(rf'({_AUTH}) \(((?:19|20)\d\d)[a-z]?\)')


def cite_key(author, year):
    return author.split()[0] + year


def cite_keys(t):
    """The REFS keys cited in t, in order of first appearance."""
    out = []
    for m in re.finditer(f'{CITE_PAREN.pattern}|{CITE_NARR.pattern}', t):
        if m.group(1):
            ks = [cite_key(a, y) for a, y in CITE_ONE.findall(m.group(1))]
        else:
            ks = [cite_key(m.group(2), m.group(3))]
        out += [k for k in ks if k not in out]
    return out


def check_citations(texts):
    """Fail on a citation with no reference; warn on a reference nothing cites."""
    cited = []
    for t in texts:
        cited += [k for k in cite_keys(t) if k not in cited]
    missing = [k for k in cited if k not in REFS]
    if missing:
        raise SystemExit(f'cited but not in REFS: {missing}')
    unused = [k for k in REFS if k not in cited]
    if unused:
        print(f'  ! in REFS but never cited: {unused}')
    print(f'  references: {len(cited)} cited, all resolved')
    return cited


GREEK = {'ρ': 'rho', 'σ': 'sigma', 'λ': 'lambda', 'α': 'alpha', 'β': 'beta',
         'χ': 'chi', 'Χ': 'chi', 'μ': 'u', 'Δ': 'delta', 'π': 'pi', 'θ': 'theta'}
SUPS = {'⁻': '-', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
        '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}


SUP_ON, SUP_OFF = '\x01', '\x02'
SUP_SCALE, SUP_RISE = .66, .32      # of the body size


def join_line(words):
    """A wrapped line as plain text, markers stripped.

    wrap() returns the words of a line, not a string, because measurement and
    justification both need them separately. Callers that just want to stamp a
    line go through here -- drawing the list itself silently stacked every word
    at one point, which is what broke the cover page.
    """
    if isinstance(words, str):
        words = [words]
    return ' '.join(words).replace(SUP_ON, '').replace(SUP_OFF, '')


def runs(word):
    """Split a word into (text, superscript?) runs at the markers."""
    out, sup = [], False
    for part in word.replace(SUP_OFF, SUP_ON).split(SUP_ON):
        if part:
            out.append((part, sup))
        sup = not sup
    return out or [('', False)]


def run_len(word, font, size):
    return sum(text_len(t, font, size * (SUP_SCALE if sup else 1.0))
               for t, sup in runs(word))


def clean(t):
    """Markdown and unicode down to what the base-14 fonts can actually set."""
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)', r'\1', t)
    t = t.replace('`', '')
    # Superscripts are MARKED, not flattened. They used to come out as the
    # literal '10^-6', which is a typesetting failure visible on every page;
    # the markers survive wrapping and are set raised and smaller at draw time.
    t = re.sub(r'([A-Za-z0-9])([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)',
               lambda m: m.group(1) + SUP_ON
               + ''.join(SUPS.get(c, c) for c in m.group(2)) + SUP_OFF, t)
    t = re.sub(r'\^\{?(-?\d+)\}?', lambda m: SUP_ON + m.group(1) + SUP_OFF, t)
    for a, b in {**GREEK, **SUPS}.items():
        t = t.replace(a, b)
    for a, b in {'—': '-', '–': '-', '×': 'x', '≈': '~', '≥': '>=', '≤': '<=',
                 '“': '"', '”': '"', '‘': "'", '’': "'", '·': '.', '√': 'sqrt',
                 '±': '+/-', '→': '->', '…': '...'}.items():
        t = t.replace(a, b)
    return t


class Doc:
    def __init__(self):
        self.d = pymupdf.open()
        self.page = None
        self.col = 0
        self.y = 0.0
        self.full = True          # front matter runs the page width
        self.pending = []         # span figures waiting for the next page top
        self.col_top = MARGIN_TOP  # where columns start on the CURRENT page
        self.new_page(first=True)

    def col_rect(self):
        # In full-width mode the "column" is the whole measure. Without this the
        # masthead paragraphs wrap at the full width but overflow into column 2,
        # which overprinted the body text on the first build.
        if self.full:
            return MARGIN_X, PW - MARGIN_X
        x = MARGIN_X + self.col * (COLW + GUTTER)
        return x, x + COLW

    def new_page(self, first=False):
        self.page = self.d.new_page(width=PW, height=PH)
        if _HAVE_AV:
            for _n, _f in _FONTFILES.items():
                self.page.insert_font(fontname=_n, fontfile=_f)
        self.col = 0
        self.col_top = MARGIN_TOP      # only page 1 reserves room for the masthead
        self.y = MARGIN_TOP
        if not first:
            self.page.draw_line(pymupdf.Point(MARGIN_X, MARGIN_TOP - 16),
                                pymupdf.Point(PW - MARGIN_X, MARGIN_TOP - 16),
                                color=(.75, .75, .75), width=.4)
            self.page.insert_text(pymupdf.Point(MARGIN_X, MARGIN_TOP - 21),
                                  'Clark et al.  |  Article', fontname=SANS,
                                  fontsize=6.4, color=(.45, .45, .45))

    def space(self, h):
        if self.y + h > PH - MARGIN_BOT:
            self.next_col()

    def next_col(self):
        if self.full:
            self.new_page()
            return
        if self.col == 0:
            self.col = 1
            self.y = self.col_top
        else:
            self.new_page()
            self._flush_pending()

    def _flush_pending(self):
        """Set one queued span figure across the top of this fresh page.

        Figures used to be placed the moment their heading appeared, and
        _figure_span began with new_page() -- so a heading could be written, the
        page immediately abandoned, and the figure started on the next one. That
        is what left page 3 holding nothing but a two-line heading. Queuing
        instead lets the text fill the page it is on, and the figure opens the
        next page with the text continuing beneath it.
        """
        if not self.pending or self.col != 0 or self.y > self.col_top + 1:
            return
        lab, path, ttl, cap = self.pending.pop(0)
        src = pymupdf.open(path)
        self._place_span(lab, src, ttl, cap)
        src.close()

    def wrap(self, text, font, size, width):
        """Greedy wrap, measuring superscript runs at their own size."""
        sp = text_len(' ', font, size)
        lines, cur, cw = [], [], 0.0
        for w in text.split():
            ww = run_len(w, font, size)
            if cur and cw + sp + ww > width:
                lines.append(cur); cur, cw = [w], ww
            else:
                cw += (sp if cur else 0) + ww
                cur.append(w)
        if cur:
            lines.append(cur)
        return lines

    def draw_line(self, words, x, width, font, size, color, justify=False):
        """One line of words, optionally justified, superscripts raised.

        Justification spreads the slack between words rather than scaling the
        glyphs. The last line of a paragraph is never stretched, and neither
        is a line whose slack would open gaps wider than a third of the
        measure -- a short line forced by a long word looks broken that way.
        """
        sp = text_len(' ', font, size)
        nat = sum(run_len(t, font, size) for t in words) + sp * (len(words) - 1)
        extra = 0.0
        if justify and len(words) > 1:
            slack = width - nat
            if 0 < slack < width / 3.0:
                extra = slack / (len(words) - 1)
        for w in words:
            for t, sup in runs(w):
                fs = size * (SUP_SCALE if sup else 1.0)
                yy = self.y - size * SUP_RISE if sup else self.y
                self.page.insert_text(pymupdf.Point(x, yy), t, fontname=font,
                                      fontsize=fs, color=color)
                x += text_len(t, font, fs)
            x += sp + extra

    def para(self, text, font=SERIF, size=BODY, lead=BODY_LEAD, indent=0.0,
             color=(0, 0, 0), gap=3.0, width=None, justify=True):
        x0, x1 = self.col_rect()
        w = width if width else (x1 - x0)
        wrapped = self.wrap(clean(text), font, size, w - indent)
        for i, ln in enumerate(wrapped):
            if self.y + lead > PH - MARGIN_BOT:
                self.next_col()
                x0, x1 = self.col_rect()
                if width is None:
                    w = x1 - x0
            self.draw_line(ln, x0 + indent, w - indent, font, size, color,
                           justify=justify and i < len(wrapped) - 1)
            self.y += lead
        self.y += gap

    def heading(self, text, size=H2, font=SANS_B, color=(0, 0, 0), pre=6.0):
        # Reserve room for the heading AND three lines of the text that follows.
        # Checking only the heading's own height lets it sit alone at the foot of
        # a column with its section starting in the next one.
        need = pre + 2 * (size + 2.0) + 3 * BODY_LEAD
        if self.y + need > PH - MARGIN_BOT:
            self.next_col()
        self.y += pre
        self.para(text, font=font, size=size, lead=size + 2.0, color=color, gap=2.5)

    def figure(self, label, path, title, caption):
        """Place a figure. Wide or large multi-panel figures span both columns.

        A 11.8 x 11.1 in figure set into a 2.3 in column is unreadable, and Cell
        runs figures of that kind across the measure. Spanning figures are placed
        at the top of a fresh page with the caption beneath them, and the
        two-column text resumes below -- which is also how the journal sets them.
        """
        src = pymupdf.open(path)
        r = src[0].rect
        span = (r.width / r.height) > 1.25 or r.width / 72.0 > 8.5
        if span:
            self._figure_span(label, src, title, caption)
        else:
            self._figure_col(label, src, title, caption)
        src.close()

    def _figure_col(self, label, src, title, caption):
        r = src[0].rect
        x0, x1 = self.col_rect()
        w = x1 - x0
        h = w * (r.height / r.width)
        maxh = PH - MARGIN_BOT - MARGIN_TOP - 110
        if h > maxh:
            h = maxh
            w = h * (r.width / r.height)
        if self.y + h + 5 * CAP_LEAD > PH - MARGIN_BOT:
            self.next_col()
            x0, x1 = self.col_rect()
        self.page.show_pdf_page(pymupdf.Rect(x0, self.y, x0 + w, self.y + h), src, 0)
        self.y += h + 6
        self.para(f'{label}. {title}', font=ARB, size=CAP + .5,
                  lead=CAP_LEAD + .6, gap=1.6)
        self.para(caption, font=SERIF, size=CAP, lead=CAP_LEAD, gap=6.0)

    def _figure_span(self, label, src, title, caption):
        self.new_page()
        self._place_span(label, src, title, caption)

    def _place_span(self, label, src, title, caption):
        r = src[0].rect
        full = PW - 2 * MARGIN_X
        w = full
        h = w * (r.height / r.width)
        maxh = PH * 0.52
        if h > maxh:
            h = maxh
            w = h * (r.width / r.height)
        x = MARGIN_X + (full - w) / 2
        self.page.show_pdf_page(pymupdf.Rect(x, self.y, x + w, self.y + h), src, 0)
        self.y += h + 8
        self.full = True
        self.para(f'{label}. {title}', font=ARB, size=CAP + .5,
                  lead=CAP_LEAD + .6, gap=1.6)
        self.para(caption, font=SERIF, size=CAP, lead=CAP_LEAD, gap=8.0)
        self.full = False
        self.col_top = self.y          # text resumes beneath the figure
        self.col = 0

    def _supp_figure(self, label, src, title, caption):
        """A supplementary figure: on its own page, with its caption beneath it
        when it fits at full width in the usual 52% of the page; otherwise the
        figure takes the whole page and its caption starts the next one."""
        r = src[0].rect
        full = PW - 2 * MARGIN_X
        h_at_full = full * (r.height / r.width)
        if h_at_full <= PH * 0.52:
            self._figure_span(label, src, title, caption)
            return
        self.new_page()
        avail = PH - MARGIN_BOT - self.y
        w, h = full, h_at_full
        if h > avail:
            h = avail
            w = h * (r.width / r.height)
        x = MARGIN_X + (full - w) / 2
        self.page.show_pdf_page(pymupdf.Rect(x, self.y, x + w, self.y + h), src, 0)
        self.new_page()
        self.full = True
        self.para(f'{label}. {title}', font=ARB, size=CAP + .5,
                  lead=CAP_LEAD + .6, gap=1.6)
        self.para(caption, font=SERIF, size=CAP, lead=CAP_LEAD, gap=8.0)
        self.full = False
        self.col_top = self.y
        self.col = 0


def parse_methods(path):
    """STAR Methods from PAPER.md, with the editorial apparatus stripped.

    MANUSCRIPT.md deliberately carries no Methods -- it points at PAPER.md, which
    holds them alongside correction notes, warnings and TODOs in blockquotes.
    Those are working-draft matter, not manuscript text, so every '>' line is
    dropped here rather than typeset into the article.

    FILENAMES ARE STRIPPED TOO. PAPER.md names the script behind each analysis,
    which is exactly what it is for -- it is the record of why the analyses are
    what they are. A manuscript does not cite its own source files, so every
    reference to a .py or .ipynb is removed on the way into the document, along
    with the parentheses or backticks left holding it. PAPER.md keeps them.
    """
    def _strip_files(t):
        """Drop source-file references on the way into the manuscript.

        PAPER.md records which script produced each analysis as an
        "Implementation: ..." sentence. That belongs to the working draft; a
        manuscript does not cite its own source files. The leading sentence is
        removed and anything substantive after it is kept, because some of
        those paragraphs continue into real methods text.
        """
        if re.match(r'\*{0,2}Implementation[.:]', t):
            parts = re.split(r'(?<=\.)\s+(?=[A-Z])', t, maxsplit=1)
            t = parts[1] if len(parts) > 1 else ''
        t = re.sub(r'\s*\((?:see\s+)?`?[\w./-]+\.(?:py|ipynb)`?\s*\)', '', t)
        t = re.sub(r'`?\b[\w./-]*[\w-]\.(?:py|ipynb)`?', '', t)
        t = re.sub(r'\(\s*[,;]?\s*\)', '', t)
        t = re.sub(r'\s{2,}', ' ', t)
        t = re.sub(r'\s+([,.;:])', r'\1', t)
        t = re.sub(r'(?:^|\s)(?:in|with|using|via|from)\s*(?=[,.;:])', '', t)
        t = re.sub(r',\s*(?=[,.;:])', '', t)          # the comma the file left behind
        t = re.sub(r'\s{2,}', ' ', t)
        return t.strip(' ,;')

    if not os.path.exists(path):
        return []
    src = open(path).read()
    try:
        body = src[src.index('## Methods'):src.index('## Open items')]
    except ValueError:
        return []
    out, buf = [], []

    def flush():
        if buf:
            out.append(('p', _strip_files(' '.join(' '.join(buf).split()))))
            buf.clear()

    for raw in body.split('\n'):
        t = raw.rstrip()
        if t.strip().startswith('>') or t.strip().startswith('|'):
            continue                      # editorial notes and tables
        if t.startswith('## '):
            flush(); out.append(('h1', 'STAR METHODS')); continue
        if t.startswith('### '):
            flush(); out.append(('h2', t[4:].strip())); continue
        if not t.strip():
            flush(); continue
        buf.append(t.strip())
    flush()
    # drop the placeholder section and anything left empty
    return [(k, v) for k, v in out
            if v and not v.startswith('[MISSING]') and '[MISSING]' not in v[:12]]


def parse(md):
    """MANUSCRIPT.md -> ordered (kind, text) blocks, editorial matter dropped."""
    body = md.split('## Editorial notes')[0]
    out = []
    for raw in body.split('\n\n'):
        t = raw.strip()
        if not t or t.startswith('*Manuscript draft') or t == '---':
            continue
        if t.startswith('# '):
            continue
        if t.startswith('## '):
            out.append(('h1', t[3:].strip()))
        elif t.startswith('### '):
            out.append(('h2', t[4:].strip()))
        else:
            out.append(('p', ' '.join(t.split('\n'))))
    return out


def main():
    md = open(SRC).read()
    blocks = parse(md)
    doc = Doc()
    p = doc.page

    # the Summary is the first body paragraph of MANUSCRIPT.md; it is set on the
    # opener page and must not be repeated when the two-column body runs
    summary = next(t for k, t in blocks if k == 'p')

    # ---- COVER PAGE (Cell Press front matter) ------------------------------
    full = PW - 2 * MARGIN_X
    doc.y = MARGIN_TOP
    p.insert_text(pymupdf.Point(MARGIN_X, doc.y + 20), JOURNAL,
                  fontname=AVB, fontsize=26, color=CELL_BLUE)
    p.insert_text(pymupdf.Point(PW - MARGIN_X - 42, doc.y + 6), 'Article',
                  fontname=AVD, fontsize=12, color=GREY)
    doc.y += 44
    doc.para(TITLE, font=AVD, size=15.0, lead=18.0, gap=14, width=full)

    # two columns: graphical abstract on the left, author matter on the right
    gx0, gx1 = MARGIN_X, MARGIN_X + full * 0.58
    rx0 = MARGIN_X + full * 0.64
    top = doc.y

    doc.para('Graphical abstract', font=AVD, size=9.5, lead=12, gap=4,
             color=CELL_BLUE, width=gx1 - gx0)
    ga = f'{FIG}/graphical_abstract.pdf'
    ga_bottom = doc.y
    if os.path.exists(ga):
        src = pymupdf.open(ga)
        r = src[0].rect
        gw = gx1 - gx0
        gh = gw * (r.height / r.width)
        maxh = 352
        if gh > maxh:
            gh = maxh; gw = gh * (r.width / r.height)
        p.draw_rect(pymupdf.Rect(gx0, doc.y, gx0 + gw, doc.y + gh),
                    color=(0, 0, 0), width=.8)
        p.show_pdf_page(pymupdf.Rect(gx0 + 3, doc.y + 3, gx0 + gw - 3,
                                     doc.y + gh - 3), src, 0)
        ga_bottom = doc.y + gh
        src.close()

    # right column
    doc.y = top
    rw = MARGIN_X + full - rx0
    saved_col = doc.col

    def rpara(txt, font=SERIF, size=8.0, lead=10.0, gap=3.0, color=(0, 0, 0)):
        for ln in doc.wrap(clean(txt), font, size, rw):
            p.insert_text(pymupdf.Point(rx0, doc.y), join_line(ln), fontname=font,
                          fontsize=size, color=color)
            doc.y += lead
        doc.y += gap

    rpara('Authors', font=AVD, size=9.5, lead=12, gap=3, color=CELL_BLUE)
    rpara(AUTHORS.replace('^', '').replace(',*', ''), size=8.4, lead=10.6, gap=9)
    rpara('Correspondence', font=AVD, size=9.5, lead=12, gap=3, color=CELL_BLUE)
    rpara('mattnolanSed.ac.uk', size=8.4, lead=10.6, gap=9, color=CELL_BLUE)
    rpara('In brief', font=AVD, size=9.5, lead=12, gap=3, color=CELL_BLUE)
    rpara(IN_BRIEF, size=8.0, lead=10.2, gap=4)

    # highlights, under the graphical abstract
    doc.y = max(ga_bottom, doc.y) + 16
    doc.para('Highlights', font=AVD, size=9.5, lead=12, gap=5,
             color=CELL_BLUE, width=full)
    for h in HIGHLIGHTS:
        by = doc.y
        p.draw_circle(pymupdf.Point(MARGIN_X + 3, by - 3), 2.1,
                      color=None, fill=CELL_BLUE)
        for ln in doc.wrap(clean(h), SERIF, 8.4, full - 16):
            p.insert_text(pymupdf.Point(MARGIN_X + 14, doc.y), join_line(ln),
                          fontname=SERIF, fontsize=8.4)
            doc.y += 10.6
        doc.y += 4

    # footer
    fy = PH - MARGIN_BOT - 4
    for i, line in enumerate(reversed(CITATION)):
        p.insert_text(pymupdf.Point(MARGIN_X, fy - i * 10), line,
                      fontname=SERIF, fontsize=7.4, color=(.25, .25, .25))
    p.insert_text(pymupdf.Point(PW - MARGIN_X - 62, fy), 'CellPress',
                  fontname=AVD, fontsize=11, color=CELL_BLUE)

    # ---- PAGE 2: article opener --------------------------------------------
    doc.new_page()
    p = doc.page
    doc.full = True
    doc.y = MARGIN_TOP
    p.insert_text(pymupdf.Point(MARGIN_X, doc.y + 14), JOURNAL,
                  fontname=AVB, fontsize=18, color=CELL_BLUE)
    p.insert_text(pymupdf.Point(PW - MARGIN_X - 86, doc.y + 4), 'CellPress',
                  fontname=AVD, fontsize=11, color=CELL_BLUE)
    p.insert_text(pymupdf.Point(PW - MARGIN_X - 86, doc.y + 15), 'OPEN ACCESS',
                  fontname=AVR, fontsize=6.6, color=GREY)
    doc.y += 34
    doc.para('Article', font=AVD, size=11, lead=13, gap=5, color=GREY, width=full)
    doc.para(TITLE, font=AVD, size=16.0, lead=19.0, gap=9, width=full)
    doc.para(AUTHORS.replace('^', ''), font=SERIF, size=8.6, lead=10.8,
             gap=2, width=full)
    for a in AFFIL:
        doc.para(a.replace('^', ''), font=SERIF, size=7.0, lead=8.4, gap=0.5,
                 width=full, color=(.3, .3, .3))
    doc.y += 10
    doc.para('SUMMARY', font=AVD, size=9.0, lead=11, gap=4, color=RULE, width=full)
    doc.para(summary, font=SERIF, size=8.3, lead=10.6, gap=8, width=full,
             color=CELL_BLUE)
    p.draw_line(pymupdf.Point(MARGIN_X, doc.y), pymupdf.Point(PW - MARGIN_X, doc.y),
                color=(.75, .75, .75), width=.5)
    doc.y += 12
    top_of_cols = doc.y

    # ---- body, two columns -------------------------------------------------
    doc.full = False
    doc.col_top = top_of_cols      # page 1 only; new_page() resets to MARGIN_TOP
    doc.col = 0
    doc.y = top_of_cols
    seen_summary = False
    figs = list(FIGURES)
    # Figures otherwise drip one per section heading, in order. That is right for
    # the Results but wrong for the speculative model figure, which argues a
    # hypothesis and has to sit with the section that states it rather than
    # landing mid-Results wherever the drip reaches. PINNED holds a figure back
    # until its heading appears; if that heading never appears it falls through
    # to the end-of-document placement below rather than being dropped.
    PINNED = {'Figure 10': 'ideas and speculation'}
    for kind, text in blocks:
        if kind == 'p' and not seen_summary and text == summary:
            seen_summary = True
            continue
        if kind == 'h1':
            if text.lower().startswith('summary'):
                continue
            doc.heading(text.upper(), size=H1, font=SANS_B, color=RULE, pre=9)
        elif kind == 'h2':
            doc.heading(text, size=H2, font=SANS_B, pre=7)
        else:
            doc.para(text)
        # drip the figures through the Results so they are not all at the end
        if kind == 'h2' and figs:
            for _i, (lab, path, ttl, cap) in enumerate(figs):
                want = PINNED.get(lab)
                if want and want not in text.lower():
                    continue        # held for its own heading; try the next one
                src = f'{FIG}/{path}'
                if os.path.exists(src):
                    doc.pending.append((lab, src, ttl, cap))
                    figs.pop(_i)
                break

    # anything still queued or unplaced, set now
    for lab, path, ttl, cap in figs:
        src = f'{FIG}/{path}'
        if os.path.exists(src):
            doc.pending.append((lab, src, ttl, cap))
    while doc.pending:
        lab, src, ttl, cap = doc.pending.pop(0)
        d_ = pymupdf.open(src)
        doc._figure_span(lab, d_, ttl, cap)
        d_.close()

    # ---- STAR Methods ------------------------------------------------------
    meth = parse_methods(f'{FIG}/PAPER.md')
    for kind, text in meth:
        if kind == 'h1':
            doc.heading(text, size=H1, font=SANS_B, color=RULE, pre=10)
        elif kind == 'h2':
            doc.heading(text, size=H2 - .4, font=SANS_B, pre=6)
        else:
            doc.para(text, size=BODY - .3, lead=BODY_LEAD - .5)
    print(f'  STAR Methods: {sum(1 for k, _ in meth if k == "h2")} subsections')

    # ---- AI declaration ----------------------------------------------------
    doc.heading('DECLARATION OF GENERATIVE AI AND AI-ASSISTED TECHNOLOGIES',
                size=H1 - 1.5, font=SANS_B, color=RULE, pre=10)
    doc.para("During the preparation of this work the authors used Claude (Anthropic) to write and debug analysis code, to compute the statistics reported here from the authors' own recordings, to generate the figures, and to draft and revise manuscript text. The scientific questions, experimental design, data collection and analysis decisions were the authors'. Every reported result was produced by code that the authors reviewed, and the authors reviewed and edited all generated text. The authors take full responsibility for the content of this publication.")

    # ---- references --------------------------------------------------------
    doc.heading('REFERENCES', size=H1, font=SANS_B, color=RULE, pre=10)
    cited = check_citations([t for _, t in blocks] + [t for _, t in meth] +
                            [' '.join(map(str, f)) for f in FIGURES + SUPPLEMENTS])
    for k in sorted(cited, key=lambda k: REFS[k].lower()):
        doc.para(REFS[k], size=7.4, lead=9.2, gap=2.0, indent=6)

    # ---- supplemental figures ----------------------------------------------
    doc.heading('SUPPLEMENTAL FIGURES', size=H1, font=SANS_B, color=RULE, pre=10)
    doc.para('Supplemental figures are referenced from the Results and are '
             'presented here in the order their main figure appears.',
             font=SERIF_I, size=7.4, lead=9.2, gap=4)
    n_supp = 0
    for lab, path, ttl, cap in SUPPLEMENTS:
        src = f'{FIG}/{path}'
        if not os.path.exists(src):
            print(f'  ! {path} missing, skipped')
            continue
        doc.pending.append((lab, src, ttl, cap))
        n_supp += 1
    while doc.pending:
        lab, src, ttl, cap = doc.pending.pop(0)
        d_ = pymupdf.open(src)
        doc._supp_figure(lab, d_, ttl, cap)
        d_.close()
    print(f'  supplemental figures: {n_supp}')

    doc.d.save(OUT, deflate=True)
    print(f'wrote {OUT}  ({doc.d.page_count} pages)')
    import shrink_pdf                     # under 25 MB, and a smaller copy
    shrink_pdf.fit(OUT)


if __name__ == '__main__':
    main()
