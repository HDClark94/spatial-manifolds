"""Compile MANUSCRIPT.md + figures + captions into a Cell-style PDF mock-up.

No LaTeX, pandoc or HTML-to-PDF engine is available on this machine, so the
document is composed directly with PyMuPDF: a small line-breaking layout engine
over the base-14 fonts, with the existing figure PDFs embedded as vector content
via show_pdf_page (so figures stay vector, not rasterised).

Cell Press conventions approximated here: running title block, author line and
affiliations, Summary, Highlights, two-column body, figures placed in-flow with
captions set beneath them in small type, and STAR Methods as a trailing section.

WHAT IS AND IS NOT REAL. The text and every number come from MANUSCRIPT.md. The
reference list does NOT come from the manuscript, because the manuscript has
none -- the working draft carries unresolved superscript placeholders and no
bibliography (see the open items in PAPER.md). References here are assembled
from works NAMED in the text, resolved against the published record. One citation
that could not be verified is listed as outstanding rather than invented.

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

TITLE = ('A global engagement state, read out by entorhinal grid cells '
         'and local inhibition rather than built by them')
AUTHORS = ('Harry Clark,^1 Wolf de Wulfe,^1 Chris Halcrow,^1 '
           'and Matthew F. Nolan^1,2,*')
AFFIL = ['^1Centre for Discovery Brain Sciences, University of Edinburgh, Edinburgh, UK',
         '^2Simons Initiative for the Developing Brain, University of Edinburgh, Edinburgh, UK',
         '*Correspondence: mattnolan@ed.ac.uk']

JOURNAL = 'Cell'
CELL_BLUE = (0.00, 0.42, 0.71)
GREY = (0.58, 0.58, 0.58)
IN_BRIEF = (
    'Clark et al. show that the anchoring of entorhinal spatial firing to a '
    'task reference frame is a population state rather than a property of single '
    'cells. Grid cells and fast-spiking interneurons express it, but their '
    'synaptic connections do not carry it, and the same state is present in every '
    'structure recorded. The state tracks attentional engagement and predicts '
    'success when the animal must navigate by path integration.')
CITATION = [
    'Clark et al., 2026, manuscript draft',
    'Style mock-up compiled from MANUSCRIPT.md - not a published article',
]

HIGHLIGHTS = [
    'Anchoring of entorhinal firing to a task reference frame is a population state',
    'Grid cells and fast-spiking interneurons follow it; spatial coding in general does not',
    'Synaptically connected pairs share the state no more than matched unconnected pairs',
    'The same state is present in every structure recorded, including cerebellum',
]

# Figure order, source file, and the caption shown beneath it.
FIGURES = [
    ('Figure 1', 'fig1_composite.pdf',
     'A population anchoring state in medial entorhinal cortex',
     '(A) The location memory task: head-fixed mice run on a treadmill through a virtual '
     'linear track, stopping in a reward zone that is visually cued on some trials and '
     'uncued on others, so that the zone can only be localised efficiently by path '
     'integration when the cue is absent. (B, C) Stopping on cued and uncued trials for one '
     'session. (D) Three example units as trial x position rate maps. (E) The MEC population '
     'anchoring state for the same session with its first principal component. (F) The '
     'fraction of cells locked to the state across sessions. (G, H) Behaviour by state and '
     'the Anchoring Dependence Index. Recording configuration and the population anchoring state. Neurons were recorded with '
     'four-shank Neuropixels 2.0 probes spanning the mediolateral extent of MEC, in an open arena '
     'and then during a virtual-track location memory task. Trial-by-trial anchoring labels are '
     'correlated across simultaneously recorded cells; the first principal component of the '
     'cell x trial label matrix exceeds a circular-shift null preserving each cell\'s anchored '
     'fraction by roughly twofold. Blocks of tens of trials share a state and the population '
     'crosses between states over about three trials. Anchored trials are more often successful, '
     'and markedly so on uncued trials (Anchoring Dependence Index +0.094, p = 0.012).'),
    ('Figure 2', 'fig2_single_units.pdf',
     'Grid cells and putative interneurons follow the population state most strongly',
     '(A, B) Two example sessions, each as the full MEC anchoring raster with PC1 of the cell x '
     'trial label matrix and example single cells of each identity, including cells locked into '
     'one mode. (C) The null for one cell: its labels circularly shifted 200 times against the '
     'population axis, with the observed value marked. Chance lies near +0.15, not at zero, '
     'because a cell anchored on most trials correlates with a population that is also anchored on '
     'most trials. (D) Agreement with the population axis by identity, as excess of |r| over each '
     'cell\'s own null. Every class exceeds chance, and they differ in degree: grid cells +0.185 '
     '(p = 1 x 10^-11, 7 of 7 mice) and putative interneurons +0.131 (p = 3 x 10^-16, 7 of 7), '
     'against non-grid spatial +0.081 (p = 6 x 10^-22, 8 of 8) and non-spatial +0.056 '
     '(p = 4 x 10^-16, 7 of 7). Grid and interneuron do not differ from one another (p = 0.54) '
     'and both exceed the other two (p <= 8 x 10^-6, Holm-corrected), while non-grid spatial '
     'exceeds non-spatial (p = 2 x 10^-6). (E) The fraction of cells individually beating their '
     'own null, per mouse, which orders the classes the same way -- 43% of grid cells and 37% of '
     'interneurons against 24% and 19% -- all far above the 5% false-positive line. (F) The same '
     'question '
     'in time: all four identities switch together over roughly three trials around a population '
     'transition, and identity does not predict switch timing (p = 0.15). (G) Agreement between '
     'identities as a 4 x 4 matrix over its own null; grid-interneuron reaches +0.109, '
     'indistinguishable from either within-group value. (H) Which identities never leave one mode: '
     'interneurons lock on (108:2) while non-spatial cells lock off (39:293). (I) The same '
     'asymmetry per mouse, so it is not carried by one animal. (J) Why they lock: cells that '
     'never vary are the speed-coding cells. Interneurons '
     'that lock on score +0.141 on the open-field speed measure against +0.065 for those that vary '
     '(p = 0.005), because a speed-tuned profile repeats on every trial and is therefore called '
     'anchored on every trial.'),
    ('Figure 3', 'fig3_anatomy.pdf',
     'The responsive subnetwork is medial, within the superficial grid population',
     'Anatomical organisation of identity and of following. (A) Open-field rate maps from four '
     'M25 sessions whose probes sat at different mediolateral positions, sampled proportionally '
     'so the grid:non-grid ratio drawn is the ratio actually recorded; the key at left gives the '
     'recording sequence, and every map comes from OF1, the arena session preceding the task. '
     '(B-D) Best-fit slices through three mice, each under a view of the probe position, with '
     'cells coloured by open-field identity over the Allen region annotation. (E-G) Identity '
     'against mediolateral and dorsoventral position and across layers: grid cells are enriched '
     'medially (median rho = -0.078, p = 0.025) and superficially (8.9% against 3.9%, p = 0.002). '
     '(H-J) The same axes against whether a cell follows the population state; the mediolateral '
     'gradient holds within session on the 27 sessions spanning at least two shank pitches '
     '(median rho = -0.109, p = 0.034), while dorsoventral position and probe depth predict '
     'neither. Layer predicts identity but not following within session (median rho = +0.028, '
     'p = 0.154), so the superficial bias in following is the one carried by identity rather '
     'than a gradient in following of its own.'),
    ('Figure 4', 'fig4_monosynaptic.pdf',
     'Monosynaptic connectivity does not account for shared anchoring',
     '(A) The interneuron call: waveform peak-to-trough duration against firing rate, with the '
     'marginal distribution and the fixed 0.4 ms boundary. (B) What a classified interneuron does '
     'to its target - a causal trough in the cross-correlogram. (C) What a broad-spiking cell does '
     '- a causal peak, on the same scale. (D) The validation: inhibitory connections originate '
     'from the cells in panel A at 3.1 times their population share, which the detector cannot '
     'produce by construction. (E, F) Both connection matrices as a percentage of the ordered '
     'pairs each identity contributes. Interneuron-to-grid is the strongest pathway in the dataset '
     '(3.93%, 5.1x over identity shuffling, p = 0.0005) and grid-to-grid the strongest excitatory '
     'one (0.217%, 2.6x, p = 0.004). (G) The two connection types are kinetically distinct, '
     'inhibition being slower (median 3.0 against 1.0 ms). (H) Detection validated against a '
     'jitter null for both signs. (I) The question the connectivity raises, and the null on which '
     'the paper turns: connected pairs share the anchoring state no more than unconnected pairs '
     'matched for session, identity pair and probe distance (grid-interneuron p = 0.39; '
     'interneuron-interneuron p = 0.90). Both comparisons resolve effects a third the size of the '
     'identity effect, so these are informative nulls; grid-grid is underpowered and is greyed.'),
    ('Figure 5', 'fig5_pupil_arousal_M21D19.pdf',
     'The state tracks attentional engagement',
     '(A) The recording arrangement, with infrared illumination and the side camera. (B) Pupil '
     'tracking: the eye with the DeepLabCut perimeter landmarks from which radius is computed. (C) '
     'One anchored and one non-anchored trial shown as seven moments of a single traversal, with '
     'pupil radius printed beneath each frame. (D) Mean pupil radius against track position for '
     'the two state blocks of this session, with the two example trials overlaid. (E) The MEC '
     'population anchoring raster for the same session, cells that vary ordered by PC1 loading and '
     'cells locked into one mode separated to the right of a gap. (F) PC1 of the same label '
     'matrix. (G) Pupil radius for every trial on the trial axis of E and F, so a state block can '
     'be read across. (H) Pupil radius by state, paired within session across 23 sessions (delta = '
     '-0.670, p = 2e-06). (I) The result that identifies this as a change in engagement rather '
     'than attention paid at particular places: the difference is present across the whole track, '
     'with the same sign in all 50 position bins and significance in 48, no trend with position '
     '(rho = -0.264, p = 0.064), and the reward zone if anything slightly less affected than the '
     'rest.'),
    ('Figure 6', 'fig2_supp_vis_anchored.pdf',
     'Visual cortex holds the task reference frame through entorhinal switches',
     'Figure 2K shows that the lock-on bias among MEC cells is carried entirely by '
     'speed-modulated cells. Repeating that count in every recorded structure turns up one '
     'exception: in visual cortex, cells with no open-field speed tuning lock into the anchored '
     'mode 98 against 13 -- an odds ratio of 9.9 against the entorhinal cells, p = 4 x 10^-19 -- and '
     'the excess is present in every one of the 15 sessions contributing such cells. (A, B) One '
     'session '
     'per row. The probe map at the left is the session\'s best-fit slice through the atlas, as '
     'in Figure S7, with every contact in pale grey and the recorded cells coloured by '
     'structure, so the rasters beside it can be read against where their cells actually sit. '
     'Then the entorhinal anchoring raster, the session PC1, and the visual raster. Both '
     'rasters are built the same way -- cells that vary in PC1-loading order, then the cells '
     'whose label never changes to the right of a gap -- and both are cut by the same dashed '
     'lines, which are the major transitions of the one session state, so the comparison is '
     'between two descriptions of the same trials rather than between two separately defined '
     'states. The entorhinal population reorganises at those lines; the visual locked block, 78 '
     'of 150 cells in A, does not. Each raster is followed by three of its own cells on the '
     'same trial axis, with the peak of its per-state mean rates in the title. The two '
     'selections are opposites, which is the '
     'point of showing them together: the entorhinal cells are the three largest PC1 loadings '
     'among the cells that vary, that is the left edge of the raster beside them, and their '
     'fields appear in one state and not the other; the visual cells are picked for never '
     'switching, and theirs are the same field in both. '
     '(C, D) Every locked-anchored cell of each structure, clustered on the SHAPE of its '
     'position profile rather than sorted by peak, since sorting imposes a diagonal whether or '
     'not the population has structure. Correlation distance, average linkage, cut into four; '
     'the dendrogram is coloured by cluster below the cut and grey above it, and each cluster '
     'mean is drawn with its SEM on its own axes so that a flat cluster reads as flat. No speed '
     'criterion is applied, because the speed relationship is measured in E rather than selected '
     'on. The two structures are concentrated at the same places on the track -- peak positions '
     'in both are far from uniform (KS p = 2 x 10^-13 and 5 x 10^-13) and fall mostly in the '
     'first 50 cm and the last 25 cm -- so the positional concentration is a property of '
     'persistently anchored cells here rather than something peculiar to visual cortex. '
     '(E) The two confounds that the task builds in, tested against a matched null. Running '
     'speed and the luminance proxy are both smooth functions of position, and any smooth '
     'function of position resembles a smooth firing profile, so a correlation against zero is '
     'not a test. Each group\'s median correlation with each regressor is compared with 500 '
     'phase-randomised surrogates of that regressor -- identical power spectrum, so identical '
     'smoothness, relationship to position destroyed -- and the grey band is the 95th percentile '
     'of what those surrogates reach. Each structure is split by open-field speed tuning here, '
     'where the split is the measurement, rather than in the selection. Entorhinal cells follow '
     'the track\'s speed profile whether or not they are speed-modulated in the open field '
     '(+0.12, p = 0.016 and +0.22, p = 0.028) and follow the light in neither case. Visual cells '
     'are the opposite, and only in one half: the cells without open-field speed tuning follow '
     'the light (+0.34, p = 0.010) and not the speed, while the speed-modulated visual cells '
     'follow neither. Pooling the two halves of visual cortex was tried and abandoned for that '
     'reason -- it reads +0.11, p = 0.21 -- and the same subset carries the beacon result: these '
     'cells\' reward-zone firing differs between cued and uncued trials at p = 0.025 in the 98 '
     'cells without speed tuning and p = 0.063 across all 308. A variance-explained version of '
     'this panel was built first and dropped: cross-validated R^2 clears this null for no '
     'regressor in any group (best p = 0.075), so the figure can say whether each relationship '
     'is present but not how much of the profile it accounts for. (F) Where the position '
     'structure of these cells sits. Variance of the profile against position, normalised to its '
     'track average, with the 30 cm black box at each end shaded and the reward zone marked. '
     'Nearly half of it falls inside the black boxes, which cover 30% of the track (48% visual, '
     '45% entorhinal), and the reward zone is the one place it is scarce: these cells are '
     'organised around where each traversal begins and ends rather than around the goal. (G) The '
     'counts by structure, among cells with no open-field speed tuning, which is the form in '
     'which Figure 2K left the question. The restriction is not what produces the contrast: '
     'across all locked cells it is 308 against 40 in visual cortex and 514 against 488 in '
     'entorhinal, 7.7:1 against 1.1:1. Together these indicate that the persistently anchored '
     'visual cells follow features of the scene -- its luminance structure and the beacon -- '
     'which repeat on every lap, rather than carrying an anchoring signal of their own. That the '
     'same reference frame remains legible in visual cortex while the entorhinal population '
     'leaves it is the point of interest: the information is available and is not being used by '
     'MEC in the non-anchored epochs.'),
    ('Figure 7', 'fig7_theta_gamma.pdf',
     'A frequency-specific change in the entorhinal field potential',
     '(A-D) One session trial by trial, with both states shaded behind every panel and dashed '
     'lines at major transitions. (A) The population axis with running speed overlaid. (B) The '
     'spectrogram, 1.9-150 Hz rebinned into 56 log-spaced bands and smoothed along the trial axis '
     'only. (C) The three band traces. (D) Theta-gamma modulation index for both bands on twinned '
     'axes, fast on the left axis and slow on the right, which they need because slow-gamma MI is '
     'roughly five times smaller. (E) The two states as spectra across 26 switching sessions. (F) '
     'The panel that excludes a gain change: theta (-0.094) and mid gamma (-0.108) fall while slow '
     'gamma rises (+0.105), so the difference reverses sign. (G) Theta-gamma coupling by band; '
     'fast gamma uncouples (-15.9%, p = 5e-6) while slow gamma does not. (H) Per-session slow '
     'against fast gamma change: the two are uncorrelated (rho = -0.08) and the quadrant count is '
     'what the marginals imply, so this is not a demonstrated redistribution. (I) The difference '
     'index per session. (J) The pooled difference profile, with the SEM across sessions, crossing '
     'zero near 67 Hz in 25 of 26 sessions. Where in the theta cycle each frequency peaks, '
     'and how that moves with the state, is shown in Figure S10.'),
    ('Figure 8', 'fig8_model.pdf',
     'A two-loop account of the anchoring state (hypothesis)',
     'A proposed mechanism, placed with Ideas and speculation rather than with the Results. '
     'Panels A and E are schematics; C, D and one feature of B carry measured values '
     'and are included so that the schematics cannot be read as evidence. (A) The circuit motif '
     'and what is proposed to set it: grid cells and fast-spiking interneurons form a recurrent '
     'loop, sensory afferents drive a feedforward one through the same interneurons, and '
     'cholinergic tone sets the balance between them. The loop is the one measured in Figure 4, '
     'where interneuron-to-grid is the strongest pathway in the dataset and the inhibitory limb '
     'the slower of the two (3.0 against 1.0 ms). (B) Why two bands rather than one rhythm of '
     'varying amplitude: gamma is generated by the inhibitory limb, so loop kinetics set '
     'frequency, and each band is nested in theta at its own phase. The offset between those '
     'phases is measured, not drawn: across 44 sessions slow gamma peaks at +94 degrees of theta '
     'and fast gamma at +48, a within-session difference of +45 degrees (resultant length 0.87, '
     'Rayleigh p = 5e-15). An earlier version of this panel drew the bands in antiphase, which '
     'only 1 of 44 sessions approaches. The states are proposed to differ in which loop dominates. (C) The field-potential changes that motivate the account '
     '(Figure 7), with slow gamma rising while fast gamma, theta and theta-fast coupling fall. '
     '(D) The cells that express the state (Figure 2) are the two populations that constitute '
     'the loop, and no others. (E) The test that would settle the weakest step. The association '
     'of slow and fast gamma with internally and externally guided processing is taken from '
     'hippocampal recordings, where fast gamma indexes entorhinal input to CA1, and cannot be '
     'carried into entorhinal cortex unexamined; in sessions recording entorhinal cortex, the '
     'subicular complex and visual cortex together, the two bands should differ in which '
     'structure they cohere with. This figure makes no claim about where the grid phase sits on '
     'a non-anchored trial: non-anchored trials show no coherent field at any offset once the '
     'unrelated-cell floor is applied, so phase-level versions of this account are excluded.'),
]

# Supplemental figures, in the order their main figure appears.
SUPPLEMENTS = [
    ('Figure S1', 'fig1_supp_classifier.pdf',
     'The anchoring classifier, step by step, against RNN ground truth',
     'An RNN trained on a 1D slice of a 2D environment with a teleport reproduces the task, '
     'giving trial labels that are known rather than inferred. The classification is shown as '
     'the steps it actually takes, from the NaN-free rate map through the trial-by-trial '
     'correlation and the per-cell null gate to the median-filtered label sequence, with '
     'accuracy and recall reported for each anchoring regime with and without the filter. The '
     'gate changes performance mainly where trials are locked in one mode or split far from '
     'even.'),
    ('Figure S2', 'fig1_supp_population_anchoring_examples.pdf',
     'Population anchoring states across sessions',
     'Anchoring rasters for further sessions, ordered by the share of label variance the first '
     'principal component explains, from sessions where nearly all cells move together to '
     'sessions where they do not. Cells on the x axis, trials on the y axis.'),
    ('Figure S3', 'fig1_supp_speed_matching.pdf',
     'Behavioural controls for the anchoring state',
     'Running speed and stopping behaviour compared between states, including the speed-matched '
     'subsets used wherever a comparison could otherwise be carried by a difference in '
     'locomotion.'),
    ('Figure S4', 'fig2_supp_nonanchored_rate.pdf',
     'What a non-anchored trial is, and whether firing rate changes with the state',
     'Two halves of one question. (A-D) Has the field moved, or has it gone? Each trial\'s map is '
     'correlated against the cell\'s anchored template at every circular offset. Zero-offset '
     'correlation collapses from +0.33 to +0.07 while the best-over-offsets correlation falls only '
     'from +0.56 to +0.53, which looks like displacement until the null is applied: a maximum over '
     '100 offsets is large by construction, two unrelated cells reach +0.500 under the same search, '
     'and non-anchored trials sit only 0.029 above that floor with a mean best offset of 47 cm '
     'against the 50 cm a uniform choice on a 200 cm track would give. Non-anchored trials show '
     'loss of spatial correspondence with no evidence of a coherent field elsewhere. (E-K) Does '
     'rate change with the state, or only where cells fire? The effect is small and its sign '
     'depends on the measurement: on the same 6,517 cells the log2 ratio is -0.090 spatially '
     'averaged and unmatched, -0.028 time averaged and unmatched, and +0.036 time averaged and '
     'speed-matched. All three are significant and they disagree about direction, because a rate '
     'read off a rate map divides by occupancy and occupancy is set by running speed. Nothing that '
     'reverses sign under a change of denominator should be reported as a rate effect, and the '
     'conclusion is that anchoring concerns where cells fire rather than how much.'),
    ('Figure S5', 'fig3_supp_laminar.pdf',
     'Recording sites are biased toward superficial layers medially, and what that does to '
     'the mediolateral results',
     'MEC is curved and a four-shank probe is flat, so shanks at different mediolateral '
     'positions do not enter the tissue at the same depth relative to the layers. (A) They do '
     'not: superficial layers are 90.3% of entorhinal recording sites in the most medial '
     'quartile and 59.8% in the most lateral (pooled rho = -0.267, p = 4e-80), a gradient '
     'present in every animal (within-mouse median rho = -0.447, negative in 6 of 6, p = '
     '0.031). (B) The same bias in the neurons actually analysed (within-session median rho = '
     '-0.402, p = 0.016). (C, D) Stratifying the mediolateral tests by layer separates the two '
     'results Figure 3 reports. The identity gradient does not survive: within superficial '
     'cells it roughly halves and loses significance (median rho = -0.051, p = 0.24, 25 '
     'sessions) against -0.078 (p = 0.025) across all layers. The following gradient does '
     'survive, and is if anything stronger within superficial cells (-0.144, p = 0.043, 22 '
     'sessions) than across all layers (-0.109, p = 0.034). Deep layers are too sparsely '
     'sampled to test alone (6-7 sessions), so those columns are underpowered rather than '
     'null. Stratification is not mediation: it shows the mediolateral effect is not carried '
     'by layer differences within a stratum, not that layer plays no part.'),
    ('Figure S6', 'fig6_cross_region.pdf',
     'One state, shared across structures, read out most strongly by one cell type',
     '(A) One session recording all three structures, with the per-trial anchored fraction '
     'computed separately from each. (B) Does each region have a state of its own? Split-half '
     'reliability of each region\'s own first principal component, size-matched to 8 cells per '
     'half: MEC +0.447, subicular +0.474, visual cortex +0.330 (p = 2.4e-5). (C) Is it the same '
     'state? Cross-region agreement divided by the geometric mean of the two reliabilities; '
     'corrected, MEC-subicular +0.920 and MEC-visual +0.863, neither differing from 1. (D) Per- '
     'cell expression of the state by group, with every cell scored out of sample against an axis '
     'built from half the entorhinal population. (E) The panel the figure turns on: grid cells '
     'exceed visual cortex by +0.149 (p = 9.9e-6) and interneurons by +0.144, against +0.050 for '
     'other entorhinal cells, while the region-level difference does not reach significance (p = '
     '0.065). The cell-type contrast is three times the regional one. (F) The Anchoring Dependence '
     'Index by region.'),
    ('Figure S7', 'fig6_region_examples.pdf',
     'The same state, described separately by each structure',
     'Two sessions bracketing the range among those recording all three structures, read left to '
     'right as entorhinal, subicular and visual cortex. Everything within a structure is built '
     'from that structure\'s own cells - the raster ordered by its own PC1 loading, the '
     'component signed to its own anchored fraction - so agreement between the groups is '
     'agreement between independent descriptions rather than a consequence of a shared frame.'),
    ('Figure S8', 'fig6_supp_region_examples.pdf',
     'The state outside MEC, where MEC does not dominate the sample',
     'Sessions chosen for their sampling rather than their result: visual-rich and '
     'subicular-rich sessions, and the two with enough cerebellar cells to build an axis. '
     'Cerebellum carries a matching state in both, having been included as a distant negative '
     'control that it turned out not to be. One visual session runs the other way, and is shown '
     'for that reason.'),
    ('Figure S9', 'fig6_supp_dimensionality.pdf',
     'One axis is an adequate description of the state',
     'Out-of-sample variance explained, with the axis built from one half of a region\'s cells '
     'and scored on the other. (A) Neuron-dropping curve per region against the circular-shift '
     'null, showing the cell-count dependence rather than hiding it behind one matched number. '
     '(B) By component at matched cell count: the first carries roughly fifty times the '
     'generalising structure of the second, so the label matrix is effectively one-dimensional.'),
    ('Figure S10', 'fig7_supp_band_examples.pdf',
     'Band power tracking the population state in individual sessions',
     'Four sessions as the population axis, the log-spaced spectrogram and the three band traces '
     'on a shared trial axis. A-C show the population pattern; D is drawn in red as a '
     'counterexample, a session with clear transitions whose bands do not follow them, included '
     'because the effect is 16 of 26 sessions rather than all of them.'),
    ('Figure S11', 'fig7_supp_comodulogram.pdf',
     'Where in the theta cycle each frequency peaks, and how little it moves with the state',
     'The band analysis in Figure 7 fixes two gamma bands in advance and asks how strongly each '
     'is nested in theta; the modulation index discards phase, so it cannot ask where in the '
     'cycle they peak. (A, B) The phase-frequency map for each state across 28 sessions. Each '
     'frequency row is normalised to sum to one, so colour is where a frequency peaks rather '
     'than how much power it carries; without that the 1/f background would dominate every row. '
     'Both states\' preferred-phase ridges are drawn on both maps, solid for anchored and dashed '
     'for non-anchored, because the two maps share 94% of their pixel variance and are not '
     'tellable apart side by side; the ridges put the only thing that differs in one place. '
     'The mains band is marked: it does not drive these panels, and is if anything less '
     'modulated than its neighbours (0.0081 against 0.0102), which is what a line uncorrelated '
     'with theta does. (C) Their difference. The state effect is a phase shift rather than a '
     'gain change, so it appears as a dipole -- anchored higher on the earlier flank and lower '
     'on the later one -- strongest where the amplitude itself is, around 50-75 Hz. It is a '
     'small difference on a noisy map, and the panel below reports what it amounts to. (D) The '
     'same information as preferred phase against frequency. The relationship is a continuous '
     'gradient rather than a step between two bands: preferred phase runs from about 130 degrees '
     'at 20-30 Hz to about 41 degrees at 60-100 Hz, and that separation is present within each '
     'state taken alone -- slow minus fast is +34 degrees when anchored (p = 5 x 10^-4, 24 of 28 '
     'sessions) and +52 when not (p = 1 x 10^-5, 26 of 28), paired within session. The STATE, by '
     'contrast, moves it very little and not reliably: within session the anchored state prefers '
     'a phase 7 degrees earlier in the slow band (p = 0.33) and 3 degrees earlier in the fast '
     'band (p = 0.16), and the separation itself is not state-dependent (-7 degrees, p = 0.27), '
     'all Wilcoxon signed-rank over 28 sessions and marked n.s. on the panel. The gradient is '
     'the result here; the state shift visible in the grand averages does not survive pairing by '
     'session, which is the test that respects how these data were collected. (E-G) The same '
     'question asked against depth rather than frequency, for three example sessions: the '
     'theta-triggered average LFP down one shank, triggered on the theta peak at the probe '
     'tip, with its current source density behind it and both states drawn on the same axes '
     '(anchored solid, non-anchored dashed). Colour limits come from the interior of the span, '
     'since theta is several times larger at its ends. (H) The laminar phase gradient over the '
     '27 sessions with a usable profile, referenced to each session\'s own layer 2 because the '
     'probe tip sits in a different lamina in different penetrations. Theta advances about 100 '
     'degrees from layer 1 to layer 6, which is the laminar signature of the entorhinal '
     'travelling wave, and the anchoring state does not move it: every layer differs by under '
     '3 degrees, all p >= 0.06. Taken with D, neither the frequency axis nor the depth axis of '
     'theta-gamma organisation is reorganised by the state, which is a constraint on what the '
     'state can be doing to this circuit rather than a failure to measure it.'),
]

REFS_FULL = [
    'Clark, H., and Nolan, M.F. (2024). Task-anchored grid cell firing is selectively '
    'associated with successful path integration-dependent behaviour. eLife 12, RP89356.',
    'Tennant, S.A., Clark, H., Hawes, I., Tam, W.K., Hua, J., Yang, W., Gerlei, K.Z., '
    'Wood, E.R., and Nolan, M.F. (2022). Spatial representation by ramping activity of '
    'neurons in the retrohippocampal cortex. Curr. Biol. 32, 4451-4464.',
    'Vollan, A.Z., Gardner, R.J., Moser, M.-B., and Moser, E.I. (2025). Left-right-alternating '
    'theta sweeps in entorhinal-hippocampal maps of space. Nature 639, 995-1005.',
    'Robinson, J., Ying, J., Hasselmo, M.E., and Brandon, M.P. (2024). Septal GABAergic '
    'control of grid cell periodicity and phase precession. Cell Rep. 43, 114590.',
    'Aston-Jones, G., and Cohen, J.D. (2005). An integrative theory of locus '
    'coeruleus-norepinephrine function: adaptive gain and optimal performance. Annu. Rev. '
    'Neurosci. 28, 403-450.',
    'Colgin, L.L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., Moser, M.-B., '
    'and Moser, E.I. (2009). Frequency of gamma oscillations routes flow of information in the '
    'hippocampus. Nature 462, 353-357.',
    'Fuhrmann, F., Justus, D., Sosulina, L., Kaneko, H., Beutel, T., Friedrichs, D., Schoch, S., '
    'Schwarz, M.K., Fuhrmann, M., and Remy, S. (2015). Locomotion, theta oscillations, and the '
    'speed-correlated firing of hippocampal neurons are controlled by a medial septal '
    'glutamatergic circuit. Neuron 86, 1253-1264.',
    'Gil, M., Ancau, M., Schlesiger, M.I., Neitz, A., Allen, K., De Marco, R.J., and Monyer, H. '
    '(2018). Impaired path integration in mice with disrupted grid cell firing. Nat. Neurosci. '
    '21, 81-91.',
    'Hallanger, A.E., and Wainer, B.H. (1988). Ascending projections from the pedunculopontine '
    'tegmental nucleus and the adjacent mesopontine tegmentum in the rat. J. Comp. Neurol. 274, '
    '483-515.',
    'Jacob, P.-Y., Gordillo-Salas, M., Facchini, J., Poucet, B., Save, E., and Sargolini, F. '
    '(2019). Path integration maintains spatial periodicity of grid cell firing in a 1D circular '
    'track. Nat. Commun. 10, 840.',
    'Justus, D., Dalugge, D., Bothe, S., Fuhrmann, F., Hannes, C., Kaneko, H., Friedrichs, D., '
    'Sosulina, L., Schwarz, I., Elliott, D.A., et al. (2017). Glutamatergic synaptic integration '
    'of locomotion speed via septoentorhinal projections. Nat. Neurosci. 20, 16-19.',
    'Kempter, R., Leibold, C., Buzsaki, G., Diba, K., and Schmidt, R. (2012). Quantifying '
    'circular-linear associations: hippocampal phase precession. J. Neurosci. Methods 207, '
    '113-124.',
    'Kropff, E., Carmichael, J.E., Moser, M.-B., and Moser, E.I. (2015). Speed cells in the medial '
    'entorhinal cortex. Nature 523, 419-424.',
    'Low, I.I.C., Williams, A.H., Campbell, M.G., Linderman, S.W., and Giocomo, L.M. (2021). '
    'Dynamic and reversible remapping of network representations in an unchanging environment. '
    'Neuron 109, 2967-2980.',
    'Tennant, S.A., Fischer, L., Garden, D.L.F., Gerlei, K.Z., Martinez-Gonzalez, C., McClure, C., '
    'Wood, E.R., and Nolan, M.F. (2018). Stellate cells in the medial entorhinal cortex are '
    'required for spatial learning. Cell Rep. 22, 1313-1324.',
    'Tort, A.B.L., Komorowski, R., Eichenbaum, H., and Kopell, N. (2010). Measuring '
    'phase-amplitude coupling between neuronal oscillations of different frequencies. J. '
    'Neurophysiol. 104, 1195-1210.',
    'Ye, J., Witter, M.P., Moser, M.-B., and Moser, E.I. (2018). Entorhinal fast-spiking speed '
    'cells project to the hippocampus. Proc. Natl. Acad. Sci. USA 115, E1627-E1636.',
    'Reimer, J., McGinley, M.J., Liu, Y., Rodenkirch, C., Wang, Q., McCormick, D.A., and '
    'Tolias, A.S. (2016). Pupil fluctuations track rapid changes in adrenergic and cholinergic '
    'activity in cortex. Nat. Commun. 7, 13289.',
]
REFS_PARTIAL = [
    'Qin, H., et al. (2018) - layer 2 stellate cell inactivation. Cited by name in the source '
    'draft; the full citation could not be verified and is left outstanding rather than '
    'reconstructed from memory.',
    'Target selectivity of septal cholinergic neurons in the medial and lateral entorhinal '
    'cortex (2018), Proc. Natl. Acad. Sci. USA - the source for septal cholinergic projections '
    'being denser to MEC than LEC. Title, journal and year verified; the author list could not '
    'be retrieved and is left outstanding rather than reconstructed.',
]

GREEK = {'ρ': 'rho', 'σ': 'sigma', 'λ': 'lambda', 'α': 'alpha', 'β': 'beta',
         'χ': 'chi', 'Χ': 'chi', 'μ': 'u', 'Δ': 'delta', 'π': 'pi', 'θ': 'theta'}
SUPS = {'⁻': '-', '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
        '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}


def clean(t):
    """Markdown and unicode down to what the base-14 fonts can actually set."""
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)', r'\1', t)
    t = t.replace('`', '')
    # superscript runs become ^-6 style so exponents survive
    t = re.sub(r'10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)',
               lambda m: '10^' + ''.join(SUPS.get(c, c) for c in m.group(1)), t)
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
        words, lines, cur = text.split(), [], ''
        for w in words:
            t = (cur + ' ' + w).strip()
            if text_len(t, font, size) <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def para(self, text, font=SERIF, size=BODY, lead=BODY_LEAD, indent=0.0,
             color=(0, 0, 0), gap=3.0, width=None):
        x0, x1 = self.col_rect()
        w = width if width else (x1 - x0)
        for i, ln in enumerate(self.wrap(clean(text), font, size, w - indent)):
            if self.y + lead > PH - MARGIN_BOT:
                self.next_col()
                x0, x1 = self.col_rect()
                if width is None:
                    w = x1 - x0
            self.page.insert_text(pymupdf.Point(x0 + indent, self.y), ln,
                                  fontname=font, fontsize=size, color=color)
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
            p.insert_text(pymupdf.Point(rx0, doc.y), ln, fontname=font,
                          fontsize=size, color=color)
            doc.y += lead
        doc.y += gap

    rpara('Authors', font=AVD, size=9.5, lead=12, gap=3, color=CELL_BLUE)
    rpara(AUTHORS.replace('^', '').replace(',*', ''), size=8.4, lead=10.6, gap=9)
    rpara('Correspondence', font=AVD, size=9.5, lead=12, gap=3, color=CELL_BLUE)
    rpara('mattnolan@ed.ac.uk', size=8.4, lead=10.6, gap=9, color=CELL_BLUE)
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
            p.insert_text(pymupdf.Point(MARGIN_X + 14, doc.y), ln,
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
    PINNED = {'Figure 8': 'ideas and speculation'}
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
    doc.para('Citations for works named in the text. The source manuscript carries '
             'unresolved superscript placeholders and no bibliography; these were '
             'resolved against the published record rather than reconstructed from '
             'memory, and should still be checked against the originals before submission:',
             font=SERIF_I, size=7.4, lead=9.2, gap=3)
    for i, r in enumerate(REFS_FULL, 1):
        doc.para(f'{i}. {r}', size=7.4, lead=9.2, gap=2.0, indent=6)
    doc.para('Still outstanding. Named in the source draft but not verifiable against the '
             'published record here, and left incomplete rather than invented:',
             font=SERIF_I, size=7.4, lead=9.2, gap=3)
    for i, r in enumerate(REFS_PARTIAL, len(REFS_FULL) + 1):
        doc.para(f'{i}. {r}', size=7.4, lead=9.2, gap=2.0, indent=6)

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
        doc._figure_span(lab, d_, ttl, cap)
        d_.close()
    print(f'  supplemental figures: {n_supp}')

    doc.d.save(OUT, deflate=True)
    print(f'wrote {OUT}  ({doc.d.page_count} pages)')


if __name__ == '__main__':
    main()
