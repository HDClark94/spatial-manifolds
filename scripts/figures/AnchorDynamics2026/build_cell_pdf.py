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

TITLE = 'Entorhinal grid codes are gated by attention'
AUTHORS = ('Harry Clark,^1 Wolf de Wulfe,^1 Chris Halcrow,^1 '
           'and Matthew F. Nolan^1,2,*')
AFFIL = ['^1Centre for Discovery Brain Sciences, University of Edinburgh, Edinburgh, UK',
         '^2Simons Initiative for the Developing Brain, University of Edinburgh, Edinburgh, UK',
         '*Correspondence: mattnolan@ed.ac.uk']

JOURNAL = 'Cell'
CELL_BLUE = (0.00, 0.42, 0.71)
GREY = (0.58, 0.58, 0.58)
# "Grid cells and fast-spiking interneurons express it" read as exclusive,
# which Figure 2D no longer supports: every class exceeds its own null and
# those two lead by about a factor of two.
IN_BRIEF = (
    'Clark et al. show that the anchoring of entorhinal spatial firing to a '
    'task reference frame is a property of the population rather than of single '
    'cells. Every cell class follows this state, grid cells and fast-spiking '
    'interneurons most strongly, yet synaptically connected pairs share it no '
    'more than matched unconnected pairs, and the same state is present in every '
    'structure recorded. It tracks attentional engagement, indexed by pupil '
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
    ('Figure 2', 'fig2_single_units.pdf',
     'Grid cells and putative interneurons follow the population state most strongly',
     '(A, B) Two example sessions: the MEC anchoring raster with PC1, and example single cells of '
     'each identity, including cells locked into one mode. (C) The null for one cell, its labels '
     'circularly shifted 200 times against the population axis, with the observed value marked. '
     'Chance lies near +0.15 rather than at zero, because a cell anchored on most trials '
     'correlates with a population that is also anchored on most trials. (D) Agreement with the '
     "population axis by identity, as excess of |r| over each cell's own null. Every class exceeds "
     'chance and they differ in degree: grid +0.185 (p = 1e-11) and putative interneuron +0.131 (p '
     '= 3e-16) against non-grid spatial +0.081 (p = 6e-22) and non-spatial +0.056 (p = 4e-16). '
     'Grid and interneuron do not differ (p = 0.54) and both exceed the other two (Holm p <= '
     '8e-6). (E) The fraction of cells individually beating their own null, per mouse: 43% of grid '
     'cells and 37% of interneurons against 24% and 19%, all above the 5% false-positive line. (F) '
     'Agreement in the trials around a population transition. All four identities switch together '
     'over about three trials, and identity does not predict switch timing (p = 0.15). (G) '
     'Agreement between identities as a 4 x 4 matrix over its own null; grid-interneuron reaches '
     '+0.109, indistinguishable from either within-group value. (H) Which identities never leave '
     'one mode: interneurons lock anchored (108:2) and non-spatial cells lock non-anchored '
     '(39:293). (I) Open-field speed score for interneurons that lock anchored against those that '
     'vary: +0.141 against +0.065, p = 0.005. (J) The same as a count across entorhinal cells: '
     'speed-modulated cells lock anchored 1.39:1 (p = 1e-4) while cells with no open-field speed '
     'tuning lock the other way at 0.76:1 (p = 0.004), interaction odds ratio 1.84 (p = 3e-6), so '
     'the lock-on bias is carried by the speed cells. (K) Open-field theta index against agreement '
     'with the population axis, pooled over all cells. Points are deciles of the index with their '
     'standard error and the line is the mixed-model fixed effect with a band of one standard '
     'error. The slope is +0.173 per unit index alone (p = 1e-11) and +0.136 with identity in the '
     'model (p = 2e-6), and holds within grid (+0.353, p = 0.037), non-grid spatial (+0.147, p = '
     '9e-5) and interneuron (+0.116, p = 0.028) classes, so theta rhythmicity predicts following '
     'in its own right rather than restating cell type. (L) Whether each '
     "class's own state predicts behaviour, as the Anchoring Dependence "
     'Index: AUC(uncued) minus AUC(cued) for a state built from five '
     'randomly drawn cells of that class, so a class with 92 cells in a '
     'session is not compared against one with five. Every class is '
     'positive -- grid +0.098, putative interneuron +0.050, non-grid '
     'spatial +0.039, non-spatial +0.024, Holm p <= 0.065 -- and no pair '
     'of classes differs (smallest p = 0.12 across the 27 sessions '
     'carrying all four). The ordering follows panel D, but this figure '
     'cannot show that one class carries the behavioural index more than '
     'another.'),
    ('Figure 3', 'fig3_anatomy.pdf',
     'Grid cells are enriched medially and superficially in MEC',
     '(A) Open-field rate maps from four M25 sessions whose probes sat at different mediolateral '
     'positions, sampled so the grid to non-grid ratio drawn is the ratio recorded. Every map comes '
     'from OF1, the arena session preceding the task. (B-D) Best-fit histological slices through '
     'three mice, each under a view of the probe position, with cells coloured by open- field '
     'identity over the Allen region annotation. (E-G) Identity against mediolateral position, '
     'dorsoventral position and layer. Grid cells are enriched medially (median within- session rho ='
     ' -0.078, p = 0.025) and superficially (8.9% against 3.9%, p = 0.002). Regions are shaded in '
     'greys throughout, as in Figure S7, so that only the cells carry colour: deep MEC (L5-6) '
     'darkest, then superficial MEC (L1-3), pre/parasubiculum, visual cortex and hippocampus.'),
    ('Figure 4', 'fig4_ml_following.pdf',
     'The responsive subnetwork is medial, within the superficial grid population',
     'Following the population state, rather than identity, organised along the mediolateral axis of '
     'MEC. (A-D) Four multi-shank sessions, each shown two ways. A-C are the three strongest of the '
     '27 multi-shank sessions by within-session rho on MEC cells, against a median of -0.11, so they '
     'show the gradient clearly rather than typically; D, from a fourth animal, is a session in which'
     ' it is absent. Left, the trial-by-cell anchoring raster split by shank from medial to lateral, '
     "cells within a shank ordered dorsal to ventral, under a strip giving each cell's region, beside"
     ' z-scored PC1, with dashed lines where PC1 changes sign and holds the new sign for at least '
     'three trials. Shanks are numbered S1 to S4 from medial to lateral, on each slice above the '
     "shank's most dorsal cell and beneath each raster block. Right, the best-fit slice with each "
     'cell coloured by its agreement with the population axis above chance: |r| minus the mean |r| of'
     " the cell's own circular-shift null, as in Figure 2D, so a cell locked to the opposite state "
     'also counts as following. The scale diverges from white at chance, green above and purple '
     'below. Cells are jittered slightly along the mediolateral axis on the slice (SD 15 um) so that '
     'cells sharing a shank do not overlap. Above the slice, agreement above chance for each shank as'
     " a boxplot at the shank's own position on the slice's axis. Cells from every region on these "
     'shanks are shown, and the per-shank means and rho quoted below include the parasubicular and '
     'visual cells. Regions are shaded in greys throughout, as in Figure S7, so that only the cells '
     'carry colour: deep MEC (L5-6) darkest, then superficial MEC (L1-3), pre/parasubiculum, visual '
     'cortex and hippocampus. (A) M25 D25, 212 cells over four shanks spanning 720 um and anchored on'
     ' most trials. Agreement is 0.31, 0.52, 0.37 and 0.04 from medial to lateral (rho = -0.255, p = '
     '0.0016). (B) M26 D19, 405 cells over four shanks spanning 720 um, in which PC1 changes sign 13 '
     'times. Following is concentrated on the two medial shanks (mean agreement 0.31 and 0.44) and '
     'nearly absent from the two lateral ones (0.09 and 0.09; rho = -0.467, p = 6e-20). (C) M28 D25, '
     '264 cells over four shanks spanning 690 um and the most evenly balanced multi-shank session '
     '(49% of trials anchored, 4 sustained PC1 transitions). Counted over every region its most '
     'medial shank looks unresponsive (mean agreement 0.13; rho = -0.107, p = 0.14), but the region '
     'strip shows that shank lies entirely in the parasubiculum. Across its three MEC shanks '
     'agreement falls from 0.36 to 0.12 and 0.18 (MEC only, rho = -0.291, p = 0.0007). (D) M29 D25, '
     '344 cells over four shanks spanning 750 um, from a fourth animal, with no gradient: agreement '
     'across its three MEC shanks is 0.16, 0.27 and 0.17 from medial to lateral (MEC only, rho = '
     '-0.015, p = 0.86, rank 18 of 27), and counted over every region the trend runs slightly the '
     "other way (rho = +0.126, p = 0.076). This animal's multi-shank sessions show the gradient "
     'weakly at best (strongest, rho = -0.19). (E) Mediolateral position against agreement above '
     'chance, MEC cells pooled over sessions (rho = -0.17, p = 5e-43); within session the gradient '
     'holds over the 27 sessions spanning at least two shank pitches (median rho = -0.061, p = '
     '0.023). (F) The same against distance from the MEC-parasubiculum border, which aligns sessions '
     'to an anatomical landmark rather than to probe placement. The border is taken from each '
     "animal's best-fit slice through the atlas, as the parasubiculum pixels adjacent to MEC; a "
     "cell's distance is its shortest distance to it, signed by the cell's own atlas label (MEC "
     'positive). Five animals have enough border in their slice to measure (3,702 cells), and '
     'positions are per-shank, so distances are no finer than the shank pitch. Agreement above chance'
     ' is highest in MEC within about 200 um of the border and roughly halves beyond 300 um (MEC '
     'side, pooled rho = -0.20, p = 2e-31), but within session the relation does not reach '
     'significance (median rho = -0.043, p = 0.20, 25 sessions), so the pooled curve partly reflects '
     'differences between sessions. Within 250 um of the border, parasubicular cells agree less than '
     'MEC cells (0.152 against 0.190, p = 0.0002). (G) Within-session rho for raw agreement on each '
     'axis: mediolateral position (median -0.109, p = 0.034, 27 sessions), dorsoventral position (p ='
     ' 0.99, 59 sessions) and probe depth (p = 0.92); only mediolateral position survives. (H, I) '
     'Controls, on the MEC cells of the 27 multi-shank sessions. (H) The gradient survives an axis to'
     " which a cell's own shank does not contribute: each cell scored against a PC1 built only from "
     'the other shanks (median within-session rho = -0.078, negative in 19 of 27 sessions, p = 0.016,'
     ' against -0.109, p = 0.034 on the usual axis). (I-K) What predicts following, in a mixed '
     'model with shanks nested in sessions (so that position, which varies mostly between shanks, '
     'is tested at the level of shanks), on the 2,257 MEC cells on 167 shanks with every factor '
     'measured. (I) The mediolateral coefficient as factors are added. In the model of identity, '
     'layer and firing rate it is -0.017 per SD lateral (p = 0.002); lateral cells are also less '
     'theta-modulated (r = -0.36) and carry less spatial information (r = -0.21), and with both '
     'added it falls to -0.011 (p = 0.036), and with every factor to -0.009 (p = 0.079). (J) '
     'Coefficients of all twelve factors in one model, continuous factors per SD; black, factors '
     'whose removal worsens the fit (likelihood ratio, Holm across factors). (K) Which factor '
     'matters most. Several overlap by construction (putative interneurons are classed on '
     'waveform and firing rate, grid cells on grid score), so a factor\'s unique share understates '
     'any that duplicate one another; factors are instead ranked by dominance analysis, each '
     'one\'s added variance averaged over every combination of the others (4,096 models, least '
     'squares with session fixed effects), which divides shared variance between them. Together '
     'the twelve explain 8.1% of within-session variance: firing rate 25% of that, mediolateral '
     'position 19%, theta modulation 15%, cell type and spatial information 11% each, and '
     'dorsoventral position, layer, head-direction and speed tuning about none. p, the unique '
     'effect as in J. Firing rate predicts following within non-grid spatial and non-spatial '
     'cells separately (p = 0.0009, 0.004; rate x type interaction p >= 0.29), so it is not a '
     'proxy for interneurons, and grid score does so within non-grid spatial cells (p = 0.02). '
     'Between sessions, in the mixed model, session accounts for 17.5% of the variance and shank '
     'for 7.9%, against 4.8% for all twelve factors.'),
    ('Figure 5', 'fig4_monosynaptic.pdf',
     'Monosynaptic connectivity does not account for shared anchoring',
     '(A) The interneuron call: waveform peak-to-trough duration against firing rate, with the '
     'marginal distributions and the fixed 0.4 ms boundary. (B) What a classified interneuron does '
     'to its target, a causal trough in the cross-correlogram. (C) What a broad-spiking cell does, '
     'a causal peak on the same scale. (D) Presynaptic enrichment by identity: inhibitory '
     'connections originate from the cells panel A called interneurons at 3.1 times their '
     'population share, which the detector cannot produce by construction since it never sees the '
     'waveform. (E, F) The inhibitory and excitatory connection matrices, each as a percentage of '
     'the ordered pairs an identity contributes. Interneuron-to-grid is the strongest pathway in '
     'the dataset (3.93%, 5.1x over identity shuffling, p = 0.0005) and grid-to-grid the strongest '
     'excitatory one (0.217%, 2.6x, p = 0.004). (G) Latency by connection type: inhibition is the '
     'slower, median 3.0 against 1.0 ms. (H) Detection against a jitter null, for both signs. (I) '
     'Whether connected pairs share the anchoring state more than unconnected pairs matched for '
     'session, identity pair and probe distance. They do not (grid-interneuron p = 0.39, '
     'interneuron-interneuron p = 0.90). Both comparisons resolve effects a third the size of the '
     'identity effect; grid-grid is underpowered and is greyed.'),
    ('Figure 6', 'fig5_pupil_arousal_M21D19.pdf',
     'The state tracks attentional engagement',
     '(A) The recording arrangement, with infrared illumination and the side camera. (B) The eye '
     'with the DeepLabCut perimeter landmarks from which pupil radius is computed. (C) One '
     'anchored and one non-anchored trial as seven moments of a single traversal, with pupil '
     'radius printed beneath each frame. (D) Mean pupil radius against track position for the two '
     'state blocks of this session, with the two example trials overlaid. (E) The MEC anchoring '
     'raster for the same session, cells that vary ordered by PC1 loading and cells locked into '
     'one mode past a gap. (F) PC1 of the same label matrix. (G) Pupil radius for every trial, on '
     'the trial axis of E and F, so a state block can be read across. (H) Pupil radius by state, '
     'paired within session over 23 sessions: -0.670, p = 2e-06. (I) Pupil radius against track '
     'position for the two states. The difference has the same sign in all 50 position bins and '
     'reaches significance in 48, with no trend along the track (rho = -0.264, p = 0.064) and the '
     'reward zone if anything less affected than the rest, so this is a tonic shift rather than '
     'attention paid at particular places.'),
    ('Figure 7', 'fig2_supp_vis_anchored.pdf',
     'Visual cortex holds the task reference frame through entorhinal switches',
     "(A, B) One session per row: the session's best-fit slice through the atlas with recorded "
     'cells coloured by structure, then the entorhinal anchoring raster, the session PC1, and the '
     'visual raster, all on one trial axis. Both rasters are ordered the same way, cells that vary '
     'by PC1 loading and cells locked into one mode past a gap, and both are cut by the same '
     'dashed lines, the trials on which PC1 changes sign for at least three trials. The entorhinal '
     'population reorganises at those lines and the visual locked block, 77 of 146 cells in A, '
     'does not. Each raster is followed by three of its own cells on the same trial axis: the '
     'entorhinal three are the largest PC1 loadings among cells that vary and their fields appear '
     'in one state only, the visual three are cells that never switch and show the same field in '
     'both. Cells on the slices are spread by 15 um of Gaussian jitter along the mediolateral '
     'axis for display only, as in Figure 4, so that cells on one shank do not overlap. '
     '(C) The census: which cells '
     'lock anchored, by structure and by open-field speed tuning. Both visual groups lock '
     'anchored, 7.5:1 without speed tuning and 7.8:1 with it, against 0.8:1 and 1.4:1 in '
     'entorhinal cortex, so the visual enrichment does not depend on the speed split. (D) What '
     "those same cells follow. Each group's median correlation with running speed and with a "
     'luminance proxy is compared against 500 phase-randomised surrogates of the same regressor, '
     'which keep its smoothness while destroying its relation to position; the block is the middle '
     '95% of the surrogates and the dot is the real regressor. Entorhinal cells follow speed '
     'whether or not they are speed-tuned in the open field (+0.12, p = 0.016 and +0.22, p = '
     '0.028) and the light in neither case; visual cells without open-field speed tuning follow '
     'the light (+0.34, p = 0.010) and not speed, while speed-tuned visual cells follow neither. '
     'Random 98-cell draws from the 308 reach +0.34 in 0.6% of cases, so the open-field '
     'classification selects the luminance-following cells specifically. (E) Where the position '
     'structure of these cells sits: variance of the profile against position, normalised to its '
     'track average, with the 30 cm black box at each end shaded. Nearly half of it falls inside '
     'the black boxes, which cover 30% of the track, and the reward zone is where it is scarcest.'),
    ('Figure 8', 'fig7_theta_gamma.pdf',
     'A frequency-specific change in the entorhinal field potential',
     '(A-D) One session trial by trial, both states shaded behind every panel and dashed lines '
     'where PC1 changes sign for at least three trials. (A) The population axis. (B) The '
     'spectrogram, 1.9-150 Hz in 56 log-spaced bands, smoothed along the trial axis only. (C) The '
     'three band traces. (D) Theta-gamma modulation index for both bands on twinned axes, fast on '
     'the left and slow on the right, which they need because slow-gamma MI is about five times '
     'smaller. (E) The two states as spectra across 26 switching sessions, the spectra above and '
     'the deviation between them below. The deviation crosses zero at a median of 67 Hz and does '
     'so in 25 of the 26 sessions. (F) The same difference band by band: theta falls (-0.094) '
     'while slow gamma rises (+0.105), so the difference reverses sign across the spectrum. (G) '
     'Theta-gamma coupling by band: fast gamma uncouples (-15.9%, p = 5e-6) and slow gamma does '
     "not. (H) The redistribution, tested pairwise within session. Left, each session's slow-gamma "
     'change against its fast-gamma change, with the equal-and-opposite line and the mean with its '
     'SEM. Right, the three contrasts drawn directly, one point per session. The difference '
     'between bands is large and reliable (slow minus fast +0.213, p = 4e-5) while the sum is not '
     'distinguishable from zero (-0.002, p = 0.86), nor is the change across 2-200 Hz (-0.031, p = '
     '0.48): power moves from fast to slow with no net change. The sum is zero on average rather '
     'than session by session, and the balance is per Hz -- weighting each band by its width moves '
     'the sum to -0.36 (p = 0.12) -- so what is supported is that power moves from fast to slow, '
     'not that the total is conserved. Where in the theta cycle each frequency peaks is shown in '
     'Figure S10.'),
    ('Figure 9', 'fig8_model.pdf',
     'Proposed engagement gate for position estimation in the entorhinal cortex',
     'A proposed mechanism, placed with Ideas and speculation rather than with the Results. Panel '
     'A is a schematic and the values on it are measured; B is drawn from measured values; C, D '
     'and E are simulated. (A) The circuit. A self-motion drive is always present and a landmark '
     'drive carrying the task frame passes through an engagement gate; the gate is what the '
     'anchoring state is, open meaning the estimate is re-referenced to the task frame on every '
     'traversal and shut meaning it is not. In this task engagement goes with a constricted pupil, '
     'not a dilated one (Figure 6). The estimate is carried by grid cells and fast-spiking '
     'interneurons, the two populations that follow the state most strongly (Figure 2), and drives '
     "where the animal slows and stops. The slope on the self-motion arrow is the hinge: theta's "
     'entrainment to running speed roughly halves when the gate opens (0.374 to 0.218, p = 0.045) '
     'while the open-field speed cells code speed just as well in both states (+11.9%, p = 0.12), '
     'so the velocity signal is intact and the rhythm is heeding it less. (B) Theta-nested gamma '
     'in the two states, each band at its measured preferred theta phase -- 95 degrees for slow '
     'gamma and 42 degrees for fast, with 0 degrees the theta peak (Figure S10). The phases are '
     'identical in the two rows because the state does not move them (paired within session, 6.6 '
     'and 2.8 degrees, p = 0.33 and 0.16). What changes is amplitude, slow gamma up and fast gamma '
     'and theta down (Figure 8), and how tightly the fast burst is confined to its phase, drawn '
     'broader when anchored because theta-fast coupling weakens 15.9% (p = 5e-6) while theta-slow '
     'does not (p = 0.71). (C) Three simulated periodic grid cells of different spacing over ten '
     'traversals. With the gate open the landmark drive re-references the attractor each lap and '
     'fields recur at the same places; with it shut the phase is drawn afresh lap by lap. The '
     'phase is drawn afresh rather than progressively displaced because non-anchored trials show '
     'no coherent field at any offset and neighbouring traversals do not align, so the correction '
     'is missing rather than moved. (D) Position decoded from a continuous attractor fed by those '
     'cells, drawn as the decoded position itself, wrapped onto the track. Open, it runs 0 to 200 '
     'cm identically every lap; shut, it starts from an arbitrary offset and integrates away '
     'within the lap. (E) Stops on uncued traversals, with the reward zone at 90-110 cm shaded. '
     'Uncued because the zone carries no visual cue there, so stopping in it requires the estimate '
     'to be on the track, and because that is where anchoring buys most of its measured '
     'behavioural advantage (Figure 1). The circuit is drawn once rather than twice because the '
     'measurements say it is the same in both states: spike coupling to gamma does not move with '
     'the state (Holm p >= 0.17), nor does the laminar theta phase gradient, and connected pairs '
     'share the state no more than matched unconnected ones (Figure 5).'),
]

# Supplemental figures, in the order their main figure appears.
SUPPLEMENTS = [
    ('Figure S1', 'fig1_supp_classifier.pdf',
     'The anchoring classifier, step by step, against RNN ground truth, related to Figure 1',
     'The classifier validated against ground truth, using an RNN trained on a 1D slice of a 2D '
     'environment with a teleport, where the trial labels are known rather than inferred. (A) The '
     "200 cm chord through the 2.2 m box that the network runs along. (B) One network unit's 2D "
     'rate map with that chord drawn on it: the 1D firing the classifier sees is this slice. (C) '
     'Position on the track against time; the dotted lines are teleport resets. (D) Decode error '
     'per trial, which shows the manipulation worked before any classifier is applied: the blind '
     'block is re-anchored at neither end. (E-H) The four steps of the classification on one cell, '
     'in order: the NaN-free trial x position rate map, the trial x trial correlation matrix, the '
     "cell's mean correlation against its own circular-shift null, and the resulting label "
     'sequence against truth, with and without the median filter. (I) Reported against true '
     'anchored fraction, with and without the gate. Ungated, a session that is 0% anchored is '
     'reported as 50.6% anchored; gated, the curve tracks the identity line. (J) Per-trial '
     'accuracy over the same range; the band marks where the validation session itself sits, which '
     'is why that session cannot test the gate. (K) Recall against the length of a non-anchored '
     'epoch: the size-5 median filter erases runs shorter than three trials.'),
    ('Figure S2', 'fig1_supp_population_anchoring_examples.pdf',
     'Population anchoring states across sessions, related to Figure 1',
     'Six further sessions, ordered by the share of label variance the first principal component '
     'explains, from sessions where nearly all cells move together to sessions where they do not. '
     '(A-F) One session each, as five panels on a shared trial axis: stops on cued trials, stops '
     'on uncued trials, the anchoring raster with cells on the x axis, PC1, and hit rate by state '
     'and cue condition. Error bars are binomial standard errors, the unit here being the trial '
     'within one session.'),
    ('Figure S3', 'fig1_supp_speed_matching.pdf',
     'Behavioural controls for the anchoring state, related to Figure 1',
     'Behavioural controls for the anchoring state, and the speed-matched subsets used wherever a '
     'comparison could otherwise be carried by a difference in locomotion. (A, B) Trial mean '
     'running speed by state, before and after matching within session in deciles of speed. (C) '
     'The same per session, before and after. (D-F) Hit rate by state and cue condition after '
     'matching, for MEC, visual cortex and the subicular complex; the cerebellar sessions are too '
     'few to match. (G) How well the state predicts a hit, as AUC, before and after matching, for '
     'each structure and cue condition. (H) The Anchoring Dependence Index per session, before and '
     'after matching. (I) The cost of matching, as the percentage of trials retained.'),
    ('Figure S4', 'fig2_supp_nonanchored_rate.pdf',
     'What a non-anchored trial is, and whether firing rate changes with the state, related'
     'to Figure 2',
     "Two halves of one question. (A-D) Has the field moved, or has it gone? (A) Each trial's map "
     "correlated against the cell's anchored template at every circular offset. (B) Zero-offset "
     'correlation by state: +0.33 to +0.07. (C) Best-over-offsets correlation by state, which '
     'falls only from +0.56 to +0.53 and looks like displacement until the null is applied. (D) '
     'That null: a maximum over 100 offsets is large by construction, two unrelated cells reach '
     '+0.500 under the same search, and non-anchored trials sit 0.029 above that floor with a mean '
     'best offset of 47 cm against the 50 cm a uniform choice would give. Non-anchored trials lose '
     'spatial correspondence with no evidence of a coherent field elsewhere. (E-K) Does rate '
     'change with the state, or only where cells fire? (E) The same cells under three measures: '
     'log2 anchored over non-anchored is -0.090 spatially averaged, -0.028 time averaged, and '
     '+0.036 time averaged and speed-matched. All three are significant as cell-level tests and '
     'they disagree about direction. (F) Why: a rate read off a rate map divides by occupancy, '
     'which running speed sets, while a rate in spikes per running second does not; the two '
     'correlate at r = 0.96 and differ in mean. (G) Trial mean speed by state, before and after '
     'matching. (H) The circularity control: splitting a cell by its own label gives +0.128 '
     'against -0.090 for the population state, so every split here uses the population state. (I, '
     'J) The same question by identity, unmatched and speed-matched, with one point per session '
     'over the cell cloud and a Holm-corrected p per class. Sessions are the unit because cells '
     'within a session share one state. On that test no class differs from zero on either measure; '
     'putative interneurons are +2.7% unmatched and +4.2% speed-matched over 34 sessions, the '
     'largest value and the only class that does not go down, but not significant. (K) Per mouse, '
     'unmatched and matched. Nothing that reverses sign under a change of denominator should be '
     'reported as a rate effect: anchoring concerns where cells fire rather than how much.'),
    ('Figure S5', 'fig3_supp_laminar.pdf',
     'Recording sites are biased toward superficial layers medially, and what that does to '
     'the mediolateral results, related to Figures 3 and 4',
     'MEC is curved and a four-shank probe is flat, so shanks at different mediolateral positions '
     'do not enter the tissue at the same depth relative to the layers. (A) They do not: '
     'superficial layers are 90.3% of entorhinal recording sites in the most medial quartile and '
     '59.8% in the most lateral (pooled rho = -0.267, p = 4e-80), a gradient present in every '
     'animal (within-mouse median rho = -0.447, p = 0.031). (B) The same bias in the neurons '
     'actually analysed (within-session median rho = -0.402, p = 0.016). (C, D) The mediolateral '
     'tests stratified by layer. The identity gradient does not survive within superficial cells '
     '(median rho = -0.051, p = 0.24, 25 sessions) against -0.078 (p = 0.025) across all layers; '
     'the following gradient does and is if anything stronger (-0.144, p = 0.043, 22 sessions '
     'against -0.109, p = 0.034). Deep layers are too sparsely sampled to test on their own.'),
    ('Figure S6', 'fig6_region_examples.pdf',
     'The same state, described separately by each structure, related to Figure 7',
     'Two sessions bracketing the range among those recording all three structures. (A, B) One '
     'session each, read left to right as entorhinal, subicular and visual cortex. Everything '
     "within a structure is built from that structure's own cells, the raster ordered by its own "
     'PC1 loading and the component signed to its own anchored fraction, so agreement between the '
     'groups is agreement between independent descriptions rather than a consequence of a shared '
     'frame.'),
    ('Figure S7', 'fig6_supp_region_examples.pdf',
     'The state outside MEC, where MEC does not dominate the sample, related to Figure 7',
     'Four sessions chosen for their sampling rather than their result. (A, B) Two visual-rich '
     'sessions. (C) A session in which visual cortex contributes more cells than entorhinal and '
     'disagrees with it. (D) A session in which the subicular complex does. Each row is built as '
     'in Figure S6, every structure described by its own cells. A cerebellar session was removed: '
     'it rested on 15 to 18 cells, too few for an axis to be estimated with confidence.'),
    ('Figure S8', 'fig6_supp_dimensionality.pdf',
     'One axis is an adequate description of the state, related to Figure 7',
     "Out-of-sample variance explained, with the axis built from one half of a region's cells and "
     'scored on the other. (A) Neuron-dropping curve per region: how much of the held-out half one '
     'axis explains as cells are added. (B) The same at the full cell count, per region. One axis '
     'is an adequate description of the state in every structure recorded.'),
    ('Figure S9', 'fig7_supp_band_examples.pdf',
     'Band power tracking the population state in individual sessions, related to Figure 8',
     'Four sessions as the population axis, the log-spaced spectrogram and the three band traces '
     'on a shared trial axis. (A-C) Sessions showing the population pattern, slow gamma rising and '
     'fast gamma falling across anchored blocks. (D) A counterexample, drawn in red: a session '
     'with clear state transitions whose band traces do not follow them. It is included because '
     'the population effect is a mean over sessions that individually scatter widely -- slow gamma '
     'rises in 21 of 26 and fast gamma falls in 19 of 26, not in all of them.'),
    ('Figure S10', 'fig7_supp_comodulogram.pdf',
     'Where in the theta cycle each frequency peaks, and how little it moves with the state,'
     'related to Figure 8',
     'Figure 8 fixes two gamma bands in advance and asks how strongly each is nested in theta; the '
     'modulation index discards phase, so it cannot ask where in the cycle they peak. This does, '
     'as a continuous function of frequency. (A, B) The phase-frequency map for each state over 28 '
     'sessions, each frequency row normalised to sum to one so colour is where a frequency peaks '
     "and not how much power it carries. Both states' preferred-phase ridges are drawn on both "
     'maps. (C) Their difference, which is where a state effect would live. The two maps share 94% '
     'of their pixel variance. (D) The same reduced to preferred phase against frequency. The '
     'relationship is a gradient, not a step between two bands: phase runs from about 130 degrees '
     'at 20-30 Hz to about 41 degrees at 60-100 Hz, and that separation holds in each state on its '
     'own (+34 degrees anchored, p = 5e-4; +52 degrees non-anchored, p = 1e-5). Paired within '
     'session the state moves it very little and not reliably: 6.6 degrees earlier in the slow '
     'band (p = 0.33) and 2.8 degrees in the fast (p = 0.16). (E-G) The same question against '
     'depth: theta-triggered LFP and CSD down one shank for three example sessions, both states '
     'overlaid. (H) The laminar phase gradient over 27 sessions by layer. Theta advances about 100 '
     'degrees across the entorhinal layers and the gradient is identical in the two states: '
     "referenced to each session's own layer 2, every layer differs by less than 3 degrees with p "
     '>= 0.06.'),
    ('Figure S11', 'fig7_supp_gamma_locking.pdf',
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
     'interneuron-minus-principal phase lag per session: a median +45 degrees in the fast band (p = 9.6e-5, later in '
     '25 of 31 sessions) and nothing in the slow band (+10 degrees, p = 0.77). The interneurons '
     'therefore FOLLOW the principal cells, the direction PING predicts, and do so in one band '
     'only. (E) Coupling by anchoring state. No band and no class moves '
     'with the state (Holm p >= 0.17). (F) Spike counts per cell and state, to show the null in E '
     'is not a power difference.'),
    ('Figure S12', 'fig7_supp_speed_tuning.pdf',
     'Speed coding does not change between the anchoring states, related to Figure 8',
     'Theta loses much of its entrainment to running speed in the anchored state, and the account '
     'in Figure 9 turns on whether the cells that carry the speed signal lose it too. (A) One '
     "session's speed and firing rate on the trial axis, with both states shaded. (B) Rate against "
     "speed for open-field speed cells, by state, on the two states' common speed support. (C) The "
     'regression slope by state across sessions: 0.1523 anchored against 0.1361, +11.9%, p = '
     '0.121, 29 sessions. (D) The same as rate-normalised gain, which separates a change in speed '
     'sensitivity from a change in overall firing: +6.8%, p = 0.565. (E) The other cell groups, '
     'none of which changes either. (F) The power of the comparison: it resolves a 17% change in '
     'slope and an 18% change in gain at 80% power, so a halving of speed drive of the size seen '
     'in the field potential is excluded in the cells that carry it.'),
    ('Figure S13', 'fig8_supp_can_model.pdf',
     'The gated attractor, simulated as a ring and as a two-dimensional grid network, related to Figure 9',
     'The mechanism Figure 9 proposes, built as a circuit to see what it produces rather than to '
     'illustrate what it claims, and built twice: as a ring carrying one module\'s phase along the track, '
     'which is enough for the field potential but assumes what a grid model ought to produce, and as a '
     'two-dimensional sheet whose units fire in hexagonal grids in an open arena. In both the gate is '
     'shut, then open, then shut again over about a hundred laps, because the recorded population state '
     'comes in blocks of tens of trials rather than as one switch. (A) The architecture. A landmark '
     'phase drives an error population with fast inhibition, which compares it against a subtractive '
     'copy of the attractor; the gate scales only how much that error is allowed to correct the '
     'attractor, which has slow inhibition. The landmark input arrives in both states, so the gate '
     'controls whether the error is acted on and not whether it is transmitted. No geometry is named '
     'here because the same circuit is built both ways below, and the one difference the geometry forces '
     '-- how the error reaches the attractor -- is described under H. (B, C) What each network is made '
     'of. The ring is 128 units carrying one module\'s phase, so one turn around it is 55 cm of track; '
     'the sheet is 1,296 units covering one cell of a hexagonal lattice 42 cm across, drawn in the '
     'coordinates in which the bump is round. Both hold the bump in a slow synaptic variable and both '
     'are moved by an exact spectral translation, so the geometry its units tile is the only thing that '
     'differs between them. (D) The run that builds the fields, and what the teleport does. The sheet is '
     'driven along a straight line through the arena at 12 degrees to a lattice axis and the animal is '
     'then put back at the start, while the integrator is not. Behind the line is one unit\'s firing in '
     'the plane; the three traces are that same unit along that same path on three successive laps with '
     'the gate shut. The path is identical every lap and the fields it crosses are not, because the '
     'integrator has carried on by the per-lap phase step G gives. That drift is what the gate has to '
     'cancel, and the diagonals in H are the same thing drawn over 86 laps. (E) One ring unit over the '
     'session, with the anchoring classifier\'s call on each lap beside it. Its fields recur in the same '
     'places while the gate is open and walk when it shuts, and nothing is reset at the transitions, so '
     'the attractor runs straight through them. The call is right on every open lap and on 90% of shut '
     'ones, and the strip shows what the remainder are: laps on which the drifting phase happens to pass '
     'close to where the fields sat when anchored. Those coincidences are periodic rather than scattered '
     '-- the phase slips a median 131 degrees per lap, so it comes back into register every 2.7 laps, '
     'and with the classifier\'s five-trial median filter switched off the shut block shows exactly that '
     'rhythm and a false-positive rate of 0.37 rather than 0.10. Their regularity is itself evidence '
     'that the bump drifts at a constant rate rather than wandering, which is what Figure S14E also '
     'finds. (F) The same session through the recordings\' own anchoring classifier and its first '
     'principal component, the code that produces Figure 1E, applied unchanged: every unit is called '
     'anchored on every lap of the open block, and PC1 is positive on 33 of 33 open laps and 4 of 53 '
     'shut. (J) The band changes, measured against both models, in the same units: a difference of log10 '
     'power between the states over the same bands, with the mains lines excluded. (G) Six sheet units '
     'in the arena with their grid scores, from the same scorer used for the recorded cells: median 1.53 '
     'over all 1,296 units, every one above 0.4. Two geometric choices matter and neither is the obvious '
     'one. How many fields a unit has per lap is set by the track\'s heading through the lattice rather '
     'than by the period: a track that grazes a lattice row gives one long field however short the '
     'period is. And what carries the bump between laps is the per-lap step in PHASE, which is larger '
     'for a shorter period, not a longer one. Because path integration here is exact, that step is '
     'identical every lap, so if either lattice axis steps only a little the unit fires in closed blobs '
     'spanning several laps instead of fields that shift. At 42 cm and a heading of 12 degrees the two '
     'axes step 0.38 and 0.30 of a cycle per lap, 89% of units have two or more fields per 200 cm, and '
     'the unit shown has six. (H, I) The same two panels as E and F for the sheet, which is driven along '
     'a line through the arena and teleported back to the start each lap, as a virtual linear track '
     'does. The teleport is what gives the gate work: the animal\'s position jumps back and its path '
     'integrator does not. The sheet also forces the one change of mechanism in this figure. On the ring '
     'the error reaches the attractor as a drive, and that drive moves the bump because it is '
     'concentrated on 128 units; spread over 1,296 it cannot. Scanned from a gain of 5 to 300 it never '
     'bought a lock without destroying what it was locking: the mean phase mismatch fell only from 54 to '
     '38 degrees, while the drive -- which is shaped like a bump at the LANDMARK phase -- painted a '
     'second bump on the sheet beside the attractor\'s own, taking the population vector from 0.27 to '
     '0.047 and, at a gain of 150, the grid score to 0.29. An earlier version of this panel ran at the '
     'low end of that scan, and its open block repeated only every few laps rather than every lap. On '
     'the sheet the error therefore corrects the bump\'s POSITION instead, driving the phase error '
     'towards zero with a time constant of 0.25 s, which is what a correction to a path integrator does '
     'and costs the bump nothing: the population vector is 0.430 in both states. The mismatch is then 6 '
     'degrees with the gate open against 139 with it shut, successive laps\' maps correlate at +0.98, and '
     'every unit is called anchored on every lap of the open block against 4% of laps outside it, and '
     'PC1 is positive on 33 of 33 open laps and 2 of 53 shut. I shows the sheet twice: as simulated, and '
     'run again with three times the noise on both populations. The noisier sheet is no worse, every '
     'unit anchored in the open block and 5% outside it. The comparison is a clean one now: what holds '
     'the phase is the correction above, whose rate does not scale with the noise, so tripling the noise '
     'does not quietly strengthen the gate along with it. It shows that the state classification is not '
     'an artefact of a quiet simulation. (K) Theta-gamma coupling in both models against the recordings. '
     '(M, N, O) Each model\'s two states as spectra and as deviations, drawn as Figure 8E draws the '
     'recorded ones, with one spectrum per lap standing in for one per trial: the ring, the sheet, and '
     'the sheet with three times the noise. (L) Where in the theta cycle each band peaks, the quantity '
     'Figure S10 measures. Taken together the two models settle different things and neither settles '
     'all. The circuit works when its units are real grid cells and the anchoring switch survives in two '
     'dimensions, which the ring could not show. The ring gets the fall in theta-fast coupling, and '
     'generates the prediction that was confirmed in the recordings (within the gate-shut state, '
     'fast-gamma amplitude against mismatch, r = +0.51); the sheet gets neither (r = 0.00), because its '
     'error population\'s inhibitory pool averages over 1,296 units rather than 128 and never reaches the '
     'strongly oscillating regime that carried them. The same division runs through J. The ring '
     'reproduces all three measured band changes in sign -- theta -0.008, slow +0.025, fast -0.204, '
     'against -0.013, +0.018 and -0.016 in the recordings -- while the sheet puts all three downwards, '
     'theta -0.093, slow -0.062 and fast -0.027, and so gets the theta and fast-gamma falls but not the '
     'slow-gamma rise. What does rise on the sheet is theta-slow COUPLING, by 9.8% as simulated and '
     '22.1% with three times the noise (K), which the recordings leave unchanged. The attractor\'s own '
     'pool barely moves between the two states -- its mean square differs by 1% -- so what band '
     'differences the sheet does produce are carried by the error population falling quiet once the '
     'correction succeeds, which is the fast-gamma fall itself, and not by any change in the attractor\'s '
     'rhythm. And both put the two gamma bands nearly antiphase in theta where the recordings place them '
     'about 40 degrees apart; sweeping a phase offset on the error population\'s theta gate over a full '
     'cycle does not move it, because the two pools\' envelopes are driven in antiphase by the '
     'construction itself, one tracking how concentrated the bump is and the other how large the error '
     'is.'),
    ('Figure S14', 'fig8_supp_attractor_state.pdf',
     'The attractor bump survives while non-anchored to the task reference frame, '
     'related to Figure 9',
     'Figure 9 and the model in Figure S13 both assume the non-anchored state is an attractor '
     'that is still running and merely uncorrected, rather than one that has come apart. That '
     'assumption is tested here. Each cell is given its phase offsets on the module\'s torus by '
     'the spectral method, the population is projected onto them, and every result is referred '
     'to a shuffle of which offset belongs to which cell, which leaves all firing rates '
     'untouched and destroys only the lattice organisation. (A) The population anchoring state '
     'in M25 D24, cell by cell and as PC1; the two lines mark the trials shown in E. (B, C) The '
     'population\'s phase along two of the module\'s lattice axes at each position of each '
     'trial. Which two matters: on a linear track the bump has one degree of freedom, so the '
     'track direction projects onto two of the three axes almost equally and onto the third '
     'hardly at all, and those two are then nearly the same phase, correlated at +0.96 and 12 '
     'degrees apart on average. B therefore shows the axis running along the track and C the one '
     'running across it, which is the independent pair (circular r = +0.05); the torus above '
     'each shows which cycle it is. (D) The bump\'s path over single trials, unwrapped across '
     'copies of the sheet, twelve trials either side of the switch, with the unit cell dashed '
     'and the mean starting point marked. This panel uses the two axes that advance, since an '
     'unwrapped path needs coordinates that move. (E) The bump through the whole session and at '
     'the switch. Above, its two sheet coordinates and the animal\'s position as continuous '
     'traces. Below left, a hundred-second window in which the bump\'s own speed is plotted '
     'against the animal\'s: the two move together. Over the session the bump advances 0.69 cm '
     'per centimetre run while the state holds and 0.38 once it is lost, and the same shortfall '
     'appears in the position domain, where the non-anchored phase advance is 0.55 to 0.73 of '
     'the anchored one. That is the mechanism rather than a random walk: over a 200 cm lap it '
     'accumulates three to five radians, half to three quarters of a cycle, so the phase at a '
     'given place on the track is effectively re-drawn by the next lap, which is also why '
     'lap-to-lap phase displacement looks uniform and cannot be unwrapped. Below right, the unit '
     'cell itself in successive windows of six trials across the switch, each a scatter of where '
     'the bump sat at 10-30 cm on the track: tight while the state holds, filling the cell once '
     'it is lost. (F) The same session\'s bump population vector per 100 ms, both states above '
     'the shuffled distribution. (G) Every session. The bump exceeds its shuffle by +0.061 when '
     'anchored and +0.041 when not, both p = 1.2e-07, so the attractor does not collapse; the '
     'states differ by +0.020, p = 1.3e-04, so it is modestly weaker, and equating the two '
     'states\' running speeds changes nothing (+0.019, p = 7.6e-05). (H) The bump aligned to the '
     '150 anchored-to-non-anchored switches, mean and SEM over switches. It does not fall '
     'abruptly at the switch (-0.005 over three trials either side, p = 0.20) and does not decay '
     'once there (slope -0.00004 per trial within non-anchored epochs, p = 0.96), so the weaker '
     'bump is a property the state holds throughout rather than a progressive deterioration.'),
    ('Figure S15', 'fig7_supp_vis_examples.pdf',
     'Entorhinal and visual anchoring in four sessions, related to Figure 7',
     'Four sessions, drawn as Figure 4 draws its examples: the two of Figure 7 (A, B) and the two '
     'others with at least 25 cells in each structure and the largest difference between them in '
     'the share of locked cells, from mice not already shown (C, D). Left, one anchoring raster '
     'split into an entorhinal (MEC) and a visual block, each under a strip giving every cell\'s '
     'region in the grey key of Figure 4; within a block, cells that vary are ordered by their '
     'correlation with the session PC1 and locked cells sit past a gap, with each structure\'s '
     'cell count and locked share above. Cells with no usable label on any trial are left out. '
     'Middle, PC1 computed separately from each structure\'s own cells, z-scored, one line per '
     'structure, with their correlation above. Dashed lines are the trials on which the session '
     'PC1 changes sign for at least three trials. Right, the session\'s best-fit atlas slice at '
     'true aspect, with every device contact drawn small and grey as in Figure S7 and depth '
     'measured from the most dorsal contact, and with '
     'every cell coloured by its agreement above chance with the session PC1, on the diverging '
     'scale of Figure 4 (white at chance), with 15 um of display jitter along the mediolateral '
     'axis. Locked cells have no agreement, since a constant label correlates with nothing, and '
     'are drawn small and grey. The visual cells that vary follow the entorhinal axis closely: '
     'the two structures\' PC1s correlate at +0.93, +0.76, +0.91 and +0.53 in A-D. What '
     'separates visual cortex is its locked block, larger than the entorhinal one in A, C and D '
     '(53% against 34%, 25% against 10%, 38% against 13%) but not in B (18% against 21%). '
     '(E-G) Whether the two structures share one state, in every session with at least ten '
     'varying cells in each (24 sessions). Each structure\'s state is its own PC1, and its '
     'switches are that PC1\'s sign changes with epochs shorter than three trials absorbed, as '
     'for the dashed lines. (E) Switch-triggered averages: the change in the fraction of each '
     'structure\'s varying cells in the anchored mode, signed so that the new state is up and '
     'referred to the five trials before the switch, averaged per session and then across '
     'sessions (mean and SEM), around MEC switches (left, 167) and visual switches (right, 234). '
     'Stippled, the middle 95% when the other structure is shifted in time 200 times. Visual '
     'cortex steps at MEC switches, 59% as far as MEC over the first five trials, and MEC steps '
     'at visual switches, 81% as far as visual cortex, both beyond every shift (p < 0.005), at '
     'the same trial and with no lead in either direction. (F) The correlation of the two PC1s '
     'against the ceiling set by their own reliability, the geometric mean of each axis\'s '
     'split-half reliability (Spearman-Brown corrected): a session on the diagonal shares as '
     'much as its noise allows. Filled, beyond a null shifting one axis in time; letters, the '
     'sessions of A-D. Where both axes are reliable (ceiling at least 0.5, 14 sessions) the '
     'correlation is a median 0.96 of its ceiling, with three exceptions (M21 D25, M28 D25, M29 '
     'D25); most sessions without a correlation are ones whose visual axis is too noisy to show '
     'one. (G) Every cell against each axis: following it (r above the 95th percentile of the '
     'cell\'s own circular-shift null), opposing it (below the 5th), neither, or locked, with a '
     'cell\'s own structure\'s axis rebuilt without it. Visual cells follow the MEC axis as often '
     'as their own (17% against 16% of 1,355) and MEC cells follow the visual axis nearly as often '
     'as their own (21% against 25% of 3,657). The state is shared between the structures; what '
     'distinguishes visual cortex is the share of its cells that never switch.'),
    ('Figure S16', 'fig8_supp_tsc.pdf',
     'Anchored and non-anchored trials are built from nearly the same kinds of theta cycle, '
     'related to Figure 8',
     'Following Lopes-dos-Santos et al. (2018) and Esparza et al. (2023), each theta cycle of the '
     'entorhinal LFP is described by its 20-120 Hz amplitude across 18 theta-phase bins, after '
     'subtracting the theta-locked waveform so that harmonics of non-sinusoidal theta cannot pose '
     'as nested gamma. Maps are normalised per session and frequency, the cycle\'s overall level '
     'is kept separately, and equal numbers of cycles from each of 44 sessions are decomposed '
     'by PCA and independent component analysis into theta-nested spectral components (tSCs). '
     '(A) The four components, the number at which every one reproduces across two disjoint '
     'halves of the sessions (split-half r above each), with a theta cycle drawn over them: '
     'slow gamma on the rising and on the falling phase, fast gamma at the trough, and a 23 Hz '
     'component across the cycle that may be residual theta harmonic. (B) Anchored minus '
     'non-anchored weight on each component and the cycle\'s gamma level, per session, after '
     'matching the two states on running speed, in units of the session\'s SD; Holm-corrected '
     'p above. Only the falling-phase slow-gamma component differs (+0.04 SD, p = 0.027) and '
     'the level is if anything lower when anchored (-0.03 SD, p = 0.053). (C) How well state '
     'can be read from a cycle\'s four weights and level: cross-validated AUC with folds by trial, '
     'per session, for single cycles (median 0.51) and for trial means (0.54), neither above '
     'chance. (D) Within each state, the partial correlation of each weight with how far the '
     'MEC population decoded on that cycle lies from the true position, controlling for speed, '
     'theta amplitude, spike count and position; stars mark Holm p < 0.05. Both slow-gamma '
     'components go with slightly better decoding in the anchored state (median rho -0.007 and '
     '-0.011), too small to carry the state difference. (E) One session\'s per-cycle MEC '
     'population vectors embedded with UMAP and coloured by position, state and the cycle\'s '
     'largest tSC: the manifold is organised by position and by state, and cycle type is '
     'scattered across it.'),
]

REFS_FULL = [
    'Lopes-dos-Santos, V., van de Ven, G.M., Morley, A., Trouche, S., Campo-Urriza, N., and '
    'Dupret, D. (2018). Parsing hippocampal theta oscillations by nested spectral components '
    'during spatial exploration and memory-guided behavior. Neuron 100, 940-952.',
    'Esparza, J., Sebastian, E.R., and de la Prida, L.M. (2023). From cell types to population '
    'dynamics: making hippocampal manifolds physiologically interpretable. Curr. Opin. '
    'Neurobiol. 83, 102800.',
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
    PINNED = {'Figure 9': 'ideas and speculation'}
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
