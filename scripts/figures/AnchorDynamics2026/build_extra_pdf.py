"""The Extra analyses PDF: material removed from the AnchorDynamics2026 paper.

Analyses taken out of the manuscript on 2026-11-01 but kept because they may
serve another paper: the open-field decomposition of each structure's cells
(XGBoost, the paper's former Figure S5), the laminar sampling of the
mediolateral results (former S8), the region-by-region example sessions
(former S11 and S12), and the Methods of every analysis that used boosted
trees. Text comes from EXTRA.md; the layout, fonts and reference list are the
Cell mock-up's own (build_cell_pdf.py), so the two documents match.

Writes AnchorDynamics2026_Extra_analyses.pdf
Usage:  python build_extra_pdf.py
"""
import os
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_cell_pdf as B                                        # noqa: E402

OUT = f'{HERE}/AnchorDynamics2026_Extra_analyses.pdf'

EXTRA_FIGURES = [
    ('Figure E1', 'fig1_supp_of_decomp.pdf',
     'Open-field coding of the cells of each structure, formerly with Figure 2',
     'Each cell\'s firing in the open field decomposed into coding of position (P), head direction '
     '(H) and running speed (S), or a combination of them, by the forward model selection of '
     'Hardcastle et al. (2017), with gradient-boosted tree models (XGBoost) in place of their '
     'linear-nonlinear models (Methods). (A) The model. A cell\'s spike counts in 20 ms bins '
     'are predicted from its own open-field variables alone: position (x, y), head direction (as '
     'its sine and cosine, since a tree cannot split across the wrap) and speed. No other cell\'s '
     'spikes enter. The model is a sum of regression trees, each as deep as the model has input '
     'columns, and the predicted rate is the exponential of that sum (Poisson objective); trees are '
     'added until the fit to one held-out training fold stops improving. The session is cut into 50 '
     'sections, and each of 10 folds tests on every tenth section, so every fold samples the whole '
     'session. A model\'s performance is the Poisson likelihood of the observed spikes in the test '
     'fold, over a constant rate fitted to the training folds, in bits per spike. Traces, 10 s of '
     'the cell of B; right, its predicted rate in held-out bins. (B) Forward selection for that '
     'cell: each model\'s held-out gain per fold (points) and mean (bars). The best single variable '
     'is taken first; the best model adding one more replaces it if it is better across the 10 '
     'folds (one-sided Wilcoxon signed-rank, p < 0.05; here PH over P, p = 0.001, and PHS over PH, '
     'p = 0.001, the smallest p ten folds allow), and so on. A cell is classified only if its '
     'selected model beats the constant rate (signed-rank against zero); otherwise it is '
     'unclassified. Hollow, a model the selection never fits. (C) Held-out gain of the selected '
     'model by class, every classified cell of every structure (boxes, quartiles; whiskers, 1.5 '
     'times the interquartile range; cells above). Conjunctive models carry the most per spike '
     '(median PH 0.10, PHS 0.09 bits/spike; P, H and S alone 0.02-0.03). (D) Four MEC cells, one '
     'each of P, H, S and PHS (the cell of B): rate maps in 2.5 cm bins, observed and predicted in '
     'held-out bins on one colour scale, and head-direction and speed tuning, observed (black) and '
     'predicted (grey). Each prediction comes from the cell\'s selected model alone, so the '
     'predicted head-direction and speed tuning of a P cell is what its position tuning and the '
     'mouse\'s path imply. (E-H) The share of each structure\'s cells in each class; grey, '
     'unclassified. All curated cells of the labelled sessions with at least 100 open-field spikes; '
     'cell and session counts above each panel. Units whose refractory-period violations rule out a '
     'neural origin (ISI violation ratio above 10) are excluded here as throughout; before that '
     'exclusion, these units supplied the strongest apparent position coding in the cerebellum '
     '(Methods). The cerebellar cells that remain in the position-only class do not carry '
     'cell-specific maps: a cell\'s map in the first open-field session correlates with other '
     'cerebellar cells\' maps in the second as well as with its own (median r 0.47 and 0.42; in '
     'MEC, 0.10 and 0.57), the shared map favours the centre of the arena, and position retains 6% '
     'of its gain once wall distance, running speed, angular head velocity and acceleration are '
     'modelled (MEC, 70%).'),
    ('Figure E2', 'fig3_supp_laminar.pdf',
     'Recording sites are biased toward superficial layers medially, and what that does to '
     'the mediolateral results, formerly with Figure 4',
     'MEC is curved and a four-shank probe is flat, so shanks at different mediolateral positions '
     'do not enter the tissue at the same depth relative to the layers. (A) They do not: '
     'superficial layers are 90.3% of entorhinal recording sites in the most medial quartile and '
     '59.8% in the most lateral (pooled rho = -0.267, p = 4e-80), a gradient present in every '
     'animal (within-mouse median rho = -0.447, p = 0.031). (B) The same bias in the neurons '
     'actually analysed (within-session median rho = -0.391, p = 0.035). (C, D) The mediolateral '
     'tests stratified by layer. The identity gradient does not survive within superficial cells '
     '(median rho = -0.054, p = 0.13, 24 sessions) against -0.078 (p = 0.028) across all layers; '
     'the following gradient does and is if anything stronger (-0.167, p = 0.009, 21 sessions '
     'against -0.094, p = 0.004). Deep layers are too sparsely sampled to test on their own.'),
    ('Figure E3', 'fig6_region_examples.pdf',
     'The same state, described separately by each structure, formerly with Figure 7',
     'Two sessions bracketing the range among those recording all three structures. (A, B) One '
     'session each, read left to right as entorhinal, subicular and visual cortex. Everything '
     "within a structure is built from that structure's own cells, the raster ordered by its own "
     'PC1 loading and the component signed to its own anchored fraction, so agreement between the '
     'groups is agreement between independent descriptions rather than a consequence of a shared '
     'frame.'),
    ('Figure E4', 'fig6_supp_region_examples.pdf',
     'The state outside MEC, where MEC does not dominate the sample, formerly with Figure 7',
     'Four sessions chosen for their sampling rather than their result. (A, B) Two visual-rich '
     'sessions. (C) A session in which visual cortex contributes more cells than entorhinal and '
     'disagrees with it. (D) A session in which the subicular complex does. Each row is built as '
     'in Figure E3, every structure described by its own cells. A cerebellar session was removed: '
     'it rested on 15 to 18 cells, too few for an axis to be estimated with confidence.'),
]


def main():
    src = open(f'{HERE}/EXTRA.md').read()
    intro = src.split('## Methods')[0].split('\n\n', 1)[1].strip()
    doc = B.Doc()
    doc.y = B.MARGIN_TOP
    doc.full = True
    doc.para('Extra analyses', font=B.AVD, size=15.0, lead=18.0, gap=10)
    doc.para(B.TITLE, font=B.SERIF_I, size=8.5, lead=11, gap=8)
    doc.para(intro, size=B.BODY, lead=B.BODY_LEAD, gap=8)
    doc.full = False
    doc.col_top = doc.y
    meth = B.parse_methods(f'{HERE}/EXTRA.md')
    for kind, text in meth:
        if kind == 'h1':
            doc.heading('METHODS', size=B.H1, font=B.SANS_B, color=B.RULE, pre=10)
        elif kind == 'h2':
            doc.heading(text, size=B.H2 - .4, font=B.SANS_B, pre=6)
        else:
            doc.para(text, size=B.BODY - .3, lead=B.BODY_LEAD - .5)
    cited = set()
    for t in [t for _, t in meth] + [' '.join(f) for f in EXTRA_FIGURES]:
        cited |= {k for k in B.cite_keys(t) if k in B.REFS}
    doc.heading('REFERENCES', size=B.H1, font=B.SANS_B, color=B.RULE, pre=10)
    for k in sorted(cited, key=lambda k: B.REFS[k].lower()):
        doc.para(B.REFS[k], size=7.4, lead=9.2, gap=2.0, indent=6)
    doc.heading('FIGURES', size=B.H1, font=B.SANS_B, color=B.RULE, pre=10)
    for lab, path, ttl, cap in EXTRA_FIGURES:
        p = f'{HERE}/{path}'
        if not os.path.exists(p):
            print(f'  ! {path} missing, skipped')
            continue
        d_ = pymupdf.open(p)
        doc._supp_figure(lab, d_, ttl, cap)
        d_.close()
    doc.d.save(OUT, deflate=True)
    print(f'  methods: {sum(1 for k, _ in meth if k == "h2")} sections; references: {len(cited)}; '
          f'figures: {len(EXTRA_FIGURES)}')
    print(f'wrote {OUT}  ({doc.d.page_count} pages)')


if __name__ == '__main__':
    main()
