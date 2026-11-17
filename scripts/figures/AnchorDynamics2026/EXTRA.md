# Extra analyses

Analyses removed from the AnchorDynamics2026 manuscript on 2026-11-01, kept here because they may serve another paper: the open-field decomposition of each structure's cells, the laminar sampling of the mediolateral results, the region-by-region example sessions, and every analysis that used gradient-boosted trees (XGBoost). Figure numbers E1-E4 replace the paper's former Figures S5, S8, S11 and S12.

## Methods

### Decomposition of open-field coding into position, head direction and speed

Each cell's open-field firing was decomposed into coding of position (P), head direction (H) and speed (S), or a combination of them, following the model-selection procedure of Hardcastle et al. (2017) with gradient-boosted tree models (XGBoost) (Chen and Guestrin, 2016) in place of their linear–nonlinear (LN) Poisson models. The analysis used the first open-field session of each recording day, the session from which cell classes are assigned.

**Data.** Spike counts were taken in 20 ms bins. Position (x, y), head direction and speed were interpolated from the tracking (~16 Hz) to the bin centres. Head direction entered as its sine and cosine, because a tree cannot split across the wrap at ±π. Theta phase, Hardcastle's fourth variable, was not included.

**Models.** For each combination of variables, spike counts were modelled with a Poisson gradient-boosted tree ensemble (`count:poisson`; learning rate 0.1, minimum leaf weight 10, row subsampling 0.8). Tree depth equalled the model's number of input columns (P 2, H 2, S 1; PH 4, PS 3, HS 3, PHS 5). This is the greatest depth that enlarges what the ensemble can represent. A root-to-leaf path of depth *d* tests at most min(*d*, *p*) of *p* columns, and a sum of depth-*p* trees already spans every piecewise-constant function on a grid of thresholds: the indicator of a box is a product of *p* interval indicators, which expands into 2*ᵖ* products of one-sided steps, each a depth-*p* tree. Depth beyond *p* therefore changes how many trees a fit needs, not what it can fit. The number of trees was set by early stopping inside the training data, the role played by the roughness penalty in the LN model. Of the nine training folds, one was held out for validation, trees were added (up to 1,000) until 30 successive trees failed to improve the validation Poisson likelihood, and the model at the best iteration was kept. This limit was set long enough that shallow models, which need many trees, were not stopped before converging.

**Cross-validation and performance.** As in Hardcastle et al. (2017), the session was divided into 50 contiguous sections and each of 10 folds took every tenth section, so that every fold sampled the whole session. For each fold, a model was fitted on the other nine and its performance was the increase in test-fold Poisson log-likelihood over a constant-rate model fitted to the same training data, in bits per spike.

**Model selection.** Selection followed Hardcastle et al. (2017). The single-variable model with the highest mean performance was taken first. The best model adding one further variable replaced it if its performance was greater across the 10 folds (one-sided Wilcoxon signed-rank test, *p* < 0.05), and this was repeated until no addition improved the fit or all three variables were included. A cell was assigned the selected model only if that model's performance exceeded the constant-rate model across folds (one-sided signed-rank test against zero, *p* < 0.05); otherwise it was left unclassified. Only the models visited by the selection were fitted. Every comparison is between test-fold likelihoods, so a model that gains only by fitting noise performs worse, not better, and is not selected.

**Cells.** All curated cells (grid, non-grid spatial and non-spatial) with at least 100 open-field spikes in the sessions carrying anchoring labels were decomposed, grouped by the region families of Figures S3 and S4 (MEC, visual cortex, the subicular complex and cerebellum).

Implementation: `of_xgb_decomp.py` (run by `run_of_xgb.sh`), writing `data/of_xgb/of_xgb_cells.csv`.

> **NOTE — depth.** Depth = number of input columns is the representational bound. In test cells, models fitted at that depth still scored below the same models at twice the depth (position alone 0.155 vs 0.177 bits/spike, cell 99 of M26 D18) even with the 1,000-tree, 30-round limit, because greedy boosting rarely assembles the 2*ᵖ* cooperating trees a box needs. Depth = 2 × columns (one tree can isolate a box with two splits per variable) was considered and not adopted. If class fractions look conservative, this is the first thing to revisit.


### Speed coding with boosted trees, and the earlier per-state slopes

These were the paper's speed analyses before the per-cell Poisson model replaced them.

**Speed tuning of entorhinal cells.** Firing rate and running speed were binned at 250 ms and restricted to running bins (≥ 3 cm/s), each bin taking the anchoring state of its trial. All curated entorhinal cells were used (grid, non-grid spatial and narrow-spiking). Speed cells were defined pooled across states, so that the definition could not itself be state-biased, as cells whose rate–speed correlation exceeded a circular-shift null of the binned rate (200 shifts, p < 0.05, positive); this null preserves the autocorrelation of firing, which makes a parametric p meaningless at this bin size. Per cell and state we computed the regression slope of rate on speed (Hz per cm/s), the gain (slope divided by mean rate, which separates a change in speed sensitivity from a change in overall firing), and the correlation, each on all running bins and again after trimming both states to their common 5–95% speed support. Sessions, not cells, are the unit of analysis: cells within a session share one behavioural state, so a cell-level test would count the same state change hundreds of times.

**Unexpected speed.** Because anchored trials have both more reliable position coding and a more stereotyped speed profile, position can stand in for speed and inflate any apparent speed coding. Position was therefore partialled out of both sides: within each state, cyclic position (sin and cos of 2π·position/L) was used to predict speed and, separately, each cell's spike count, both cross-validated (three folds; xgboost, 50 rounds at depth 3, `count:poisson` for rate and squared error for speed). The residuals — unexpected speed and unexpected rate — were then regressed on one another. The speed residual is orthogonal to position by construction, so a difference between states cannot arise from position substituting for speed. Sensitivity is reported both per cm/s (physical gain) and per SD of residual speed, because the two answer different questions when the states differ in how much unexpected speed they contain.

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


## Open items
