# MEC shank-reconstruction — population statistics

Built while you were asleep. `population_stats.py` aggregates every cached session
(`data/xgboost_shank_anchoring/M*D*_shank_reconstruction.pkl`) into a tidy table
(`population_stats.csv`, one row per **target × covariate-shank**) and produces two
figure sheets + a stats report in `figures/`.

Shanks are ranked **medial → lateral** within each session by mean `|coord_SCs_x|`
(so it's ML-correct even for reverse-implanted probes). Same-shank predictions use
leave-target-out.

> **Status when written:** only **3 of 6** multi-shank switchers were cached
> (M25D25, M28D16, M28D17 — 173 targets, 844 rows). M29D17 / M29D25 / M26D19 were
> still running. **Re-run `python population_stats.py`** once they finish — everything
> updates automatically.

---

## Your two questions — answers so far

### Q1. Does mediolateral shank position affect reconstruction?  → **Yes, for the spike code; not really for the anchoring state.**

`figures/population_ml_reconstruction.png`

- **Reconstruction is local.** Same-shank pR² = **0.062** vs other-shank **0.040**, and pR²
  falls monotonically with mediolateral distance (Spearman ρ = **−0.21**, p = 2e-9;
  mixed-model slope **−0.012 / shank-step**, p = 3e-16). The target×covariate matrix is
  diagonal-dominant.
- **But the anchoring *label* is global.** The predicted-vs-observed anchoring agreement
  (MCC) does **not** significantly decay with ML distance (ρ = −0.04, p = 0.22).
  → Interpretation worth highlighting: **fine spike prediction is local (ML-dependent),
  but the trial-by-trial anchoring *state* can be read out about equally well from any
  shank** — consistent with anchoring being a population/session-wide variable.
- Directional panel: pR² peaks at the same shank and is roughly symmetric medial vs
  lateral (no strong side asymmetry with these 3 sessions).

### Q2. pR² anchored vs non-anchored  → **Anchored reconstructs better.**

`figures/population_anch_vs_nonanch.png`

- Median pR² **anchored 0.026 > non-anchored 0.018**; most target×shank points sit above
  the y=x line. Mixed model: non-anchored is **−0.011** lower (p = 4e-6). The gap holds
  for **same and other shanks** and at **every ML distance**.
- Per-session medians all point the same way (n = 3 → Wilcoxon p = 0.25, just underpowered;
  will firm up with 6 sessions).
- Reading: during anchored (task-locked) epochs the population is more coherent, so one
  cell is more predictable from its neighbours.

---

## Good ways to display medial↔lateral shank reconstruction

**Built (in the two figure sheets):**
1. **Target × covariate-shank matrix** (medial→lateral both axes), coloured by mean pR²
   and by label-MCC — the single clearest view of ML structure (diagonal = local).
2. **pR² / MCC vs |ML distance|** decay curves (overall / anchored / non-anchored).
3. **Directional (signed ML)** — medial vs lateral covariates relative to the target.
4. **Anchored-vs-non paired scatter + Δ violin + Δ by shank relation + anch/non decay.**

**Further ideas worth adding (say the word):**
5. **Per-session small-multiples of the matrix** — check the diagonal holds in every animal.
6. **ΔpR² relative to the same-shank baseline** (`pR²(cov) − pR²(same-shank)` per target) —
   normalises out each cell's overall predictability and isolates the *cost of ML distance*.
7. **Continuous ML displacement in µm** (not shank steps) on the x-axis, using `dml_abs`.
8. **Split targets by medial vs lateral position** — are medial-MEC targets more reconstructable?
9. **Grid vs non-grid targets** (needs the cell-type merge) — does grid anchoring behave differently?
10. **Qualitative label rasters**: observed vs predicted anchoring side-by-side, ordered by
    shank distance, for a few example cells.

---

## Caveats
- **Pseudoreplication**: 844 rows are not independent (many per target, many targets per
  session). The pooled Wilcoxons (p≈1e-28) are inflated — trust the **mixed model** and the
  **per-session medians** instead. With only 3 sessions the session-level tests are underpowered.
- The **|distance| = 3** bin is sparse (few shank pairs, one session) — the uptick there is noisy.
- Covariates are the Figure-4 setup: 16-cell stratified ENT+PARA sample, 5-filter 1000 ms
  history, 10-fold CV.

## Re-run
```
cd scripts/figures/figure_6_task_anchoring/xgboost_shank_anchoring
python population_stats.py      # rebuilds CSV + both figure sheets + report on all cached sessions
```
Outputs: `figures/population_ml_reconstruction.{png,pdf}`,
`figures/population_anch_vs_nonanch.{png,pdf}`,
`figures/population_stats_report.txt`, `data/xgboost_shank_anchoring/population_stats.csv`.
