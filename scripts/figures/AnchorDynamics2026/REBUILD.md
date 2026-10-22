# Rebuilding the AnchorDynamics2026 figures and paper

This folder holds every script behind the paper *AnchorDynamics2026* (the
mock-up PDF is `AnchorDynamics2026_Cell_mockup.pdf`). This guide explains how
to rebuild the figures and the PDF on your own machine from the shared data.

One script does the work: `rebuild_paper.py`. The steps below are all you
need, in order.

---

## 1. What you need

| What | Where it comes from | Needed for |
|---|---|---|
| **clark2025** | DataStore (Harry will give you the path). 81 VR sessions, each with `VR/`, `OF1/`, `OF2/` and an `anatomy_M*_D*.csv`. | everything |
| **clark2025_lfp** | DataStore, alongside clark2025. Same sessions, with the raw LFP. | Figure 8 and Figure S9 only, plus the LFP steps of `derive` |
| **Allen mouse atlas, 10 µm** | Public. Install with `pip install brainglobe-atlasapi` then `brainglobe install -a allen_mouse_10um`. It lands in `~/.brainglobe/allen_mouse_10um_v1.2` (about 5 GB). | anatomy figures (3, 4, S5, S15) |
| **This repository** | `git clone https://github.com/HDClark94/spatial-manifolds.git`, branch `spatial` | everything |
| **Python 3.11** with the packages in `requirements-paper.txt` (in the repository root) | `pip install -r requirements-paper.txt` | everything |

Disk: the repository is a few GB because it carries the intermediate results.
Copying clark2025 locally is faster than reading it over the network, but
either works.

### Things that are *not* in clark2025, and are in the repository instead

Almost everything is computed from clark2025 and clark2025_lfp. A few inputs
cannot be, so they are committed to the repository. `rebuild_paper.py check`
confirms they are there:

- **RNN simulation caches** for Figure S1 (`data/rnn_xgboost/net7_1D_track_sim/`).
  Figure S1 validates the anchoring classifier on a simulated grid network, not
  on recorded data.
- **Hand-drawn schematics**: `fig1_schem.pdf` (Figure 1) and
  `camera_and_IR_mouse.pdf` and `pupilometry.pdf` (Figure 6).
- **Eye-video stills** for Figure 6 (`eye_frames_M21D19_trials.npz`). The pupil
  *measurements* come from clark2025; only these example frames come from raw
  videos, which are not part of the dataset.
- **First-pass anatomy table** `data/cell_classifications.csv`, which the
  classification step reads alongside clark2025.
- **3D brain renders** for Figure 3 (`data/anatomy_renders/*.png`). These are
  only re-rendered if you delete them, which needs `brainrender` and a display.

---

## 2. Point the scripts at your data (once)

The scripts were written on Harry's laptop and contain his paths. This command
replaces them with yours, everywhere they occur:

```bash
cd scripts/figures/AnchorDynamics2026
python rebuild_paper.py configure \
    --clark2025 /path/to/clark2025 \
    --lfp       /path/to/clark2025_lfp \
    --atlas     ~/.brainglobe/allen_mouse_10um_v1.2
```

Add `--dry-run` first if you want to see what it will change. It edits text
inside the scripts' string literals only, and records what it did in
`rebuild_paths.json`, so you can run it again later if the data moves.

## 3. Check

```bash
python rebuild_paper.py check
```

Every line should say `ok`. If you don't have clark2025_lfp yet, use
`check --skip-lfp`. A missing Arial font is only a note: the figures still
build, in a fallback font. To fix it on Linux, install the Microsoft core
fonts (`ttf-mscorefonts-installer`) and delete `~/.cache/matplotlib`.

## 4. Rebuild the figures and the paper

```bash
python rebuild_paper.py figures
```

This redraws every figure from the intermediate results in the repository and
then compiles `AnchorDynamics2026_Cell_mockup.pdf`. It takes under an hour.
Without clark2025_lfp, add `--skip-lfp`; Figures 8 and S9 then keep whatever
version is already in the folder.

Each step writes a log to `rebuild_logs/<step>.log`. A failed step doesn't
stop the rest, and the summary at the end lists anything that failed. To
re-run one step: `python rebuild_paper.py figures --only fig3_4`. To resume
from a step: `--from fig7`.

## 5. (Optional) Recompute the intermediate results from the raw data

```bash
python rebuild_paper.py derive          # then: python rebuild_paper.py figures
```

Step 4 uses the intermediate results already committed to the repository:
anchoring labels, the unit table, LFP spectra and so on. `derive` recomputes
all of them from clark2025 and clark2025_lfp, in order, so you can confirm the
committed versions follow from the data. It overwrites them in place, so
commit or copy `data/` first if you want to compare. It is slow, most of a
day, almost all of it the LFP steps (`--skip-lfp` leaves those out).
`monosyn` alone takes several hours. Steps can be resumed with `--from`.

---

## Which script makes which figure

| Figure | Script |
|---|---|
| 1 | `figure1_schematics.ipynb` (panels), then `fig1_composite.py` |
| 2 | `fig2_single_units.py` |
| 3 and 4 | `fig3_anatomy.py` (one script, two figures) |
| 5 | `fig4_monosynaptic.py` |
| 6 | `fig5_pupil_arousal.py` |
| 7 | `fig2_supp_vis_anchored.py` |
| 8 | `fig7_theta_gamma.py` |
| 9 | `fig8_model.py` |
| S1–S3 | `fig1_supp_classifier.py`, `fig1_supp_rasters.py`, `fig1_supp_speedmatch.py` |
| S4–S5 | `fig2_supp_nonanchored_rate.py`, `fig3_supp_laminar.py` |
| S7–S8 | `fig6_supp_region_examples.py`, `region_state_dimensionality.py` |
| S9–S12 | `fig7_supp_band_examples.py`, `fig7_supp_comodulogram.py`, `fig7_supp_gamma_locking.py`, `speed_tuning_fig.py` |
| S13–S16 | `fig8_supp_can_model.py`, `fig8_supp_attractor_state.py`, `fig7_supp_vis_examples.py`, `fig8_supp_tsc.py` |
| paper PDF | `build_cell_pdf.py` (text from `MANUSCRIPT.md` and `PAPER.md`, captions inside the script) |

The file names lag the figure numbers because the figures were renumbered
during writing. Each script's opening docstring explains what it computes and
why.

## If something goes wrong

- **A path error naming `/Users/harryclark/...`**: a path the `configure` step
  doesn't know about. Search for it (`grep -rn "/Users/harryclark" .`) and tell
  Harry. The ones that are expected to remain are only used when a cached
  input is missing (eye videos, old exploratory notebooks).
- **`check` reports clark2025 with 0 sessions**: the path should be the folder
  that contains `M20/`, `M21/`, ...
- **Figure 3 tries to open a 3D window**: a render PNG is missing from
  `data/anatomy_renders/`; restore it from git rather than re-rendering.
- **Numbers in a rebuilt figure differ slightly from the PDF**: if you ran
  `derive`, shuffles are seeded, so differences should be at most at the last
  printed digit. Larger ones are worth telling Harry about.
