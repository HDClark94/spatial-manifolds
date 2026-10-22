"""Rebuild the AnchorDynamics2026 figures and paper from the shared datasets.

    python rebuild_paper.py configure --clark2025 /path/to/clark2025 \
                                      --lfp /path/to/clark2025_lfp \
                                      [--atlas ~/.brainglobe/allen_mouse_10um_v1.2]
    python rebuild_paper.py check
    python rebuild_paper.py derive  [--skip-lfp] [--from STEP] [--only STEP]
    python rebuild_paper.py figures [--skip-lfp] [--from STEP] [--only STEP]
    python rebuild_paper.py all     [--skip-lfp]

Read REBUILD.md first; it says what each stage needs and how long it takes.

WHAT EACH COMMAND DOES

  configure  The analysis scripts were written on one laptop and carry its
             paths as literal strings ('/Users/harryclark/...'). This replaces
             those prefixes, in every script and notebook the paper uses, with
             the locations on your machine: the repository itself (found
             automatically), clark2025, clark2025_lfp and the Allen atlas. It is
             a plain text substitution inside string literals, so it cannot
             break the code, and the current mapping is kept in
             rebuild_paths.json so it can be run again to move things later.

  check      Confirms the datasets, the atlas, the Python packages and the
             cached inputs that cannot be derived from clark2025 are present.
             Run it before anything else; it changes nothing.

  derive     Recomputes every intermediate result the figures read (anchoring
             labels, unit table, LFP spectra, ...) from clark2025 and
             clark2025_lfp, in dependency order. Slow: most of a day, mostly
             the LFP stages. Not needed to redraw the figures, because the
             repository already holds these results; run it to check that they
             follow from the data.

  figures    Redraws every figure from the intermediate results and compiles
             AnchorDynamics2026_Cell_mockup.pdf. Under an hour.

Every step writes a log to rebuild_logs/, and a failed step does not stop the
others; the summary at the end lists what failed and where its log is.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PATHS_JSON = os.path.join(HERE, 'rebuild_paths.json')
LOGS = os.path.join(HERE, 'rebuild_logs')

# The paths as they were on the machine the analysis was written on. configure
# replaces these (or whatever a previous configure set) with the new ones.
ORIGINAL = {
    'repo': '/Users/harryclark/Documents/spatial-manifolds',
    'clark2025': '/Users/harryclark/Downloads/clark2025',
    'lfp': '/Volumes/datasetsSD/clark2025_lfp',
    'lfp_alt': '/Volumes/INCR-NolanLab/ActiveProjects/Wolf/harry/data/clark2025_lfp',
    'atlas': '/Users/harryclark/.brainglobe/allen_mouse_10um_v1.2',
}

# Files whose string literals carry those paths.
def _target_files():
    fs = glob.glob(os.path.join(HERE, '*.py')) + glob.glob(os.path.join(HERE, '*.ipynb'))
    fs += glob.glob(os.path.join(REPO, 'src', 'spatial_manifolds', '**', '*.py'), recursive=True)
    fs += glob.glob(os.path.join(REPO, 'scripts', 'figures', 'figure_0_classification', '*.ipynb'))
    return [f for f in fs if os.path.basename(f) != 'rebuild_paper.py']


# ---------------------------------------------------------------------------
# The pipeline. Each step: (name, kind, file, args, needs_lfp, what it makes)
# kind is 'py' (run as a script) or 'nb' (executed headless with nbconvert).
# Order matters: every step reads only what the steps above it wrote.
# ---------------------------------------------------------------------------
DERIVE = [
    ('contacts', 'py', 'build_device_contacts.py', [], False,
     'every probe contact in atlas coordinates (from clark2025 anatomy_*.csv)'),
    ('classification', 'nb', '../figure_0_classification/create_classification_datasheet_v2.ipynb', [], False,
     'cell_classifications_v2.csv: open-field identity, scores and anatomy per cell'),
    ('labels', 'py', 'build_anchoring_labels.py', [], False,
     'per-cell, per-trial anchoring labels and the population state'),
    ('labels_merge', 'merge', None, [], False,
     'anchoring_cells.csv / anchoring_trials.csv from the per-worker files'),
    ('per_cell_pc1', 'py', 'build_per_cell_pc1.py', [], False,
     'each cell\'s agreement with the population axis'),
    ('heldout_pc1', 'py', 'build_heldout_pc1.py', [], False, 'held-out PC1'),
    ('trial_table', 'py', 'build_trial_table.py', [], False,
     'behaviour joined to the anchoring state, per trial'),
    ('of_speed', 'py', 'of_speed_score.py', [], False, 'open-field speed scores'),
    ('of_theta', 'py', 'of_theta_index.py', [], False, 'open-field theta index'),
    ('unit_table', 'py', 'build_unit_table.py', [], False,
     'one row per cell: identity, waveform, interneuron call, agreement'),
    ('trial_maps', 'py', 'build_trial_maps.py', [], False, 'per-cell trial-map summaries'),
    ('adi', 'py', 'adi_by_identity.py', [], False, 'anchoring dependence index'),
    ('hit_rate', 'py', 'hit_rate_chance.py', [], False, 'hit rate against chance'),
    ('fig4_controls', 'py', 'fig4_controls.py', [], False, 'Figure 4 controls'),
    ('fig4_factors', 'py', 'fig4_factors.py', [], False, 'Figure 4 I-K mixed model'),
    ('attractor', 'py', 'attractor_coherence.py', [], False, 'attractor bump (Figure S14)'),
    ('monosyn', 'py', 'monosyn.py', [], False, 'monosynaptic pairs (Figure 5); slow'),
    ('speed_tuning', 'py', 'speed_tuning.py', [], False, 'speed tuning'),
    ('dimensionality', 'py', 'region_state_dimensionality.py', [], False, 'Figure S8'),
    ('decoding', 'nb', 'position_decoding_by_anchoring.ipynb', [], False,
     'position decoding by state'),
    ('eye', 'nb', 'figure5_pupil_arousal.ipynb', [], False, 'pupil tables and profiles'),
    ('session_validity', 'nb', 'session_validity_pc1_significance.ipynb', [], False,
     'per-session PC1 significance'),
    # --- everything below reads clark2025_lfp ---
    ('phase_precession', 'py', 'phase_precession.py', [], True, 'phase precession'),
    ('lfp_power', 'py', 'lfp_batch.py', [], True, 'LFP power by state'),
    ('spectrum_pac', 'py', 'lfp_spectrum_pac.py', [], True, 'spectra and PAC by state'),
    ('gamma_events', 'py', 'gamma_events.py', [], True, 'gamma by position'),
    ('gamma_locking', 'py', 'gamma_locking.py', [], True, 'spike-gamma locking'),
    ('gamma_decode', 'py', 'gamma_vs_decode_error.py', [], True, 'gamma vs decoding error'),
    ('theta_comod', 'py', 'theta_phase_comodulogram.py', [], True, 'comodulogram'),
    ('theta_depth', 'py', 'theta_depth_profile.py', [], True, 'theta depth profile'),
    ('theta_residual', 'py', 'theta_residual.py', [], True, 'theta vs speed residuals'),
    ('theta_extra', 'py', 'theta_extra.py', [], True, 'theta frequency extras'),
    ('tsc_extract', 'py', 'tsc_extract.py', [], True, 'theta-cycle spectral maps'),
    # these two read tsc_extract's output, so they go with the LFP stages
    ('tsc_analyse', 'py', 'tsc_analyse.py', [], True, 'tSC components'),
    ('tsc_neural', 'py', 'tsc_neural.py', [], True, 'tSC vs decoding'),
    # simulations; they need no data
    ('can_model', 'py', 'can_gate_model.py', [], False, 'gated attractor model'),
    ('can_model_2d', 'py', 'can_gate_model_2d.py', [], False, '2D gated attractor model'),
]

FIGURES = [
    ('fig1_panels', 'nb', 'figure1_schematics.ipynb', [], False, 'Figure 1 panels'),
    ('fig1', 'py', 'fig1_composite.py', [], False, 'Figure 1'),
    ('figS1', 'py', 'fig1_supp_classifier.py', [], False, 'Figure S1 (needs the RNN caches)'),
    ('figS2', 'py', 'fig1_supp_rasters.py', [], False, 'Figure S2'),
    ('figS3', 'py', 'fig1_supp_speedmatch.py', [], False, 'Figure S3'),
    ('fig2', 'py', 'fig2_single_units.py', [], False, 'Figure 2'),
    ('figS4', 'py', 'fig2_supp_nonanchored_rate.py', [], False, 'Figure S4'),
    ('fig3_4', 'py', 'fig3_anatomy.py', [], False, 'Figures 3 and 4'),
    ('figS5', 'py', 'fig3_supp_laminar.py', [], False, 'Figure S5'),
    ('fig5', 'py', 'fig4_monosynaptic.py', [], False, 'Figure 5'),
    ('fig6', 'py', 'fig5_pupil_arousal.py', [], False, 'Figure 6 (pupil)'),
    ('fig7', 'py', 'fig2_supp_vis_anchored.py', [], False, 'Figure 7'),
    ('figS7', 'py', 'fig6_supp_region_examples.py', [], False, 'Figure S7'),
    ('figS8', 'py', 'region_state_dimensionality.py', [], False, 'Figure S8'),
    ('fig_regions', 'py', 'fig6_region_examples.py', [], False, 'region examples'),
    ('figS15', 'py', 'fig7_supp_vis_examples.py', [], False, 'Figure S15'),
    ('figS14', 'py', 'fig8_supp_attractor_state.py', [], False, 'Figure S14'),
    ('fig8', 'py', 'fig7_theta_gamma.py', [], True, 'Figure 8 (theta-gamma)'),
    ('figS9', 'py', 'fig7_supp_band_examples.py', [], True, 'Figure S9'),
    ('figS10', 'py', 'fig7_supp_comodulogram.py', [], False, 'Figure S10'),
    ('figS11', 'py', 'fig7_supp_gamma_locking.py', [], False, 'Figure S11'),
    ('figS12', 'py', 'speed_tuning_fig.py', [], False, 'Figure S12'),
    ('figS16', 'py', 'fig8_supp_tsc.py', [], False, 'Figure S16'),
    ('fig9', 'py', 'fig8_model.py', [], False, 'Figure 9 (model)'),
    ('figS13', 'py', 'fig8_supp_can_model.py', [], False, 'Figure S13'),
    ('graphical_abstract', 'py', 'graphical_abstract.py', [], False, 'graphical abstract'),
    ('paper', 'py', 'build_cell_pdf.py', [], False, 'compile AnchorDynamics2026_Cell_mockup.pdf'),
]

# Inputs that clark2025 / clark2025_lfp cannot regenerate. They are committed
# to the repository; check() confirms they are present.
NOT_DERIVABLE = [
    ('data/cell_classifications.csv', 'first-pass anatomy per cell, read by the classification notebook'),
    ('data/rnn_xgboost/net7_1D_track_sim/sim_cache.npz', 'RNN simulation, Figure S1'),
    ('data/rnn_xgboost/net7_1D_track_sim/switch_cache.npz', 'RNN simulation, Figure S1'),
    ('data/rnn_xgboost/net7_1D_track_sim/of_ratemaps_cache.npz', 'RNN simulation, Figure S1'),
    ('scripts/figures/AnchorDynamics2026/fig1_schem.pdf', 'Figure 1 schematic (drawn by hand; source fig1_schem.svg)'),
    ('scripts/figures/AnchorDynamics2026/camera_and_IR_mouse.pdf', 'Figure 6 apparatus drawing'),
    ('scripts/figures/AnchorDynamics2026/pupilometry.pdf', 'Figure 6 apparatus drawing'),
    ('scripts/figures/AnchorDynamics2026/eye_frames_M21D19_trials.npz', 'Figure 6 eye-video stills (videos are not in clark2025)'),
    ('data/anatomy_renders/brain_all.png', 'Figure 3 3D brain render (brainrender; regenerated only if deleted)'),
]


def load_paths():
    if os.path.exists(PATHS_JSON):
        return json.load(open(PATHS_JSON))
    return dict(ORIGINAL)


def cmd_configure(a):
    cur = load_paths()
    new = dict(cur)
    new['repo'] = REPO
    if a.clark2025:
        new['clark2025'] = os.path.abspath(os.path.expanduser(a.clark2025)).rstrip('/')
    if a.lfp:
        new['lfp'] = new['lfp_alt'] = os.path.abspath(os.path.expanduser(a.lfp)).rstrip('/')
    if a.atlas:
        new['atlas'] = os.path.abspath(os.path.expanduser(a.atlas)).rstrip('/')
    # longest first, so a path is never half-replaced by a shorter one inside it
    pairs = sorted(((cur[k], new[k]) for k in cur if cur[k] != new[k]),
                   key=lambda p: -len(p[0]))
    if not pairs:
        print('paths already configured:'); print(json.dumps(new, indent=1)); return
    n_files = n_hits = 0
    for f in _target_files():
        t = open(f, encoding='utf-8').read()
        t2 = t
        for old, nw in pairs:
            t2 = t2.replace(old, nw)
        if t2 != t:
            hits = sum(t.count(o) for o, _ in pairs)
            n_files += 1; n_hits += hits
            if not a.dry_run:
                open(f, 'w', encoding='utf-8').write(t2)
    print(f'{"would rewrite" if a.dry_run else "rewrote"} {n_hits} paths in {n_files} files:')
    for old, nw in pairs:
        print(f'   {old}\n -> {nw}')
    if not a.dry_run:
        json.dump(new, open(PATHS_JSON, 'w'), indent=1)


def cmd_check(a):
    P = load_paths()
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= good
        print(('  ok   ' if good else '  MISSING ') + msg)

    print('datasets')
    n = len(glob.glob(os.path.join(P['clark2025'], 'M*', 'D*', 'VR')))
    say(n > 0, f'clark2025 at {P["clark2025"]}: {n} VR sessions (expect 81)')
    na = len(glob.glob(os.path.join(P['clark2025'], 'M*', 'D*', 'anatomy_M*_D*.csv')))
    say(na > 0, f'clark2025 anatomy files: {na} (expect 81)')
    nl = len(glob.glob(os.path.join(P['lfp'], 'M*', 'D*', 'VR')))
    say(nl > 0 or a.skip_lfp, f'clark2025_lfp at {P["lfp"]}: {nl} VR sessions'
        + (' (skipped: --skip-lfp)' if a.skip_lfp and not nl else ''))
    say(os.path.exists(os.path.join(P['atlas'], 'annotation.tiff')),
        f'Allen atlas annotation.tiff under {P["atlas"]} (brainglobe allen_mouse_10um)')
    if P['repo'] != REPO:
        say(False, f'repository paths still point at {P["repo"]} -- run configure first')
    print('inputs not derivable from clark2025 (committed to the repository)')
    for rel, what in NOT_DERIVABLE:
        say(os.path.exists(os.path.join(REPO, rel)), f'{rel}  ({what})')
    print('python packages')
    for mod in ('numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
                'statsmodels', 'pynapple', 'pymupdf', 'umap', 'tifffile', 'h5py',
                'nbconvert', 'xgboost'):
        try:
            __import__(mod); say(True, mod)
        except Exception as e:
            say(False, f'{mod} ({type(e).__name__})')
    try:
        from matplotlib import font_manager
        f = font_manager.findfont('Arial', fallback_to_default=False)
        say(True, f'Arial font ({os.path.basename(f)})')
    except Exception:
        print('  note    Arial not installed: figures render in a fallback font')
    print('\nall present' if ok else '\nsomething is missing -- see above')
    return ok


def _run_step(step, a):
    name, kind, fn, args, lfp, what = step
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f'{name}.log')
    if lfp and a.skip_lfp:
        return name, 'skipped (LFP)', 0.0, log
    t0 = time.time()
    if kind == 'merge':
        try:
            import pandas as pd
            ps = os.path.join(REPO, 'data', 'population_state')
            for stem in ('anchoring_cells', 'anchoring_trials'):
                ws = sorted(glob.glob(os.path.join(ps, f'{stem}_w*.csv')))
                pd.concat([pd.read_csv(w) for w in ws], ignore_index=True).to_csv(
                    os.path.join(ps, f'{stem}.csv'), index=False)
            open(log, 'w').write('merged\n')
            return name, 'ok', time.time() - t0, log
        except Exception as e:
            open(log, 'w').write(repr(e))
            return name, 'FAILED', time.time() - t0, log
    path = os.path.normpath(os.path.join(HERE, fn))
    if kind == 'py':
        cmd = [sys.executable, path] + list(args)
    else:
        out_dir = os.path.join(LOGS, 'executed_notebooks')
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
               '--execute', '--ExecutePreprocessor.timeout=-1',
               '--output-dir', out_dir, path]
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=os.path.join(REPO, 'src')
               + os.pathsep + os.environ.get('PYTHONPATH', ''))
    with open(log, 'w') as lf:
        lf.write(' '.join(cmd) + '\n\n'); lf.flush()
        r = subprocess.run(cmd, cwd=os.path.dirname(path), stdout=lf,
                           stderr=subprocess.STDOUT, env=env)
    return name, 'ok' if r.returncode == 0 else 'FAILED', time.time() - t0, log


def _run(steps, a, label):
    names = [s[0] for s in steps]
    if a.only:
        steps = [s for s in steps if s[0] in a.only.split(',')]
    elif a.from_:
        if a.from_ not in names:
            raise SystemExit(f'unknown step {a.from_}; steps: {", ".join(names)}')
        steps = steps[names.index(a.from_):]
    print(f'{label}: {len(steps)} steps' + ('  (LFP steps skipped)' if a.skip_lfp else ''))
    res = []
    for st in steps:
        print(f'  {st[0]:<18} {st[5]} ...', end='', flush=True)
        r = _run_step(st, a)
        print(f' {r[1]} ({r[2] / 60:.1f} min)')
        res.append(r)
    bad = [r for r in res if r[1] == 'FAILED']
    print(f'\n{label}: {len(res) - len(bad)} of {len(res)} steps ok')
    for r in bad:
        print(f'  FAILED {r[0]} -- see {r[3]}')
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('configure'); c.add_argument('--clark2025'); c.add_argument('--lfp')
    c.add_argument('--atlas'); c.add_argument('--dry-run', action='store_true')
    for nm in ('check', 'derive', 'figures', 'all'):
        p = sub.add_parser(nm)
        p.add_argument('--skip-lfp', action='store_true')
        p.add_argument('--from', dest='from_'); p.add_argument('--only')
    a = ap.parse_args()
    if a.cmd == 'configure':
        cmd_configure(a)
    elif a.cmd == 'check':
        sys.exit(0 if cmd_check(a) else 1)
    elif a.cmd == 'derive':
        _run(DERIVE, a, 'derive')
    elif a.cmd == 'figures':
        _run(FIGURES, a, 'figures')
    elif a.cmd == 'all':
        if not cmd_check(a):
            raise SystemExit('fix the missing inputs first')
        _run(DERIVE, a, 'derive') and None
        _run(FIGURES, a, 'figures')


if __name__ == '__main__':
    main()
