'''Take local copies of the eye videos, so Figures 7 and S20 never need the lab volume.

The eye videos are not in clark2025; they live on /Volumes/INCR-NolanLab. For
every session with pupil data (eye_trials()):
  eye_zone videos (already cropped to the eye, ~50 MB each) are copied whole to
    data/eye_frames/video/;
  full-frame videos (1440x1080, ~6 GB each, ~2 MB/s from the share) are too big
    to copy, so the eye crop (all_eye_crops.csv) at every SHOW_POS frame of every
    trial is saved instead, in greyscale (the camera is IR), to
    data/eye_frames/M{mo}D{dy}_crops.npz.
fig5_pupil_arousal.get_frames reads both, so any trial a later relabelling picks
is covered. Resumable: a session already done is skipped, and the crop archive
is saved every 200 frames. The DLC samples choosing the frame come from
clark2025, so this only goes stale if the pupil tracking is redone.

Run with the lab volume mounted: python3 cache_eye_frames.py
'''
import glob
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig5_pupil_arousal as F                                         # noqa: E402

VID_DIR = f'{F.EYE_LOCAL}/video'
# the figure sessions first
FIRST = [(21, 19), (21, 22), (29, 17), (26, 18), (28, 19), (27, 19), (28, 25)]


def remote_video(mo, dy):
    for pat in (f'{F.VROOT}/M{mo}_D{dy}_*/*eye_zone.avi',
                f'{F.VROOT}/M{mo}_D{dy}_*/sub-*video.avi'):
        g = glob.glob(pat)
        if g:
            return g[0]
    return None


def copy_zone(mo, dy, src):
    dst = f'{VID_DIR}/{os.path.basename(src)}'
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    t0 = time.time()
    shutil.copyfile(src, dst + '.part'); os.replace(dst + '.part', dst)
    print(f'  M{mo}D{dy}: copied {os.path.basename(src)} '
          f'({os.path.getsize(dst)/1e6:.0f} MB, {time.time()-t0:.0f}s)', flush=True)


def crop_full(mo, dy, src):
    S = F.load_session(mo, dy)
    want = sorted({f for u in np.unique(S['tr']) for f in F.trial_frames(S, u)
                   if f is not None})
    out = F.archive_path(mo, dy)
    store = {}
    if os.path.exists(out):
        z = np.load(out); store = {int(k): z[k] for k in z.files}
    todo = [i for i in want if i not in store]
    if not todo:
        return
    C = F.crops()
    r = C[(C.mouse == mo) & (C.day == dy) &
          (C.session.astype(str).str.lower() == 'vr')].iloc[0]
    x0, y0, w, h = int(r.x), int(r.y), int(r.w), int(r.h)
    print(f'  M{mo}D{dy}: {len(todo)} of {len(want)} frames to read', flush=True)
    cap = cv2.VideoCapture(src)
    t0 = time.time()

    def save():
        np.savez_compressed(out + '.part.npz', **{str(k): v for k, v in store.items()})
        os.replace(out + '.part.npz', out)
    for n, i in enumerate(todo, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
        if ok:
            store[i] = fr[y0:y0 + h, x0:x0 + w, 0].copy()
        if n % 200 == 0:
            save()
            print(f'    {n}/{len(todo)}, {time.time()-t0:.0f}s', flush=True)
    cap.release()
    save()
    print(f'  M{mo}D{dy}: {len(store)} frames, {os.path.getsize(out)/1e6:.1f} MB, '
          f'{time.time()-t0:.0f}s', flush=True)


def main():
    if not os.path.isdir(F.VROOT):
        raise SystemExit(f'{F.VROOT} not found -- mount /Volumes/INCR-NolanLab')
    os.makedirs(VID_DIR, exist_ok=True)
    if not os.path.exists(F.CROPS_LOCAL):
        shutil.copyfile(F.CROPS_PATH, F.CROPS_LOCAL)
    E = F.eye_trials()
    ss = sorted(set(zip(E.mouse.astype(int), E.day.astype(int))))
    ss = [s for s in FIRST if s in ss] + [s for s in ss if s not in FIRST]
    srcs = {s: remote_video(*s) for s in ss}
    print(f'{len(ss)} sessions: '
          f'{sum("eye_zone" in (v or "") for v in srcs.values())} eye_zone, '
          f'{sum(v is not None and "eye_zone" not in v for v in srcs.values())} full frame, '
          f'{sum(v is None for v in srcs.values())} no video', flush=True)
    # the small copies first, then the slow crops
    for (mo, dy), src in srcs.items():
        if src and 'eye_zone' in src:
            copy_zone(mo, dy, src)
    for (mo, dy), src in srcs.items():
        if src and 'eye_zone' not in src:
            crop_full(mo, dy, src)
    print('done')


if __name__ == '__main__':
    main()
