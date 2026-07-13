"""
run_border_scores_eddie.py
───────────────────────────
Submit shifted border score jobs (OF1 + OF2) for all COHORT12 sessions to
Eddie via qsub. One job per mouse × day × session. Each job stages its output
to the datastore when complete.

Run from the Eddie login node:
    cd /exports/eddie/scratch/hclark3/spatial-manifolds
    python scripts/eddie/run_border_scores_eddie.py
"""

import os
import subprocess
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
USERNAME      = "chalcrow"
SCRATCH_BASE  = f"/exports/eddie/scratch/{USERNAME}"
SOURCE_PATH   = f"{SCRATCH_BASE}/COHORT12/"
SCRATCH_OUT   = f"{SCRATCH_BASE}/data/border_scores_of/"
DATASTORE_OUT = (
    "/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Harry/"
    "SpatialLocationManifolds2025/data/border_scores_of/"
)
SCRIPT_PATH   = (
    f"{SCRATCH_BASE}/spatial-manifolds/scripts/eddie/"
    "compute_shifted_border_scores.py"
)

SESSIONS  = ['OF1', 'OF2']
H_RT      = "23:59:59"
H_VMEM    = 16   # GB per core
CORES     = 16

mouse_days = {
    20: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    21: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    22: [33, 34, 35, 36, 37, 38, 39, 40, 41],
    25: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    26: [11, 12, 13, 14, 15, 16, 17, 18, 19],
    27: [16, 17, 18, 19, 20, 21, 22, 23, 24, 26],
    28: [16, 17, 18, 19, 20, 21, 22, 23, 25],
    29: [16, 17, 18, 19, 20, 21, 22, 23, 25],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_and_qsub(content, name):
    path = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sh"
    with open(path, 'w') as f:
        f.write(content)
    subprocess.run(['qsub', path])


def submit_compute(mouse, day, session):
    job_name = f"BS_M{mouse}D{day:02}{session}"
    python_arg = (
        f"{SCRIPT_PATH} "
        f"--mouse={mouse} --day={day} --session={session} "
        f"--source_path={SOURCE_PATH} --output_path={SCRATCH_OUT}"
    )
    script = f"""#!/bin/bash
#$ -cwd -pe sharedmem {CORES} -l rl9=true,h_vmem={H_VMEM}G,h_rt={H_RT} -N {job_name}
source $HOME/.bashrc
/home/{USERNAME}/.local/bin/uv run {python_arg}"""
    _save_and_qsub(script, job_name)
    return job_name


def submit_stageout(hold_jid):
    job_name = f"stage_{hold_jid}"
    script = f"""#!/bin/sh
#$ -cwd -q staging -l h_rt=00:29:59 -hold_jid {hold_jid} -N {job_name}
cp -rn {SCRATCH_OUT} {DATASTORE_OUT}"""
    _save_and_qsub(script, job_name)


# ── Main ──────────────────────────────────────────────────────────────────────
os.makedirs(SCRATCH_OUT, exist_ok=True)

total = 0
for mouse, days in mouse_days.items():
    for day in days:
        for session in SESSIONS:
            session_folder = f"{SOURCE_PATH}M{mouse}/D{day:02}/{session}/"
            if not os.path.isdir(session_folder):
                print(f"SKIP  M{mouse} D{day:02} {session} — folder not found")
                continue
            jid = submit_compute(mouse, day, session)
            submit_stageout(jid)
            total += 1
            print(f"Submitted  M{mouse} D{day:02} {session}  →  {jid}")

print(f"\nTotal jobs submitted: {total} compute + {total} stageout")
