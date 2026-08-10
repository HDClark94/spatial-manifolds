"""
Submit the batch->per-cell migration as an Eddie STAGING job, so it runs in-place on
the datastore (only staging nodes can see /exports/cmvm/datastore) — no download/upload.

The migration is idempotent (skips per-cell files that already exist), so if a job hits
the staging walltime it can simply be re-run to continue. Use --njobs N to submit N jobs
chained with hold_jid, which auto-continues across the walltime until everything is split.

Usage (on an Eddie login node):
    python run_eddie_migration.py                 # one staging job
    python run_eddie_migration.py --njobs 4       # 4 chained jobs (survives walltime)
    python run_eddie_migration.py --dry-run       # count only, write nothing
    python run_eddie_migration.py --clean-empty   # also delete 0-byte failed batches
"""
import sys
import subprocess
from datetime import datetime

USERNAME       = "hclark3"
EMAIL          = "hclark3@ed.ac.uk"
SCRATCH_REPO   = "/exports/eddie/scratch/hclark3/spatial-manifolds"
MIGRATE_SCRIPT = f"{SCRATCH_REPO}/scripts/eddie/migrate_batch_to_percell.py"
DATASTORE_BASE = "/exports/cmvm/datastore/sbms/groups/INCR-NolanLab/ActiveProjects/Harry/COHORT12/eddie"
H_RT           = "01:59:59"   # per staging job; jobs chain so total time is N * this


def main():
    passthrough = " ".join(a for a in sys.argv[1:] if a in ("--dry-run", "--clean-empty"))
    njobs = 1
    for a in sys.argv[1:]:
        if a.startswith("--njobs="):
            njobs = int(a.split("=")[1])
        elif a == "--njobs":
            i = sys.argv.index(a)
            njobs = int(sys.argv[i + 1])

    prev = None
    for k in range(njobs):
        job_name = "migrate_percell" if njobs == 1 else f"migrate_percell_{k}"
        hold = f" -hold_jid {prev}" if prev else ""
        script = (
            "#!/bin/bash\n"
            f"#$ -cwd -q staging -l rl9=true,h_rt={H_RT} -N {job_name} -M {EMAIL} -m e{hold}\n"
            "source $HOME/.bashrc\n"
            f"/home/{USERNAME}/.local/bin/uv run {MIGRATE_SCRIPT} {DATASTORE_BASE} {passthrough}\n"
        )
        path = f"{job_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.sh"
        with open(path, "w") as f:
            f.write(script)
        subprocess.run(["qsub", path])
        print(f"Submitted staging migration job {job_name}{' (holds on '+prev+')' if prev else ''} "
              f"-> {DATASTORE_BASE} {passthrough}")
        prev = job_name

    print(f"\n{njobs} staging job(s) submitted. Migration runs in-place on the datastore; "
          f"re-run this any time — already-split cells are skipped.")


if __name__ == "__main__":
    main()
