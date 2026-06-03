from datetime import datetime
import subprocess
import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')


# ── Shared HPC helpers (copied from run_eddie_cell_number_assay_extra.py) ─────

def make_run_python_script(python_arg, username, venv=None, cores=None, email=None,
                            h_rt=None, h_vmem=None, hold_jid=None, job_name=None, staging=False):
    if hold_jid is not None:
        hold_script = f" -hold_jid {hold_jid}"
    else:
        hold_script = ""
    if email is not None:
        email_script = f" -M {email} -m e"
    else:
        email_script = ""
    if venv is None:
        venv = "elrond"
    if cores is None:
        cores = 32
    if h_rt is None:
        h_rt = "47:59:59"
    if h_vmem is None:
        h_vmem = 19
    if job_name is not None:
        name_script = f" -N {job_name}"
    else:
        name_script = ""
    if staging:
        staging_script = " -q staging"
        core_script    = ""
        vmem_script    = ""
    else:
        staging_script = ""
        core_script    = f" -pe sharedmem {cores}"
        vmem_script    = f",h_vmem={h_vmem}G"

    script_content = (
        f"#!/bin/bash\n"
        f"#$ -cwd{staging_script}{core_script} -l rl9=true{vmem_script}"
        f",h_rt={h_rt}{hold_script}{email_script}{name_script}\n"
        f"source $HOME/.bashrc\n"
        f"/home/{username}/.local/bin/uv run {python_arg}"
    )
    return script_content


def run_python_script(python_arg, username, venv=None, cores=None, email=None,
                       h_rt=None, h_vmem=None, hold_jid=None,
                       script_file_path=None, staging=False, job_name=None):
    if job_name is None:
        job_name = "run_python"
    script_content = make_run_python_script(
        python_arg, username, venv=venv, cores=cores, email=email,
        h_rt=h_rt, h_vmem=h_vmem, hold_jid=hold_jid, staging=staging, job_name=job_name,
    )
    if script_file_path is None:
        script_file_path = f"{job_name}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh"
    save_script(script_content, script_file_path)
    run_script(script_file_path)


def run_stage_script(stageout_dict, script_file_path=None, hold_jid=None, job_name=None):
    if hold_jid is not None:
        hold_script = f" -hold_jid {hold_jid}"
    else:
        hold_script = ""
    if job_name is None:
        job_name = "stage"
    name_script = f" -N {job_name}"

    script_text = (
        f"#!/bin/sh\n"
        f"#$ -cwd\n"
        f"#$ -q staging\n"
        f"#$ -l h_rt=00:29:59{hold_script}{name_script}"
    )
    if script_file_path is None:
        script_file_path = f"{job_name}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh"
    for source, dest in stageout_dict.items():
        script_text += "\ncp -rn " + str(source) + " " + str(dest)
    save_script(script_text, script_file_path)
    run_script(script_file_path)


def save_script(script_content, script_file_path):
    with open(script_file_path, "w") as f:
        f.write(script_content)


def run_script(script_file_path):
    subprocess.run(("qsub " + script_file_path).split())


# ══════════════════════════════════════════════════════════════════════════════
# Anchor Assay — job submission
# ══════════════════════════════════════════════════════════════════════════════

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

# ── Paths on Eddie scratch ────────────────────────────────────────────────────
cell_class_path = "/exports/eddie/scratch/hclark3/spatial-manifolds/data/cell_classifications.csv"
analysis_script = "/exports/eddie/scratch/hclark3/spatial-manifolds/scripts/figures/xgboost_anchor_assay.py"
data_path       = "/exports/eddie/scratch/hclark3/data/xgboost_anchor_assay/"
datastore_path  = (
    "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/"
    "ActiveProjects/Harry/SpatialLocationManifolds2025/data/xgboost_anchor_assay/"
)

stageout_dict = {data_path: datastore_path}

# ── Load cell classifications for batching ────────────────────────────────────
cell_class_df = pd.read_csv(cell_class_path)

BATCH_SIZE = 20

for mouse, days in mouse_days.items():
    for day in days:
        # Only batch over GC + NG targets (the cell types actually processed)
        session_cells = cell_class_df[
            (cell_class_df['mouse']     == mouse) &
            (cell_class_df['day']       == day) &
            (cell_class_df['cell_type'].isin(['GC', 'NG']))
        ]
        n_cells = len(session_cells)
        if n_cells == 0:
            print(f"M{mouse} D{day:02d}: no GC/NG cells — skipping.")
            continue

        n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"M{mouse} D{day:02d}: {n_cells} target cells → {n_batches} batch(es)")

        for batch_idx in range(n_batches):
            cell_start = batch_idx * BATCH_SIZE
            cell_end   = min((batch_idx + 1) * BATCH_SIZE, n_cells)
            job_name   = f"M{mouse}D{day}A_{cell_start}_{cell_end}"

            run_python_script(
                (
                    f"{analysis_script}"
                    f" --mouse={mouse}"
                    f" --day={day}"
                    f" --data_path={data_path}"
                    f" --cell_class_path={cell_class_path}"
                    f" --cell_start={cell_start}"
                    f" --cell_end={cell_end}"
                ),
                username="hclark3",
                email="hclark3@ed.ac.uk",
                cores=16,
                job_name=job_name,
            )
            run_stage_script(stageout_dict, hold_jid=job_name)
