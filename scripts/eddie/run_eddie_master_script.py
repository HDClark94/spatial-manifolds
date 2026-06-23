from datetime import datetime
import subprocess
import pandas as pd
import pynapple as nap
import warnings
import os


def curate_clusters(clusters) -> nap.TsGroup:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Some epochs have no duration",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in scalar divide",
        )
    return clusters[
        (clusters["isi_violations_ratio"] < 0.5)
        # & (clusters['amplitude_cutoff'] < 0.1)
        & (clusters["presence_ratio"] > 0.9)
        & (clusters["firing_rate"] > 0.5)
        & (clusters["snr"] > 1)
    ]


def run_python_script(python_arg, username, venv=None, cores=None, email=None, h_rt=None, h_vmem=None, hold_jid=None, script_file_path=None, staging=False, job_name=None):

    if job_name is None:
        job_name = "run_python"

    script_content = make_run_python_script(python_arg, username, venv=venv, cores=cores, email=email, h_rt=h_rt, h_vmem=h_vmem, hold_jid=hold_jid, staging=staging, job_name=job_name)

    if script_file_path is None:
        script_file_path = f"{job_name}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh"

    save_script(script_content, script_file_path)
    run_script(script_file_path)

    return


def run_stage_script(stageout_dict, script_file_path=None, hold_jid=None, job_name=None):

    if hold_jid is not None:
        hold_script = f" -hold_jid {hold_jid}"
    if job_name is None:
        job_name = "stage"

    name_script = f" -N {job_name}"

    script_text=f"""#!/bin/sh
#$ -cwd
#$ -q staging
#$ -l h_rt=00:29:59{hold_script}{name_script}"""

    if script_file_path is None:
        script_file_path = f"{job_name}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh"


    for source, dest in stageout_dict.items():
        script_text = script_text + "\ncp -rn " + str(source) + " " + str(dest)

    save_script(script_text, script_file_path)
    run_script(script_file_path)

    return

def make_run_python_script(python_arg, username, venv=None, cores=None, email=None, h_rt=None, h_vmem=None, hold_jid=None, job_name=None, staging=False):

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
        h_vmem=19
    if job_name is not None:
        name_script = f" -N {job_name}"
    else:
        name_script = ""
    if staging:
        staging_script = " -q staging"
        core_script = ""
        vmem_script = ""
    else:
        staging_script = ""
        core_script = f" -pe sharedmem {cores}"
        vmem_script = f",h_vmem={h_vmem}G"

    script_content = f"""#!/bin/bash
#$ -cwd{staging_script}{core_script} -l rl9=true{vmem_script},h_rt={h_rt}{hold_script}{email_script}{name_script}
source $HOME/.bashrc
/home/{username}/.local/bin/uv run {python_arg}"""

    return script_content

def save_script(script_content, script_file_path):

    if script_file_path is None:
        script_file_path = f"run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".sh"

    f = open(script_file_path, "w")
    f.write(script_content)
    f.close()

    return

def run_script(script_file_path):

    compute_string = "qsub " + script_file_path
    subprocess.run( compute_string.split() )

    return


#=================================================================================================================

all_mouse_days = {20: [14,15,16,17,18,19,20,21,22,23,24,25,26],
                  21: [15,16,17,18,19,20,21,22,23,24,25,26],
                  22: [33,34,35,36,37,38,39,40,41,42,43,44,45,46],
                  25: [16,17,18,19,20,21,22,23,24,25],
                  26: [11,12,13,14,15,16,17,18,19,20,21,22,23],
                  27: [16,17,18,19,20,21,22,23,24,25,26],
                  28: [16,17,18,19,20,21,22,23,24],
                  29: [16,17,18,19,20,21,22,23,25],
                 }

# multishank sessions only
multishank_mouse_days = {20: [14,15,16,17,18,19],
                         21: [15,16,17,18,22,25,26],
                         22: [37,38,39,40,41,42,43,44],
                         25: [16,17,18,25,26],
                         26: [13,15,16,19,20,21,22],
                         27: [16,17,18,19,23,24,26],
                         28: [16,17,18,23],
                         29: [16,17,18,20,23,25],
                        }

cell_class_path = "/exports/eddie/scratch/hclark3/spatial-manifolds/data/cell_classifications.csv"
cell_class_df = pd.read_csv(cell_class_path)

scripts_base = "/exports/eddie/scratch/hclark3/spatial-manifolds/scripts/figures"
datastore_base = "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/SpatialLocationManifolds2025/data"

# all sessions: VR and OF extra assays
all_session_configs = [
    ("xgboost_cell_number_assay_extra.py",     "xgboost_cell_number_assay",    "X"),
    ("xgboost_cell_number_assay_extra_of.py",  "xgboost_cell_number_assay_of", "XOF"),
]

# multishank only: medial/lateral NGS assays
medlat_configs = [
    ("xgboost_cell_number_assay_medlat_ngs_vr.py",  "xgboost_medlat_ngs_vr",  "mlVR"),
    ("xgboost_cell_number_assay_medlat_ngs_of.py",  "xgboost_medlat_ngs_of",  "mlOF"),
]

BATCH_SIZE = 20

def submit_assay(mouse, day, script_name, data_subfolder, job_prefix):
    session_cells = cell_class_df[(cell_class_df['mouse'] == mouse) & (cell_class_df['day'] == day)]
    n_cells = len(session_cells)
    n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
    data_path = f"/exports/eddie/scratch/hclark3/data/{data_subfolder}/"
    stageout_dict = {
        data_path: f'{datastore_base}/{data_subfolder}/'
    }
    for batch_idx in range(n_batches):
        cell_start = batch_idx * BATCH_SIZE
        cell_end = min((batch_idx + 1) * BATCH_SIZE, n_cells)
        job_name = f"M{mouse}D{day}{job_prefix}_{cell_start}_{cell_end}"
        run_python_script(
            f"{scripts_base}/{script_name} --mouse={mouse} --day={day} --data_path={data_path} --cell_start={cell_start} --cell_end={cell_end}",
            username="hclark3", email="hclark3@ed.ac.uk", cores=16, job_name=job_name
        )
        run_stage_script(stageout_dict, hold_jid=job_name)

# submit VR and OF extra assays for all sessions
for mouse, days in all_mouse_days.items():
    for day in days:
        for script_name, data_subfolder, job_prefix in all_session_configs:
            submit_assay(mouse, day, script_name, data_subfolder, job_prefix)

# submit medial/lateral NGS assays for multishank sessions only
for mouse, days in multishank_mouse_days.items():
    for day in days:
        for script_name, data_subfolder, job_prefix in medlat_configs:
            submit_assay(mouse, day, script_name, data_subfolder, job_prefix)
