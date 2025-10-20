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

    """
    makes a stage out script from a stageout_dict of the form
    {'path/to/file/on/eddie': 'path/to/destination/on/datastore'}
    Note: let's never stageout to the raw data folder, to avoid risk of deletion
    """

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
    """
    Makes a python script, which will run
    >>  python python_arg

    If nothing else is supplied, this will run on the venv 'elrond' with 19 cores, 19GB of RAM per core, with a 
    hard runtime limit of 48 hours.    
    """

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
        h_rt = "167:59:59"
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

#=================================================================================================================

#=================================================================================================================

#=================================================================================================================

#=================================================================================================================


#=================================================================================================================


#=================================================================================================================


#=================================================================================================================


#=================================================================================================================



#=================================================================================================================



#=================================================================================================================




#=================================================================================================================
#=================================================================================================================
# these sessions actually have task-independent firing grid cells
mouse_days = {27: [23,26],
              21: [21,26],
              25: [19,20,24,25],
              26: [18,19],
            }

# there are all the sessions
mouse_days = {20: [14,15,16,17,18,19,20,21,22,23,24,25,26],
              21: [15,16,17,18,19,20,21,22,23,24,25,26],
              25: [16,17,18,19,20,21,22,23,24,25],
              26: [11,12,13,14,15,16,17,18,19],
              27: [16,17,18,19,20,21,22,23,24,26],
              28: [16,17,18,19,20,21,22,23,25],
              29: [16,17,18,19,20,21,22,23,25],
            }

mouse_days = {29: [23,19,20,25],
              25: [24,25],
              27: [26,23,24],
              21: [16,21,25,26],
              26: [15,18,19],
              28: [19,23,25],
            }

mouse_days = {27: [23,24],
              21: [16,21],
              26: [19],
            }

mouse_days = {27: [24],
              26: [19],
            }
print('')

# there are all the sessions
mouse_days = {20: [14,15,16,17,18,19,20,21,22,23,24,25,26],
              21: [15,16,17,18,19,20,21,22,23,24,25,26],
              25: [16,17,18,19,20,21,22,23,24,25],
              26: [11,12,13,14,15,16,17,18,19],
              27: [16,17,18,19,20,21,22,23,24,26],
              28: [16,17,18,19,20,21,22,23,25],
              29: [16,17,18,19,20,21,22,23,25],
            }

# there are sessions with detected grid modules
mouse_days = {21: [16,21,25,26],
              20: [25],
              25: [19, 20,21,22,24,25],
              26: [15, 18,19],
              27: [20,23,24,26],
              28: [17,19,23,25],
              29: [19,20,23,25],
            }

source_path = '/exports/eddie/scratch/hclark3/COHORT12/'
for mouse, days in mouse_days.items():
    for day in days:
        data_path = f"/exports/eddie/scratch/hclark3/COHORT12/xgboost_task_anchoring/"
        job_name = f"M{mouse}D{day}_xgb"
        if mouse == 21:
            cores = 48
        else:
            cores = 32 # 32 for mouse 21, 16 for everything else

        run_python_script(f"/exports/eddie/scratch/hclark3/spatial-manifolds/scripts/figures/xgboost_task_anchoring_assay.py --mouse={mouse} --day={day} --data_path={data_path}", username="hclark3", email="hclark3@ed.ac.uk", cores=cores, job_name=job_name)
        #run_stage_script(stageout_dict, hold_jid=job_name)

print('This script does not run any jobs, it is just a template for running jobs on Eddie.')