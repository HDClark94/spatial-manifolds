from datetime import datetime
import subprocess

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

    If nothing else is supplied, this will run on the venv 'elrond' with 8 cores, 19GB of RAM per core, with a 
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
        cores = 8

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
