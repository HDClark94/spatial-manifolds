#!/bin/sh
#$ -cwd
#$ -l h_vmem=10G
#$ -l h_rt=04:00:00

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load anaconda

# Run script
/exports/eddie3_homes_local/wdewulf/.local/bin/uv run --link-mode=copy --cache-dir $STORAGE/uv_cache/ scripts/tuning_scores/script.py --mouse $MOUSE --day $DAY --storage $STORAGE --session_type $SESSION_TYPE
