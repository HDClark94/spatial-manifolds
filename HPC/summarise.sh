#!/bin/sh
#$ -cwd
#$ -l h_vmem=1G
#$ -l h_rt=00:10:00

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load anaconda

storage=$1

# Run script
/exports/eddie3_homes_local/wdewulf/.local/bin/uv run --link-mode=copy --cache-dir $storage/uv_cache/ scripts/tuning_scores/summarise.py --storage $storage
