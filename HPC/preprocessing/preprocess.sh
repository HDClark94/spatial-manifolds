#!/bin/sh
#$ -cwd
#$ -l h_vmem=6G
#$ -l h_rt=03:00:00

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load anaconda

mouse=$1
day=$2
storage=$3

echo "Formatting session $mouse $day"
/exports/eddie3_homes_local/wdewulf/.local/bin/uv run --link-mode=copy --cache-dir $storage/uv_cache/ scripts/preprocessing/behaviour.py --mouse $mouse --day $day --storage $storage
/exports/eddie3_homes_local/wdewulf/.local/bin/uv run --link-mode=copy --cache-dir $storage/uv_cache/ scripts/preprocessing/sorting.py --mouse $mouse --day $day --storage $storage
