#!/bin/bash
#$ -cwd -pe sharedmem 1 -l h_vmem=19G,h_rt=0:29:59 -N my_job

/home/hclark3/.local/bin/uv run /exports/eddie/scratch/hclark3/spatial-manifolds/scripts/eddie/test_eddie_script.py