#!/bin/bash
#$ -cwd -pe sharedmem 1 -l h_vmem=19G,h_rt=0:29:59 -N my_job

cd /exports/eddie/scratch/hclark3/spatial-manifolds/scripts/eddie
/home/hclark3/.local/bin/uv run test_eddie_script.py