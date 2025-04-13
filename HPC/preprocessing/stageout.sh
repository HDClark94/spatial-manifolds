#!/bin/bash
#$ -cwd
#$ -q staging
#$ -l h_rt=00:10:00

# This script stages out all preprocessed data from a given path to the given path on the DataStore

input=$1
output=$2

mkdir -p /exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/${output}
cp -r $input/** /exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/${output}
