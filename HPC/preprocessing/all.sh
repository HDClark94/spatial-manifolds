#!/bin/bash

# This script loops over all sessions and queues stagein and format_single tasks

storage="/exports/eddie/scratch/wdewulf"
datastore="/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects"
sessions=(
    "20 14"
    "20 15"
    "20 16"
    "20 17"
    "20 18"
    "20 19"
    "20 20"
    "20 21"
    "20 22"
    "20 23"
    "20 24"
    "20 25"
    "20 26"
    "20 27"
    "20 28" # 15
    "21 15"
    "21 16"
    "21 17"
    "21 18"
    "21 19"
    "21 20"
    "21 21"
    "21 22"
    "21 23"
    "21 24"
    "21 25"
    "21 26"
    "21 27"
    "21 28" # 14
    "22 33"
    "22 34"
    "22 35"
    "22 36"
    "22 37"
    "22 38"
    "22 39"
    "22 40"
    "22 41" # --
    "25 16"
    "25 17"
    "25 18"
    "25 19"
    "25 20"
    "25 21"
    "25 22"
    "25 23"
    "25 24"
    "25 25"
    "26 11" # --
    "26 12"
    "26 13"
    "26 14"
    "26 15"
    "26 16"
    "26 17"
    "26 18"
    "26 19"
    "27 16" # --
    "27 17"
    "27 18"
    "27 19"
    "27 20"
    "27 21"
    "27 22"
    "27 23"
    "27 24"
    "27 26"
    "28 16" # --
    "28 17"
    "28 18"
    "28 19"
    "28 20"
    "28 21"
    "28 22"
    "28 23"
    "28 25"
    "29 16" # --
    "29 17"
    "29 18"
    "29 19"
    "29 20"
    "29 21"
    "29 22"
    "29 23"
    "29 25"
)
sessions=(
    "22 42"
    "22 43"
    "22 44"
    "22 45"
    "22 46"
    "25 26" #---
    "25 27"
    "25 28"
    "25 29"
    "25 30"
    "25 31" 
    "26 20" # --
    "26 21"
    "26 22"
    "26 23"
    "26 24"
    "27 25" #---
    "27 27"
    "27 28"
    "27 29"
    "27 30"
    "28 24" #---
    "28 26"
    "28 27"
    "28 28"
    "28 29"
    "29 24" #---
    "29 26"
    "29 27"
    "29 28"
    "29 29"
  )
batch_size=18  # Number of jobs per batch
num_sessions=${#sessions[@]}  # Total number of sessions
prev_batch_jobs=""  # Track job IDs of the previous batch

rm -rf "${datastore}/tmp/"
for ((i = 0; i < num_sessions; i += batch_size)); do
    curr_batch_jobs=""  # Store job IDs for this batch

    # Submit tasks in sequence for each session (stagein -> formatting -> cleanup)
    for ((j = i; j < i + batch_size && j < num_sessions; j++)); do
        read -r mouse day <<< "${sessions[j]}"

        # Stagein task
        if [[ -n "$prev_batch_jobs" ]]; then
          qsub -N "s${mouse}${day}" -hold_jid "$prev_batch_jobs" -e "logs/${mouse}${day}.log" -o "logs/${mouse}${day}.log" HPC/preprocessing/stagein.sh $mouse $day $storage $datastore
        else
          qsub -N "s${mouse}${day}" -e "logs/${mouse}${day}.log" -o "logs/${mouse}${day}.log" HPC/preprocessing/stagein.sh $mouse $day $storage $datastore
        fi
        
        # Preprocessing task
        qsub -N "p${mouse}${day}" -hold_jid "s${mouse}${day}" -e "logs/${mouse}${day}.log" -o "logs/${mouse}${day}.log" HPC/preprocessing/preprocess.sh $mouse $day $storage
        
        # Cleanup task
        qsub -N "c${mouse}${day}" -hold_jid "p${mouse}${day}" -e "logs/${mouse}${day}.log" -o "logs/${mouse}${day}.log" HPC/preprocessing/cleanup.sh $mouse $day $storage

        # Store the last job ID of the current session (cleanup job)
        curr_batch_jobs+="c${mouse}${day},"
    done

    # Remove trailing comma
    curr_batch_jobs=${curr_batch_jobs%,}

    # Set this batch as the dependency for the next batch
    prev_batch_jobs="$curr_batch_jobs"
done
