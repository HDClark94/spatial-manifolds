#!/bin/bash
#$ -cwd
#$ -q staging
#$ -l h_rt=02:00:00

# This script stages data for a single mouse and day to the given path

mouse=$1
day=$2
storage=$3
datastore=$4

# Check for suspicious substrings in the storage path
forbidden_strings=("NolanLab" "ActiveProjects")

for forbidden in "${forbidden_strings[@]}"; do
    if [[ "$storage" == *"$forbidden"* ]]; then
        echo "Warning: The storage path '$storage' contains '$forbidden', did you swap destination and source directories? We don't want to overwrite the raw data!"
        exit 0
    fi
done

# ---------------------------
# RAW
SOURCE_PATH="${datastore}/Harry/EphysNeuropixelData/./"
SUBDIRS=("vr" "of" "vr_multi_context")

RSYNC_CMD="rsync -vrR --prune-empty-dirs --ignore-existing"

for subdir in "${SUBDIRS[@]}"; do
    RSYNC_CMD+=" --include '${subdir}/'"
    RSYNC_CMD+=" --include '${subdir}/M${mouse}_D${day}_*/'"
    RSYNC_CMD+=" --include '${subdir}/M${mouse}_D${day}_*/**'"
done

RSYNC_CMD+=" --exclude '*' ${SOURCE_PATH} ${storage}/raw/"

# Execute the rsync command
echo $RSYNC_CMD
eval $RSYNC_CMD

# ---------------------------
# SORTING
SOURCE_PATH="$datastore/Chris/Cohort12/derivatives/./"
SUBDIRS=("of1" "of2")

RSYNC_CMD="rsync -vrR --prune-empty-dirs --ignore-existing \
  --include 'M${mouse}/' \
  --include 'M${mouse}/D${day}/' \
  --include 'M${mouse}/D${day}/full/' \
  --include 'M${mouse}/D${day}/full/kilosort4/' \
  --include 'M${mouse}/D${day}/full/kilosort4/kilosort4_sa/' \
  --include 'M${mouse}/D${day}/vr/' \
  --include 'M${mouse}/D${day}/vr/licks/' \
  --include 'M${mouse}/D${day}/vr/licks/lick_mask.csv' \
  --include 'M${mouse}/D${day}/vr/pupil_dilation/' \
  --include 'M${mouse}/D${day}/vr/pupil_dilation/*.csv' \
  --include 'M$mouse/D$day/full/kilosort4/kilosort4_sa/**' \
  --include 'labels/' \
  --include 'labels/anatomy/' \
  --include 'labels/anatomy/mouse_day_channel_ids_brain_location.csv'"

for subdir in "${SUBDIRS[@]}"; do
    RSYNC_CMD+=" --include 'M$mouse/D$day/${subdir}/'"
    RSYNC_CMD+=" --include 'M$mouse/D$day/${subdir}/dlc/'"
    RSYNC_CMD+=" --include 'M$mouse/D$day/${subdir}/dlc/**'"
done

RSYNC_CMD+=" --exclude '*' ${SOURCE_PATH} $storage/derivatives/"

# Execute the rsync command
echo $RSYNC_CMD
eval $RSYNC_CMD
