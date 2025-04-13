#!/bin/bash

# Define the base directory, target folder name, script to run, and script arguments
DATA_DIR="./data"
TARGET_FOLDER_NAME=$1
SCRIPT_TO_RUN=$2
shift 2
SCRIPT_ARGS="$@"

find "$DATA_DIR" -maxdepth 4 -type d -name "$TARGET_FOLDER_NAME" | while read -r dir; do
    parent_dir=$(dirname "$dir")
    echo "Running script $SCRIPT_TO_RUN for $parent_dir"
    python "$SCRIPT_TO_RUN" --session "$parent_dir" $SCRIPT_ARGS
done