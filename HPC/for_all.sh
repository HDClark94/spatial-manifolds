#!/bin/sh

script=$1
log_dir=$2
storage=$3
for session in $(find $storage/sessions/ -path "*/*/*" -name "*.nwb"); do

    # Extract the filename
    filename=$(basename -- "$session")
    
    # Extract mouse, day, and task from the filename using regex
    if [[ $filename =~ M([0-9]+)D([0-9]+)([A-Za-z0-9_]+)\.nwb ]]; then
        mouse="${BASH_REMATCH[1]}"
        day="${BASH_REMATCH[2]}"
        session_type="${BASH_REMATCH[3]}"
    else
        echo "Filename $filename does not match expected pattern!"
        continue
    fi

    # Construct a tag for the session
    log_file="${log_dir}/${session_type}/M${mouse}D${day}.log"

    # Submit the job with the extracted variables
    qsub -N "M${mouse}D${day}${session_type}" -e ${log_file} -o ${log_file} \
      -v MOUSE=${mouse},DAY=${day},SESSION_TYPE=${session_type},STORAGE=$storage $script
done
