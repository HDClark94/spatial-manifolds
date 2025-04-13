#!/bin/bash
#$ -cwd

mouse=$1
day=$2
storage=$3

echo "Cleaning session $mouse $day"
rm -rf ${storage}/raw/*/M${mouse}_D${day}*/ ${storage}/derivatives/M${mouse}/D${day}/
