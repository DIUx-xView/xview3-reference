#!/bin/bash
set -eo pipefail

# Execute xView3 inference pipeline using CLI arguments passed in
# 1) Path to directory with all data files for inference
# 2) Scene ID
# 3) Path to output directory

if [ $# -lt 3 ]; then 
    echo "run_inference.sh: [#1 Path to directory with all data files for inference] [#2 Scene ID] [#3 Path to output directory]"
else

    conda run --no-capture-output -n xview3 python3 baseline/inference.py --image_folder "$1" --scene_ids "$2" --output "$3" --weights /home/xview3/baseline/trained_model_5_epochs.pth --chips_path /home/ubuntu/data-pipe-chips 2>&1 | tee -a /mnt/log.txt

fi
