#!/bin/bash

# eval_all.sh
# Orchestrates the execution of adversarial probing pipeline for all models

set -e

# Setup parameters
DATA_FILE=""  # Add path here, e.g. "path/to/queries" empty will use internal dummy ones
RESULTS_DIR="../results"

cd src

echo "============================================="
echo "Starting Modular Adversarial Probing Overview"
echo "============================================="

# Ensure results dir is created
mkdir -p "$RESULTS_DIR"

# Define model paths based on folder structure
declare -a base_models=(
    "base_fp16|../models/base/base_gemma3_1b_it_fp16|auto"
    "base_int4|../models/base/base_gemma3_1b_it_int4|auto"
    "base_int8|../models/base/base_gemma3_1b_it_int8|auto"
)

declare -a npo_models=(
    "npo_fp16|../models/npo/fp16_unlearnt_model|auto"
    "npo_int4|../models/npo/int4_unlearnt_model|auto"
    "npo_int8|../models/npo/int8_unlearnt_model|auto"
)

declare -a ta_models=(
    "ta_fp16|../models/task_arithmetic/fp16_unlearned_model|auto"
)

# Combine them
declare -a all_models=("${base_models[@]}" "${npo_models[@]}" "${ta_models[@]}")

for model_info in "${all_models[@]}"; do
    # Split string by |
    IFS="|" read -r model_name model_path state <<< "${model_info}"

    # Check if directory exists and has config.json (a real model)
    if [ -d "$model_path" ] && [ -f "$model_path/config.json" ]; then
        echo ""
        echo "---------------------------------------------------------"
        echo "Discovered model: $model_name"
        echo "Running pipeline for $model_name..."
        echo "---------------------------------------------------------"
        
        # Execute pipeline
        python pipeline.py \
            --model_name "$model_name" \
            --model_path "$model_path" \
            --state "$state" \
            --results_dir "$RESULTS_DIR" ${DATA_FILE:+--data_file "$DATA_FILE"}
            
        echo "Completed pipeline for $model_name."
    else
        echo ""
        echo "Skipping $model_name - Target directory missing or empty: $model_path"
    fi
done

echo ""
echo "============================================="
echo "All Evaluations Complete!"
echo "Check the $RESULTS_DIR folder for outputs."
echo "============================================="
