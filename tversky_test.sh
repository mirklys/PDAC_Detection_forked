#!/bin/bash

# Activate the correct Conda environment
CONDA_BASE_DIR=$(conda info --base)
source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
conda activate pdac_tversky 

echo "Successfully activated Conda environment: $CONDA_DEFAULT_ENV"
echo "-------------------------------------------"

# --- Configuration ---
# Use absolute paths or paths relative to the script for robustness
# The following assumes you run this script from your home directory (~)
cd repos/PDAC_Detection_forked
echo "Changed directory to: $PWD"

export nnUNet_raw=./tversky_workspace/nnUNet_preprocessed
export nnUNet_preprocessed=./tversky_workspace/nnUNet_preprocessed
export nnUNet_results=./tversky_workspace/nnUNet_results
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=6
export OMP_NUM_THREADS=6


# --- Path Definitions ---
# Source directory containing all the .nii.gz files to be processed
SOURCE_DATA_DIR="$PWD/workspace/test/imagesTs/" 

# Main output directory where batch folders will be created
MAIN_OUTPUT_DIR="$PWD/workspace/test/labelsTs/tversky_predicted"

# Temporary directory for processing batches
BATCH_PROCESSING_DIR="$PWD/workspace/test/temp_batch_processing"

# Batch size
BATCH_SIZE=5

# --- Script Logic ---
mkdir -p "$BATCH_PROCESSING_DIR"
mkdir -p "$MAIN_OUTPUT_DIR"

echo "Source data directory: $SOURCE_DATA_DIR"
echo "Main output directory: $MAIN_OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "-------------------------------------------"

mapfile -t all_files < <(find "$SOURCE_DATA_DIR" -type f -name "*.nii.gz")

num_files=${#all_files[@]}
echo "Total files found: $num_files"

num_batches=$(( (num_files + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Total batches to run: $num_batches"
echo "==========================================="

for (( i=0; i<$num_files; i+=BATCH_SIZE )); do
    current_batch_num=$(( i / BATCH_SIZE + 1 ))
    echo "Processing Batch $current_batch_num of $num_batches..."

    BATCH_OUTPUT_DIR="$MAIN_OUTPUT_DIR/batch_${current_batch_num}"
    mkdir -p "$BATCH_OUTPUT_DIR"
    echo "Output for this batch will be saved in: $BATCH_OUTPUT_DIR"

    batch=("${all_files[@]:i:BATCH_SIZE}")

    for file in "${batch[@]}"; do
        cp "$file" "$BATCH_PROCESSING_DIR/"
    done

    echo "Running inference on ${#batch[@]} files..."
    python -m main_tversky -i "$BATCH_PROCESSING_DIR" -o "$BATCH_OUTPUT_DIR" -m $PWD/tversky_workspace/nnUNet_results

    echo "-------------------------------------------"

    echo "Cleaning up temporary processing directory..."
    rm -f "$BATCH_PROCESSING_DIR"/*
done

# Final cleanup
rmdir "$BATCH_PROCESSING_DIR"

echo "==========================================="
echo "All batches processed successfully!"
echo "All outputs are saved in separate subdirectories inside $MAIN_OUTPUT_DIR"