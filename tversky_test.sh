#!/bin/bash

CONDA_BASE_DIR=$(conda info --base)
source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
conda activate pdac_tversky 

export nnUNet_raw=./workspace/nnUNet_preprocessed
export nnUNet_preprocessed=./workspace/nnUNet_preprocessed
export nnUNet_results=./workspace/nnUNet_results
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export OMP_NUM_THREADS=2

echo 'RAW=' $nnUNet_raw
python -m main -i ./workspace/test/imagesTs/ -o ./workspace/test/labelsTs/tversky_predicted -m ./workspace/nnUNet_results
