#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduIMC037
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err
#SBATCH --mail-user=giedrius.mirklys@ru.nl
#SBATCH --mail-type=END,FAIL

CONDA_BASE_DIR=$(conda info --base)
source "${CONDA_BASE_DIR}/etc/profile.d/conda.sh"
conda activate pdac_tversky 
export nnUNet_raw=./workspace/nnUNet_preprocessed
export nnUNet_preprocessed=./workspace/nnUNet_preprocessed
export nnUNet_results=./tversky_workspace/nnUNet_results
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export OMP_NUM_THREADS=2

echo 'RAW=' $nnUNet_raw
python -m main -i /home/bobby/repos/PDAC_Detection_forked/workspace/test/imagesTs/ -o /home/bobby/repos/PDAC_Detection_forked/workspace/test/labelsTs/tversky_predicted 
