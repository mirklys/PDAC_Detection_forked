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


export nnUNet_raw=/home/bobby/repos/PDAC_Detection_forked/workspace/nnUNet_preprocessed
export nnUNet_preprocessed=/home/bobby/repos/PDAC_Detection_forked/workspace/nnUNet_preprocessed
export nnUNet_results=/home/bobby/repos/PDAC_Detection_forked/workspace/nnUNet_results

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export OMP_NUM_THREADS=2

echo "RAW =" $nnUNet_raw

python -m main \
    -i ./workspace/test/imagesTs/ \
    -o ./workspace/test/labelsTs/predicted 
