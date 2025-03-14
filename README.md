# 🏆 1st place in the PANORAMA Challenge (Team DTI)

Report is available [here](https://github.com/han-liu/PDAC_Detection/blob/main/%5BTeam%20DTI%5D%20AI-assisted%20Early%20Detection%20of%20Pancreatic%20Ductal%20Adenocarcinoma%20on%20Contrast-enhanced%20CT.pdf)

Code and models will be released soon.


Installation guide

1. Install virtual environment:
conda create --prefix /pct_wbo2/home/han.l/envs/pdac python=3.12 -y
conda activate /pct_wbo2/home/han.l/envs/pdac

module load cuda/cuda-11.1
module load cudnn/9.0.0-cuda-12

cd temp
pip install -r requirements.txt

cd packages/nnunetv2
pip install -e . 

cd packages/report-guided-annotation
pip install -e . 


export nnUNet_raw="./workspace/nnUNet_raw" 
export nnUNet_preprocessed="./workspace/nnUNet_preprocessed" 
export nnUNet_results="./workspace/nnUNet_results"

python main.py -i ${INPUT_DIR} -o ${OUTPUT_DIR} --inv_alpha ${INV_ALPHA}

python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output 