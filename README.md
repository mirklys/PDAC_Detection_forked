# 🏆 1st Place in the PANORAMA Challenge (Team DTI)

We are proud to share the code behind our **1st place** solution for the PANORAMA Challenge: AI-assisted Early Detection of Pancreatic Ductal Adenocarcinoma (PDAC) from contrast-enhanced CT scans.

**Report**: [AI-assisted Early Detection of PDAC on Contrast-enhanced CT](https://github.com/han-liu/PDAC_Detection/blob/main/%5BTeam%20DTI%5D%20AI-assisted%20Early%20Detection%20of%20Pancreatic%20Ductal%20Adenocarcinoma%20on%20Contrast-enhanced%20CT.pdf)

Code and models will be released soon.

---

**Installation Guide**

Create a Conda environment (example path shown, adapt as needed):

    conda create --prefix /pct_wbo2/home/han.l/envs/pdac python=3.12 -y
    conda activate /pct_wbo2/home/han.l/envs/pdac

If your environment requires modules, load them:

    module load cuda/cuda-11.1
    module load cudnn/9.0.0-cuda-12

Install dependencies:

    cd temp
    pip install -r requirements.txt

Install additional packages:

    cd packages/nnunetv2
    pip install -e .

    cd ../report-guided-annotation
    pip install -e .

Set environment variables:

    export nnUNet_raw="./workspace/nnUNet_raw"
    export nnUNet_preprocessed="./workspace/nnUNet_preprocessed"
    export nnUNet_results="./workspace/nnUNet_results"

Run the main script:

    python main.py -i ${INPUT_DIR} -o ${OUTPUT_DIR} --inv_alpha ${INV_ALPHA}

For a quick test:

    python main.py -i ./workspace/test_example/input \
                   -o ./workspace/test_example/output

Adjust your `${INPUT_DIR}`, `${OUTPUT_DIR}`, and `${INV_ALPHA}` parameters as needed.

Feel free to open an issue or submit a pull request if you have questions or suggestions!