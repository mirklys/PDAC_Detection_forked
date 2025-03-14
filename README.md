# :trophy: 1st place in the PANORAMA Challenge (Team DTI)
[![arXiv](https://img.shields.io/badge/arXiv-2311.12437-blue)](https://arxiv.org/abs/2503.10068) [![cite](https://img.shields.io/badge/cite-BibTex-yellow)](xx) [![website](https://img.shields.io/badge/Challenge%20website-50d13d)](https://panorama.grand-challenge.org/)

This is the implementation for the paper:
[AI-assisted Early Detection of Pancreatic Ductal Adenocarcinoma on Contrast-enhanced CT](https://arxiv.org/abs/2503.10068)

If you find our code/paper helpful for your research, please kindly consider citing our work:
```
@incollection{liu2023learning,
  title={AI-assisted Early Detection of Pancreatic Ductal Adenocarcinoma on Contrast-enhanced CT},
  author={Liu, Han and Gao, Riqiang and Grbic, Sasa},
  journal={arXiv preprint arXiv:2503.10068},
  year={2025}
}
```

If you have any questions, feel free to contact han.liu@siemens-healthineers.com or open an Issue in this repo. 

---

### Installation
- Requirements: cuda-11.1, cudnn/9.0.0-cuda-12
- Create a virtual environment:
```
conda create pdac python=3.12 -y
conda activate pdac
```

- Install dependencies
```
cd PDAC_detection
pip install -r requirements.txt

cd packages/nnunetv2
pip install -e .
    
cd ../report-guided-annotation
pip install -e .
```

### Inference
- Set up environment variables for nnU-Net
```
export nnUNet_raw="./workspace/nnUNet_raw"
export nnUNet_preprocessed="./workspace/nnUNet_preprocessed"
export nnUNet_results="./workspace/nnUNet_results"
```

- Run our models
```
python main.py -i ${INPUT_DIR} -o ${OUTPUT_DIR} --inv_alpha ${INV_ALPHA}
```

- We prepare an testing example in workspace/test_example. For a quick test:
```
python main.py -i ./workspace/test_example/input -o ./workspace/test_example/output
```
inv_alpha controls the expansion of the predicted lesion. Larger inv_alpha will predict larger lesions. Defaults to 15.


### Acknowledgement
This code is built upon the following works. We gratefully acknowledge their contribution and encourage users to cite their original work:
1. Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods
2. Bosma, Joeran S, et al. "Semi-supervised learning with report-guided pseudo labels for deep learning–based prostate cancer detection using biparametric MRI." Radiology AI
3. Alves, Natália,  et al. "Fully automatic deep learning framework for pancreatic ductal adenocarcinoma detection on computed tomography." Cancers