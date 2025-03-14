# nnUNetMD
This is a repository for self-adaptive nnU-Net models with incomplete modalities.

## Code structure
>Our codes are built upon the state-of-the-art medical image segmentation framework, nnU-Net. 
>- We create additional trainers to train self-adaptive models. The trainers can be found [here](https://github.com/han-liu/nnUNetMD/tree/main/nnunetv2/training/nnUNetTrainer/variants/missing_modality).
>- We slightly modify the base [nnUNetTrainer](https://github.com/han-liu/nnUNetMD/blob/main/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py)
>- We update the [inference code](https://github.com/han-liu/nnUNetMD/blob/main/nnunetv2/inference/predict_from_raw_data.py) so that we can predict with incomplete modalities.

## Usage
>The commands for preprocessing, training, and inference are the same as nnU-Net.
Examples of training scripts can be found [here](https://github.com/han-liu/nnUNetMD/tree/main/scripts).

#### Training
```
nnUNetv2_train 4 3d_fullres 0 -tr ${trainer}
```

#### Inference
```
nnUNetv2_predict -i ${input_dir} -o "${output_dir}" -d 4 -c 3d_fullres -f 0 -tr ${trainer} -code ${code} --disable_tta
```
