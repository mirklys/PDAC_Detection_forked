import argparse
from glob import glob
import os
import os.path as osp
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import json
import SimpleITK as sitk
import subprocess
import time
import shutil
import warnings
from report_guided_annotation.extract_lesion_candidates import extract_lesion_candidates
import re
from pathlib import Path
warnings.filterwarnings("ignore")


_EXT_RE = re.compile(r'\.(nii(\.gz)?|mha)(\(.+)?\)?$', re.IGNORECASE)
def is_image_file(path: os.PathLike) -> bool:
    """Return True if *path* looks like a medical image we can read."""
    return bool(_EXT_RE.search(Path(path).name))

def find_images(root):
    """Return every file under *root* that looks like an image."""
    return sorted(p for p in Path(root).rglob('*') if p.is_file() and is_image_file(p))


# ONE regex that matches .nii, .nii.gz, .mha – *with* or *without* GC’s "(…)"
def strip_id(fp: str) -> str:
    """
    Returns the *filename without extension*, trimming both ".nii" / ".nii.gz"
    / ".mha" and the optional GC wrapper "(...)".
    """
    name = Path(fp).name
    # drop everything from the first matching extension onwards
    m = _EXT_RE.search(name)
    return name[:m.start()] if m else Path(name).stem


def get_args_parser():
    parser = argparse.ArgumentParser("PDAC detection", add_help=True)
    parser.add_argument("-i", "--input_dir",
                    type=str,
                    default="/input",          
                    help="input directory that contains CECT and metadata")

    parser.add_argument("-o", "--output_dir",
                    type=str,
                    default="/output",         
                    help="output directory that saves prediction results")
    parser.add_argument("-m", "--model_dir",   type=str, default="./workspace/nnUNet_results", help="model directory for nnU-Net")
    parser.add_argument("--inv_alpha",   type=int, default=15, help="parameter to control the expansion of the lesion")
    args = parser.parse_args()
    return args


def get_file_extension(image_fp):
    base, ext = osp.splitext(image_fp)
    if ext == ".gz" and base.endswith(".nii"):
        return ".nii.gz"
    return ext


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False, out_size = [], out_origin = [], out_direction= []):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    if not out_size:
        out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                    int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                    int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    if not out_direction:
        out_direction = itk_image.GetDirection()
    resample.SetOutputDirection(out_direction)
    if not out_origin:
        out_origin = itk_image.GetOrigin()
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)
    return itk_image


def downsample_panorama_dataset(img_dir, img_save_dir, resample=(4.5, 4.5, 9.0)):
    assert osp.exists(img_dir), f'image directory does not exist: {img_dir}'
    if not osp.exists(img_save_dir):
        os.mkdir(img_save_dir)
    # downsample_panorama_dataset
    img_paths = find_images(img_dir)
    if len(img_paths) == 0:
        print('No images found in input directory')
    with tqdm(total=len(img_paths)) as pbar:
        for img_path in img_paths:
            # ext = get_file_extension(img_path)
            out_name = strip_id(img_path) + '_0000.nii.gz'
            itk_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            image_resampled = resample_img(itk_img, resample, is_label=False, out_size = [])
            # sitk.WriteImage(image_resampled, osp.join(img_save_dir, osp.basename(img_path).replace(ext, '_0000.nii.gz')))
            sitk.WriteImage(image_resampled, Path(img_save_dir) / out_name)
            pbar.update(1)


def crop_roi(img_dir, low_msk_dir, save_img_dir, margins=[100, 50, 15]):
    if not osp.exists(save_img_dir):
        os.mkdir(save_img_dir)
    img_paths = find_images(img_dir)

    crop_coordinates = {}
    with tqdm(total=len(img_paths)) as pbar:
        for img_path in img_paths:
            ext = get_file_extension(img_path)
            # low_msk_path = osp.join(low_msk_dir, osp.basename(img_path).replace(ext, '.nii.gz'))
            low_msk_path = Path(low_msk_dir) / (strip_id(img_path) + '.nii.gz')
            
            img = sitk.ReadImage(img_path, sitk.sitkFloat32)
            low_msk_path = Path(low_msk_dir) / (strip_id(img_path) + '.nii.gz')
            low_msk = sitk.ReadImage(low_msk_path)
            pancreas_mask_np = (sitk.GetArrayFromImage(low_msk) != 0).astype(np.uint8)
            pancreas_mask_nonzeros = np.nonzero(pancreas_mask_np)
            min_x = min(pancreas_mask_nonzeros[2])
            min_y = min(pancreas_mask_nonzeros[1])
            min_z = min(pancreas_mask_nonzeros[0])
            max_x = max(pancreas_mask_nonzeros[2])
            max_y = max(pancreas_mask_nonzeros[1])
            max_z = max(pancreas_mask_nonzeros[0])
            start_point_coordinates = (int(min_x), int(min_y), int(min_z))
            finish_point_coordinates = (int(max_x), int(max_y), int(max_z))          
            start_point_physical = low_msk.TransformIndexToPhysicalPoint(start_point_coordinates)
            finish_point_physical = low_msk.TransformIndexToPhysicalPoint(finish_point_coordinates)
            start_point = img.TransformPhysicalPointToIndex(start_point_physical)
            finish_point = img.TransformPhysicalPointToIndex(finish_point_physical)
            spacing = img.GetSpacing()
            size = img.GetSize()
            marginx = int(margins[0]/spacing[0])
            marginy = int(margins[1]/spacing[1])
            marginz = int(margins[2]/spacing[2])
            x_start = max(0, start_point[0] - marginx)
            x_finish = min(size[0], finish_point[0] + marginx)
            y_start = max(0, start_point[1] - marginy)
            y_finish = min(size[1], finish_point[1] + marginy)
            z_start = max(0, start_point[2] - marginz)
            z_finish = min(size[2], finish_point[2] + marginz)
            cropped_image = img[x_start:x_finish, y_start:y_finish, z_start:z_finish]
            crop_coordinates[strip_id(img_path)] = {
                'x_start': x_start, 'x_finish': x_finish,
                'y_start': y_start, 'y_finish': y_finish,
                'z_start': z_start, 'z_finish': z_finish}
            cropped_name = strip_id(img_path) + '_0000.nii.gz'
            sitk.WriteImage(cropped_image, Path(save_img_dir) / cropped_name)
            # sitk.WriteImage(cropped_image, osp.join(save_img_dir, osp.basename(img_path).replace(ext, '_0000.nii.gz')))
            pbar.update(1)
    return crop_coordinates


def predict(nnunet_model_dir, input_dir, output_dir, task:int="107", trainer:str="nnUNetTrainerTverskyLoss", plan:str="nnUNetPlans",
            configuration="3d_fullres", checkpoint="checkpoint_final.pth", 
            folds="0", store_probability_maps=True, tta=False):
    
    os.environ['RESULTS_FOLDER'] = str(nnunet_model_dir)
    cmd = [
        'nnUNetv2_predict',
        '-d',  str(task),
        '-i',  str(input_dir),
        '-o',  str(output_dir),
        '-c',  configuration,
        '-tr', trainer,
        '-p',  plan,
        '--disable_tta',
        '--disable_progress_bar',
        '--continue_prediction'
    ]
    if folds:
        cmd.append('-f')
        cmd.extend(folds.split(','))
    if checkpoint:
        cmd.append('-chk')
        cmd.append(checkpoint)
    if store_probability_maps:
        cmd.append('--save_probabilities')
    if not tta:
        cmd.append('--disable_tta')

    cmd_str = " ".join(cmd)
    subprocess.check_call(cmd_str, shell=True)


def ensemble(nnunet_model_dir, input_dirs, output_dir):
    os.environ['RESULTS_FOLDER'] = str(nnunet_model_dir)
    cmd = [
        'nnUNetv2_ensemble',
        '-i',  str(" ".join(input_dirs)),
        '-o',  str(output_dir),
        '--save_npz',
    ]
    cmd_str = " ".join(cmd)
    subprocess.check_call(cmd_str, shell=True)


def PostProcessing(cropped_prediction, pred_path_nifti):
    prediction_np = cropped_prediction['probabilities'][1]
    prediction_np = prediction_np.astype(np.float32)
    return prediction_np


def GetFullSizDetectionMap(prediction_np, crop_coordinates, full_image, inv_alpha=15):
    lesion_candidates, confidences, indexed_pred = extract_lesion_candidates(prediction_np, dynamic_threshold_factor=inv_alpha)  
    patient_level_prediction = float(np.max(lesion_candidates))
    full_size_detection_map = np.zeros(sitk.GetArrayFromImage(full_image).shape)
    full_size_detection_map = full_size_detection_map.astype(np.float32)
    # Use integer slicing, ensuring no slice is empty
    z_slice = slice(int(crop_coordinates['z_start']), int(crop_coordinates['z_finish']))
    y_slice = slice(int(crop_coordinates['y_start']), int(crop_coordinates['y_finish']))
    x_slice = slice(int(crop_coordinates['x_start']), int(crop_coordinates['x_finish']))
    full_size_detection_map[z_slice, y_slice, x_slice] = lesion_candidates
    full_size_detection_map = full_size_detection_map.astype(np.float32)
    detection_map_image = sitk.GetImageFromArray(full_size_detection_map)
    detection_map_image.CopyInformation(full_image)
    return detection_map_image, patient_level_prediction


def write_json_file(*, location, content):
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))


def run(args):
    start = time.time()
    working_folder = osp.join(args.output_dir, "itm")
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not osp.exists(working_folder):
        os.mkdir(working_folder)

    image_folder = osp.join(args.input_dir)
    clinical_info_path = osp.join(args.input_dir, "clinical-information-pancreatic-ct.json")

    try:
        with open(clinical_info_path, 'r') as file:
            clinical_info = json.load(file)
        print('Clinical Information:')
        print('age:', clinical_info['age'])
        print('sex:',clinical_info['sex'])
        print('study date:',clinical_info['study_date'])
        print('scanner:',clinical_info['scanner'])
    except:
        pass
    
    # Step 1: downsample the dataset 
    print("Step 1/4: downsample the input image...")
    low_image_folder = osp.join(working_folder, 'LowImagesTr')
    downsample_panorama_dataset(image_folder, low_image_folder)

    # Step 2: inference on low resolution images using nnU-Net
    print("Step 2/4: predict on the low-resolution images...")
    low_pred_folder = osp.join(working_folder, 'LowPred')
    predict(
        nnunet_model_dir=args.model_dir, 
        input_dir=low_image_folder, 
        output_dir=low_pred_folder, 
        task=107,
        store_probability_maps=False)

    # Step 3: crop high resolution ROI
    print("Step 3/4: crop the ROI from the high-resolution image...")
    cropped_image_folder = osp.join(working_folder, 'CroppedImages')
    crop_coordinates = crop_roi(
        img_dir=image_folder, 
        low_msk_dir=low_pred_folder, 
        save_img_dir=cropped_image_folder, 
        margins=[100, 50, 15])

    # Step 4: predict on high resolution ROI using nnU-Net
    print("Step 4/4: detect PDAC on the high-resolution ROI...")
    cropped_pred_folder = osp.join(working_folder, 'CroppedPred')
    predict(
        nnunet_model_dir=args.model_dir, 
        input_dir=cropped_image_folder, 
        output_dir=cropped_pred_folder,
        task=107, 
        trainer="nnUNetTrainerTverskyLoss",
        plan="nnUNetPlans",
        folds="0",
        store_probability_maps=False)

    npz_fps = sorted(Path(cropped_pred_folder).glob('*.npz'))
    img_fps = find_images(image_folder)
    if len(img_fps) != len(npz_fps):
        raise RuntimeError(f"Got {len(img_fps)} images but {len(npz_fps)} predictions")
    patient_scores = []
    images_dir = osp.join(args.output_dir, "images", "pdac-detection-map")
    os.makedirs(images_dir, exist_ok=True)
    for npz_fp, img_fp in zip(npz_fps, img_fps):
        filename = osp.basename(npz_fp)[:-4]
        # ext = get_file_extension(img_fp)
        # assert osp.basename(img_fp)[:-len(ext)] == filename, f"{osp.basename(img_fp)[:-len(ext)]} and {filename}"
        case_id = strip_id(img_fp)
        # itk_img = sitk.ReadImage(img_fp, sitk.sitkFloat32)
        itk_img  = sitk.ReadImage(str(img_fp), sitk.sitkFloat32)
        mask_nifti = str(npz_fp).replace('.npz','.nii.gz')  # same path but .nii.gz
        mask_img   = sitk.ReadImage(mask_nifti)
        prediction_postprocessed = sitk.GetArrayFromImage(mask_img)  
        detection_map, pat_pred = GetFullSizDetectionMap(prediction_postprocessed, crop_coordinates[case_id], itk_img, args.inv_alpha)
        
        # Write detection map to the correct directory
        sitk.WriteImage(detection_map, os.path.join(images_dir, f"{case_id}.mha"))
        patient_scores.append(float(pat_pred))
    
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    
    # Write json to root output directory
    if len(patient_scores) == 1:
        payload = patient_scores[0]
    else:
        payload = patient_scores
    write_json_file(
        location=Path(args.output_dir) / "pdac-likelihood.json",
        content=payload
    )

    
    print(f"\nInference time: {(time.time()-start)/len(npz_fps):.2f} seconds per image")
    print(f"Patient-level scores are saved at: {osp.join(args.output_dir, 'pdac-likelihood.json')}")
    print(f"Lesion detection maps are saved at: {images_dir}\n")


def print_info():
    print("#############################################################################################################################################################################\n")
    print("Welcome to use Team DTI's PDAC detection models (1st place in the PANORAMA challenge)\n")
    print("Please cite the following paper when using our code and models:")

    print('\n\nLiu, H., et al. "AI-assisted Early Detection of Pancreatic Ductal Adenocarcinoma on Contrast-enhanced CT"\n\n')
    print("If you have questions or suggestions, feel free to open an issue at https://github.com/han-liu/PDAC_Detection/\n")

    print('This code is built upon the following works. We gratefully acknowledge their contribution and encourage users to cite their original work:')
    print('1. Isensee, Fabian, et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature methods')
    print('2. Bosma, Joeran S, et al. "Semi-supervised learning with report-guided pseudo labels for deep learning–based prostate cancer detection using biparametric MRI." Radiology AI')
    print('3. Alves, Natália,  et al. "Fully automatic deep learning framework for pancreatic ductal adenocarcinoma detection on computed tomography." Cancers\n')
    print("#############################################################################################################################################################################\n")



if __name__ == "__main__":
    args = get_args_parser()
    print_info()
    run(args)
