import json
import os
import subprocess

ROOT_FOLDER = "/home/bobby/repos/PDAC_Detection_forked/"
TEST_FOLDER = os.path.join(ROOT_FOLDER, "workspace/test")

os.makedirs(TEST_FOLDER, exist_ok=True)
os.makedirs(os.path.join(TEST_FOLDER, "imagesTs"), exist_ok=True)
os.makedirs(os.path.join(TEST_FOLDER, "labelsTs"), exist_ok=True)
os.makedirs(os.path.join(TEST_FOLDER, "labelsTs/tversky_predicted"), exist_ok=True)


def get_testing_fold(fold: int) -> list:
    SPLITS_PATH = "./workspace/nnUNet_preprocessed/splits_final.json"
    with open(SPLITS_PATH, "r") as f:
        folds = json.load(f)

    testing_fold = folds[fold].get("val", [])
    assert len(testing_fold) > 0, "No validation data found for the specified fold."

    return testing_fold


def get_images(data_folder: str, fold: int = 0) -> list:
    """Gets all the testing images/labels from the specified folder."""
    listing = os.listdir(data_folder)
    testing_fold = get_testing_fold(fold)

    if not listing:
        raise ValueError(f"The directory {data_folder} is empty or does not exist.")

    prefixes = tuple(testing_fold)
    images = [
        os.path.join(data_folder, file) for file in listing if file.startswith(prefixes)
    ]

    if not images:
        raise ValueError(f"No images found in {data_folder} for the specified fold.")
    return images


def move_images_to_test(images: list):
    for image in images:
        image_name = os.path.basename(image)
        new_image_path = os.path.join(TEST_FOLDER, "imagesTs", image_name)
        if not os.path.exists(new_image_path):
            os.symlink(image, new_image_path)


def create_tester_shell_script(
    test_folder: str, env_vars: dict, output_file: str = "test.sh"
) -> None:
    """Creates a shell script to run the testing process."""
    with open(output_file, "w") as f:
        f.write(
            """#!/bin/bash
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
conda activate pdac_tversky \n"""  ### WRITE YOUR ENVIRONEMNT
        )
        f.write(f"export nnUNet_raw={env_vars['nnUNet_raw']}\n")
        f.write(f"export nnUNet_preprocessed={env_vars['nnUNet_preprocessed']}\n")
        f.write(f"export nnUNet_results={env_vars['nnUNet_results']}\n")
        f.write(
            f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={env_vars['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']}\n"
        )
        f.write(f"export OMP_NUM_THREADS={env_vars['OMP_NUM_THREADS']}\n")
        f.write("\n")
        f.write(f"echo 'RAW=' $nnUNet_raw\n")
        f.write(
            f"""python -m main -i {test_folder}/imagesTs/ -o {test_folder}/labelsTs/tversky_predicted \n"""
        )
    os.chmod(output_file, 0o777)


def run_test(env_vars: dict, data_folder: str, bash_test_name: str):
    images = get_images(data_folder)
    move_images_to_test(images)
    create_tester_shell_script(TEST_FOLDER, env_vars, output_file=bash_test_name)
    # subprocess.run([os.path.join(ROOT_FOLDER, bash_test_name)])


if __name__ == "__main__":
    env_vars = {
        "nnUNet_raw": "./workspace/nnUNet_preprocessed",
        "nnUNet_preprocessed": "./workspace/nnUNet_preprocessed",
        "nnUNet_results": "./tversky_workspace/nnUNet_results",
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2",
        "OMP_NUM_THREADS": "2",
    }
    data_folder = os.path.join(ROOT_FOLDER, "data")
    bash_test_name = "tversky_test.sh"
    run_test(data_folder=data_folder, env_vars=env_vars, bash_test_name=bash_test_name)
