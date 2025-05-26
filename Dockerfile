#FROM --platform=linux/amd64 pytorch/pytorch
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 AS base

# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore"


RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    wget \
    unzip \
    libopenblas-dev \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    nano \
    && \
    python3.10 -m ensurepip --upgrade && \
    python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/algorithm
RUN chown -R user /opt/algorithm
ENV PATH="/home/user/.local/bin:${PATH}"

USER user

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python3.10 -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt


### Clone nnUNet
# Configure Git, clone the repository without checking out, then checkout the specific commit
RUN git config --global advice.detachedHead false && \
    git clone https://github.com/MIC-DKFZ/nnUNet.git /opt/algorithm/nnunet/ 

# Install a few dependencies that are not automatically installed
RUN pip3 install \
    -e /opt/algorithm/nnunet \
    graphviz \
    onnx \
    SimpleITK && \
    rm -rf ~/.cache/pip

COPY --chown=user:user ./packages/report-guided-annotation/ /opt/algorithm/report-guided-annotation/

# Install 'report-guided-annotation' in editable mode
RUN cd /opt/algorithm/report-guided-annotation && \
    python3.10 -m pip install --user --no-cache-dir -e . && \
    cd /

RUN mkdir -p /opt/algorithm/nnunet/nnunetv2/training/nnUNetTrainer/variants/loss/
COPY --chown=user:user ./packages/nnunetv2/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py /opt/algorithm/nnunet/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerCELoss.py

COPY --chown=user:user ./workspace/nnUNet_results/ /opt/algorithm/nnunet/nnUNet_results/

### Define workdir
WORKDIR /opt/app
COPY --chown=user:user ./main.py /opt/app/main.py

RUN rm -rf ~/.cache/pip
### Set environment variable defaults
ENV nnUNet_raw="/opt/algorithm/nnunet/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnunet/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnunet/nnUNet_results"

ENTRYPOINT [ "python3.10", "-m", "main"]