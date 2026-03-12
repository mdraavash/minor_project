# ============================================================
# Dockerfile: Endometrial Cancer Pipeline
# Base: NVIDIA PyTorch + CUDA 12.1 (compatible with RTX 3060 Ti)
# ============================================================

FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL maintainer="Endometrial Cancer Pipeline"
LABEL description="nnU-Net + 3D Classifier for CPTAC-UCEC endometrial cancer grading/segmentation"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ─── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget curl git unzip \
    libgl1-mesa-glx libglib2.0-0 \
    dcmtk \
    && rm -rf /var/lib/apt/lists/*

# ─── Python packages ──────────────────────────────────────────────────────────
RUN pip install --upgrade pip

# Core medical imaging
RUN pip install \
    nibabel==5.2.0 \
    pydicom==2.4.4 \
    SimpleITK==2.3.1 \
    scikit-image==0.22.0 \
    scipy==1.12.0

# nnU-Net v2 (pinned for stability)
RUN pip install \
    nnunetv2==2.4.1 \
    hiddenlayer==0.3

# MONAI for 3D DenseNet classifier
RUN pip install \
    monai[all]==1.3.0

# Training utilities
RUN pip install \
    pandas==2.2.0 \
    scikit-learn==1.4.0 \
    tensorboard==2.16.2 \
    tqdm==4.66.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.2

# ─── nnU-Net environment variables ────────────────────────────────────────────
# Paths match the server's shared directory mounted into the container
ENV nnUNet_raw="/data/nnunet_raw"
ENV nnUNet_preprocessed="/data/nnunet_preprocessed"
ENV nnUNet_results="/data/nnunet_results"

# ─── Copy pipeline scripts ────────────────────────────────────────────────────
WORKDIR /workspace
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true

# ─── Internal data mount points ───────────────────────────────────────────────
# Actual data lives on server at /mnt/data/shared-data/endometrial-cancer/
# and is mounted at runtime — nothing is copied into the image
RUN mkdir -p /data/nifti_output \
             /data/nnunet_raw \
             /data/nnunet_preprocessed \
             /data/nnunet_results \
             /data/classification \
             /data/inference/input \
             /data/inference/output

CMD ["/bin/bash"]