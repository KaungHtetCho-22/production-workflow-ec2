# Start from the PyTorch image with CUDA 11.8
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Install system packages required for PyTorch and librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch \
    torchaudio \
    torchvision \
    numpy==1.22.4 \
    pandas==1.4.3 \
    scikit-learn==1.1.3 \
    librosa==0.10.2.post1 \
    tqdm==4.64.0 \
    timm==0.6.12 \
    audiomentations==0.36.0 \
    torch-audiomentations==0.11.1

RUN pip install --no-cache-dir matplotlib==3.7.3 scipy==1.9.3 seaborn==0.12.1 pyyaml fastapi uvicorn

WORKDIR /home/work

EXPOSE 5000
