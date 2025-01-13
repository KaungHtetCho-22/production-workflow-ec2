FROM python:3.12 AS devel

ENV TZ=Asia/Bangkok

RUN apt-get update && apt-get install --no-install-recommends -y sudo git curl wget lsb-release \
    software-properties-common

# tvm dependencies
RUN apt-get install -y make cmake libpolly-14-dev llvm clang libc++-dev libc++abi-dev lld zlib1g-dev
# RUN apt-get install -y make cmake zlib1g-dev

# WORKDIR /llvm-installer

# RUN curl -L -O https://apt.llvm.org/llvm.sh && \
#     chmod +x llvm.sh && \
#     sudo ./llvm.sh 15 all

# RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
#     update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
#     update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-15 100

# # Installing TVM
WORKDIR /tvm-builder

RUN git clone --recursive -b v0.18.0.rc0 https://github.com/apache/tvm tvm

RUN cd tvm && rm -rf build && mkdir build && cd build && \
    cp ../cmake/config.cmake . && \
    echo "set(CMAKE_BUILD_TYPE Release)" >> config.cmake && \
    echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake && \
    echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake && \
    echo "set(USE_CUDA   OFF)" >> config.cmake && \
    echo "set(USE_METAL  OFF)" >> config.cmake && \
    echo "set(USE_VULKAN OFF)" >> config.cmake && \
    echo "set(USE_OPENCL OFF)" >> config.cmake && \
    echo "set(USE_CUBLAS OFF)" >> config.cmake && \
    echo "set(USE_CUDNN  OFF)" >> config.cmake && \
    echo "set(USE_CUTLASS OFF)" >> config.cmake && \
    cmake .. && cmake --build . --parallel $(nproc) && \
    export TVM_LIBRARY_PATH=/tvm-builder/tvm/build && \
    pip install -e /tvm-builder/tvm/python

# WORKDIR /arm-toolchain

# RUN curl -O -L https://github.com/ARM-software/LLVM-embedded-toolchain-for-Arm/releases/download/release-14.0.0/LLVMEmbeddedToolchainForArm-14.0.0-linux.tar.gz && \
#     tar -xvf LLVMEmbeddedToolchainForArm-14.0.0-linux.tar.gz

ARG USERNAME=monsoon
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd -g $USER_GID $USERNAME

RUN useradd -rm -d /home/$USERNAME -s /bin/bash -g $USER_GID -G sudo -u $USER_UID $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# tvm needs xgboost to run static shape tuning
RUN pip install --no-cache-dir xgboost

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

USER $USERNAME

WORKDIR /home/$USERNAME


FROM python:3.12 AS prod

ENV TZ=Asia/Bangkok \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "monsoon_audio_biodiversity/audio_processor/audio_process.py"]
