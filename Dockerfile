# Official NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Remove need for interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /service

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Clear apt cache

# Install Mambaforge
ENV MAMBA_ROOT_PREFIX=/opt/mambaforge
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ARG MAMBA_VERSION=25.3.0-3
RUN wget https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Miniforge3-${MAMBA_VERSION}-Linux-x86_64.sh -O mambaforge.sh && \
    bash mambaforge.sh -b -p $MAMBA_ROOT_PREFIX && \
    rm mambaforge.sh
# Remove installer

# Create conda environment with dependencies
RUN mamba create -y -c conda-forge -p /opt/env \
    boost-cpp \
    benchmark \
    gtest \
    cmake \
    hdf5 \
    hdf5-external-filter-plugins \
    compilers \
    bitshuffle \
    spdlog \
    gemmi \
    python=3.11 \
    pip \
    zocalo \
    workflows \
    pydantic \
    rich \
    && mamba clean -afy
# Clean conda cache at the end

# Add conda environment to PATH
ENV PATH=/opt/env/bin:$PATH
ENV CONDA_PREFIX=/opt/env

# Copy source and submodules
COPY . .
RUN git submodule update --init --recursive

# Build the C++/CUDA backend
RUN mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Set environment variables for the service
ENV SPOTFINDER=/service/build/spotfinder
ENV ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml

# Expose service port?
#EXPOSE 8000

# Start the service
CMD ["zocalo.service", "-s", "GPUPerImageAnalysis"]