#!/bin/bash
echo "Setting up environment for MinkLoc3Dv2 with RTX 3090 Ti..."
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 设置GPU架构（支持RTX 3090 Ti的8.6）
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# 设置OpenMP线程数，消除MinkowskiEngine警告
export OMP_NUM_THREADS=12

# 激活conda环境
conda activate minkloc3dv2

# 验证环境
echo "CUDA version:"
nvcc --version
echo "GPU Architecture Support:"
echo "RTX 3090 Ti (Compute 8.6) should now be supported"
echo "OpenMP threads: $OMP_NUM_THREADS"
