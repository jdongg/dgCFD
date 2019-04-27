#!/bin/bash

# Request a CPU partition and access to 1 GPU

#SBATCH --nodelist=gpu1212
#SBATCH -p gpu

# Request 1 CPU core
#SBATCH -n 1

#SBATCH -t 00:05:00
#SBATCH -o output

# Load CUDA module
module load cuda/10.0.130
module load gcc/7.2

# Compile CUDA program and run
# nvcc -c -g cuda/DGsolveCUDA.cu -I./inc
g++ -c -O3 -std=c++11 src/parameters.cpp -I./inc
nvcc -g -Xptxas -O3 -use_fast_math  main.cu -I./inc parameters.o

nvidia-smi

# nvprof -o SpMv_unified.nvprof ./main_SpMv
# nvprof --kernels triad_kernel --metrics all ./example4
# rm main.nvprof
nvprof ./a.out
# nvprof --print-api-trace ./example4
