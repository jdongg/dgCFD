#!/bin/bash

# Request a CPU partition and access to 1 GPU

#SBATCH --nodelist=gpu1212
#SBATCH -p gpu
#SBATCH --gres=gpu:2

# Request 1 CPU core
#SBATCH -n 1
#SBATCH -c 2

#SBATCH --mem 8000

#SBATCH -t 10:00:00
#SBATCH -o output

# Load CUDA module
module load cuda/10.0.130
module load gcc/7.2

# Compile CUDA program and run
g++ -c -O3 -std=c++11 src/parameters.cpp -I./inc
nvcc -g -Xptxas -O3 -use_fast_math -Xcompiler -fopenmp main.cu -I./inc parameters.o

nvidia-smi

time nvprof ./a.out
