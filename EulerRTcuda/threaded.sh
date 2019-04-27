#!/bin/bash

# Request an hour of runtime:
#SBATCH --time=04:00:00

# Default resources are 1 core with 2.8GB of memory per core.

# Use more cores (16):
#SBATCH -c 24
#SBATCH --mem 7000

#SBATCH -p apma2822

# Specify a job name:
#SBATCH -J MyThreadedJob

# Specify an output file
#S`BATCH -o MyThreadedJob-%j.out
#SBATCH -e MyThreadedJob-%j.out

export OMP_NUM_THREADS=24
export GOMP_CPU_AFFINITY=0-23

# Run a command
cd results
rm results*
cd ..

make main
time ./main -threads 24 > output
