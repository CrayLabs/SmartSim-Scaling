#!/bin/bash

#SBATCH -N 4
#SBATCH -C "P100*4"
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py inference_colocated --nodes=[4]
