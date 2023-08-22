#!/bin/bash

#SBATCH -N 16
#SBATCH -C "P100*16"
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
python driver.py inference_colocated --nodes=[4, 8, 12, 16]
