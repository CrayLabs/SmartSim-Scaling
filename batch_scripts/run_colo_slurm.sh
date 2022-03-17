#!/bin/bash

#SBATCH -N 16
#SBATCH --exclusive
#SBATCH -C P100
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py inference_colocated --clients_per_node=[24,28] \
                                     --nodes=[16] --db_tpq=[2] \
                                     --db_cpus=[4] --pin_app_cpus=[False]
