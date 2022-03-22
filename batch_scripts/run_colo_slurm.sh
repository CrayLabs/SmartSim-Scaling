#!/bin/bash

#SBATCH -N 12
#SBATCH --exclusive
#SBATCH -C P100
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py inference_colocated --clients_per_node=[16,18,24] \
                                     --nodes=[12] --db_tpq=[2] \
                                     --db_cpus=[4] --pin_app_cpus=[True]
