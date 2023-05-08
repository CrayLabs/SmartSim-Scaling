#!/bin/bash

#SBATCH -N 60
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py inference_standard --client_nodes=[20,40,60] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[8,16]
