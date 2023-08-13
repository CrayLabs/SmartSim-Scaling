#!/bin/bash

#SBATCH -N 8
#SBATCH -C "[P100*4&SK48*4]"
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py inference_standard --client_nodes=[4] \
                                    --db_nodes=[4] --db_tpq=[1] \
                                    --db_cpus=[8]
