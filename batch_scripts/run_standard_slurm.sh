#!/bin/bash

#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -t 10:00:00

cd ..
module load slurm
python driver.py inference_standard --client_nodes=[2] \
                                    --db_nodes=[1] --db_tpq=[1] \
                                    --db_cpus=[8] --batch_args='{"C":"P100", "exclusive": None}'
