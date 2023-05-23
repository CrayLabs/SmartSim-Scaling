#!/bin/bash

#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p allgriz
#SBATCH -t 1:00:00

module load cudatoolkit/11.7 cudnn PrgEnv-intel
source ~/pyenvs/smartsim-dev/bin/activate

cd ..
python driver.py inference_colocated --clients_per_node=[12] \
                                     --nodes=[1] --db_tpq=[2] \
                                     --db_cpus=[12] --pin_app_cpus=[True] \
				     --net_type="uds" --node_feature='{}' --languages=['fortran']
