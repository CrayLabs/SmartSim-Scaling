#!/bin/bash

#SBATCH -N 93
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -C SK48
#SBATCH --oversubscribe

cd ..
module load slurm
python driver.py aggregation_scaling --client_nodes=[60] \
                                     --clients_per_node=[48] \
                                     --db_nodes=[16,32] \
                                     --db_cpus=32 --net_ifname=ipogif0 \
                                     --run_db_as_batch=False \
                                     --tensors_per_dataset=[1,4]

