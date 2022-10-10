#!/bin/bash

#SBATCH -N 93
#SBATCH --exclusive
#SBATCH -t 24:00:00

cd ..
module load slurm
python driver.py aggregation_scaling_python --exp_name='aggregation-scaling-py-batch' \
                                            --client_nodes=[60] \
                                            --clients_per_node=[48] \
                                            --db_nodes=[16] \
                                            --db_cpus=32 \
                                            --net_ifname=ipogif0 \
                                            --run_db_as_batch=False \
                                            --tensors_per_dataset=[1,4] \
                                            --tensor_bytes=[1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
                                            --client_threads=[1,2,4,8,16,32] \
                                            --cpu_hyperthreads=2 \
                                            --iterations=20

