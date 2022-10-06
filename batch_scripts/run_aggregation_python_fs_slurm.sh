#!/bin/bash

#SBATCH -N 61
#SBATCH --exclusive
#SBATCH -t 24:00:00

cd ..
module load slurm
python driver.py aggregation_scaling_python_fs --exp_name='aggregation-scaling-py-fs-batch' \
                                               --client_nodes=[60] \
                                               --clients_per_node=[48] \
                                               --tensors_per_dataset=[1,4] \
                                               --tensor_bytes=[1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
                                               --client_threads=[1,2,4,8,16,32] \
                                               --cpu_hyperthreads=2 \
                                               --iterations=20