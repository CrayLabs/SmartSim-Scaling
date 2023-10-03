#!/bin/bash

#SBATCH -N 92
#SBATCH --exclusive
#SBATCH -t 10:00:00
#SBATCH -C SK48
#SBATCH --oversubscribe
echo "Note: The flag net_ifname should be replaced with the appropriate value on the target system"
cd ..
python driver.py throughput_standard --client_nodes=[60] \
                                    --clients_per_node=[48] \
                                    --db_nodes=[32] \
                                    --db_cpus=[32] --net_ifname=ipogif0 \
                                    --run_db_as_batch=False

