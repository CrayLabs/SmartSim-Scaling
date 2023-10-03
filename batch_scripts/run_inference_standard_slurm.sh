#!/bin/bash

#SBATCH -N 116
#SBATCH -C "[P100*16&SK48*100]"
#SBATCH --exclusive
#SBATCH -t 10:00:00
echo "Note: The flag net_ifname should be replaced with the appropriate value on the target system"
cd ..
python driver.py inference_standard --client_nodes=[25, 50, 75, 100] \
                                    --db_nodes=[4, 8, 16] --db_tpq=[1] \
                                    --db_cpus=[8]
