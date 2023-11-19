#!/bin/bash

#SBATCH -N 16
#SBATCH -C "P100*16"
#SBATCH --exclusive
#SBATCH -t 10:00:00
echo "Note: The flag net_ifname should be replaced with the appropriate value on the target system"
cd ..
python driver.py inference_colocated --nodes=[4, 8, 12, 16]
