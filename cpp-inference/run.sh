#!/bin/bash

#SBATCH -N 16
#SBATCH --exclusive
#SBATCH -C P100
#SBATCH -t 10:00:00

source ../use_keydb.sh
python driver.py resnet
