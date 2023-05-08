#PBS -S /bin/bash
#PBS -l select=576:ncpus=36
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -o throughput.out
#PBS -N smartsim-throughput
#PBS -V

PYTHON=/lus/snx11242/spartee/miniconda/envs/0.4.0/bin/python
cd $PBS_O_WORKDIR/../
$PYTHON driver.py throughput_standard --client_nodes=[128,256,512] \
                                     --clients_per_node=[36] \
                                     --db_nodes=[16,32,64] \
                                     --db_cpus=36 --net_ifname=ipogif0 \
                                     --run_db_as_batch=False

