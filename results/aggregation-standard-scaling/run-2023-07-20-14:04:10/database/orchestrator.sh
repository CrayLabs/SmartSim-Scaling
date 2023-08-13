#!/bin/bash

#SBATCH --output=/lus/cls01029/richaama/smartsim-scaling/SmartSim-Scaling/results/aggregation-standard-scaling/run-2023-07-20-14:04:10/database/orchestrator.out
#SBATCH --error=/lus/cls01029/richaama/smartsim-scaling/SmartSim-Scaling/results/aggregation-standard-scaling/run-2023-07-20-14:04:10/database/orchestrator.err
#SBATCH --job-name=orchestrator-CU78YGO7H8LB
#SBATCH --nodes=16
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=36

cd /lus/cls01029/richaama/smartsim-scaling/SmartSim-Scaling/results/aggregation-standard-scaling/run-2023-07-20-14:04:10/database ; /opt/slurm/20.11.5/bin/srun --output /lus/cls01029/richaama/smartsim-scaling/SmartSim-Scaling/results/aggregation-standard-scaling/run-2023-07-20-14:04:10/database/orchestrator.out --error /lus/cls01029/richaama/smartsim-scaling/SmartSim-Scaling/results/aggregation-standard-scaling/run-2023-07-20-14:04:10/database/orchestrator.err --job-name orchestrator-CU78YGO7HZ52 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_0-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_1-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_2-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_3-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_4-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_5-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_6-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_7-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_8-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_9-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_10-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_11-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_12-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_13-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_14-6780.conf : --ntasks=1 --ntasks-per-node=1 --cpus-per-task=36 --job-name orchestrator-CU78YGO7HZ52 /lus/scratch/richaama/miniconda3/envs/plz3/bin/python -m smartsim._core.entrypoints.redis +ifname=ipogif0 +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --cluster-enabled yes --cluster-config-file nodes-orchestrator_15-6780.conf &

wait
