#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

export SMARTSIM_LOG_LEVEL=debug
/lus/scratch/richaama/miniconda3/envs/plz23/bin/python -m smartsim._core.entrypoints.colocated +lockfile smartsim-88a48c0.lock +db_cpus 8 +ifname lo +command /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/bin/redis-server /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/config/redis.conf --loadmodule /lus/cls01029/richaama/smartsim-scaling/SmartSim/smartsim/_core/lib/redisai.so --port 6780 --logfile /lus/cls01029/richaama/smartsim-scaling/SmartSim-Scaling/results/throughput-colocated-scaling/run-2023-08-07-10:03:47/throughput-sess-colo-cpp-N128-T48-DBCPU8-PINFalse-ITER100-TB1024-452f/throughput-sess-colo-cpp-N128-T48-DBCPU8-PINFalse-ITER100-TB1024-452f-db.log --loglevel notice &
DBPID=$!

$@

