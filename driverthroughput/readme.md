### Throughput

The throughput tests run as an MPI program where a single SmartRedis C++ client
is initialized on every rank.

Each client performs 10 executions of the following commands

  1) ``put_tensor``     (send image to database)
  2) ``unpack_tensor``  (Retrieve the inference result)


The input parameters to the test are used to generate permutations
of tests with varying configurations.

```text

NAME
    driver.py throughput_scaling - Run the throughput scaling tests

SYNOPSIS
    driver.py throughput_scaling <flags>

DESCRIPTION
    Run the throughput scaling tests

FLAGS
    --exp_name=EXP_NAME
        Default: 'throughput-scaling'
        name of output dir
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --run_db_as_batch=RUN_DB_AS_BATCH
        Default: True
        run database as separate batch submission each iteration
    --batch_args=BATCH_ARGS
        Default: {}
        additional batch args for the database
    --db_hosts=DB_HOSTS
        Default: []
        optionally supply hosts to launch the database on
    --db_nodes=DB_NODES
        Default: [12]
        number of compute hosts to use for the database
    --db_cpus=DB_CPUS
        Default: [2]
        number of cpus per compute host for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --clients_per_node=CLIENTS_PER_NODE
        Default: [32]
        client tasks per compute node for the synthetic scaling producer app
    --client_nodes=CLIENT_NODES
        Default: [128, 256, 512]
        number of compute nodes to use for the synthetic scaling producer app
    --tensor_bytes=TENSOR_BYTES
        Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
        list of tensor sizes in bytes
```
