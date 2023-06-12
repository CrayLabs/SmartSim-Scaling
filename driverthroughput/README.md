### Throughput

The throughput tests run as an MPI program where a single SmartRedis C++ client
is initialized on every rank.

Each client performs 10 executions of the following commands

  1) ``put_tensor``     (send image to database)
  2) ``unpack_tensor``  (Retrieve the inference result)

The input parameters to the test are used to generate permutations
of tests with varying configurations.

## Co-located throughput

Co-located Orchestrators are deployed on the same nodes as the
application. This improves inference performance as no data movement
"off-node" occurs with co-located deployment. For more information
on co-located deployment, see [our documentation](https://www.craylabs.org/docs/orchestrator.html)

Below is the help output. The arguments which are lists control
the possible permutations that will be run.

```text
NAME
    driver.py inference_colocated - Run ResNet50 inference tests with colocated Orchestrator deployment

SYNOPSIS
    driver.py inference_colocated <flags>

DESCRIPTION
    Run ResNet50 inference tests with colocated Orchestrator deployment

FLAGS
    --exp_name=EXP_NAME
        Default: 'throughput-colocated-scaling'
        name of output dir
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --node_feature=NODE_FEATURE
        Default: {}
        dict of runsettings for both app and db
    --nodes=NODES
        Default: []
        compute nodes to use for synthetic scaling app with
                      a co-located orchestrator database
    --db_cpus=DB_CPUS
        Default: []
        number of cpus per compute host for the database
    --db_port_DB_PORT
        Default: 6780
        port to use for the database
    --net_ifname=NET_IFNAME
        Default: 'lo'
        network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
    --clients_per_node=CLIENT_PER_NODE
        Default: [48]
        client tasks per compute node for the synthetic scaling app
    --pin_app_cpus=PINE_APP_CPUS
        Default: [False]
        pin the threads of the application to 0-(n-db_cpus)
    --iterations=ITERATIONS
        Default: 100
        number of put/get loops run by the applications
    --tensor_bytes=TENSOR_BYTES
        Default: []
        list of tensor sizes in bytes
    --languages=LANGUAGES
        Default: []
        which language to use for the tester "cpp" or "fortran"
```

So for example, the following command could be run to execute a battery of
tests in the same allocation

```bash
python driver.py throughput_colocated --client_nodes=[20,40,60] \
                                     --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                     --db_cpus=[8,16]
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
#!/bin/bash

#SBATCH -N 16
#SBATCH --exclusive
#SBATCH -t 10:00:00

module load slurm
python driver.py throughput_colocated --client_nodes=[20,40,60] \
                                     --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                     --db_cpus=[8,16]
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory

## Standard throughput

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
        Default: 'throughput-standard-scaling'
        name of output dir
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --run_db_as_batch=RUN_DB_AS_BATCH
        Default: True
        run database as separate batch submission each iteration
    --node_feature=NODE_FEATURE
      Default: {}
      dict of runsettings for both app
    --db_node_feature=DB_NODE_FEATURE
      Default: {}
      dict of runsettings for the db
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
    --iterations=ITERATIONS
        Default: 100
        number of put/get loops run by the applications
    --tensor_bytes=TENSOR_BYTES
        Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
        list of tensor sizes in bytes
    --languages=LANGUAGES
        Default: []
        which language to use for the tester "cpp" or "fortran"
    --wall_time=WALL_TIME
        Default: '05:00:00'
        allotted time for database launcher to run
```

The standard throughput tests will spin up a database for each iteration in the
battery of tests chosen by the user. There are multiple ways to run this.

1. Everything in the same interactive (or batch file) without caring about placement
```bash
# alloc must contain at least 120 (max client_nodes) + 16 nodes (max db_nodes)
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --run_db_as_batch=False \
                                    --net_ifname=ipogif0 --device=GPU
```

This option is recommended as it's easy to launch in interactive allocations and
as a batch submission, but if you need to specify separate hosts for the database
you can look into the following two methods.

A batch submission for this first option would look like the following for Slurm
based systems.

```bash
#!/bin/bash

#SBATCH -N 136
#SBATCH --exclusive
#SBATCH -t 10:00:00

python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --run_db_as_batch=False
                                    --net_ifname=ipogif0 --device=CPU
```

2. Same as 1, but specify hosts for the database
```bash
# alloc must contain at least 120 (max client_nodes) + 16 nodes (max db_nodes)
# db nodes must be fixed if hostlist is specified
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --db_hosts=[nid0001, ...] \
                                    --net_ifname=ipogif0 --device=CPU

```

3. Launch database as a separate batch submission each time
```bash
# must obtain separate allocation for client driver through interactive or batch submission
# if batch submission, compute nodes must have access to slurm
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --batch_args='{"C":"V100", "exclusive": None}' \
                                    --net_ifname=ipogif0 --device=GPU
```

All three options will conduct ``n`` scaling tests where ``n`` is the multiple of
all lists specified as options.
