# Data Aggregation Scaling Tests

SmartSim-Scaling currently offers three data aggregation test versions:

 1. Data Aggregation Standard               (C++ Application, SmartRedis Orchestrator)
 2. Data Aggregation Standard Python        (Python Application, SmartRedis Orchestrator)
 3. Data Aggregation Standard File System   (Python Application, File System)

## Client Description

The data aggregation scaling test runs two applications. The first
is an MPI application that produces datasets that are added to an aggregation list.
In this producer application, each MPI rank has a single-threaded client.  The second
application is a consumer application. This application consumes the aggregation
lists that are produced by the first application.  The consumer application
can be configured to use multiple threads for data aggregation.  The producer and consumer
applications are running at the same time, but the producer application waits for the
consumer application to finish an aggregation list before starting to produce
the next aggregation list.

By default, the clients in the producer application perform an adjustable 100 executions of the following command:

  1) ``append_to_list`` (add dataset to the aggregation list)

Note that the client on rank 0 of the producer application performs a ``get_list_length()``
function invocation prior to a ``MPI_BARRIER`` in order to only produce the next aggregation
list after the previous aggregation list was consumed by the consumer application.

There is only a single MPI rank for the consumer application, which means there is only
one SmartRedis client active for the consumer application.  The consumer application client
invokes the following SmartRedis commands:

  1) ``poll_list``  (check when the next aggregation list is ready)
  2) ``get_list``   (retrieve the data from the aggregation list)

The input parameters to the test are used to generate permutations
of tests with varying configurations.

## Why Standard Deployment for Data Aggregation vs Colocated?

SmartSim-Scaling supports standard deployment for all three data aggregation tests. For standard deployement, the Orchestrator database and the application will be launched on different nodes: database nodes and client nodes. The client performing the data aggregation will spawn a number of threads, determined by the environment variable `SR_THREAD_COUNT` to retrieve datasets from the list in parallel. Each thread is assigned to a specific database shard, so increasing `SR_THREAD_COUNT` higher than the amount of shards is not likely to increase performance.

For example, if you had a dataset list of length 8 (each dataset stored on a separate shard), then you could pull all 8 of them simultaneously if you had 8 threads for SmartRedis which is hypothetically 8x faster than requesting them one at a time with colocated deployement.

Colocated deployment supports a database on the same node as the application. If we used 
colocated deployement for data aggregation, there would be no performance benefit since it would aggregate the data in serial like mentioned previously.

## Data Aggregation Standard - C++ Client

The ``aggregation_scaling`` test uses a c++ client and a SmartRedis Orchestrator.

> Note that the number of threads per client should be less than or equal 
(most performant) to the number of database shards.

```text

NAME
    driver.py aggregation_scaling - Run the data aggregation scaling tests with standard Orchestrator deployment

SYNOPSIS
    driver.py aggregation-scaling <flags>

DESCRIPTION
    Run the data aggregation scaling tests with standard Orchestrator deployment

FLAGS
    --exp_name=EXP_NAME
        Default: 'aggregation-standard-scaling'
        name of output dir
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --run_db_as_batch=RUN_DB_AS_BATCH
        Default: True
        run database as separate batch submission each iteration
    --db_node_feature=DB_NODE_FEATURE
        Default: {}
        dict of runsettings for the db
    --prod_node_feature=PROD_NODE_FEATURE
        Default: {}
        dict of runsettings for the producer
    --cons_node_feature=CONS_NODE_FEATURE
        Default: {}
        dict of runsettings for the consumer
    --db_hosts=DB_HOSTS
        Default: []
        optionally supply hosts to launch the database on
    --db_nodes=DB_NODES
        Default: [16]
        number of compute hosts to use for the database
    --db_cpus=DB_CPUS
        Default: 36
        number of cpus per compute host for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --clients_per_node=CLIENTS_PER_NODE
        Default: [48]
        client tasks per compute node for the synthetic scaling app
    --client_nodes=CLIENT_NODES
      Default: [128, 256, 512]
        number of compute nodes to use for the synthetic scaling app
    --iterations=ITERATIONS
        Default: 20
        number of append/retrieve loops run by the applications
    --tensor_bytes=TENSOR_BYTES
        Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
        list of tensor sizes in bytes
    --tensors_per_dataset=TENSORS_PER_DATASET
        Default: [1, 2, 4]
        list of number of tensors per dataset
    --client_threads=CLIENT_THREADS
        Default: [1, 2, 4, 8, 16, 32]
        list of the number of client threads used for data aggregation
    --cpu_hyperthreads==CPU_HYPERTHREADS
        Default: 2
        the number of hyperthreads per cpu.  This is done
                                 to request that the consumer application utilizes
                                 all physical cores for each client thread.
    --languages=LANGUAGES
        Default: ['cpp']
        list of languages to use for the tester "cpp" and/or "fortran"
    --wall_time=WALL_TIME
        Default: '05:00:00'
        allotted time for database launcher to run
    --plot=PLOT
        Default: 'database_nodes'
        flag to plot against in process results
```
For example, the following command could be run to execute a battery of
tests in the same allocation. The battery of test will be determined by the number 
of permutations computed based on the list inputs.

> The interface name may be different on your target system. Please update the `net_ifname` flag to the appropriate value.

```bash
python driver.py aggregation_scaling --client_nodes=[60] \
                                     --clients_per_node=[48] \
                                     --db_nodes=[16,32] \
                                     --db_cpus=32 --net_ifname="ipogif0" \
                                     --run_db_as_batch=False \
                                     --tensors_per_dataset=[1,4]
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
#!/bin/bash

#SBATCH -N 93
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -C SK48
#SBATCH --oversubscribe

cd ..
module load slurm
python driver.py aggregation_scaling --client_nodes=[60] \
                                     --clients_per_node=[48] \
                                     --db_nodes=[16,32] \
                                     --db_cpus=32 --net_ifname="ipogif0" \
                                     --run_db_as_batch=False \
                                     --tensors_per_dataset=[1,4]
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory


## Data Aggregation Standard - Python Client

The ``aggregation_scaling_python`` test uses a Python client and a SmartRedis Orchestrator.


```text

NAME
    driver.py aggregation_scaling_python - Run the data aggregation scaling tests with standard Orchestrator deployment

SYNOPSIS
    driver.py aggregation_scaling_python <flags>

DESCRIPTION
    Run the data aggregation scaling tests with standard Orchestrator deployment

FLAGS
  --exp_name=EXP_NAME
      Default: 'aggregation-standard-scaling-py'
      name of output dir
  --launcher=LAUNCHER
      Default: 'auto'
      workload manager i.e. "slurm", "pbs"
  --run_db_as_batch=RUN_DB_AS_BATCH
      Default: True
      run database as separate batch submission
                                    each iteration
  --db_node_feature=DB_NODE_FEATURE
      Default: {}
      dict of runsettings for db
  --prod_node_feature=PROD_NODE_FEATURE
      Default: {}
      dict of runsettings for the producer
  --cons_node_feature=CONS_NODE_FEATURE
      Default: {}
      dict of runsettings for the consumer
  --db_hosts=DB_HOSTS
      Default: []
      optionally supply hosts to launch the database on
  --db_nodes=DB_NODES
      Default: [16]
      number of compute hosts to use for the database
  --db_cpus=DB_CPUS
      Default: 36
      number of cpus per compute host for the database
  --db_port=DB_PORT
      Default: 6780
      port to use for the database
  --net_ifname=NET_IFNAME
      Default: 'ipogif0'
      network interface to use i.e. "ib0" for infiniband or
                            "ipogif0" aries networks
  --clients_per_node=CLIENTS_PER_NODE
      Default: [48]
      client tasks per compute node for the aggregation
                                    producer app
  --client_nodes=CLIENT_NODES
      Default: [128, 256, 512]
      number of compute nodes to use for the aggregation
                                producer app
  --iterations=ITERATIONS
      Default: 20
      number of append/retrieve loops run by the applications
  --tensor_bytes=TENSOR_BYTES
      Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
      list of tensor sizes in bytes
  --tensors_per_dataset=TENSORS_PER_DATASET
      Default: [1, 2, 4]
      list of number of tensors per dataset
  --client_threads=CLIENT_THREADS
      Default: [1, 2, 4, 8, 16, 32]
      list of the number of client threads used for data
                                aggregation
  --cpu_hyperthreads=CPU_HYPERTHREADS
      Default: 2
      the number of hyperthreads per cpu.  This is done
                                    to request that the consumer application utilizes
                                    all physical cores for each client thread
    --languages=LANGUAGES
        Default: ['cpp']
        list of languages to use for the tester "cpp" and/or "fortran"
    --wall_time=WALL_TIME
        Default: '05:00:00'
        allotted time for database launcher to run
    --plot=PLOT
        Default: 'database_nodes'
        flag to plot against in process results
```

For example, the following command could be run to execute a battery of
tests in the same allocation. The number of tests executed will be computed based
on the number of permutations from the list inputs given.

> The interface name may be different on your target system. Please update the `net_ifname` flag to the appropriate value.

```bash
python driver.py aggregation_scaling_python --exp_name='aggregation-scaling-py-batch' \
                                            --client_nodes=[60] \
                                            --clients_per_node=[48] \
                                            --db_nodes=[16] \
                                            --db_cpus=32 \
                                            --net_ifname="ipogif0" \
                                            --run_db_as_batch=False \
                                            --tensors_per_dataset=[1,4] \
                                            --tensor_bytes=[1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
                                            --client_threads=[1,2,4,8,16,32] \
                                            --cpu_hyperthreads=2 \
                                            --iterations=20
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
#!/bin/bash

#SBATCH -N 93
#SBATCH --exclusive
#SBATCH -t 24:00:00

cd ..
module load slurm
python driver.py aggregation_scaling_python --exp_name='aggregation-scaling-py-batch' \
                                            --client_nodes=[60] \
                                            --clients_per_node=[48] \
                                            --db_nodes=[16] \
                                            --db_cpus=32 \
                                            --net_ifname="ipogif0" \
                                            --run_db_as_batch=False \
                                            --tensors_per_dataset=[1,4] \
                                            --tensor_bytes=[1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
                                            --client_threads=[1,2,4,8,16,32] \
                                            --cpu_hyperthreads=2 \
                                            --iterations=20
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory


## Data Aggregation Standard - Python Client and File System

The ``aggregation_scaling_python_fs`` test uses a Python client with the file system in replacement of SmartRedis.

```text

NAME
    aggregation_scaling_python_fs - Run the data aggregation scaling tests with standard File System deployment

SYNOPSIS
    driver.py aggregation_scaling_python_fs <flags>

DESCRIPTION
    Run the data aggregation scaling tests with standard File System deployment

FLAGS
  --exp_name=EXP_NAME
      Default: 'aggregation-standard-scaling-py-fs'
      name of output dir
  --launcher=LAUNCHER
      Default: 'auto'
      workload manager i.e. "slurm", "pbs"
  --prod_node_feature=PROD_NODE_FEATURE
      Default: {}
      dict of runsettings for the producer
  --cons_node_feature=CONS_NODE_FEATURE
      Default: {}
      dict of runsettings for the consumer
  --clients_per_node=CLIENTS_PER_NODE
      Default: [48]
      client tasks per compute node for the aggregation
                                 producer app
  --client_nodes=CLIENT_NODES
      Default: [128, 256, 512]
      number of compute nodes to use for the aggregation
                             producer app
  --iterations=ITERATIONS
      Default: 20
      number of append/retrieve loops run by the applications
  --tensor_bytes=TENSOR_BYTES
      Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
      list of tensor sizes in bytes
  --tensors_per_dataset=TENSORS_PER_DATASET
      Default: [1, 2, 4]
      list of number of tensors per dataset
  --client_threads=CLIENT_THREADS
      Default: [1, 2, 4, 8, 16, 32]
      list of the number of client threads used for data
                               aggregation
  --cpu_hyperthreads=CPU_HYPERTHREADS
      Default: 2
      the number of hyperthreads per cpu.  This is done
                                 to request that the consumer application utilizes
                                 all physical cores for each client thread
  --languages=LANGUAGES
        Default: ['cpp']
        list of languages to use for the tester "cpp" and/or "fortran"
    --plot=PLOT
        Default: 'clients_per_node'
        flag to plot against in process results
```

For example, the following command could be run to execute a battery of
tests in the same allocation. As mentioned before, the number of tests executed are
made up of all permutations of the given list inputs.

```bash
python driver.py aggregation_scaling_python_fs --exp_name='aggregation-scaling-py-fs-batch' \
                                               --client_nodes=[60] \
                                               --clients_per_node=[48] \
                                               --tensors_per_dataset=[1,4] \
                                               --tensor_bytes=[1024,8192,16384,32769,65538,131076,262152,524304,1024000] \
                                               --client_threads=[1,2,4,8,16,32] \
                                               --cpu_hyperthreads=2 \
                                               --iterations=20
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
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
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory