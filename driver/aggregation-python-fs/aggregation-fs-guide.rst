### Data aggregation

The data aggregation scaling test runs two applications.  The first application
is an MPI application that produces datasets that are added to an aggregation list.
In this producer application, each MPI rank has a single-threaded client.  The second
application is a consumer application.  This application consumes the aggregation
lists that are produced by the first application.  The consumer application
can be configured to use multiple threads for data aggregation.  The producer and consumer
applications are running at the same time, but the producer application waits for the
consumer application to finish an aggregation list before starting to produce
the next aggregation list.

By default, the clients in the producer application perform 100 executions of the following command:

  1) ``append_to_list`` (add dataset to the aggregation list)

Note that the client on rank 0 of the producer application performs a ``get_list_length()``
function invocation prior to an ``MPI_BARRIER`` in order to only produce the next aggregation
list after the previous aggregation list was consumed by the consumer application.

There is only a single MPI rank for the consumer application, which means there is only
one SmartRedis client active for the consumer application.  The consumer application client
invokes the following SmartRedis commands:

  1) ``poll_list_length`` (check when the next aggregation list is ready)
  2) ``get_datasets_from_list`` (retrieve the data from the aggregation list)


The input parameters to the test are used to generate permutations
of tests with varying configurations.

```text

NAME
    driver.py aggregation-scaling - Run the data aggregation scaling tests

SYNOPSIS
    driver.py aggregation-scaling <flags>

DESCRIPTION
    Run the data aggregation scaling tests

FLAGS
    --exp_name=EXP_NAME
        Default: 'aggregation-scaling'
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
        Default: 36
        number of cpus per compute host for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --clients_per_node=CLIENTS_PER_NODE
        Default: [32]
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
```