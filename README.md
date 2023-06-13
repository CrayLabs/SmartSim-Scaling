# SmartSim Scaling

This repository holds all of the scripts and materials for testing
the scaling of SmartSim and the SmartRedis clients.


## Scaling Tests

There are two types of scaling tests in the repository.

 1. Inference
 2. Throughput

Both applications use a MPI + C++ application to mimic an HPC workload
making calls to SmartSim infrastructure. These applications are used
to test the performance of SmartSim across various system types.

## Building

To run the scaling tests, SmartSim and SmartRedis will need to be
installed. See [our installation docs](https://www.craylabs.org/docs/installation.html)
for instructions.

For the inference tests, be sure to have installed SmartSim with support
for the device (CPU or GPU) you wish to run the tests on, as well as
have built support for the PyTorch backend.

This may look something like the following:

```bash
pip install smartsim
smart build --device gpu
```

But please consult the documentation for other peices like specifying compilers,
CUDA, cuDNN, and other build settings.

Once SmartSim is installed, the Python dependencies for the scaling test and
result processing/plotting can be installed with

```bash
cd SmartSim-Scaling
pip install -r requirements.txt
```

You will need to install ``mpi4py`` in your python environment. The install instructions
can be found by selecting [mpi4py docs](https://mpi4py.readthedocs.io/en/stable/install.html).

Lastly, the C++ applications themselves need to be built. One CMake edit is required.
Near the top of the CMake file, change the path to the ``SMARTREDIS`` variable to
the top level of the directory where you built or installed the SmartRedis library.

After the cmake edit, both tests can be built by running

```bash
  cd cpp-<test name> # ex. cpp-inference for the inference tests
  mkdir build && cd build
  cmake ..
  make
```

## Running

A single SmartSim driver script can be used to launch both tests. The ``Fire`` CLI
specifies the options for the scaling tests.

```text
SYNOPSIS
    driver.py COMMAND | VALUE

COMMANDS
    COMMAND is one of the following:

     process_scaling_results
       Create a results directory with performance data and plots

     inference_colocated
       Run ResNet50 inference tests with colocated Orchestrator deployment

     inference_standard
       Run ResNet50 inference tests with standard Orchestrator deployment

     throughput_scaling
       Run the throughput scaling tests
```

Each of the command provides their own help menu as well that shows the
arguments possible for each.

### Inference

The inference tests run as an MPI program where a single SmartRedis C++ client
is initialized on every rank.

Currently supported inference tests

  1) Resnet50 CNN with ImageNet dataset

Each client performs 101 executions of the following commands. The first iteration is a warmup;
the next 100 are measured for inference throughput. 

  1) ``put_tensor``     (send image to database)
  2) ``run_script``     (preprocess image)
  3) ``run_model``      (run resnet50 on the image)
  4) ``unpack_tensor``  (Retrieve the inference result)

The input parameters to the test are used to generate permutations
of tests with varying configurations.

### The model
As Neural Network, we use Pytorch's implementation of Resnet50. The script `imagenet/model_saver.py`
can be used to generate the model for CPU or GPU. By navigating to the `imagenet` folder, the CPU model
can be created running 

```bash
python model_saver.py
```

and the GPU model can be created running

```bash
python model_saver.py --device=GPU
```


If the benchmark driver is executed and
no model exists, an attempt is made to generate the model on the fly. In both cases,
the specified device must be available on the node where the script is called (this
means that it could be required to run the script through the workload manager launcher
to execute it on a node with a GPU, for example).

### Co-located inference

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
        Default: 'inference-scaling'
        name of output dir, defaults to "inference-scaling"
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --nodes=NODES
        Default: [12]
        compute nodes to use for synthetic scaling app with a co-located orchestrator database
    --clients_per_node=CLIENTS_PER_NODE
        Default: [18]
        client tasks per compute node for the synthetic scaling app
    --db_cpus=DB_CPUS
        Default: [2]
        number of cpus per compute host for the database
    --db_tpq=DB_TPQ
        Default: [1]
        number of device threads to use for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --pin_app_cpus=PIN_APP_CPUS
        Default: [False]
        pin the threads of the application to 0-(n-db_cpus)
    --batch_size=BATCH_SIZE
        Default: [1]
        batch size to set Resnet50 model with
    --device=DEVICE
        Default: 'GPU'
        device used to run the models in the database
    --num_devices=NUM_DEVICES
        Default: 1
        number of devices per compute node to use to run ResNet
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --rebuild_model=FORCE_REBUILD
        Default: False
        force rebuild of PyTorch model even if it is available
```

So for example, the following command could be run to execute a battery of
tests in the same allocation

```bash
python driver.py inference_colocated --clients_per_node=[24,28] \
                                     --nodes=[16] --db_tpq=[1,2,4] \
                                     --db_cpus=[1,2,4,8] --net_ifname=ipogif0 \
                                     --device=GPU
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
#!/bin/bash

#SBATCH -N 16
#SBATCH --exclusive
#SBATCH -C P100
#SBATCH -t 10:00:00

module load slurm
python driver.py inference_colocated --clients_per_node=[24,28] \
                                     --nodes=[16] --db_tpq=[1,2,4] \
                                     --db_cpus=[1,2,4,8] --net_ifname=ipogif0
                                     --device=GPU
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory


### Standard Inference

Co-locacated deployment is the preferred method for running tightly coupled
inference workloads with SmartSim, however, if you want to deploy the Orchestrator
database and the application on different nodes you may want to use standard
deployment.

For example, if you only have a small number of GPU nodes and want to test a large
CPU application you may want to use standard deployment. For more information
on Orchestrator deployment methods, please see
[our documentation](https://www.craylabs.org/docs/orchestrator.html)

Like the above inference scaling tests, the standard inference tests also provide
a method of running a battery of tests all at once. Below is the help output.
The arguments which are lists control the possible permutations that will be run.

```text
NAME
    driver.py inference_standard - Run ResNet50 inference tests with standard Orchestrator deployment

SYNOPSIS
    driver.py inference_standard <flags>

DESCRIPTION
    Run ResNet50 inference tests with standard Orchestrator deployment

FLAGS
    --exp_name=EXP_NAME
        Default: 'inference-scaling'
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
    --db_tpq=DB_TPQ
        Default: [1]
        number of device threads to use for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --batch_size=BATCH_SIZE
        Default: [1000]
        batch size to set Resnet50 model with
    --device=DEVICE
        Default: 'GPU'
        device used to run the models in the database
    --num_devices=NUM_DEVICES
        Default: 1
        number of devices per compute node to use to run ResNet
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --clients_per_node=CLIENTS_PER_NODE
        Default: [48]
        client tasks per compute node for the synthetic scaling app
    --client_nodes=CLIENT_NODES
        Default: [12]
        number of compute nodes to use for the synthetic scaling app
    --rebuild_model=FORCE_REBUILD
        Default: False
        force rebuild of PyTorch model even if it is available
```

The standard inference tests will spin up a database for each iteration in the
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

### Collecting Performance Results

The ``process_scaling_results`` function will collect the timings for each
of the tests and make a series of plots for each client function called in
each run as well as make a collective csv of timings for all runs. These
artifacts will be in a ``results`` folder inside the directory where the
function was pointed to the scaling data with the ``scaling_dir`` flag
shown below. The default will work for the inference tests.

Similar to the inference and throughput tests, the result collection has
it's own options for execution

```text
NAME
    driver.py process_scaling_results - Create a results directory with performance data and plots

SYNOPSIS
    driver.py process_scaling_results <flags>

DESCRIPTION
    With the overwrite flag turned off, this function can be used
    to build up a single csv with the results of runs over a long
    period of time.

FLAGS
    --scaling_dir=SCALING_DIR
        Default: 'inference-scaling'
        directory to create results from
    --overwrite=OVERWRITE
        Default: True
        overwrite any existing results
```

For example for the inference tests (if you don't change the output dir name)
you can run

```bash
python driver.py process_scaling_results
```

For the throughput tests
```bash
python driver.py process_scaling_results --scaling_dir=throughput-scaling
```


## Performance Results

The performance of SmartSim is detailed below across various types of systems.

### Inference

The following are scaling results from the cpp-inference scaling tests with ResNet-50
and the imagenet dataset. For more information on these scaling tests, please see
the SmartSim paper on arXiv

![Inference plots dark theme](/figures/all_in_one_violin_dark.png#gh-dark-mode-only "Standard inference")
![Inference plots ligh theme](/figures/all_in_one_violin_light.png#gh-light-mode-only "Standard inference")

### Colocated Inference

The following are scaling results for a colocated inference test, run on 12 36-core Intel Broadwell nodes,
each one equipped with 8 Nvidia V100 GPUs. On each node, 28 client threads were run, and the databases
were run on 8 CPUs and 8 threads per queue. 

Note that the first iteration can take longer (up to several seconds) than the rest of the execution. This
is due to the DB loading libraries when the first RedisAI call is made. In the following plots, we excluded
the first iteration time.

![Colocated inference plots dark theme](/figures/colo_dark.png#gh-dark-mode-only "Colocated inference")
![Inference plots ligh theme](/figures/colo_light.png#gh-light-mode-only "Colocated inference")

### Throughput

The following are results from the throughput tests for Redis. For results obtained using KeyDB, see section below.

All the throughput data listed here is based on the ``loop time`` which is the time to complete a single put and get. Each client
in the test performs 100 loop iterations and the aggregate throughput for all clients is displayed in the plots below.

Each test has three lines for the three database sizes tested: 16, 32, 64. Each of the plots represents a different number of total clients
the first is 4096 clients (128 nodes x 32 ranks per node), followed by 8192 (256 nodes x 32 ranks per node) and lastly 16384 clients
(512 nodes x 32 ranks per node)

![Throughput plots dark theme](/figures/loop_time-128-redis_dark.png#gh-dark-mode-only "Throughput scaling for 128 node Redis DB")
![Throughput plots light theme](/figures/loop_time-128-redis_light.png#gh-light-mode-only "Throughput scaling for 128 node Redis DB")

![Throughput plots dark theme](/figures/loop_time-256-redis_dark.png#gh-dark-mode-only "Throughput scaling for 256 node Redis DB")
![Throughput plots light theme](/figures/loop_time-256-redis_light.png#gh-light-mode-only "Throughput scaling for 256 node Redis DB")

![Throughput plots dark theme](/figures/loop_time-512-redis_dark.png#gh-dark-mode-only "Throughput scaling for 512 node Redis DB")
![Throughput plots light theme](/figures/loop_time-512-redis_light.png#gh-light-mode-only "Throughput scaling for 512 node Redis DB")

### Using KeyDB

KeyDB is a multithreaded version of Redis with some strong performance claims. Luckily, since
KeyDB is a drop in replacement for Redis, it's fairly easy to test. If you are looking for
extreme performance, especially in throughput for large data sizes,
we recommend building SmartSim with KeyDB.

In future releases, switching between Redis and KeyDB will be achieved by setting an environment variable specifying the backend.

The following plots show the results for the same throughput tests of previous section, using KeyDB as a backend.


![Throughput plots dark theme](/figures/loop_time-128-keydb_dark.png#gh-dark-mode-only "Throughput scaling for 128 node KeyDB DB")
![Throughput plots light theme](/figures/loop_time-128-keydb_light.png#gh-light-mode-only "Throughput scaling for 128 node KeyDB DB")

![Throughput plots dark theme](/figures/loop_time-256-keydb_dark.png#gh-dark-mode-only "Throughput scaling for 256 node KeyDB DB")
![Throughput plots light theme](/figures/loop_time-256-keydb_light.png#gh-light-mode-only "Throughput scaling for 256 node KeyDB DB")

![Throughput plots dark theme](/figures/loop_time-512-keydb_dark.png#gh-dark-mode-only "Throughput scaling for 512 node KeyDB DB")
![Throughput plots light theme](/figures/loop_time-512-keydb_light.png#gh-light-mode-only "Throughput scaling for 512 node KeyDB DB")

### Result analysis

> :warning: from the above plots, it is clear that there is a performance decrease at 64 and 128 KiB, which is visible in all cases,
but is most relevant for large DB node counts and for KeyDB. We are currently investigating this behavior, as we are not exactly
sure of what the root cause could be.

A few interesting points:

 1. Client connection time: KeyDB connects client MUCH faster than base Redis. At this time, we
    are not exactly sure why, but it does. So much so, that if you are looking to use the SmartRedis
    clients in such a way that you will be disconnecting and reconnecting to the database, you
    should use KeyDB instead of Redis with SmartSim.

 2. In general, according to the throughput scaling tests, KeyDB has roughly up to 2x the throughput
    of Redis and reaches a maximum throughput of ~125 Gb/s, whereas we could not get Redis to achieve
    more than ~90 Gb/s.

 3. KeyDB seems to handle higher numbers of clients better than Redis does.

 4. There is an evident bottleneck on throughput around 128 kiB


## Advanced Performance Tips

There are a few places users can look to get every last bit of performance.

 1. TCP settings
 2. Database settings

The communication goes over the TCP/IP stack. Because of this, there are
a few settings that can be tuned

 - ``somaxconn`` - The max number of socket connections. Set this to at least 4096 if you can
 - ``tcp_max_syn_backlog`` - Raising this value can help with really large tests.

The database (Redis or KeyDB) has a number of different settings that can increase
performance.

For both Redis and KeyDB:
  - ``maxclients`` - This should be raised to well above what you think the max number of clients will be for each DB shard
  - ``threads-per-queue`` - can be set in ``Orchestrator()`` init. Helps with GPU inference performance (set to 4 or greater)
  - ``inter-op-threads`` - can be set in ``Orchestrator()`` init. helps with CPU inference performance
  - ``intra-op-threads`` - can be set in ``Orchestrator()`` init. helps with CPU inference performance

For Redis:
  - ``io-threads`` - we set to 4 by default in SmartSim
  - ``io-use-threaded-reads`` - We set to yes (doesn't usually help much)

For KeyDB:
  - ``server-threads`` - Makes a big difference. We use 8 on HPC hardware. Set to 4 by default.

