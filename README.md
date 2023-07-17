<div align="center">
    <a href="https://github.com/CrayLabs/SmartSim"><img src="https://raw.githubusercontent.com/CrayLabs/SmartSim/master/doc/images/SmartSim_Large.png" width="90%"><img></a>
    <br />
    <br />
    <div display="inline-block">
        <a href="https://github.com/CrayLabs/SmartRedis"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/docs/installation.html#smartredis"><b>Install</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/docs/smartredis.html"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://join.slack.com/t/craylabs/shared_invite/zt-nw3ag5z5-5PS4tIXBfufu1bIvvr71UA"><b>Slack</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/CrayLabs"><b>Cray Labs</b></a>&nbsp;&nbsp;&nbsp;
    </div>
    <br />
    <br />
</div>

----------

# SmartSim Scaling

This repository holds all of the scripts and materials for testing
the scaling of SmartSim and the SmartRedis clients.

## Scalability Tests

The SmartSim-Scaling repo offers three scalability tests with 
six total permutations shown below:

#### Inference Tests

| Inference Type | Client | Message Passing Interface |
| --- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### Throughput Tests

| Throughput Type | Client | Message Passing Interface |
| --- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### Data Aggregation Tests

| Data Aggregation Type | Client | Message Passing Interface |
| --- | --- | --- |
| Standard | C++ | MPI |
| Standard | Python | MPI |
| Standard | Python | File System |

The scalability tests mimic an HPC workload by making calls to SmartSim 
and SmartRedis infrastructure to complete highly complex, data-intensive 
tasks that are spread across compute resources, which each run parts of the task in parallel. 
These applications are used
to test the performance of SmartSim and SmartRedis across various system types.

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

     inference_colocated
       Run ResNet50 inference tests with colocated Orchestrator deployment
       Client: C++
       
     inference_standard
       Run ResNet50 inference tests with standard Orchestrator deployment
       Client: C++
       
     throughput_colocated
       Run throughput scaling tests with colocated Orchestrator deployment
       Client: C++
       
     throughput_standard
       Run throughput scaling tests with standard Orchestrator deployment
       Client: C++
       
     aggregation_scaling
       Run aggregation scaling tests with standard Orchestrator deployment
       Client: C++
       
     aggregation_scaling_python
       Run aggregation scaling tests with standard Orchestrator deployment
       Client: Python
       
     aggregation_scaling_python_fs
       Run aggregation scaling tests with standard File System deployment
       Client: Python

     process_scaling_results
       Create a results directory with performance data and performance plots
       Client: None
       
     scaling_plotter
       Create just performance plots
       Client:  None
     
```

Each of the command provides their own help menu as well that shows the
arguments possible for each.

## Results 

1. where the results are stored
2. the format of folder names
3. what is stored

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

