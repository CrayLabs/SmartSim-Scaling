<div align="center">
    <a href="https://github.com/CrayLabs/SmartSim"><img src="https://raw.githubusercontent.com/CrayLabs/SmartSim/master/doc/images/SmartSim_Large.png" width="90%"><img></a>
    <br />
    <br />
    <div display="inline-block">
        <a href="https://github.com/CrayLabs/SmartSim-Scaling"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/docs/installation_instructions/basic.html"><b>Install</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://www.craylabs.org/docs/overview.html"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://join.slack.com/t/craylabs/shared_invite/zt-nw3ag5z5-5PS4tIXBfufu1bIvvr71UA"><b>Slack</b></a>&nbsp;&nbsp;&nbsp;
        <a href="https://github.com/CrayLabs"><b>Cray Labs</b></a>&nbsp;&nbsp;&nbsp;
    </div>
    <br />
    <br />
</div>

----------

# SmartSim Scaling

This repository holds all of the scripts and materials for testing
the scaling of SmartSim and SmartRedis clients.

## Scalability Tests Supported

The SmartSim-Scaling repo offers three scalability tests with 
six total versions:

#### `Inference Tests`

| Inference Type | Client | Message Passing Interface |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### `Throughput Tests`

| Throughput Type | Client | Message Passing Interface |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### `Data Aggregation Tests`

| Data Aggregation Type | Client | Message Passing Interface |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Standard | Python | MPI |
| Standard | Python | File System |

#### `combined graph test`

| Scaling Type | Type | Client | Message Passing Interface |
| :--- | --- | --- | --- |
| Inference | Standard | C++ | MPI |
| Inference | Colocated | C++ | MPI |
| Throughput | Standard | C++ | MPI |
| Throughput | Colocated | C++ | MPI |
| Data Aggregation | Standard | C++ | MPI |
| Data Aggregation | Standard | Python | MPI |
| Data Aggregation | Standard | Python | File System |

The scalability tests mimic an HPC workload by making calls to SmartSim 
and SmartRedis infrastructure to complete in parallel highly complex, data-intensive 
tasks that are spread across compute resources. 
These applications are used to test the performance of SmartSim and 
SmartRedis across various system types.

## Difference between colocated and standard?

The scaling repo offers two types of Orchestrator deployement options: Standard and Colocated.

1. `Colocated (non-Clustered Deployement)`
⋅⋅* When running a Colocated test, your database will be deployed on the same node as your application.

2. `Standard (Clustered Deployement)`
⋅⋅* When running with Standard deployement, your database will be deployed on different compute nodes
than your application. You will notice that all Standard tests share a `db_nodes` flag. By setting the flag to `db_nodes=[4,8]` - you are telling the program to split up your database to four shards on the first permutation, then eight shards on the second permutation.

For more information on Clustered and Colocated Orchestrator deployement - please select here.
https://www.craylabs.org/docs/orchestrator.html

## Building

To run the scaling tests, SmartSim and SmartRedis will need to be
installed. See [our installation docs](https://www.craylabs.org/docs/installation_instructions/basic.html) for instructions.

For the inference tests, be sure to have installed SmartSim with support
for the device (CPU or GPU) you wish to run the tests on, as well as
have built support for the PyTorch backend. If you would also like to 
run the Fortran tests, make sure to run `make lib-with-fortran` from 
the intall docs for SmartRedis.

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
Near the top of the `CMakeLists.txt`, on the line
`set(SMARTREDIS "../../SmartRedis" CACHE PATH "Path to SmartRedis root directory")` - 
change the path to the ``SMARTREDIS`` variable to
the top level of the directory where you built or installed the SmartRedis library. 
You will need to complete this task per scaling study you would like to run. 
The paths are listed as:

1. Inference
  - `cpp-inference/CMakeLists.txt`
1. Throughput
  - `cpp-throughput/CMakeLists.txt`
1. Data Aggregation 
  - `cpp-data-aggregation/CMakeLists.txt`
  - `cpp-py-data-aggregation/db/CMakeLists.txt`
  - `cpp-py-data-aggregation/fs/CMakeLists.txt`

> Note that there are three different `CMakeLists.txt` files for the Data Aggregation tests.
You will need to setup one per Data Aggregation scaling test you would like to run.


After the cmake edit, all tests can be built by running

```bash
  cd <language name>-<test name> # ex. cpp-inference for the cpp inference tests
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

The output of each Scalability Test is detail below.

### Where the results are stored

When a scalability test is first initialized, a nested folder named `results/'exp_name'`
is created. The `exp_name` is captured by the `exp_name` flag value when you run your 
scaling test. Such that if I ran `python driver.py inference_standard` with the default
`exp_name`, the path to my results would be `results/inference-standard-scaling`. 

Each time you run a scalability test it is considered a single run. This is how the 
`results/'exp_name'` is organized. Per time you run a scalability test, the results 
will be within a folder named `run-YEAR-MONTH-DAY-TIME`. A results folder with multiple
runs of inference standard with the default `exp_name` would look like:

results/
├─ inference-standard-scaling/
│  ├─ run-2023-07-17-13:21:17/
│  │  ├─ database/
│  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4/
│  │  ├─ infer-sess-fortran-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-f8a6/
│  │  ├─ run.cfg
│  │  ├─ scaling-2023-07-19.log
│  ├─ run-2023-07-19-11:33:57
│  ├─ run-2023-07-19-11:40:08

Within each run folder there is a subset of files that will be useful to you.
There is a file named `run.cfg` that contains the flag
parameter information of the run. There is a `infer-sess-...` folder
which contains all the timings for each rank, a `run.cfg` file that contains 
the permutation values for that session as well as useful debugging files.
Lastly, there is a file named `scaling-DATE.log` that contains all the output
from your terminal during the scalability test run.

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

![Colocated inference plots dark theme](/figures/colo_dark.png#gh-light-mode-only "Colocated inference")
![Inference plots ligh theme](/figures/colo_light.png#gh-light-mode-only "Colocated inference")

### Throughput

#### Throughput Standard
![Throughput Std Unpack](/figures/unpack_tensor_thro_std.png#gh-light-mode-only "Throughput Standard")
![Throughput Std Put](/figures/put_tensor_thro_std.png#gh-light-mode-only "Throughput Standard")

#### Throughput Colocated
![Throughput colo Unpack](/figures/unpack_tensor_thro_colo.png#gh-light-mode-only "Colocated Throughput")
![Throughput colo put](/figures/put_tensor_thro_colo.png#gh-light-mode-only "Colocated Throughput")


### Data Aggregation
Input 

#### Data Aggregation
Info on test
![Data Agg Get List](/figures/get_list_data_agg.png#gh-light-mode-only "Data Aggregation Standard")

#### Data Aggregation Py
Info on test
![Data Agg Py Get List](/figures/get_list_data_agg_py.png#gh-light-mode-only "Data Aggregation Py Standard")
![Data Agg Py Poll List](/figures/poll_list_data_agg_py.png "Data Aggregation Py Standard")

#### Data Aggregation Py Fs
Info on test
![Data Agg Py Fs Get List](/figures/get_list_data_agg_fs.png#gh-light-mode-only "Data Aggregation Py Fs Standard")
![Data Agg Py Fs Poll List](/figures/poll_list_data_agg_fs.png#gh-light-mode-only "Data Aggregation Py Fs Standard")

#### Data Aggregation Performance Analysis

Info on test

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

