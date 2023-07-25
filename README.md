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

The scalability tests mimic an HPC workload by making calls to SmartSim 
and SmartRedis infrastructure to complete in parallel highly complex, data-intensive 
tasks that are spread across compute resources. 
These applications are used to test the performance of SmartSim and 
SmartRedis across various system types.

## Colocated vs Standard Deployement

The scaling repo offers two types of Orchestrator deployement options: Standard and Colocated.

1. `Colocated (non-Clustered Deployement)`
⋅⋅* When running a Colocated test, your database will be deployed on the same node as your application.

2. `Standard (Clustered Deployement)`
⋅⋅* When running with Standard deployement, your database will be deployed on different compute nodes
than your application. You will notice that all Standard tests share a `db_nodes` flag. By setting the flag to `db_nodes=[4,8]` - you are telling the program to split up your database to four shards on the first permutation, then eight shards on the second permutation.

See [our installation docs](https://www.craylabs.org/docs/orchestrator.html) for 
more information on clustered and colocated deployement

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

Lastly, the C++ applications themselves need to be built. To complete this, 
one CMake edit is required. Navigate to your respective tests `CMakeLists.txt` file.
If you need help finding the file path, please reference the chart below.
Near the top of `CMakeLists.txt`, on line
`set(SMARTREDIS "../../SmartRedis" CACHE PATH "Path to SmartRedis root directory")`, 
change the path to the ``SMARTREDIS`` variable to
the top level of the directory where you built or installed the SmartRedis library. 

1. Inference
   - `cpp-inference/CMakeLists.txt`
2. Throughput
   - `cpp-throughput/CMakeLists.txt`
3. Data Aggregation
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
       Client: None
     
```

Each of the command provides their own help menu that show the
arguments possible for each.

## Results

The output of each Scalability Test is detail below.

### Results File Structure

When a scalability test is first initialized, a nested folder named `results/'exp_name'`
is created. The `exp_name` is captured by the `exp_name` flag value when you run your 
scaling test. Such that if I ran `python driver.py inference_standard` with the default
`exp_name`, the path to my results would be `results/inference-standard-scaling`. 

Each time you run a scalability test it is considered a single run. This is how the 
`results/'exp_name'` is organized. Per time you run a scalability test, the results 
will be within a folder named `run-YEAR-MONTH-DAY-TIME`. A results folder with multiple
runs of inference standard with the default `exp_name` would look like:

```bash
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
```

Within each run folder there is a subset of files that will be useful to you.
There is a file named `run.cfg` that contains the flag
parameter information of the run. There is a `infer-sess-...` folder
which contains all the timings for each rank, a `run.cfg` file that contains 
the permutation values for that session as well as useful debugging files.
Lastly, there is a file named `scaling-DATE.log` that contains all the output
from your terminal during the scalability test run.