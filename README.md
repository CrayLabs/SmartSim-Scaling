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

The scalability tests mimic an HPC workload by making calls to SmartSim 
and SmartRedis infrastructure to complete in parallel highly complex, data-intensive 
tasks that are spread across compute resources. 
These applications are used to test the performance of SmartSim and 
SmartRedis across various system types.

## Scalability Tests Supported

The SmartSim-Scaling repo offers three scalability tests with 
six total versions detailed below:

#### `Inference Tests`

| Inference Type | Client | Parallelization |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### `Throughput Tests`

| Throughput Type | Client | Parallelization |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### `Data Aggregation Tests`

| Data Aggregation Type | Client | Parallelization |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Standard | Python | MPI |
| Standard | Python | File System |

## Colocated vs Standard Deployement

The scaling repo offers two types of Orchestrator deployement: Standard and Colocated.

> Note that the Orchestrator is the name in SmartSim for a Redis or KeyDB database with a RedisAI module built into it with the ML runtimes.

1. `Colocated (non-Clustered Deployement)`
  : A colocated Orchestrator is a special type of Orchestrator that is deployed on the same compute hosts as the application. 
  This is particularly important for GPU-intensive workloads which require frequent communication with the database. You can specify the number of nodes via the `client_nodes` flag. 

2. `Standard (Clustered Deployement)`
  : When running with Standard deployement, your database will be deployed on different compute nodes
than your application. You will notice that all Standard scaling tests share a `db_nodes` flag. By setting the flag to `db_nodes=[4,8]` - you are telling the program to split up your database to four shards on the first permutation, then eight shards on the second permutation. Each shard of the database will communicate with each application node. You can specify the number of application nodes via the `client_nodes` flag in each scaling test.

See [our installation docs](https://www.craylabs.org/docs/orchestrator.html) for 
more information on clustered and colocated deployement

## Building

**To run the scaling tests, SmartSim and SmartRedis will need to be
installed.** See [our installation docs](https://www.craylabs.org/docs/installation_instructions/basic.html) for instructions.

For the inference tests, be sure to have installed SmartSim with support
for the device (CPU or GPU) you wish to run the tests on, as well as
have built support for the PyTorch backend. If you would also like to 
run the Fortran tests, make sure to run `make lib-with-fortran` from 
the install docs for SmartRedis. For C++ scaling tests, run
`make lib`.

Installing SmartSim and SmartRedis may look something like:

```bash
# Create a python environment to install packages
python -m venv /path/to/new/environment
source /path/to/new/environment/bin/activate

# Install SmartRedis and build the library
pip install smartredis
make lib

# Install SmartSim
pip install smartsim

# Build SmartSim and install ML Backends for GPU
smart build --device gpu
```

But please consult the documentation for other pieces like specifying compilers,
CUDA, cuDNN, and other build settings. 

Once SmartSim is installed, the Python dependencies for the scaling test and
result processing/plotting can be installed with

```bash
cd SmartSim-Scaling
pip install -r requirements.txt
```

> Note that if you are using a Cray machine, you will need to run `CC=cc CXX=CC pip install -r requirements.txt`.

Lastly, the C++ applications themselves need to be built. To complete this, 
one CMake edit is required. When running `cmake ..`, 
change the path to the ``SMARTREDIS`` variable to the top level of the directory 
where you built or installed the SmartRedis library using the ``-DSMARTREDIS`` flag.
An example of this is shown below.

All tests can be built by running

```bash
  cd <language name>-<test name> # ex. cpp-inference for the cpp inference tests
  mkdir build && cd build
  cmake .. -DSMARTREDIS=/path/to/SmartRedis
  make
```

The app locations are shown below: 

1. Inference
   - `cpp-inference/CMakeLists.txt`
   - `fortran-inference/CMakeLists.txt`
2. Throughput
   - `cpp-throughput/CMakeLists.txt`
3. Data Aggregation
   - `cpp-data-aggregation/CMakeLists.txt`
   - `cpp-py-data-aggregation/db/CMakeLists.txt`
   - `cpp-py-data-aggregation/fs/CMakeLists.txt`

> Note that there are three different `CMakeLists.txt` files for the Data Aggregation tests.
You will need to build per Data Aggregation scaling test. This is the same for the 
C++ and Fortran inference test.

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
       
     scaling_read_data
       Read performance results and store to file system
       Client: None
      
     scaling_plotter
       Create just performance plots
       Client: None
     
```

Each of the command provides their own help menu that show the
arguments possible for each.

## Results

The output organization of the performance results is detail below.

### Results File Structure

When a scaling test is first initialized, a nested folder named `results/'exp_name'`
is created. The `exp_name` is captured by the `exp_name` flag value when you run your 
scaling test. For example, running the standard inference test via 
`python driver.py inference_standard` with the default name `exp_name=inference-standard-scaling`, 
places results in the `results/inference-standard-scaling` directory.

Each time you run a scaling test it is considered a single run. This is how the 
`results/'exp_name'` is organized. The results will be within a folder named 
`run-YEAR-MONTH-DAY-TIME`. A results folder with multiple
runs of inference standard with the default `exp_name` would look like:

```bash
results/
├─ inference-standard-scaling/ # name of scaling test
│  ├─ run-2023-07-17-13:21:17/ #
│  │  ├─ database/ # 
│  │  │  ├─  orchestrator.err
│  │  │  ├─  orchestrator.out
│  │  │  ├─  smartsim_db.dat
│  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4/
│  │  │  ├─ cat.raw # include?
│  │  │  ├─ data_processing_script.txt
│  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4.err
│  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4.out
│  │  │  ├─ rank_0_timing.csv
│  │  │  ├─ resnet50.GPU.pt
│  │  │  ├─ run.cfg
│  │  │  ├─ srlog.out
│  │  ├─ infer-sess-fortran-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-f8a6/
│  │  ├─ run.cfg
│  │  ├─ scaling-2023-07-19.log
│  ├─ stats/
│  │  ├─ run-2023-07-17-13:21:17/
│  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4/
│  │  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4.csv
│  │  │  │  ├─ put_tensor.pdf
│  │  │  │  ├─ run_model.pdf
│  │  │  │  ├─ run_script.pdf
│  │  │  │  ├─ unpack_tensor.pdf
│  │  │  ├─ infer-sess-fortran-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-f8a6/
│  │  ├─ dataframe.csv.gz
│  │  ├─ put_tensor.png
│  │  ├─ run_model.png
│  │  ├─ run_script.png
│  │  ├─ unpack_tensor.png
```

Within each run folder there is a subset of files that will be useful to you. 