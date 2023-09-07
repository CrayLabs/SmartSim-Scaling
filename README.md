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

The scaling tests mimic an HPC workload by making calls to SmartSim 
and SmartRedis infrastructure to complete parallel highly complex, data-intensive 
tasks that are spread across compute resources. 
These applications are used to test the performance of SmartSim and 
SmartRedis across various system types.

## Scalability Tests Supported

The SmartSim-Scaling repo offers three scalability tests with 
six total versions detailed below:

#### `Inference Tests`

| Inference Database | Client Languages | Parallelization |
| :--- | --- | --- |
| Standard | C++, Fortran | MPI |
| Colocated | C++, Fortran | MPI |

#### `Throughput Tests`

| Throughput Database | Client Languages | Parallelization |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Colocated | C++ | MPI |

#### `Data Aggregation Tests`

| Data Aggregation Database | Client Languages | Parallelization |
| :--- | --- | --- |
| Standard | C++ | MPI |
| Standard | Python | MPI |
| Standard | Python | File System |

## Colocated vs Standard Deployement

The scaling repo offers two types of Orchestrator deployments: Standard and Colocated.

> The Orchestrator is a SmartSim term for a Redis or KeyDB database with a RedisAI module built into it with the ML runtimes.

1. `Standard (Clustered Deployement)`
  : When running with Standard deployment, your Orchestrator will be deployed on different compute nodes
than your application. You will notice that all Standard scaling tests share a `db_nodes` flag. By setting the flag to `db_nodes=[4,8]` - you are telling the program to split up your database to four shards on the first permutation, then eight shards on the second permutation. Each shard of the database will communicate with each application node. You can specify the number of application nodes via the `client_nodes` flag in each scaling test.

2. `Colocated (non-Clustered Deployement)`
  : A Colocated Orchestrator is deployed on the same compute hosts as the application. This differs from standard deployment that launches the database on separate database nodes.
  Colocated deployment is particularly important for GPU-intensive workloads which require frequent communication with the database. You can specify the number of nodes to launch both the database and application on via the `client_nodes` flag. 

See [our installation docs](https://www.craylabs.org/docs/orchestrator.html) for 
more information on clustered and colocated deployment

## Building

**To run the scaling tests, SmartSim and SmartRedis will need to be
installed.** See [our installation docs](https://www.craylabs.org/docs/installation_instructions/basic.html) for instructions.

For the inference tests, be sure to have installed SmartSim with support
for the device (CPU or GPU) you wish to run the tests on, as well as
have built support for the PyTorch backend.

Installing SmartSim and SmartRedis may look something like:

```bash
# Create a python environment to install packages
python -m venv /path/to/new/environment
source /path/to/new/environment/bin/activate

# Install SmartRedis and build the library
pip install smartredis
# If you are running a Fortran app - use `make lib-with-fortran`
make lib # or make lib-with-fortran

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

> If you are using a Cray machine, you will need to run `CC=cc CXX=CC pip install -r requirements.txt`.

Lastly, the C++ applications themselves need to be built. To complete this, 
one CMake edit is required. When running `cmake ..`, 
change the path to the ``SMARTREDIS`` variable to the top level of the directory 
where you built or installed the SmartRedis library using the ``-DSMARTREDIS`` flag.
An example of this is shown below. If no SmartRedis path is specified, the program
will look for the SmartRedis library in path ``"../../SmartRedis"``.

All tests can be built by running

```bash
  cd <language name>-<test name> # ex. cpp-inference for the cpp inference tests
  mkdir build && cd build
  cmake .. -DSMARTREDIS=/path/to/SmartRedis
  make
```

The CMake files used to build the various apps are shown below:

1. Inference
   - `cpp-inference/CMakeLists.txt`
   - `fortran-inference/CMakeLists.txt`
2. Throughput
   - `cpp-throughput/CMakeLists.txt`
3. Data Aggregation
   - `cpp-data-aggregation/CMakeLists.txt`
   - `cpp-py-data-aggregation/db/CMakeLists.txt`
   - `cpp-py-data-aggregation/fs/CMakeLists.txt`

> There are three different `CMakeLists.txt` files for the Data Aggregation tests.
A separate build folder will need to be created within each CMake folder if you plan to run
all three data agg tests. You will need to navigate into the respective CMake file per Data Aggregation scaling test and run the app steps above. This is the same for the C++ and Fortran inference tests.

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

Each of the command provides their own help menu that shows the
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
`run-YEAR-MONTH-DAY-TIME`. A result's folder with multiple
runs of inference standard with the default `exp_name` would look like:

```bash
results/
├─ inference-standard-scaling/ # Holds all the runs for a scaling test
│  ├─ run-2023-07-17-13:21:17/ # Holds all information for a specific run
│  │  ├─ database/ # Holds orchestrator information
│  │  │  ├─  orchestrator.err # Will output an error within the Orchestrator
│  │  │  ├─  orchestrator.out # Will output messages pushed during an Orchestrator run
│  │  │  ├─  smartsim_db.dat 
│  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4/ # Holds all information for a session
│  │  │  ├─ cat.raw # Holds all timings from infer run
│  │  │  ├─ data_processing_script.txt # Script used during the infer session
│  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4.err # Stores error messages during inf session
│  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4.out # Stores print messages during inf session
│  │  │  ├─ rank_0_timing.csv # Holds timings for the given rank, in this case rank 0
│  │  │  ├─ resnet50.GPU.pt # The model used for the infer session
│  │  │  ├─ run.cfg # Setting file for the infer session
│  │  │  ├─ srlog.out # Log file for SmartRedis
│  │  ├─ infer-sess-fortran-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-f8a6/
│  │  ├─ run.cfg # Setting file for the run
│  │  ├─ scaling-2023-07-19.log # Log file for information on run
│  ├─ stats/ # Holds all the statistical results per run
│  │  ├─ run-2023-07-17-13:21:17/ # The run 
│  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4/ # certain sessiom
│  │  │  │  ├─ infer-sess-cpp-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-80e4.csv
│  │  │  │  ├─ put_tensor.pdf # PDF version of b/w plots
│  │  │  │  ├─ run_model.pdf # PDF version of b/w plots
│  │  │  │  ├─ run_script.pdf # PDF version of b/w plots
│  │  │  │  ├─ unpack_tensor.pdf # PDF version of b/w plots
│  │  │  ├─ infer-sess-fortran-N4-T18-DBN4-DBCPU8-ITER100-DBTPQ8-f8a6/
│  │  ├─ dataframe.csv.gz # Dataframe wit
│  │  ├─ put_tensor.png # Violin plot for put_tensor timings
│  │  ├─ run_model.png # Violin plot for run_model timings
│  │  ├─ run_script.png # Violin plot for run_script timings
│  │  ├─ unpack_tensor.png # Violin plot for unpack_tensor timings
```

Within each run folder there is a subset of files that will be useful to you. 