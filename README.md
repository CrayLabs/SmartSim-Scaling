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
