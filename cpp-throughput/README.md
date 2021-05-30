
## Throughput Tests

The throughput tests run as an MPI program where a single SmartRedis C++ client
is initialized on every rank.

Each client performs 10 executions of the following commands

  1) ``put_tensor``     (send image to database)
  2) ``unpack_tensor``  (Retrieve the inference result)

The tests are currently designed to run on Slurm or PBS based
systems.

The input parameters to the test are used to generate permutations
of tests with varying configurations.

The lists of clients_per_node, tensor_bytes and client_nodes will
be permuted into 3-tuples and run as an individual test.

The number of tests will be client_nodes * clients_per_node * tensor_bytes

if multiple db sizes are selected, a database cluster will be spun
up for all permutations for each database size.

Each run, the database will launch as a batch job that will wait until
its running (e.g. not queued) before running the client driver.

Resource constraints listed in this module are specific to in house
systems and will need to be changed for your system.

Please note that in the default configuration, this will launch
many batch jobs on the system and users should be wary of launching
this on systems supporting multiple users.

### Building the Throughput Tests

To run the scaling tests, SmartSim and SmartRedis will need to be
installed. Follow the [instructions for the full installation](https://www.craylabs.org/docs/installation.html) of
both libraries.

In addition, when installing SmartSim, be sure to install the
developer dependencies by specifying ``[dev]`` as shown in the
installation instructions.

Lastly, one extra library ``fire`` is needed to run the tests.
To install fire, activate your python environment and run.

```bash
pip install fire
```

Next, the throughput tests themselves need to be built.
One Cmake edit is required. Near the top of the CMake file, change the
path to the ``SMARTREDIS`` variable to the top level of the directory where
you installed SmartRedis.

```text
  set(SMARTREDIS <path to top level SmartRedis install dir>)
```

then, build the scaling tests with CMake.

```bash
    cd cpp-throughput/
    mkdir build
    cd build
    cmake ..
    make
```

### Running the Throughput Tests

For help running the tests, execute the following after installation

```bash
python driver.py throughput --help
```

Make sure to adapt the test to both fit the system you are running
on in terms of resources and scheduler (PBS or Slurm)

Which will show the following help output and demonstate how to
run the scaling test with varying parameters.

```
NAME
    driver.py throughput - Run the throughput scaling tests

SYNOPSIS
    driver.py throughput <flags>

DESCRIPTION
    The lists of clients_per_node, tensor_bytes and client_nodes will
    be permuted into 3-tuples and run as an individual test.

    The number of tests will be client_nodes * clients_per_node * tensor_bytes

    if multiple db sizes are selected, a database cluster will be spun
    up for all permutations for each database size.

    Each run, the database will launch as a batch job that will wait until
    its running (e.g. not queued) before running the client driver.

    Resource constraints listed in this module are specific to in house
    systems and will need to be changed for your system.

FLAGS
    --db_nodes=DB_NODES
        Default: [16, 32, 64]
        list of db node sizes
    --db_cpus=DB_CPUS
        Default: 32
        number of cpus per db shard
    --db_port=DB_PORT
        Default: 6780
        database port
    --clients_per_node=CLIENTS_PER_NODE
        Default: [32]
        list of ranks per node
    --client_nodes=CLIENT_NODES
        Default: [128, 256, 512]
        list of client node counts
    --tensor_bytes=TENSOR_BYTES
        Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
        list of tensor sizes in bytes
```
