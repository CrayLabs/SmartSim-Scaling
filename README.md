
# SmartSim Scaling

This repository holds all of the scripts and materials for testing
the scaling of SmartSim and the SmartRedis clients.


## Inference Tests

the inference tests run as an MPI program where a single SmartRedis client
is initialized on every rank.

Currently supported inference tests

  1) Resnet50 CNN with ImageNet dataset

Each client performs 10 executions of the following commands

  1) put_tensor     (send image to database)
  2) run_script     (preprocess image)
  3) run_model      (run resnet50 on the image)
  4) unpack_tensor  (Retrieve the inference result)

The tests are currently designed to run on a Slurm system but
can be adapted to a PBSPro, or Cobalt system.

The input parameters to the test are used to generate permutations
of tests with varying configurations.

The lists of ``clients_per_node``, ``db_nodes``, and ``client_nodes`` will
be permuted into 3-tuples and run as an individual test.

The number of tests will be ``client_nodes`` * ``clients_per_node`` * ``db_nodes``

An allocation will be obtained of size ``max(client_nodes)`` and will be
used for each run of the client driver

Each run, the database will launch as a batch job that will wait until
its running (e.g. not queued) before running the client driver.

Please note that in the default configuration, this will launch
many batch jobs on the system and users should be wary of launching
this on systems supporting multiple users.

### Building the Inference Tests

To run the scaling tests, SmartSim and SmartRedis will need to be
installed. Follow the instructions for the full installation of
both libraries and be sure to build for the architecture you
want to run the tests on (e.g. CPU or GPU)

In addition, when installing SmartSim, be sure to install the
developer dependencies by specifying ``[dev]`` as shown in the
installation instructions. This will install PyTorch 1.7.1 which
is needed to run the tests.

Lastly, one extra library ``fire`` is needed to run the tests.
To install fire, activate your python environment and run.

```bash
pip install fire
```

Next, the inference tests themselves need to be built.
One Cmake edit is required. Near the top of the CMake file, change the
path to the ``SMARTREDIS`` variable to the top level of the directory where
you installed SmartRedis.

```text
  set(SMARTREDIS <path to top level SmartRedis install dir>)
```

then, build the scaling tests with CMake.

```bash
    cd cpp-inference/
    mkdir build
    cd build
    cmake ..
    make
```

### Running the Inference Tests

For help running the tests, execute the following after installation

```bash
python run-scaling.py resnet --help
```

Which will show the following help output

```
NAME
    run-inference-session-imagenet.py resnet - Run the resnet50 inference tests.

SYNOPSIS
    run-inference-session-imagenet.py resnet <flags>

DESCRIPTION
    The lists of clients_per_node, db_nodes, and client_nodes will
    be permuted into 3-tuples and run as an individual test.

    The number of tests will be client_nodes * clients_per_node * db_nodes

    An allocation will be obtained of size max(client_nodes) and will be
    used for each run of the client driver

    Each run, the database will launch as a batch job that will wait until
    its running (e.g. not queued) before running the client driver.

    Resource constraints listed in this module are specific to in house
    systems and will need to be changed for your system.

FLAGS
    --db_nodes=DB_NODES
        Default: [4, 8, 16]
        list of db node sizes
    --db_cpus=DB_CPUS
        Default: 36
        number of cpus per db shard
    --db_tpq=DB_TPQ
        Default: 4
        device threads per database shard
    --db_port=DB_PORT
        Default: 6780
        database port
    --batch_size=BATCH_SIZE
        Default: 1000
        batch size for inference
    --device=DEVICE
        Default: 'GPU'
        CPU or GPU
    --model=MODEL
        Default: '../imagenet/resnet5...
        path to serialized model
    --clients_per_node=CLIENTS_PER_NODE
        Default: [48]
        list of ranks per node
    --client_nodes=CLIENT_NODES
        Default: [20, 40, 60, 80...
        list of client node counts
```

