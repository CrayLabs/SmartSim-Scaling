
## Inference Scaling Tests


The example within this directory performs a batch of parallel inference
tests with a Pytorch model (either MNIST or Resnet) with a single
image corresponding to the dataset on which the model was trained

The test currently runs the clients as a C++ MPI program using the
C++ SILC client to perform the tensor commands to the Redis Database.

The script can launch multiple sequential inference sessions on the
same allocations.

Note that the following examples are written to be run on a Slurm
system.

### Training the Models

There are two models in the inference scaling example section:

  1. Resnet50 trained on Imagenet
  2. CNN trained on the MNIST dataset

#### Picking your Architecture

Depending on wether you want to run the scaling tests on CPU or GPU
you may want to edit the files within this directory.

For MNIST on CPU
 - Make sure to train, jit-trace and save a copy of the trained mnist
   model. A script for this can be found in ``/mnist/mnist.py``.
 - The ``inference_scaling.cpp`` file will also need to be changed
   in places where the device is specified. This is all calls to
   ``Client.set_model`` and ``Client.set_script``

For MNIST on GPU
 - The script in ``mnist/mnist.py`` is already setup to obtain the
   MNIST model for GPU. Just run that script on a GPU system.
 - The ``inference_scaling.cpp`` is already setup to run both
   script and models on GPU. No edits are necessary.

For Imagenet on GPU
 - Currently the imagenet example should only be run on GPU. To train
   the imagenet model, run the ``imagenet/model_saver.py`` script on
   a GPU enabled system.

To train the MNIST model, use the script ``/MNIST/mnist.py``

```bash

    cd mnist/
    python mnist.py
    cp mnist_cnn.pt mnist_data_gpu # or mnist_data if you trained the cnn on cpu
```

To obtain the trained Resnet model, call the following

```bash
    cd imagenet/
    python model_saver.py
```

Both of these scripts are currently written to run on a GPU
with the CUDA libraries installed. Both examples can be
changed to run on CPU by adapting these scripts.


### Running the Scaling Tests

The following will run through how to run the cpp inference scaling
tests. Ensure that Smartsim and SILC have been installed as per their
installation instructions.

#### Building the Scaling Tests

Next, we build the scaling tests themselves with SILC C++ client
included.

Prior to this step, ensure that SILC has been installed as follows

```bash

# inside top-level silc dir
make lib
make pyclient
source setup_env.sh
```

One Cmake edit is required. Near the top of the CMake file, change the
path to the SILC variable to the top level of the directory where
you installed SILC.

```text
set(SILC <path to top level SILC dir>)
```

The following will build both the Resnet and MNIST examples

```bash
    cd cpp-inference/
    mkdir build
    cd build
    cmake ..
    CC=cc CXX=CC cmake .. # for Cray machines (we actually build with GNU I think)
    make
```

#### Set the Experiment Parameters

There are two places to set the experiment parameters

    1) Inside the C++ program we just built
    2) Inside the SmartSim script that runs the tests.

The ``inference-scaling.cpp`` program includes two places
where arguments can be changed for the scaling tests. The
same holds for ``inference-scaling-imagenet.cpp``.

  - Batch Size (controls the size of inference batch)
  - Number of iterations (number of inferences performed per run)

the batch size is currently set to ``10`` and the number
of iterations is set to ``50`` for the mnist test. For Imagenet
the batch size is set to ``10`` and the number of iterations
is set to ``10``.

The SmartSim Script also includes parameters for the
inference scaling tests at the top of the file.

```python
    # Constants
    DB_NODES = 3                           # number of database nodes
    DPN = 1                                # number of databases per node
    CLIENT_ALLOC = 40                      # number of nodes in client alloc
    CLIENT_NODES = [20, 40]                # list of node sizes to run clients within client alloc
    CPN = [80]                             # clients per node
    NAME = "infer-scaling"                 # name of experiment directory
    MODEL_DIR = "../mnist/mnist_data_gpu"  # Name of directory housing trained model.
```

In the current setup, 2 runs of 1600 (80 x 20) and 3200 (80 x 40) clients
respectively will be executed. Since there are 50 iterations of inference
per client, these runs will execute 80000 and 160000 inferences respectively.

All runs will execute on the same allocations which can be customized in the
script to obtain whatever allocation suits your machine.

#### Post-Processing

The inference program in C++ will output a number of CSV files for each MPI rank
that contain timings for the inference workload. these times are collected by a
post-processing script that collects the results into a single CSV that includes
the experiemnt summary.

The inference statistics will be under ``<NAME>.csv`` after a successful run.


#### Running the SmartSim Driver

To submit the inference scaling tests to slurm (or pbs if you change the launcher),
simply run the SmartSim script corresponding to the model you would
like the scaling results of.

For example, for MNIST

```bash
    python run-inference-session.py
```

or for Resnet

```bash
    python run-inference-session-resnet.py
```
