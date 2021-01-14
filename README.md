
## SmartSim Scaling

This repository holds all of the scripts and materials for testing
the scaling of SmartSim.

### Inference Tests

the inference tests run as an MPI program where a single SILC client
is initialized on every rank. Each client performs some number of
inferences with the same piece of data (an image).

Currently supported inference tests

  1) MNIST CNN
  2) Resnet50 CNN with ImageNet dataset

For more information on the scaling tests, see the directory corresponding
to the client langauge type (e.g. cpp-inference )

### Scaling data

Scaling data will also be included in the repository.

### Requirements

Both SmartSim and SILC should be installed and SmartSim will need
to have it's environment setup. Follow the instructions in the documentation
for building SmartSim and SILC.

A python environment with the `requirements-dev.txt` listed in SmartSim
is required as well.

```bash
# in a python env
pip install -r requirements-dev.txt
```

Lastly, git-lfs will have to be installed on the machine to get the model
files. This is installable on most package managers for major OS's as well
as conda if you are not the root.
