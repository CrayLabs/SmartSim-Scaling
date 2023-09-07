# Inference Scaling Test

SmartSim-Scaling offers two inference versions:

 1. Inference Standard              (C++ client and SmartRedis Orchestrator)
 2. Inference Colocated Python     (C++ client and SmartRedis Orchestrator)

Continue below for more information on both respective tests.

## Client Description

The inference tests run as an MPI program where a single SmartRedis C++ client
is initialized on every rank.

Supported inference tests:

  1) Resnet50 CNN with ImageNet dataset

Each client performs 101 executions of the following commands. The first iteration is a warmup;
the next 100 are measured for inference throughput. 

  1) ``put_tensor``     (send image to database)
  2) ``run_script``     (preprocess image)
  3) ``run_model``      (run resnet50 on the image)
  4) ``unpack_tensor``  (Retrieve the inference result)

The input parameters to the test are used to generate permutations
of tests with varying configurations.

## The model
As Neural Network, we use Pytorch's implementation of Resnet50. The script `imagenet/model_saver.py`
can be used to generate the model for CPU or GPU. By navigating to the `imagenet` folder, the CPU model
can be created running 

```bash
python model_saver.py
```

and the GPU model can be created running

```bash
python model_saver.py --device=GPU
```

> Note that if you would like to generate the GPU model, you must run the 
command on a GPU node.

If the benchmark driver is executed and
no model exists, an attempt is made to generate the model on the fly. In both cases,
the specified device must be available on the node where the script is called (this
means that it could be required to run the script through the workload manager launcher
to execute it on a node with a GPU, for example).


## Colocated inference

Colocated Orchestrators are deployed on the same nodes as the
application. This improves inference performance as no data movement
"off-node" occurs with colocated deployment. For more information
on colocated deployment, see [our documentation](https://www.craylabs.org/docs/orchestrator.html)

Below is the help output. The arguments which are lists control
the possible permutations that will be run.

```text
NAME
    driver.py inference_colocated - Run ResNet50 inference tests with colocated Orchestrator deployment

SYNOPSIS
    driver.py inference_colocated <flags>

DESCRIPTION
    Run ResNet50 inference tests with colocated Orchestrator deployment

FLAGS
    --exp_name=EXP_NAME
        Default: 'inference-colocated-scaling'
        name of output dir, defaults to "inference-scaling"
    --node_feature=NODE_FEATURE
        Default: {'constraint': 'P100'}
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --nodes=NODES
        Default: [4,8,16,32,64,128]
        compute nodes to use for synthetic scaling app with a colocated orchestrator database
    --clients_per_node=CLIENTS_PER_NODE
        Default: [18]
        client tasks per compute node for the synthetic scaling app
    --db_cpus=DB_CPUS
        Default: [8]
        number of cpus per compute host for the database
    --db_tpq=DB_TPQ
        Default: [8]
        number of device threads to use for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --batch_size=BATCH_SIZE
        Default: [96]
        batch size to set Resnet50 model with
    --device=DEVICE
        Default: 'GPU'
        device used to run the models in the database
    --num_devices=NUM_DEVICES
        Default: 1
        number of devices per compute node to use to run ResNet
    --net_type=NET_TYPE
        Default: 'uds'
        type of connection to use ("tcp" or "uds")
    --net_ifname=NET_IFNAME
        Default: 'lo'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --iterations=ITERATIONS
        Default: 100
        number of put/get loops run by the applications
    --languages=LANGUAGES
        Default: ['cpp','fortran']
        list of languages to use for the tester "cpp" and/or "fortran"
    --plot=PLOT
        Default: 'database_cpus'
        flag to plot against in process results
```

> The interface name may be different on your target system. Please update the `net_ifname` flag to the appropriate value.

So for example, the following command could be run to execute a battery of
tests in the same allocation

```bash
# alloc must contain at least 16 GPU nodes
python driver.py inference_colocated --clients_per_node=[18] \
                                     --nodes=[16] --db_tpq=[1,2,4] \
                                     --db_cpus=[1,2,4,8] --net_ifname="ipogif0" \
                                     --device="GPU"
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
#!/bin/bash

#SBATCH -N 16
#SBATCH --exclusive
#SBATCH -C P100
#SBATCH -t 10:00:00

python driver.py inference_colocated --clients_per_node=[18] \
                                     --nodes=[16] --db_tpq=[1,2,4] \
                                     --db_cpus=[1,2,4,8] --net_ifname="ipogif0" \
                                     --device="GPU"
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory


## Standard Inference

Colocated deployment is the preferred method for running tightly coupled
inference workloads with SmartSim, however, if you want to deploy the Orchestrator
database and the application on different nodes, you want to use standard
deployment.

For example, if you only have access to a small number of GPU nodes and want to test a large
CPU application, standard deployment is optimal. For more information
on Orchestrator deployment methods, please see
[our documentation](https://www.craylabs.org/docs/orchestrator.html)

Like the above colocated inference tests, the standard inference tests also provide
a method of running a battery of tests all at once. Below is the help output.
The arguments which are lists control the possible permutations that will be run.

```text
NAME
    driver.py inference_standard - Run ResNet50 inference tests with standard Orchestrator deployment

SYNOPSIS
    driver.py inference_standard <flags>

DESCRIPTION
    Run ResNet50 inference tests with standard Orchestrator deployment

FLAGS
    --exp_name=EXP_NAME
        Default: 'inference-standard-scaling'
        name of output dir
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --run_db_as_batch=RUN_DB_AS_BATCH
        Default: False
        run database as separate batch submission each iteration
    --db_node_feature=DB_NODE_FEATURE
        Default: {'constraint': 'P100'}
        dict of runsettings for the database
    --node_feature=NODE_FEATURE
        Default: {}
        dict of runsettings for the app
    --db_hosts=DB_HOSTS
        Default: []
        optionally supply hosts to launch the database on
    --db_nodes=DB_NODES
        Default: [4,8,16]
        number of compute hosts to use for the database
    --db_cpus=DB_CPUS
        Default: [8]
        number of cpus per compute host for the database
    --db_tpq=DB_TPQ
        Default: [8]
        number of device threads to use for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --batch_size=BATCH_SIZE
        Default: [1000]
        batch size to set Resnet50 model with
    --device=DEVICE
        Default: 'GPU'
        device used to run the models in the database
    --num_devices=NUM_DEVICES
        Default: 1
        number of devices per compute node to use to run ResNet
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --clients_per_node=CLIENTS_PER_NODE
        Default: [18]
        client tasks per compute node for the synthetic scaling app
    --client_nodes=CLIENT_NODES
        Default: [4,8,16,32,64,128]
        number of compute nodes to use for the synthetic scaling app
    --iterations=ITERATIONS
        Default: 100
        number of put/get loops run by the applications
    --wall_time=WALL_TIME
        Default: "05:00:00"
        allotted time for database launcher to run
    --languages=LANGUAGES
        Default: ['cpp','fortran']
        list of languages to use for the tester "cpp" and/or "fortran"
    --plot=PLOT
        Default: 'database_nodes'
        flag to plot against in process results
```

The standard inference tests will spin up a database for each iteration in the
battery of tests chosen by the user. There are multiple ways to run this.

> The interface name may be different on your target system. Please update the `net_ifname` flag to the appropriate value.

1. Everything in the same interactive (or batch file) without caring about placement
```bash
# alloc must contain at least 120 (max client_nodes) + 16 GPU nodes (max db_nodes)
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --run_db_as_batch=False \
                                    --net_ifname="ipogif0" --device="GPU"
```

This option is recommended as it's easy to launch in interactive allocations and
as a batch submission, but if you need to specify separate hosts for the database
you can look into the following two methods.

A batch submission for this first option would look like the following for Slurm
based systems.

```bash
#!/bin/bash

#SBATCH -N 136
#SBATCH -C "[P100*16&SK48*120]"
#SBATCH --exclusive
#SBATCH -t 10:00:00

python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --run_db_as_batch=False \
                                    --net_ifname="ipogif0" --device="GPU"
```

2. Same as 1, but specify hosts for the database
```bash
# alloc must contain at least 120 (max client_nodes) + 16 nodes (max db_nodes)
# db nodes must be fixed if hostlist is specified
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --db_hosts=[nid0001, ...] \
                                    --net_ifname="ipogif0" --device="GPU"

```

3. Launch database as a separate batch submission each time
```bash
# must obtain separate allocation for client driver through interactive or batch submission
# if batch submission, compute nodes must have access to slurm
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --batch_args='{"C":"V100", "exclusive": None}' \
                                    --net_ifname="ipogif0" --device="GPU" \
                                    --run_db_as_batch=True
```

All three options will conduct ``n`` scaling tests where ``n`` is the product of
all lists specified as options.