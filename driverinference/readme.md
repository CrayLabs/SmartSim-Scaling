# Inference Scaling

## Co-located inference

Co-located Orchestrators are deployed on the same nodes as the
application. This improves inference performance as no data movement
"off-node" occurs with co-located deployment. For more information
on co-located deployment, see [our documentation](https://www.craylabs.org/docs/orchestrator.html)

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
        Default: 'inference-scaling'
        name of output dir, defaults to "inference-scaling"
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --nodes=NODES
        Default: [12]
        compute nodes to use for synthetic scaling app with a co-located orchestrator database
    --clients_per_node=CLIENTS_PER_NODE
        Default: [18]
        client tasks per compute node for the synthetic scaling app
    --db_cpus=DB_CPUS
        Default: [2]
        number of cpus per compute host for the database
    --db_tpq=DB_TPQ
        Default: [1]
        number of device threads to use for the database
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --pin_app_cpus=PIN_APP_CPUS
        Default: [False]
        pin the threads of the application to 0-(n-db_cpus)
    --batch_size=BATCH_SIZE
        Default: [1]
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
    --rebuild_model=FORCE_REBUILD
        Default: False
        force rebuild of PyTorch model even if it is available
    --iterations=ITERATIONS
        Default: 100
        number of put/get loops run by the applications
    --languages=LANGUAGES
        Default: ['fortran','cpp']
        list of languages to use for the tester "cpp" and/or "fortran"
    --wall_time=WALL_TIME
        Default: "05:00:00"
        allotted time for database launcher to run
```

So for example, the following command could be run to execute a battery of
tests in the same allocation

```bash
python driver.py inference_colocated --clients_per_node=[24,28] \
                                     --nodes=[16] --db_tpq=[1,2,4] \
                                     --db_cpus=[1,2,4,8] --net_ifname=ipogif0 \
                                     --device=GPU
```

This command can be executed in a terminal with an interactive allocation
or used in a batch script such as the following for Slurm based systems

```bash
#!/bin/bash

#SBATCH -N 16
#SBATCH --exclusive
#SBATCH -C P100
#SBATCH -t 10:00:00

module load slurm
python driver.py inference_colocated --clients_per_node=[24,28] \
                                     --nodes=[16] --db_tpq=[1,2,4] \
                                     --db_cpus=[1,2,4,8] --net_ifname=ipogif0
                                     --device=GPU
```

Examples of batch scripts to use are provided in the ``batch_scripts`` directory


## Standard Inference

Co-locacated deployment is the preferred method for running tightly coupled
inference workloads with SmartSim, however, if you want to deploy the Orchestrator
database and the application on different nodes you may want to use standard
deployment.

For example, if you only have a small number of GPU nodes and want to test a large
CPU application you may want to use standard deployment. For more information
on Orchestrator deployment methods, please see
[our documentation](https://www.craylabs.org/docs/orchestrator.html)

Like the above inference scaling tests, the standard inference tests also provide
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
        Default: 'inference-scaling'
        name of output dir
    --launcher=LAUNCHER
        Default: 'auto'
        workload manager i.e. "slurm", "pbs"
    --run_db_as_batch=RUN_DB_AS_BATCH
        Default: True
        run database as separate batch submission each iteration
    --batch_args=BATCH_ARGS
        Default: {}
        additional batch args for the database
    --db_hosts=DB_HOSTS
        Default: []
        optionally supply hosts to launch the database on
    --db_nodes=DB_NODES
        Default: [12]
        number of compute hosts to use for the database
    --db_cpus=DB_CPUS
        Default: [2]
        number of cpus per compute host for the database
    --db_tpq=DB_TPQ
        Default: [1]
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
        Default: [48]
        client tasks per compute node for the synthetic scaling app
    --client_nodes=CLIENT_NODES
        Default: [12]
        number of compute nodes to use for the synthetic scaling app
    --rebuild_model=FORCE_REBUILD
        Default: False
        force rebuild of PyTorch model even if it is available
```

The standard inference tests will spin up a database for each iteration in the
battery of tests chosen by the user. There are multiple ways to run this.

1. Everything in the same interactive (or batch file) without caring about placement
```bash
# alloc must contain at least 120 (max client_nodes) + 16 nodes (max db_nodes)
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --run_db_as_batch=False \
                                    --net_ifname=ipogif0 --device=GPU
```

This option is recommended as it's easy to launch in interactive allocations and
as a batch submission, but if you need to specify separate hosts for the database
you can look into the following two methods.

A batch submission for this first option would look like the following for Slurm
based systems.

```bash
#!/bin/bash

#SBATCH -N 136
#SBATCH --exclusive
#SBATCH -t 10:00:00

python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --run_db_as_batch=False
                                    --net_ifname=ipogif0 --device=CPU
```

2. Same as 1, but specify hosts for the database
```bash
# alloc must contain at least 120 (max client_nodes) + 16 nodes (max db_nodes)
# db nodes must be fixed if hostlist is specified
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --db_hosts=[nid0001, ...] \
                                    --net_ifname=ipogif0 --device=CPU

```

3. Launch database as a separate batch submission each time
```bash
# must obtain separate allocation for client driver through interactive or batch submission
# if batch submission, compute nodes must have access to slurm
python driver.py inference_standard --client_nodes=[20,40,60,80,100,120] \
                                    --db_nodes=[4,8,16] --db_tpq=[1,2,4] \
                                    --db_cpus=[1,4,8,16] --batch_args='{"C":"V100", "exclusive": None}' \
                                    --net_ifname=ipogif0 --device=GPU
```

All three options will conduct ``n`` scaling tests where ``n`` is the multiple of
all lists specified as options.
