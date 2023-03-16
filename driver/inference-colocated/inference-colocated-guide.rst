### Co-located inference

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