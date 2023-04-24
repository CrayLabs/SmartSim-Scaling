db_nodes: is the number of shards for the db in standard

colocated: #no pin app cpus? YES ADD IT : launch this program on these cores and dont let it move
colocated: #48 compute cores (compute core on the processor which is on the node), launching 24 instances of the program, 
db_cpus: #how many cpus each db is getting
clients_per_node=[3], #how many cpus the app is getting per node
net_ifname#network interface that the db will be listing over

colo- 
# 12 nodes for the app
        # 24 clients p N
        # 16 nodes given to db
        #shard: over multiple nodes, each shard is a portion of one db

# 48 clients OR 48 apps
# 48 cores OR 48 cpus
# clients are consuming a full core
# db is also tyeing to consume a full 48 cores
# why do we need to specify both db_cpus and clients_per_node for colo


 #model launched on multiple nodes - redis database setup to speak with the mdoel on each node
        #much faster bc not going off the network
        model.colocate_db(port=db_port, #this is the model, colocating the db with the model: the db belongs to the model object and not the entire exp
                        db_cpus=db_cpus,
                        ifname=net_ifname,
                        limit_app_cpus=pin_app_cpus,
                        #threads_per_queue=db_tpq,
                        # turning this to true can result in performance loss
                        # on networked file systems(many writes to db log file)
                        debug=True,
                        loglevel="notice")
exp.generate(model, overwrite=True)#creates the run directory, on the file system







### Throughput

The throughput tests run as an MPI program where a single SmartRedis C++ client
is initialized on every rank. 

Each client performs 10 executions of the following commands

  1) ``put_tensor``     (send image to database)
  2) ``unpack_tensor``  (Retrieve the inference result)

### Standard throughput

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


The input parameters to the test are used to generate permutations
of tests with varying configurations.

```text

NAME
    driver.py throughput_standard - Run the throughput standard scaling tests

SYNOPSIS
    driver.py throughput_standard <flags>

DESCRIPTION
    Run the throughput standard scaling tests

FLAGS
    --exp_name=EXP_NAME
        Default: 'throughput-standard-scaling'
        name of /results/output dir
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
    --db_port=DB_PORT
        Default: 6780
        port to use for the database
    --net_ifname=NET_IFNAME
        Default: 'ipogif0'
        network interface to use i.e. "ib0" for infiniband or "ipogif0" aries networks
    --clients_per_node=CLIENTS_PER_NODE
        Default: [32]
        client tasks per compute node for the synthetic scaling producer app
    --client_nodes=CLIENT_NODES
        Default: [128, 256, 512]
        number of compute nodes to use for the synthetic scaling producer app
    --tensor_bytes=TENSOR_BYTES
        Default: [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 10...
        list of tensor sizes in bytes
```