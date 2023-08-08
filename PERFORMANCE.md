# Performance Results

The performance of SmartSim is detailed below across various types of systems.

Note that the first iteration can take longer (up to several seconds) than the rest of the execution. This
is due to the DB loading libraries when the first RedisAI call is made. In the following plots, we excluded
the first iteration time.

## Inference Standard

The following are scaling results from the cpp-inference scaling tests with ResNet-50
and the imagenet dataset. For more information on these scaling tests, please see
the SmartSim paper on arXiv

![Inference Colo](/figures/put_tensor_inf_std.png "Inference Colocated")
![Inference Colo](/figures/unpack_tensor_inf_std.png "Inference Colocated")
![Inference Colo](/figures/run_model_inf_std.png "Inference Colocated")
![Inference Colo](/figures/run_script_inf_std.png "Inference Colocated")

## Colocated Inference

The following are scaling results for a colocated inference test, run on 12 36-core Intel Broadwell nodes,
each one equipped with 8 Nvidia V100 GPUs. On each node, 28 client threads were run, and the databases
were run on 8 CPUs and 8 threads per queue. 

![Inference Colo](/figures/inf_colo.png "Inference Colocated")

## Inference Performance Analysis

INSERT ANALYSIS - am rerunning the inference tests : ignore Inference section for this review

## Throughput Standard
The following are scaling results for a standard throughput test, run with 48 client threads on each node. The databases were run on 8 CPUs and 8 threads per queue. 

```bash
[run]
name = run-2023-07-05-21:26:18
path = results/throughput-standard-scaling/run-2023-07-05-21:26:18
smartsim_version = 0.4.2
smartredis_version = 0.3.1
db = redis-server
date = 2023-07-05
language = ['cpp']

[attributes]
colocated = 0
client_per_node = [48]
client_nodes = [4, 8, 16, 32, 64, 128]
database_nodes = [4, 8, 16]
database_cpus = [8]
iterations = 100
tensor_bytes = [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 1024000]
language = ['cpp']
wall_time = 05:00:00
```

#### Put Tensor & Unpack Tensor
![Throughput Std Unpack](/figures/new_std_thro.png "Throughput Standard")

## Throughput Colocated
The following are scaling results for a colocated throughput test, run with 48 client threads on each node. The databases were run on 8 CPUs and 8 threads per queue. 

```bash
[run]
name = run-2023-08-07-10:03:47
path = results/throughput-colocated-scaling/run-2023-08-07-10:03:47
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-07
language = ['cpp']

[attributes]
colocated = 1
custom_pinning = [False]
client_per_node = [48]
client_nodes = [4, 8, 16, 32, 64, 128]
database_cpus = [8]
iterations = 100
tensor_bytes = [1024, 8192, 16384, 32769, 65538, 131076, 262152, 524304, 1024000]
language = ['cpp']
```

#### Put Tensor & Unpack Tensor
![Throughput colo Unpack](/figures/new_colo_thro.png "Colocated Throughput")

## Throughput Performance Analysis

1. The Throughput Standard test density decreases as the database nodes increase. Outliers become greater as client total increases. This is because as there are more clients, there is a longer wait time. We can also tell that the standard takes longer than colocated because the client nodes are having to travel offsight to the database client node. Each database is having to handle more requests since they do not 
have their own clients. 

2. Put/Unpack tensor for Throughput Colocated stays consistent with density across all client totals.
It takes much less time to complete using a colocated orchestrator since all the computations are happening on the same node. Less time it takes to communicate since the db and app are on the same node.

## Data Aggregation Standard

The following are scaling results for a data aggregation test, run with 32 client threads on each node. The databases were run on 36 CPUs and 4 threads per dataset. 

```bash
[run]
name = run-2023-07-20-14:04:10
path = results/aggregation-standard-scaling/run-2023-07-20-14:04:10
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-07-20
language = ['cpp']

[attributes]
colocated = 0
client_per_node = [32]
client_nodes = [4, 8, 16, 32, 64, 128]
db_cpus = 36
iterations = 100
tensor_bytes = [1024]
tensors_per_dataset = [4]
client_threads = [32]
cpu_hyperthreads = 2
language = ['cpp']
wall_time = 10:00:00
```
#### Get List - retrieve the data from the aggregation list
![Data Agg Get List](/figures/get_list_data_agg.png "Data Aggregation Standard")

## Data Aggregation Standard Py
The following are scaling results for a data aggregation py test, run with 32 client threads on each node. The databases were run on 36 CPUs and 4 threads per dataset.

```bash
[run]
name = run-2023-07-20-14:51:41
path = results/aggregation-standard-scaling-py/run-2023-07-20-14:51:41
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-07-20
language = ['cpp']

[attributes]
colocated = 0
client_per_node = [32]
client_nodes = [4, 8, 16, 32, 64, 128]
db_cpus = 32
iterations = 100
tensor_bytes = [1024]
tensors_per_dataset = [4]
client_threads = [32]
cpu_hyperthreads = 2
language = ['cpp']
wall_time = 05:00:00
```

![Data Agg Py Poll List](/figures/data_agg_py.png "Data Aggregation Py Standard")

## Data Aggregation Standard Py Fs
The following are scaling results for a data aggregation py fs test, run with 32 client threads on each node.

```bash
[run]
name = run-2023-07-20-15:56:58
path = results/aggregation-standard-scaling-py-fs/run-2023-07-20-15:56:58
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-07-20
language = ['cpp']

[attributes]
colocated = 0
client_per_node = [32]
client_nodes = [4, 8, 16, 32, 64, 128]
iterations = 100
tensor_bytes = [1024]
tensors_per_dataset = [4]
client_threads = [32]
cpu_hyperthreads = 2
language = ['cpp']
```

![Data Agg Py Fs Get List](/figures/data_agg_fs.png "Data Aggregation Py Fs Standard")

## Data Aggregation Performance Analysis

1. The Standard Data Aggregation test performs the fastest get_list() client function. We
can assume this is due to the fact that there is one less communication layer
than the other two tests that use python consumer clients with c++ producer clients. 
We can also see that as the database node count increases, the violin plot
density decreases. We can assume this is happening because there are more shards
of the database avaiable to complete the function get_list().

2. The Standard Data Aggregation Python test seems to be less stable than the previous test using all C++ clients. Elements of the list might be distributed unevenly causing a contention between this test and the
previous.

3. The Standard Data Aggregation File System test shows much faster results in the poll_list()
chart than the previous test. This is because it is much faster to write to a file system
than using a database since there is signigicantly less nodes. 
However, it is much faster to read from a database than the file system.

## Advanced Performance Tips

There are a few places users can look to get every last bit of performance.

 1. TCP settings
 2. Database settings

The communication goes over the TCP/IP stack. Because of this, there are
a few settings that can be tuned

 - ``somaxconn`` - The max number of socket connections. Set this to at least 4096 if you can
 - ``tcp_max_syn_backlog`` - Raising this value can help with really large tests.

The database (Redis or KeyDB) has a number of different settings that can increase
performance.

For both Redis and KeyDB:
  - ``maxclients`` - This should be raised to well above what you think the max number of clients will be for each DB shard
  - ``threads-per-queue`` - can be set in ``Orchestrator()`` init. Helps with GPU inference performance (set to 4 or greater)
  - ``inter-op-threads`` - can be set in ``Orchestrator()`` init. helps with CPU inference performance
  - ``intra-op-threads`` - can be set in ``Orchestrator()`` init. helps with CPU inference performance

For Redis:
  - ``io-threads`` - we set to 4 by default in SmartSim
  - ``io-use-threaded-reads`` - We set to yes (doesn't usually help much)

For KeyDB:
  - ``server-threads`` - Makes a big difference. We use 8 on HPC hardware. Set to 4 by default.

