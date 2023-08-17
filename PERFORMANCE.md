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

```bash
[run]
name = run-2023-08-13-21:29:20
path = results/inference-colocated-scaling/run-2023-08-13-21:29:20
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-13
language = ['cpp', 'fortran']

[attributes]
colocated = 1
client_per_node = [18]
client_nodes = [4, 8, 12, 16]
database_cpus = [8]
database_port = 6780
batch_size = [96]
device = GPU
num_devices = 1
iterations = 100
language = ['cpp', 'fortran']
node_feature = {'constraint': 'P100'}
```

![Inference Colo](/figures/put_tensor.png "Inference Colocated")
![Inference Colo](/figures/run_model.png "Inference Colocated")
![Inference Colo](/figures/run_script.png "Inference Colocated")
![Inference Colo](/figures/unpack_tensor.png "Inference Colocated")

## Inference Performance Analysis

> Note that Inference is the process of running data points into a machine learning model to calculate an output such as a single numerical score. The SmartSim-Scaling tests use Pytorch's implementation of Resnet50 to...

## Throughput Standard
The following configuration file for the example Throughput Standard scaling test is shown below. 

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
The following configuration file for the example Throughput Colocated scaling test is shown below. 

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

> Note that Throughput measures the total time it takes to push and pull data from a database.
The SmartSim Scaling studies produces..

Starting with the Throughput Standard tests, we first notice the outside points for both `put_tensor` and 
`unpack_tensor` are notabley higher than the median for each plot. The outliers increase with the less
database nodes we use. We predict that this is because...

Moving on to the Throughput Colocated tests, the lower and upper adjacent values are 



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

> Note that Data Aggregation is the process of summarizing a large pool of data for high level analysis.
For the aggregation tests below...

Looking at the data agg violin plots above, we present the argument that retrieving tensors from 
the database shows no large performance difference when comparing a C++ client and a Python client. 
More specifically, using a Python Client instead of a C++ Client will leave you with a `get_list()` performance hit
to the hundredth of a second. However, there is a large performance hit when working with the file system with 
outside points of the violin plots reaching to 2.50 / 1.20 seconds in comparison to the C++ Client: 0.10 seconds and Python Client: 0.18 seconds. We can infer that reading from a database is much faster than from the file system.

If we compare the `poll_list()` violin plots, pulling tensors from the file system proves to be significanlty faster than pulling tensors from a database. 

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

