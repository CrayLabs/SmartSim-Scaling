# Performance Results

The performance of SmartSim is detailed below across various types of systems.

Note that the first iteration can take longer (up to several seconds) than the rest of the execution. This
is due to the DB loading libraries when the first RedisAI call is made. In the following plots, we excluded
the first iteration time.

## Inference Standard

> Note that Inference is the process of running data points into a machine learning model to calculate an output such as a single numerical score. The SmartSim-Scaling tests use Pytorch's implementation of Resnet50 to...

The following are scaling results from the cpp-inference and fortran-inference scaling tests with ResNet-50
and the imagenet dataset. For more information on these scaling tests, please see
the SmartSim paper on [arXiv](https://arxiv.org/pdf/2104.09355.pdf).

```bash
[run]
name = run-2023-08-17-16:10:12
path = results/inference-standard-scaling/run-2023-08-17-16:10:12
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-17
language = ['cpp', 'fortran']

[attributes]
colocated = 0
client_per_node = [18]
client_nodes = [25, 50, 75, 100]
database_hosts = []
database_nodes = [4, 8, 16]
database_cpus = [8]
database_port = 6780
batch_size = [1000]
device = GPU
num_devices = 1
iterations = 100
language = ['cpp', 'fortran']
db_node_feature = {'constraint': 'P100'}
node_feature = {'constraint': 'SK48'}
wall_time = 15:00:00
```

#### Put Tensor
![Inference Colo](/figures/put_tensor_inf_std.png "Inference Colocated")

#### Unpack Tensor
![Inference Colo](/figures/unpack_tensor_inf_std.png "Inference Colocated")

#### Run Model
![Inference Colo](/figures/run_model_inf_std.png "Inference Colocated")

#### Run Script
![Inference Colo](/figures/run_script_inf_std.png "Inference Colocated")

## Colocated Inference

```bash
[run]
name = run-2023-08-17-18:23:38
path = results/inference-colocated-scaling/run-2023-08-17-18:23:38
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-17
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

#### Put Tensor
![Inference Colo](/figures/put_tensor_inf_colo.png "Inference Colocated")

#### Unpack Tensor
![Inference Colo](/figures/unpack_tensor_inf_colo.png "Inference Colocated")

#### Run Model
![Inference Colo](/figures/run_model_inf_colo.png "Inference Colocated")

#### Run Script
![Inference Colo](/figures/run_script_inf_colo.png "Inference Colocated")

## Inference Performance Analysis

In this section, we will compare inference clients: `put-tensor`, `unpack-tensor`, `run_model` and `run_script`,
for colocated and standard deployement.

- `put-tensor`: Colo deployement offers a consistent median for put_tensor times. Std deployement shows a slight
increase in median as client count grows. However, due to machine constraints, colo maxes at 288 clients while
std maxes at 1800. We can conclude that there is not a significant performance hit putting information into 
the database when comparing std and colo.

- `unpack-tensor`: There is no significant performance advantage when using colo deployement vs std for the client
unpack_tensor. However, std offers higher times concerning outside points than colo. 

- `run_model`: Colo deployment offers a significanlty faster run_model client than std deployment. We can 
infer colo deployement is able to transfer information faster when running a ML model than std deployement.

- `run_script`: Colo deployment offers a significanlty faster run_script client than std deployment. We can 
infer colo deployement is able to transfer information faster when running a ML script than std deployement.

## Throughput Standard

> Note that Throughput measures the total time it takes to push and pull data from a database.
The SmartSim Scaling studies produces..

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
[run]
name = run-2023-08-20-21:03:55
path = results/throughput-colocated-scaling/run-2023-08-20-21:03:55
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-20
language = ['cpp']

[attributes]
colocated = 1
custom_pinning = [False]
client_per_node = [32]
client_nodes = [16, 32, 64, 128]
database_cpus = [8]
iterations = 100
tensor_bytes = [1024]
language = ['cpp']
```

#### Put Tensor & Unpack Tensor
![Throughput colo Unpack](/figures/test.png "Colocated Throughput")

## Throughput Performance Analysis

In this section, we will compare throughput clients: `put-tensor` and `unpack-tensor`,
for colocated and standard deployement.

Moving on to the Throughput Colocated tests, the lower and upper adjacent values are 
- `put_tensor`: 

- `unpack_tensor`: We notice that for both colo and std, unpack_tensor is much faster than put_tensor. 

## Data Aggregation Standard

The following are scaling results for a data aggregation test, run with 32 client threads on each node. The databases were run on 36 CPUs and 4 threads per dataset. 

```bash
[run]
name = run-2023-08-20-21:55:15
path = results/aggregation-standard-scaling/run-2023-08-20-21:55:15
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-20
language = ['cpp']

[attributes]
colocated = 0
client_per_node = [32]
client_nodes = [16, 32, 64, 128]
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
![Data Agg Get List](/figures/std1_data_agg.png "Data Aggregation Standard")

## Data Aggregation Standard Py

> Note that Data Aggregation is the process of summarizing a large pool of data for high level analysis.
For the aggregation tests below...

The following are scaling results for a data aggregation py test, run with 32 client threads on each node. The databases were run on 36 CPUs and 4 threads per dataset.

```bash
[run]
name = run-2023-08-20-22:47:22
path = results/aggregation-standard-scaling-py/run-2023-08-20-22:47:22
smartsim_version = 0.5.0
smartredis_version = 0.3.1
db = redis-server
date = 2023-08-20
language = ['cpp']

[attributes]
colocated = 0
client_per_node = [32]
client_nodes = [16, 32, 64, 128]
db_cpus = 32
iterations = 100
tensor_bytes = [1024]
tensors_per_dataset = [4]
client_threads = [32]
cpu_hyperthreads = 2
language = ['cpp']
wall_time = 05:00:00
```

![Data Agg Py Poll List](/figures/std1_py_data_agg.png.png "Data Aggregation Py Standard")

## Data Aggregation Performance Analysis

In this section, we will compare throughput clients: `get-list` and `poll-list`,
for colocated and standard deployement.

Looking at the data agg violin plots above, we present the argument that retrieving tensors from 
the database shows no large performance difference when comparing a C++ client and a Python client. 
More specifically, using a Python Client instead of a C++ Client will leave you with a `get_list()` performance hit
to the hundredth of a second. However, there is a large performance hit when working with the file system with 
outside points of the violin plots reaching to 2.50 / 1.20 seconds in comparison to the C++ Client: 0.10 seconds and Python Client: 0.18 seconds. We can infer that reading from a database is much faster than from the file system.

If we compare the `poll_list()` violin plots, pulling tensors from the file system proves to be significanlty faster than pulling tensors from a database. 
`get_list`:
`poll_list`:
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

