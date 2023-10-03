# Performance Results

We have collected SmartSim performance results on Horizon, a Cray XC50 supercomputer.

Horizon Node Hardware Summary:

| Nodes | Cores | Threads | Processor | Memory | GPU |
| :--- | --- | --- | --- | --- | --- |
| 34 | 18 | 36 | Xeon E5-2699 v4 @ 2.20GHz BDW | 64 GB DDR4-2400 | --- |
| 16 | 18 | 36 | Xeon E5-2699 v4 @ 2.20GHz BDW | 64 GB DDR4-2400 | 1 Nvidia Tesla_P100-PCIE-16GB |
| 100 | 48 | 96 | Xeon 8160 CPU @ 2.10GHz SKL | 192 GB DDR4-2666 | --- |
| 60 | 56 | 112 | Xeon 8176 CPU @ 2.10GHz SKL | 192 GB DDR4-2666 | --- |
| 48 | 48 | 96 | Xeon 8260 CPU @ 2.40GHz CSL | 192 GB DDR4-2666 | --- |
| 53 | 48 | 96 | Xeon 8260 CPU @ 2.40GHz CSL | 384 GB DDR4-2933 | --- |
| 28 | 64 | 256 | ThunderX2 CN9980 v2.2 @ 2.50GHz | 256 GB DDR4-2666 | --- |

We have provided scaling results represented in the form of violin plots for the following:

1. Inference Standard & Colocated
2. Throughput Standard & Colocated
3. Data Aggregation Standard

All scaling tests utilize a redis database excluding the last data aggregation test that uses the file system. Note that the first iteration can take longer (up to several seconds) than the rest of the execution. This
is due to the DB loading libraries when the first RedisAI call is made. In the following plots, we excluded
the first iteration time.

## Inference Standard

The following are standard deployment scaling results from the cpp-inference and fortran-inference scaling tests using the resNet-50 model and the imagenet dataset. ResNet-50 model is a convolutional neural network that is 50 layers deep. We train the model using more than a million images from the imageNet database. The imageNet database holds 14 million hand annotated images that are used for visual object recognition software research. For more information on these scaling tests, please see
the SmartSim paper on [arXiv](https://www.sciencedirect.com/science/article/pii/S1877750322001065).

#### Inference Standard Run Configuration File
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

#### Put Tensor (send image to database)
![Inference Colo](/figures/put_tensor_inf_std.png "Inference Colocated")

#### Run Script (preprocess image)
![Inference Colo](/figures/run_script_inf_std.png "Inference Colocated")

#### Run Model (run resnet50 on the image)
![Inference Colo](/figures/run_model_inf_std.png "Inference Colocated")

#### Unpack Tensor (retrieve the inference result)
![Inference Colo](/figures/unpack_tensor_inf_std.png "Inference Colocated")

## Colocated Inference

The following are colocated deployment scaling results from the cpp-inference and fortran-inference scaling tests with ResNet-50 and the imagenet dataset. For more information on these scaling tests, please see
the SmartSim paper on [arXiv](https://arxiv.org/pdf/2104.09355.pdf).

#### Inference Colocated Run Configuration File
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

#### Put Tensor (send image to database)
![Inference Colo](/figures/put_tensor_inf_colo.png "Inference Colocated")

#### Run Script (preprocess image)
![Inference Colo](/figures/run_script_inf_colo.png "Inference Colocated")

#### Run Model (run resnet50 on the image)
![Inference Colo](/figures/run_model_inf_colo.png "Inference Colocated")

#### Unpack Tensor (retrieve the inference result)
![Inference Colo](/figures/unpack_tensor_inf_colo.png "Inference Colocated")

## Inference Performance Analysis

In this section, we compare the performance results of client operations: `put_tensor`, `unpack_tensor`, `run_model` and `run_script`
for colocated and standard deployment.

> Inference is the process of running data points into a machine learning model to calculate an output such as a single numerical score. 

- `put-tensor` : Colocated deployment offers a consistent median for put_tensor times. Standard deployment shows a slight
increase in median as client count grows. However, due to machine constraints, colocated is maxed at 288 clients while
standard maxes at 1800 clients. Due to Horizon offering 16 GPU nodes, there is no significant performance hit comparing the 
graphs above. However, we do know that there is a delay in network communication when using standard deployment.

- `run_script` : Colocated deployment offers a faster run_script function than standard deployment. We can 
infer colocated deployment is able to transfer information faster when processing data than standard deployment. 
This is likely because communication time is cut when using colocated deployment. There are also not as many requests sent using colocated deployment than standard. This is because the database is split across multiple shards when using standard, the clients must communicate between all shards, adding additional network latency.

- `run_model` : Colocated deployment demonstrates a faster run_model client than standard deployment. Like mentioned before,
there is no additional network latency. By using standard deployment, you increase the number of requests sent during the runtime. This is because the database is split up into multiple shards instead of being centralized on the same node in colocated deployment.

- `unpack-tensor` : There is no significant performance advantage when using colocated deployment vs standard for the client
unpack_tensor. However, standard shows larger outside points than colocated. This is likely because the number of requests is greater during standard deployment. Those requests, as they wait to be processed, add additional network communication time.

Due to machine constraints, there is not a large performance hit with `put-tensor` or `unpack-tensor` when using standard versus colocated deployment. Our testing constraints limited the scaling study tests to 16 GPU nodes. Therefore, we were not able to fully scale the colocated deployment to the node size of standard. Future expansive testing may indicate a larger performance hit. We do however notice a colocated deployment advantage with clients `run_model` and `run_script`. We can infer that this is due to the fact that the process is able to complete faster during colocated deployment due to the app and database being centralized on the same nodes. During standard deployment, the database is split into multiple shards. The application node has to travel to the database nodes to complete the `run_model` and `run_script` functions, earning the greater completion time. We can therefore conclude that there is a performance benefit using colocated deployment during functions `run_model` and `run_script`.

## Throughput Standard

The following are standard deployment scaling results from the cpp-throughput.

#### Throughput Standard Run Configuration File
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

#### Put Tensor (send image to database) & Unpack Tensor (retrieve the image)
![Throughput Std Unpack](/figures/new_std_thro.png "Throughput Standard")

## Throughput Colocated

The following are colocated deployment scaling results from the cpp-throughput.

#### Throughput Colocated Run Configuration File

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

#### Put Tensor (send image to database) & Unpack Tensor (retrieve the image)
![Throughput colo Unpack](/figures/test.png "Colocated Throughput")

## Throughput Performance Analysis

In this section, we will compare client operations: `put-tensor` and `unpack-tensor`,
for colocated and standard deployment.

> Throughput measures the total time it takes to push and pull data from a database.
The SmartSim Scaling studies produces a series of generated tensors to add (put_tensor) and retrieve from (unpack_tensor) a Redis Database.

- `put_tensor` : We notice that for both colocated and standard deployment, put_tensor completes
extremely quickly with both medians performing faster than .001 seconds. The difference here lies 
within the outside points. Looking at the standard violin plots, the high-end distribution values are much
higher than colocated. We can attribute this to the network latency added when using standard orchestrator deployment.
Through colocated deployment, no additional communication time is added since the application and database are
centralized to the same nodes.

- `unpack_tensor` : We notice that for both colocated and standard, unpack_tensor completes faster than put_tensor. However,
both deployment options perform similarly to each other with the difference being highlighted in the outside points.
As mentioned before, standard shows larger outside points than colocated. We can once again attribute this to the added
network latency during standard deployment.

Since we do not see a significant performance difference with colocated vs standard, in the future we plan
to expand testing to compare Throughput with a Redis Database and KeyDB. 

## Data Aggregation Standard

The following are standard deployment scaling results from the cpp-data-aggregation.

#### Data Agg Standard Run Configuration File
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
#### Poll List (check when the next list is ready) & Get List (retrieve the data from the aggregation list)
![Data Agg Get List](/figures/std1_data_agg.png "Data Aggregation Standard")

## Data Aggregation Standard Py

The following are standard deployment scaling results from the cpp-py-data-aggregation/db.

#### Data Agg Py Standard Run Configuration File
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
#### Poll List (check when the next list is ready) & Get List (retrieve the data from the aggregation list)
![Data Agg Py Poll List](/figures/std1_py_data_agg.png "Data Aggregation Py Standard")

## Data Aggregation Standard Py File System

The following are standard deployment scaling results from the cpp-py-data-aggregation/fs.

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

#### Poll List (check when the next list is ready) & Get List (retrieve the data from the aggregation list)
![Data Agg Py Fs Get List](/figures/data_agg_fs.png "Data Aggregation Py Fs Standard")

## Data Aggregation Performance Analysis

In this section, we will compare client operations: `get-list` and `poll-list`,
for standard deployment with a Python and C++ client.

> Data Aggregation is the process of summarizing a large pool of data for high level analysis.
For the data aggregation tests, we produce and store tensors in the database to poll and get.

- `poll_list` : Polling tensors from the database reveals no large performance difference when comparing the use of a C++ client and a Python client. However, there is a large performance contrast when polling from a file system instead of a database. The file system expectedly demonstrates faster polling of tensors. This is expected because no network communication adds additional time to the completion time but instead local on the machine. Knowing the location of the file, the file system is able to poll quickly, however, you lose the ability to manage complex relationships as well as ensure data accuracy, completeness, and correctness.

- `get_list` : Retrieving tensors from the database demonstrates no performance benefit when comparing a C++ client and a Python client. However, comparing the use of a file system over the database discloses a substantial performance hit. Using a file system adds a significant amount of time since there is no efficient way to query process. A database supports parsing, and optimizing the query contributing to faster retrieval of tensors. The orchestrator supports indexing on any attribute. This helps fast retrieval of data and is not supported by the file system.

Although there is no note-able performance hit when comparing a C++ client and Python client, using the file system over the database adds substantial time to a program's completion. By using the SmartRedis Orchestrator, not only are we able to efficiently query data, validate data concurrency, but also data can be shared easily due to a centralized system. We may also manipulate the data, and rely on secure data recover and data backup options offered by the database. Using a database is especially important when running a large scale test that cannot be stored on a file system.

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

