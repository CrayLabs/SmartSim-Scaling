# Performance Results

The performance of SmartSim is detailed below across the Super Computer Horizon.

Note that the first iteration can take longer (up to several seconds) than the rest of the execution. This
is due to the DB loading libraries when the first RedisAI call is made. In the following plots, we excluded
the first iteration time.

## Inference Standard

The following are standard deployment scaling results from the cpp-inference and fortran-inference scaling tests with ResNet-50 and the imagenet dataset. For more information on these scaling tests, please see
the SmartSim paper on [arXiv](https://arxiv.org/pdf/2104.09355.pdf).

#### Inference Std Run Configuration File
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

#### Inference Colo Run Configuration File
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

In this section, we will compare inference clients: `put-tensor`, `unpack-tensor`, `run_model` and `run_script`
for colocated and standard deployment.

> Note that Inference is the process of running data points into a machine learning model to calculate an output such as a single numerical score. The SmartSim-Scaling tests use Pytorch's implementation of Resnet50 to feed 
the ImageNet dataset through. We put the ImageNet data into the database via put_tensor, process the data via run_script, run a machine learning model with the data via run_model then retrieve the data via unpack_tensor.

- `put-tensor` : Colo deployment offers a consistent median for put_tensor times. Std deployment shows a slight
increase in median as client count grows. However, due to machine constraints, colo is maxed at 288 clients while
std maxes at 1800 clients. We can conclude that there is not a significant performance hit putting information into the database when comparing std and colo.

- `run_script` : Colo deployment offers a faster run_script client than std deploymnt. We can 
infer colo deployment is able to transfer information faster when processing data than std deployment. This
is likely because communiction time is cut when using colo deployment.

- `run_model` : Colo deployment offers a faster run_model client than std deployment. Like mentioned before,
the communication time is cut when using colocated since the app and database are on the same node.

- `unpack-tensor` : There is no significant performance advantage when using colo deployment vs std for the client
unpack_tensor. However, std shows larger outside points than colo. 

There is no `put-tensor` or `unpack-tensor` performance hit when using standard versus colocated deployment
shown in the violin plots above. This is likely due to our testing constraints as the number of available
GPUs on Horizon is 16 nodes. Therefore, we were not able to fully scale the colocated deployment to the node
size of standard. Future expansive testing may indicate a larger performance hit.

We do however notice a colocated deployment advantage with clients `run_model` and `run_script`. We can infer
that this is because the model and script are on the same node, therefore, it takes less time to communicate.

## Throughput Standard

The following are standard deployment scaling results from the cpp-throughput.

#### Throughput Std Run Configuration File
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

#### Throughput Colo Run Configuration File

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

In this section, we will compare throughput clients: `put-tensor` and `unpack-tensor`,
for colocated and standard deployment.

> Note that Throughput measures the total time it takes to push and pull data from a database.
The SmartSim Scaling studies produces a series of generated tensors to add and pull from a Redis Database.

- `put_tensor` : We notice that for both colocated and standard deployment, put_tensor performance
extremley quickly with both medians performing faster than .001 seconds. The difference here lies 
within the outside points. Looking at the standard violin plots, the outside point values are much
higher than colocated.

- `unpack_tensor` : We notice that for both colo and std, unpack_tensor is much faster than put_tensor. However,
both deployments performance similarly to eachother with the difference being highlighted in the outside points.
Like mentioned before, standard shows larger outside points than colocated.

Since we do not see a significant performance difference with colo vs std, in the future we plan
to expand testing to compare Throughput with a Redis Database and KeyDB. 

## Data Aggregation Standard

The following are standard deployment scaling results from the cpp-data-aggregation.

#### Data Agg Std Run Configuration File
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

#### Data Agg Py Std Run Configuration File
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

## Data Aggregation Performance Analysis

In this section, we will compare throughput clients: `get-list` and `poll-list`,
for colocated and standard deployment.

> Note that Data Aggregation is the process of summarizing a large pool of data for high level analysis.
For the data agg tests, we produce and store tensors in the database to poll and get.

- `poll_list` : Polling tensors from the database shows no large performance difference when comparing a C++ client and a Python client.

- `get_list` : Retrieving tensors from the database shows a very small large performance difference when comparing a C++ client and a Python client. The C++ client performs a hundreth of a second faster than the Python client.

Overall, we can conslude that there is not a notable performance hit when comparing the use of a Python client and
a C++ client. In the future, it is worth expanding the data aggregation testing to a larger number of clients which
might show otherwise.

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

