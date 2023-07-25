# Performance Results

The performance of SmartSim is detailed below across various types of systems.

Note that the first iteration can take longer (up to several seconds) than the rest of the execution. This
is due to the DB loading libraries when the first RedisAI call is made. In the following plots, we excluded
the first iteration time.

## Inference Standard

The following are scaling results from the cpp-inference scaling tests with ResNet-50
and the imagenet dataset. For more information on these scaling tests, please see
the SmartSim paper on arXiv

![Inference plots dark theme](/figures/all_in_one_violin_dark.png#gh-dark-mode-only "Standard inference")
![Inference plots ligh theme](/figures/all_in_one_violin_light.png#gh-light-mode-only "Standard inference")

## Colocated Inference

The following are scaling results for a colocated inference test, run on 12 36-core Intel Broadwell nodes,
each one equipped with 8 Nvidia V100 GPUs. On each node, 28 client threads were run, and the databases
were run on 8 CPUs and 8 threads per queue. 

![Colocated inference plots dark theme](/figures/colo_dark.png "Colocated inference")
![Inference plots ligh theme](/figures/colo_light.png "Colocated inference")

## Inference Performance Analysis

INSERT ANALYSIS

## Throughput Standard
The following are scaling results for a standard throughput test, run on !12 36-core Intel Broadwell nodes!. On each node, 48 client threads were run, and the databases
were run on 8 CPUs and 8 threads per queue. 

#### Unpack Tensor - retrieve the data
![Throughput Std Unpack](/figures/unpack_tensor_thro_std.png "Throughput Standard")

#### Put Tensor - send image to database
![Throughput Std Put](/figures/put_tensor_thro_std.png "Throughput Standard")

## Throughput Colocated
The following are scaling results for a colocated throughput test, !run on 12 36-core Intel Broadwell nodes!. On each node, 48 client threads were run.

#### Unpack Tensor - retrieve the data
![Throughput colo Unpack](/figures/unpack_tensor_thro_colo.png "Colocated Throughput")

#### Put Tensor - send image to database
![Throughput colo put](/figures/put_tensor_thro_colo.png "Colocated Throughput")

## Throughput Performance Analysis

INSERT ANALYSIS

## Data Aggregation Standard

The following are scaling results for a colocated throughput test, !run on 12 36-core Intel Broadwell nodes!. On each node, 48 client threads were run.

#### Get List - retrieve the data from the aggregation list
![Data Agg Get List](/figures/get_list_data_agg.png "Data Aggregation Standard")

## Data Aggregation Standard Py
The following are scaling results for a colocated throughput test, !run on 12 36-core Intel Broadwell nodes!. On each node, 48 client threads were run.

#### Get List - retrieve the data from the aggregation list
![Data Agg Py Get List](/figures/get_list_data_agg_py.png "Data Aggregation Py Standard")

#### Poll List - check when the next aggregation list is ready
![Data Agg Py Poll List](/figures/poll_list_data_agg_py.png "Data Aggregation Py Standard")

## Data Aggregation Standard Py Fs
The following are scaling results for a colocated throughput test, !run on 12 36-core Intel Broadwell nodes!. On each node, 48 client threads were run.

#### Get List - retrieve the data from the aggregation list
![Data Agg Py Fs Get List](/figures/get_list_data_agg_fs.png "Data Aggregation Py Fs Standard")

#### Poll List - check when the next aggregation list is ready
![Data Agg Py Fs Poll List](/figures/poll_list_data_agg_fs.png "Data Aggregation Py Fs Standard")

## Data Aggregation Performance Analysis

INSERT ANALYSIS

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

