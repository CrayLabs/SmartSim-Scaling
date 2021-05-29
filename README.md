
## SmartSim Scaling

This repository holds all of the scripts and materials for testing
the scaling of SmartSim and the SmartRedis clients.


## Scaling Tests

There are two types of scaling tests in the repository.

 1. Inference
 2. Throughput

### Inference

The inference scaling tests perform full inference loops (put, process,
infer, get) on CPU or GPU. The tests use a C++ MPI program to spawn
clients out onto a Slurm based system. There is also a version of the inference
tests for single node scaling.

More information can be found in the ``cpp-inference`` directory.

### Throughput

The throughput tests are similar to that of the inference scaling tests
except for they only test ``put_tensor`` and ``unpack_tensor`` (get_tensor).
The throughput tests are written to run on either a Slurm or PBS based
system.

More information can be found in the ``cpp-throughput`` directory.

## General Performance Tips

There are a few places users can look to get every last bit of performance.

 1. TCP settings
 2. Database settings

The communication goes over the TCP/IP stack. Because of this, there are
a few settings that can be tuned

 - ``somaxconn`` - The max number of socket connections. Set this to at least 4096 if you can
 - ``tcp_max_syn_backlog`` - Raising this value can help with really large tests.

The database (Redis or KeyDB) has a number of different settings that can increase
performance.

For Redis:
  - ``io-threads`` - we set to 4 by default in SmartSim
  - ``io-use-threaded-reads`` - We set to yes (doesn't usually help much)
  - ``maxclients`` - This should be raised to well above what you think the max number of clients will be for each DB shard
  - ``threads-per-queue`` - can be set in ``Orchestrator()`` init. Helps with GPU inference performance (set to 4 or greater)
  - ``inter-op-threads`` - can be set in ``Orchestrator()`` init. helps with CPU inference performance
  - ``intra-op-threads`` - can be set in ``Orchestrator()`` init. helps with CPU inference performance

For KeyDB:
  - ``server-threads`` - Makes a big difference. We use 8 on HPC hardware. Set to 4 by default.


## Scaling Results

We present some of the scaling test numbers for both the throughput
and the inference scaling tests so that users can get a sense of what
kind of performance to expect.

### Inference

### Throughput


## Using KeyDB

KeyDB is a multithreaded version of Redis with some strong performance claims. Luckily, since
KeyDB is a drop in replacement for Redis, it's fairly easy to test. If you are looking for
extreme performance, especially in throughput for large data sizes,
we recommend building SmartSim with KeyDB.

In future releases, switching between Redis and KeyDB will be an ``Orchestrator`` parameter.

Below we compare KeyDB and Redis for the general throughput tests.


A few interesting points:

 1. Client connection time: KeyDB connects client MUCH faster than base Redis. At this time, we
    are not exactly sure why, but it does. So much so, that if you are looking to use the SmartRedis
    clients in such a way that you will be disconnecting and reconnecting to the database, you
    should use KeyDB instead of Redis with SmartSim.

 2. In general, according to the throughput scaling tests, KeyDB has roughly 2x the throughput
    of Redis for data sizes over 1Mb. Redis seems to perform better than KeyDB for smaller data
    sizes (2kiB - 256kiB)

 3. KeyDB seems to handle higher numbers of clients better than Redis does.