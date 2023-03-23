# import fire

# class DataAggregation:
#     def aggregation_scaling(self,
#                             exp_name="aggregation-scaling",
#                             launcher="auto",
#                             run_db_as_batch=True,
#                             batch_args={},
#                             db_hosts=[],
#                             db_nodes=[12],
#                             db_cpus=36,
#                             db_port=6780,
#                             net_ifname="ipogif0",
#                             clients_per_node=[32],
#                             client_nodes=[128, 256, 512],
#                             iterations=20,
#                             tensor_bytes=[1024, 8192, 16384, 32769, 65538,
#                                           131076, 262152, 524304, 1024000,
#                                           2048000],
#                             tensors_per_dataset=[1,2,4],
#                             client_threads=[1,2,4,8,16,32],
#                             cpu_hyperthreads=2):

#         """Run the data aggregation scaling tests.  Permutations of the test
#         include client_nodes, clients_per_node, tensor_bytes,
#         tensors_per_dataset, and client_threads.

#         :param exp_name: name of output dir
#         :type exp_name: str, optional
#         :param launcher: workload manager i.e. "slurm", "pbs"
#         :type launcher: str, optional
#         :param run_db_as_batch: run database as separate batch submission
#                                 each iteration
#         :type run_db_as_batch: bool, optional
#         :param batch_args: additional batch args for the database
#         :type batch_args: dict, optional
#         :param db_hosts: optionally supply hosts to launch the database on
#         :type db_hosts: list, optional
#         :param db_nodes: number of compute hosts to use for the database
#         :type db_nodes: list, optional
#         :param db_cpus: number of cpus per compute host for the database
#         :type db_cpus: list, optional
#         :param db_port: port to use for the database
#         :type db_port: int, optional
#         :param net_ifname: network interface to use i.e. "ib0" for infiniband or
#                            "ipogif0" aries networks
#         :type net_ifname: str, optional
#         :param clients_per_node: client tasks per compute node for the aggregation
#                                  producer app
#         :type clients_per_node: list, optional
#         :param client_nodes: number of compute nodes to use for the aggregation
#                              producer app
#         :type client_nodes: list, optional
#         :param iterations: number of append/retrieve loops run by the applications
#         :type iterations: int
#         :param tensor_bytes: list of tensor sizes in bytes
#         :type tensor_bytes: list[int], optional
#         :param tensors_per_dataset: list of number of tensors per dataset
#         :type tensor_bytes: list[int], optional
#         :param client_threads: list of the number of client threads used for data
#                                aggregation
#         :type client_threads: list[int], optional
#         :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
#                                  to request that the consumer application utilizes
#                                  all physical cores for each client thread.
#         :type cpu_hyperthreads: int, optional
#         """
#         logger.info("Starting dataset aggregation scaling tests")
#         logger.info(f"Running with database backend: {_get_db_backend()}")
#         logger.info(f"Running with launcher: {launcher}")

#         exp = create_folder(self, exp_name, launcher)

#         for db_node_count in db_nodes:

#             # start the database only once per value in db_nodes so all permutations
#             # are executed with the same database size without bringin down the database
#             db = start_database(exp,
#                                 db_port,
#                                 db_node_count,
#                                 db_cpus,
#                                 None, # not setting threads per queue in throughput tests
#                                 net_ifname,
#                                 run_db_as_batch,
#                                 batch_args,
#                                 db_hosts)


#             perms = list(product(client_nodes, clients_per_node,
#                                  tensor_bytes,tensors_per_dataset, client_threads))
#             for perm in perms:
#                 c_nodes, cpn, _bytes, t_per_dataset, c_threads = perm
#                 logger.info(f"Running with threads: {c_threads}")
#                 # setup a an instance of the C++ driver and start it
#                 aggregation_producer_sessions = \
#                     create_aggregation_producer_session(exp, c_nodes, cpn,
#                                                         db_node_count,
#                                                         db_cpus, iterations,
#                                                         _bytes, t_per_dataset)

#                 # setup a an instance of the C++ driver and start it
#                 aggregation_consumer_sessions = \
#                     create_aggregation_consumer_session(exp, c_nodes, cpn,
#                                                         db_node_count, db_cpus,
#                                                         iterations, _bytes, t_per_dataset,
#                                                         c_threads, cpu_hyperthreads)

#                 exp.start(aggregation_producer_sessions,
#                           aggregation_consumer_sessions,
#                            summary=True)

#                 # confirm scaling test run successfully
#                 stat = exp.get_status(aggregation_producer_sessions)
#                 if stat[0] != status.STATUS_COMPLETED:
#                     logger.error(f"ERROR: One of the scaling tests failed \
#                                   {aggregation_producer_sessions.name}")
#                 stat = exp.get_status(aggregation_consumer_sessions)
#                 if stat[0] != status.STATUS_COMPLETED:
#                     logger.error(f"ERROR: One of the scaling tests failed \
#                                   {aggregation_consumer_sessions.name}")

#             # stop database after this set of permutations have finished
#             exp.stop(db)

#     def aggregation_scaling_python(self,
#                             exp_name="aggregation-scaling-py-db",
#                             launcher="auto",
#                             run_db_as_batch=True,
#                             batch_args={},
#                             db_hosts=[],
#                             db_nodes=[12],
#                             db_cpus=36,
#                             db_port=6780,
#                             net_ifname="ipogif0",
#                             clients_per_node=[32],
#                             client_nodes=[128, 256, 512],
#                             iterations=20,
#                             tensor_bytes=[1024, 8192, 16384, 32769, 65538,
#                                           131076, 262152, 524304, 1024000,
#                                           2048000],
#                             tensors_per_dataset=[1,2,4],
#                             client_threads=[1,2,4,8,16,32],
#                             cpu_hyperthreads=2):
#         """Run the data aggregation scaling tests with python consumer.
#         Permutations of the test include client_nodes, clients_per_node, 
#         tensor_bytes, tensors_per_dataset, and client_threads.

#         :param exp_name: name of output dir
#         :type exp_name: str, optional
#         :param launcher: workload manager i.e. "slurm", "pbs"
#         :type launcher: str, optional
#         :param run_db_as_batch: run database as separate batch submission
#                                 each iteration
#         :type run_db_as_batch: bool, optional
#         :param batch_args: additional batch args for the database
#         :type batch_args: dict, optional
#         :param db_hosts: optionally supply hosts to launch the database on
#         :type db_hosts: list, optional
#         :param db_nodes: number of compute hosts to use for the database
#         :type db_nodes: list, optional
#         :param db_cpus: number of cpus per compute host for the database
#         :type db_cpus: list, optional
#         :param db_port: port to use for the database
#         :type db_port: int, optional
#         :param net_ifname: network interface to use i.e. "ib0" for infiniband or
#                            "ipogif0" aries networks
#         :type net_ifname: str, optional
#         :param clients_per_node: client tasks per compute node for the aggregation
#                                  producer app
#         :type clients_per_node: list, optional
#         :param client_nodes: number of compute nodes to use for the aggregation
#                              producer app
#         :type client_nodes: list, optional
#         :param iterations: number of append/retrieve loops run by the applications
#         :type iterations: int
#         :param tensor_bytes: list of tensor sizes in bytes
#         :type tensor_bytes: list[int], optional
#         :param tensors_per_dataset: list of number of tensors per dataset
#         :type tensor_bytes: list[int], optional
#         :param client_threads: list of the number of client threads used for data
#                                aggregation
#         :type client_threads: list[int], optional
#         :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
#                                  to request that the consumer application utilizes
#                                  all physical cores for each client thread.
#         :type cpu_hyperthreads: int, optional
#         """
#         logger.info("Starting dataset aggregation scaling with python tests")
#         logger.info(f"Running with database backend: {_get_db_backend()}")
#         logger.info(f"Running with launcher: {launcher}")

#         exp = create_folder(self, exp_name, launcher)

#         for db_node_count in db_nodes:

#             # start the database only once per value in db_nodes so all permutations
#             # are executed with the same database size without bringin down the database
#             db = start_database(exp,
#                                 db_port,
#                                 db_node_count,
#                                 db_cpus,
#                                 None, # not setting threads per queue in throughput tests
#                                 net_ifname,
#                                 run_db_as_batch,
#                                 batch_args,
#                                 db_hosts)

#             for c_nodes, cpn, bytes_, t_per_dataset, c_threads in product(
#                 client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset, client_threads
#             ):
#                 logger.info(f"Running with threads: {c_threads}")
#                 # setup a an instance of the C++ producer and start it
#                 aggregation_producer_sessions = \
#                     create_aggregation_producer_session_python(exp, c_nodes, cpn,
#                                                                db_node_count,
#                                                                db_cpus, iterations,
#                                                                bytes_, t_per_dataset)

#                 # setup a an instance of the python consumer and start it
#                 aggregation_consumer_sessions = \
#                     create_aggregation_consumer_session_python(exp, c_nodes, cpn,
#                                                                db_node_count, db_cpus,
#                                                                iterations, bytes_,
#                                                                t_per_dataset, c_threads,
#                                                                cpu_hyperthreads)

#                 exp.start(aggregation_producer_sessions,
#                           aggregation_consumer_sessions,
#                           summary=True)

#                 # confirm scaling test run successfully
#                 stat = exp.get_status(aggregation_producer_sessions)
#                 if stat[0] != status.STATUS_COMPLETED:
#                     logger.error(f"ERROR: One of the scaling tests failed \
#                                   {aggregation_producer_sessions.name}")
#                 stat = exp.get_status(aggregation_consumer_sessions)
#                 if stat[0] != status.STATUS_COMPLETED:
#                     logger.error(f"ERROR: One of the scaling tests failed \
#                                   {aggregation_consumer_sessions.name}")

#             # stop database after this set of permutations have finished
#             exp.stop(db)

#     def aggregation_scaling_python_fs(self,
#                             exp_name="aggregation-scaling-py-fs",
#                             launcher="auto",
#                             clients_per_node=[32],
#                             client_nodes=[128, 256, 512],
#                             iterations=20,
#                             tensor_bytes=[1024, 8192, 16384, 32769, 65538,
#                                           131076, 262152, 524304, 1024000,
#                                           2048000],
#                             tensors_per_dataset=[1,2,4],
#                             client_threads=[1,2,4,8,16,32],
#                             cpu_hyperthreads=2):
#         """Run the data aggregation scaling tests with python consumer using the
#         file system in place of the orchastrator. Permutations of the test include 
#         client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset,
#         and client_threads.

#         :param exp_name: name of output dir
#         :type exp_name: str, optional
#         :param launcher: workload manager i.e. "slurm", "pbs"
#         :type launcher: str, optional
#         :param clients_per_node: client tasks per compute node for the aggregation
#                                  producer app
#         :type clients_per_node: list, optional
#         :param client_nodes: number of compute nodes to use for the aggregation
#                              producer app
#         :type client_nodes: list, optional
#         :param iterations: number of append/retrieve loops run by the applications
#         :type iterations: int
#         :param tensor_bytes: list of tensor sizes in bytes
#         :type tensor_bytes: list[int], optional
#         :param tensors_per_dataset: list of number of tensors per dataset
#         :type tensor_bytes: list[int], optional
#         :param client_threads: list of the number of client threads used for data
#                                aggregation
#         :type client_threads: list[int], optional
#         :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
#                                  to request that the consumer application utilizes
#                                  all physical cores for each client thread.
#         :type cpu_hyperthreads: int, optional
#         """
#         logger.info("Starting dataset aggregation scaling with python on file system tests")
#         logger.info(f"Running with database backend: None (data to file system)")
#         logger.info(f"Running with launcher: {launcher}")

#         exp = create_folder(self, exp_name, launcher)

#         for c_nodes, cpn, bytes_, t_per_dataset, c_threads in product(
#             client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset, client_threads
#         ):
#             logger.info(f"Running with processes: {c_threads}")
#             # setup a an instance of the C++ producer and start it
#             aggregation_producer_sessions = \
#                 create_aggregation_producer_session_python_fs(exp, c_nodes, cpn,
#                                                               iterations, bytes_,
#                                                               t_per_dataset)

#             # setup a an instance of the python consumer and start it
#             aggregation_consumer_sessions = \
#                 create_aggregation_consumer_session_python_fs(exp, c_nodes, cpn,
#                                                               iterations, bytes_,
#                                                               t_per_dataset, c_threads,
#                                                               cpu_hyperthreads)

#             # Bad SmartSim access to set up env vars
#             # so that producer writes files to same location
#             # that the consumer reads files
#             shared_scratch_dir = os.path.join(exp.exp_path, "scratch")
#             if os.path.exists(shared_scratch_dir):
#                 shutil.rmtree(shared_scratch_dir)
#             os.mkdir(shared_scratch_dir)
#             aggregation_producer_sessions.run_settings.env_vars[
#                 "WRITE_TO_DIR"
#             ] = aggregation_consumer_sessions.run_settings.env_vars[
#                 "READ_FROM_DIR"
#             ] = shared_scratch_dir

#             exp.start(aggregation_producer_sessions,
#                       aggregation_consumer_sessions,
#                       summary=True)

#             # confirm scaling test run successfully
#             stat = exp.get_status(aggregation_producer_sessions)
#             if stat[0] != status.STATUS_COMPLETED:
#                 logger.error(f"ERROR: One of the scaling tests failed \
#                                 {aggregation_producer_sessions.name}")
#             stat = exp.get_status(aggregation_consumer_sessions)
#             if stat[0] != status.STATUS_COMPLETED:
#                 logger.error(f"ERROR: One of the scaling tests failed \
#                                 {aggregation_consumer_sessions.name}")