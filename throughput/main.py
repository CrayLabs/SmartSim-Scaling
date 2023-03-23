# import fire

# class Throughput:
    
#     def throughput_scaling(self,
#                            exp_name="standard-throughput-scaling",
#                            launcher="auto",
#                            run_db_as_batch=True,
#                            batch_args={},
#                            db_hosts=[],
#                            db_nodes=[12],
#                            db_cpus=36,
#                            db_port=6780,
#                            net_ifname="ipogif0",
#                            clients_per_node=[32],
#                            client_nodes=[128, 256, 512],
#                            iterations=100,
#                            tensor_bytes=[1024, 8192, 16384, 32768, 65536, 131072,
#                                          262144, 524288, 1024000, 2048000, 4096000]):

#         """Run the throughput scaling tests

#         :param exp_name: name of output dir
#         :type exp_name: str, optional
#         :param launcher: workload manager i.e. "slurm", "pbs"
#         :type launcher: str, optional
#         :param run_db_as_batch: run database as separate batch submission each iteration
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
#         :param clients_per_node: client tasks per compute node for the synthetic scaling app
#         :type clients_per_node: list, optional
#         :param client_nodes: number of compute nodes to use for the synthetic scaling app
#         :type client_nodes: list, optional
#         :param iterations: number of put/get loops run by the applications
#         :type iterations: int
#         :param tensor_bytes: list of tensor sizes in bytes
#         :type tensor_bytes: list[int], optional
#         """
#         logger.info("Starting throughput scaling tests")
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


#             perms = list(product(client_nodes, clients_per_node, tensor_bytes))
#             for perm in perms:
#                 c_nodes, cpn, _bytes = perm

#                 # setup a an instance of the C++ driver and start it
#                 throughput_session = create_throughput_session(exp,
#                                                                c_nodes,
#                                                                cpn,
#                                                                db_node_count,
#                                                                db_cpus,
#                                                                iterations,
#                                                                _bytes)
#                 exp.start(throughput_session, summary=True)

#                 # confirm scaling test run successfully
#                 stat = exp.get_status(throughput_session)
#                 if stat[0] != status.STATUS_COMPLETED:
#                     logger.error(f"ERROR: One of the scaling tests failed {throughput_session.name}")

#             # stop database after this set of permutations have finished
#             exp.stop(db)