import fire
from utils import *
from utils import _get_db_backend
from utils import _check_model
from utils import _get_uuid

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")

class Throughput:
    
    def throughput_standard(self,
                           exp_name="throughput-standard-scaling",
                           launcher="auto",
                           run_db_as_batch=True,
                           batch_args={},
                           db_hosts=[],
                           db_nodes=[12],
                           db_cpus=36,
                           db_port=6780,
                           net_ifname="ipogif0",
                           clients_per_node=[32],
                           client_nodes=[128, 256, 512],
                           iterations=100,
                           tensor_bytes=[1024, 8192, 16384, 32768, 65536, 131072,
                                         262144, 524288, 1024000, 2048000, 4096000]):

        """Run the throughput scaling tests

        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param run_db_as_batch: run database as separate batch submission each iteration
        :type run_db_as_batch: bool, optional
        :param batch_args: additional batch args for the database
        :type batch_args: dict, optional
        :param db_hosts: optionally supply hosts to launch the database on
        :type db_hosts: list, optional
        :param db_nodes: number of compute hosts to use for the database
        :type db_nodes: list, optional
        :param db_cpus: number of cpus per compute host for the database
        :type db_cpus: list, optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list, optional
        :param client_nodes: number of compute nodes to use for the synthetic scaling app
        :type client_nodes: list, optional
        :param iterations: number of put/get loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        """
        logger.info("Starting throughput scaling tests")
        logger.info(f"Running with database backend: {_get_db_backend()}")
        logger.info(f"Running with launcher: {launcher}")

        exp = create_folder(exp_name, launcher)

        for db_node_count in db_nodes:

            # start the database only once per value in db_nodes so all permutations
            # are executed with the same database size without bringin down the database
            db = start_database(exp,
                                db_port,
                                db_node_count,
                                db_cpus,
                                None, # not setting threads per queue in throughput tests
                                net_ifname,
                                run_db_as_batch,
                                batch_args,
                                db_hosts)


            perms = list(product(client_nodes, clients_per_node, tensor_bytes))
            for perm in perms:
                c_nodes, cpn, _bytes = perm

                # setup a an instance of the C++ driver and start it
                throughput_session = self.create_throughput_session(exp,
                                                               c_nodes,
                                                               cpn,
                                                               db_node_count,
                                                               db_cpus,
                                                               iterations,
                                                               _bytes)
                exp.start(throughput_session, summary=True)

                # confirm scaling test run successfully
                stat = exp.get_status(throughput_session)
                if stat[0] != status.STATUS_COMPLETED:
                    logger.error(f"ERROR: One of the scaling tests failed {throughput_session.name}")

            # stop database after this set of permutations have finished
            exp.stop(db)
    
    @classmethod
    def create_throughput_session(cls,
                              exp,
                              nodes,
                              tasks,
                              db_nodes,
                              db_cpus,
                              iterations,
                              _bytes):
        """Create a Model to run a throughput scaling session

        :param exp: Experiment object for this test
        :type exp: Experiment
        :param nodes: number of nodes for the synthetic throughput application
        :type nodes: int
        :param tasks: number of tasks per node for the throughput application
        :type tasks: int
        :param db_nodes: number of database nodes
        :type db_nodes: int
        :param db_cpus: number of cpus used on each database node
        :type db_cpus: int
        :param iterations: number of put/get loops by the application
        :type iterations: int
        :param _bytes: size in bytes of tensors to use for throughput scaling
        :type _bytes: int
        :return: Model reference to the throughput session to launch
        :rtype: Model
        """
        settings = exp.create_run_settings("./cpp-throughput/build/throughput", str(_bytes))
        settings.set_tasks(nodes * tasks)
        settings.set_tasks_per_node(tasks)
        settings.update_env({
            "SS_ITERATIONS": str(iterations)
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(nodes)

        name = "-".join((
            "throughput-sess",
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
            "ITER"+str(iterations),
            "TB"+str(_bytes),
            _get_uuid()
            ))

        model = exp.create_model(name, settings)
        exp.generate(model, overwrite=True)
        write_run_config(model.path,
                    client_total=tasks*nodes,
                    client_per_node=tasks,
                    client_nodes=nodes,
                    database_nodes=db_nodes,
                    database_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=_bytes)
        return model