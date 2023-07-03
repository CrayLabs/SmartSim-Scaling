if __name__ == "__main__":
    """ Takes the pwd, then navigates to the root to append packages.
    Python is then able to find our *.py files in that directory.
    """
    sys.path.append("..")

from utils import *
from driverprocessresults.main import *
import sys

if __name__ == "__main__":
    """The file may run directly without invoking driver.py and the scaling
    study can still be run.
    """
    import fire
    fire.Fire(Throughput())

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")

class Throughput:
    def throughput_standard(self,
                           exp_name="throughput-standard-scaling",
                           launcher="auto",
                           run_db_as_batch=True,
                           node_feature={},
                           db_node_feature={},
                           db_hosts=[],
                           db_nodes=[4,8,16],
                           db_cpus=[2],
                           db_port=6780,
                           net_ifname="ipogif0",
                           clients_per_node=[32],
                           client_nodes=[10],
                           iterations=3,
                           tensor_bytes=[1024,8192,16384,32769,65538,
                                          131076,262152,524304,1024000],
                           languages=["cpp"],
                           wall_time="05:00:00",
                           plot="database_nodes",
                           smartsim_logging=False):

        """Run the throughput scaling tests with standard Orchestrator deployment

        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param run_db_as_batch: run database as separate batch submission each iteration
        :type run_db_as_batch: bool, optional
        :param node_feature: dict of runsettings for the app
        :type node_feature: dict[str,str], optional
        :param db_node_feature: dict of runsettings for the db
        :type db_node_feature: dict[str,str], optional
        :param db_hosts: optionally supply hosts to launch the database on
        :type db_hosts: list[str], optional
        :param db_nodes: number of compute hosts to use for the database
        :type db_nodes: list[int], optional
        :param db_cpus: number of cpus per compute host for the database
        :type db_cpus: int, optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list[int], optional
        :param client_nodes: number of compute nodes to use for the synthetic scaling app
        :type client_nodes: list[int], optional
        :param iterations: number of put/get loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        :param languages: which language to use for the tester "cpp" or "fortran"
        :type languages: str
        :param wall_time: allotted time for database launcher to run
        :type wall_time: str
        :param plot: flag to plot against in process results
        :type plot: str
        """
        logger.info("Starting throughput standard scaling tests")
        check_node_allocation(client_nodes, db_nodes)
        logger.info("Experiment allocation passed check")

        exp, result_path = create_experiment_and_dir(exp_name, launcher)
        write_run_config(result_path,
                    colocated=0,
                    client_per_node=clients_per_node,
                    client_nodes=client_nodes,
                    database_nodes=db_nodes,
                    database_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=tensor_bytes,
                    language=languages,
                    wall_time=wall_time)
        print_yml_file(Path(result_path) / "run.cfg", logger)
        first_perms = list(product(db_nodes, db_cpus))
        for i, first_perm in enumerate(first_perms, start=1):
            dbn, dbc = first_perm
            # start the database only once per value in db_nodes so all permutations
            # are executed with the same database size without bringin down the database
            db = start_database(exp,
                                db_node_feature,
                                db_port,
                                dbn,
                                dbc,
                                None, # not setting threads per queue in throughput tests
                                net_ifname,
                                run_db_as_batch,
                                db_hosts,
                                wall_time)
            logger.debug("database created and returned")
            
            second_perms = list(product(client_nodes, clients_per_node, tensor_bytes, db_cpus, languages))
            for j, second_perm in enumerate(second_perms, start=1):
                c_nodes, cpn, _bytes, db_cpu, language = second_perm
                logger.info(f"Running permutation {i} of {len(second_perms)} for database node index {j} of {len(first_perms)}")
                # setup a an instance of the C++ driver and start it
                throughput_session = self._create_throughput_session(exp,
                                                               node_feature,
                                                               c_nodes,
                                                               cpn,
                                                               dbn,
                                                               db_cpu,
                                                               iterations,
                                                               _bytes,
                                                               language)
                logger.debug("Throughput session created")
                exp.start(throughput_session, summary=True)
                logger.debug("experiment started")
                # confirm scaling test run successfully
                stat = exp.get_status(throughput_session)
                if stat[0] != status.STATUS_COMPLETED: # might need to add error check to inference tests
                    logger.error(f"ERROR: One of the scaling tests failed {throughput_session.name}")

            # stop database after this set of permutations have finished
            exp.stop(db)
            #Added to clean up db folder bc of issue with exp.stop()
            time.sleep(5)
            check_database_folder(result_path, logger)
        self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)
    
    @classmethod
    def _create_throughput_session(cls,
                              exp,
                              node_feature,
                              nodes,
                              tasks,
                              db_nodes,
                              db_cpus,
                              iterations,
                              _bytes,
                              language):
        """Create a Model to run a standard throughput scaling session

        :param exp: Experiment object for this test
        :type exp: Experiment
        :param node_feature: dict of runsettings for the app
        :type node_feature: dict[str,str], optional
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
        settings = exp.create_run_settings(f"./{language}-throughput/build/throughput", str(_bytes), run_args=node_feature)
        cluster = 1 if db_nodes > 1 else 0
        settings.set_tasks(nodes * tasks)
        settings.set_tasks_per_node(tasks)
        settings.update_env({
            "SS_ITERATIONS": str(iterations),
            "SS_CLUSTER": cluster
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(nodes)

        name = "-".join((
            "throughput-sess",
            language,
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
            "DBCPU"+str(db_cpus),
            "ITER"+str(iterations),
            "TB"+str(_bytes),
            get_uuid()
            ))

        model = exp.create_model(name, settings)
        exp.generate(model, overwrite=True)
        write_run_config(model.path,
                    colocated=0,
                    client_total=tasks*nodes,
                    client_per_node=tasks,
                    client_nodes=nodes,
                    database_nodes=db_nodes,
                    database_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=_bytes,
                    language=language)
        return model
    
    def throughput_colocated(self,
                           exp_name="throughput-colocated-scaling",
                           launcher="auto",
                           node_feature={},
                           nodes=[10],
                           db_cpus=[5],
                           db_port=6780,
                           net_ifname="lo",
                           clients_per_node=[48],
                           pin_app_cpus=[False],
                           iterations=3,
                           tensor_bytes=[1024,8192,16384,32769,65538,
                                          131076,262152,524304,1024000],
                           languages=["cpp"],
                           plot="database_cpus"):


        """Run the throughput scaling tests with colocated Orchestrator deployment

        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param node_feature: dict of runsettings for both app and db
        :type node_feature: dict[str,str], optional
        :param nodes: compute nodes to use for synthetic scaling app with
                      a co-located orchestrator database
        :type nodes: list, optional
        :param db_cpus: number of cpus per compute host for the database
        :type db_cpus: list, optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list[int], optional
        :param pin_app_cpus: pin the threads of the application to 0-(n-db_cpus)
        :type pin_app_cpus: list[bool], optional
        :param iterations: number of put/get loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        :param languages: list of languages to use for the tester "cpp" and/or "fortran"
        :type languages: str
        :param plot: flag to plot against in process results
        :type plot: str
        """
        logger.info("Starting throughput colocated scaling tests")
        check_node_allocation(nodes, [0])
        logger.info("Experiment allocation passed check")
        
        exp, result_path = create_experiment_and_dir(exp_name, launcher)
        write_run_config(result_path,
                    colocated=1,
                    pin_app_cpus=str(pin_app_cpus),
                    client_per_node=clients_per_node,
                    client_nodes=nodes,
                    database_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=tensor_bytes,
                    language=languages)
        print_yml_file(Path(result_path) / "run.cfg", logger)

        perms = list(product(nodes, clients_per_node, db_cpus, tensor_bytes, pin_app_cpus, languages))
        for i, perm in enumerate(perms, start=1):
            c_nodes, cpn, dbc, _bytes, pin_app, language = perm
            logger.info(f"Running permutation {i} of {len(perms)}")

            # setup a an instance of the C++ driver and start it
            throughput_session = self._create_colocated_throughput_session(exp,
                                                            node_feature,
                                                            c_nodes,
                                                            cpn,
                                                            dbc,
                                                            db_port,
                                                            iterations,
                                                            _bytes,
                                                            pin_app,
                                                            net_ifname,
                                                            language)
            logger.debug("Throughput session created")
            exp.start(throughput_session, block = True, summary=True)

            # confirm scaling test run successfully
            stat = exp.get_status(throughput_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"ERROR: One of the scaling tests failed {throughput_session.name}")
        self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)
    
    @classmethod
    def _create_colocated_throughput_session(cls,
                              exp,
                              node_feature,
                              nodes,
                              tasks,
                              db_cpus,
                              db_port,
                              iterations,
                              _bytes,
                              pin_app_cpus,
                              net_ifname,
                              language):
        """Run the throughput scaling tests with colocated Orchestrator deployment

        :param exp: Experiment object for this test
        :type exp: Experiment
        :param node_feature: dict of runsettings for app and db
        :type node_feature: dict[str,str]
        :param nodes: number of nodes for the synthetic throughput application
        :type nodes: int
        :param tasks: number of tasks per node for the throughput application
        :type tasks: int
        :param db_cpus: number of cpus used on each database node
        :type db_cpus: int
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param iterations: number of put/get loops by the application
        :type iterations: int
        :param _bytes: size in bytes of tensors to use for throughput scaling
        :type _bytes: int
        :param pin_app_cpus: pin the threads of the application to 0-(n-db_cpus)
        :type pin_app_cpus: bool, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :return: Model reference to the throughput session to launch
        :rtype: Model
        """
        settings = exp.create_run_settings(f"./{language}-throughput/build/throughput", str(_bytes), run_args=node_feature)
        settings.set_tasks(nodes * tasks)
        settings.set_tasks_per_node(tasks)
        settings.update_env({
            "SS_ITERATIONS": str(iterations),
            "SS_CLUSTER": "0"
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(nodes)

        name = "-".join((
            "throughput-sess-colo",
            language,
            "N"+str(nodes),
            "T"+str(tasks),
            "DBCPU"+str(db_cpus),
            "PIN"+str(pin_app_cpus),
            "ITER"+str(iterations),
            "TB"+str(_bytes),
            get_uuid()
            ))
        
        model = exp.create_model(name, settings)
        
        model.colocate_db(port=db_port,
                        db_cpus=db_cpus,
                        ifname=net_ifname,
                        limit_app_cpus=pin_app_cpus,
                        debug=True,
                        loglevel="notice")
        
        exp.generate(model, overwrite=True)
        
        write_run_config(model.path,
                    colocated=1,
                    pin_app_cpus=int(pin_app_cpus),
                    client_total=tasks*nodes,
                    client_per_node=tasks,
                    client_nodes=nodes,
                    database_nodes=nodes,
                    database_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=_bytes,
                    language=language)
        return model