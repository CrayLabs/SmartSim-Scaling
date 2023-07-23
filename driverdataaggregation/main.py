if __name__ == "__main__":
    """ Takes the pwd, then navigates to the root to append packages.
    Python is then able to find our *.py files in that directory.
    """
    sys.path.append("..")

from utils import *
from driverprocessresults.main import *

if __name__ == "__main__":
    """The file may run directly without invoking driver.py and the scaling
    study can still be run.
    """
    import fire
    fire.Fire(DataAggregation())

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


class DataAggregation:
    def aggregation_scaling(self,
                            exp_name="aggregation-standard-scaling",
                            launcher="auto",
                            run_db_as_batch=True,
                            db_node_feature = {},
                            prod_node_feature = {},
                            cons_node_feature = {},
                            db_hosts=[],
                            db_nodes=[4,8,16],
                            db_cpus=36,
                            db_port=6780,
                            net_ifname="ipogif0",
                            clients_per_node=[32],
                            client_nodes=[4,8,16,32,64,128],
                            iterations=100,
                            tensor_bytes=[1024],
                            tensors_per_dataset=[4],
                            client_threads=[32],
                            cpu_hyperthreads=2,
                            languages=["cpp"],
                            wall_time="10:00:00",
                            plot="database_nodes"):

        """Run the data aggregation scaling tests.  Permutations of the test
        include client_nodes, clients_per_node, tensor_bytes,
        tensors_per_dataset, and client_threads.
        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param run_db_as_batch: run database as separate batch submission
                                each iteration
        :type run_db_as_batch: bool, optional
        :param db_node_feature: dict of runsettings for the db
        :type db_node_feature: dict[str,str], optional
        :param prod_node_feature: dict of runsettings for the producer
        :type prod_node_feature: dict[str,str], optional
        :param cons_node_feature: dict of runsettings for the consumer
        :type cons_node_feature: dict[str,str], optional
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
        :param clients_per_node: client tasks per compute node for the aggregation
                                 producer app
        :type clients_per_node: list[int], optional
        :param client_nodes: number of compute nodes to use for the aggregation
                             producer app
        :type client_nodes: list[int], optional
        :param iterations: number of append/retrieve loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        :param tensors_per_dataset: list of number of tensors per dataset
        :type tensor_per_dataset: list[int], optional
        :param client_threads: list of the number of client threads used for data
                               aggregation
        :type client_threads: list[int], optional
        :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
                                 to request that the consumer application utilizes
                                 all physical cores for each client thread.
        :type cpu_hyperthreads: int, optional
        :param languages: list of languages to use for the tester "cpp" and/or "fortran"
        :type languages: str
        :param wall_time: allotted time for database launcher to run
        :type wall_time: str
        :param plot: flag to plot against in process results
        :type plot: str
        """
        logger.info("Starting dataset aggregation scaling tests")
        check_node_allocation(client_nodes, db_nodes)
        logger.info("Experiment allocation passed check")

        exp, result_path = create_experiment_and_dir(exp_name, launcher)
        write_run_config(result_path,
                    colocated=0,
                    client_per_node=clients_per_node,
                    client_nodes=client_nodes,
                    db_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=tensor_bytes,
                    tensors_per_dataset=tensors_per_dataset,
                    client_threads=client_threads,
                    cpu_hyperthreads=cpu_hyperthreads,
                    language=languages,
                    wall_time=wall_time)
        print_yml_file(Path(result_path) / "run.cfg", logger)

        for i, db_node_count in enumerate(db_nodes, start=1):
            logger.info(f"Running permutation {i} of {len(db_nodes)}")

            # start the database only once per value in db_nodes so all permutations
            # are executed with the same database size without bringin down the database
            db = start_database(exp,
                                db_node_feature,
                                db_port,
                                db_node_count,
                                db_cpus,
                                None, # not setting threads per queue in throughput tests
                                net_ifname,
                                run_db_as_batch,
                                db_hosts,
                                wall_time)

            for c_nodes, cpn, _bytes, t_per_dataset, c_threads, language in product(
                client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset, client_threads, languages
            ):
            #create new var and set it to a copy of node_feature to pass into either
                logger.info(f"Running with threads: {c_threads}")
                # setup a an instance of the C++ driver and start it
                aggregation_producer_sessions = \
                    self._create_aggregation_producer_session_cpp(exp, prod_node_feature, c_nodes, cpn,
                                                        db_node_count,
                                                        db_cpus, iterations,
                                                        _bytes, t_per_dataset, language)
                logger.debug("Producer session created")

                # setup a an instance of the C++ driver and start it
                aggregation_consumer_sessions = \
                    self._create_aggregation_consumer_session_cpp(exp, cons_node_feature, c_nodes, cpn,
                                                        db_node_count, db_cpus,
                                                        iterations, _bytes, t_per_dataset,
                                                        c_threads, cpu_hyperthreads, language)
                logger.debug("Consumer session created")
                exp.start(aggregation_producer_sessions,
                          aggregation_consumer_sessions,
                           summary=True)

                # confirm scaling test run successfully
                stat = exp.get_status(aggregation_producer_sessions)
                if stat[0] != status.STATUS_COMPLETED:
                    logger.error(f"ERROR: One of the scaling tests failed \
                                  {aggregation_producer_sessions.name}")
                stat = exp.get_status(aggregation_consumer_sessions)
                if stat[0] != status.STATUS_COMPLETED:
                    logger.error(f"ERROR: One of the scaling tests failed \
                                  {aggregation_consumer_sessions.name}")

            # stop database after this set of permutations have finished
            exp.stop(db)
            #Added to clean up db folder bc of issue with exp.stop()
            #time.sleep(5)
            #check_database_folder(result_path, logger)
        #self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)
    
    @classmethod
    def _create_aggregation_producer_session_cpp(cls, exp, prod_node_feature, nodes, tasks, db_nodes, db_cpus,
                                        iterations, _bytes, t_per_dataset, language):
        return cls._create_aggregation_producer_session(
            name="aggregate-sess-prod",
            exe=f"./{language}-data-aggregation/build/aggregation_producer",
            exe_args=[str(_bytes), str(t_per_dataset)], run_args=prod_node_feature,
            exp=exp, nodes=nodes, tasks=tasks, db_nodes=db_nodes, db_cpus=db_cpus,
            iterations=iterations, bytes_=_bytes, t_per_dataset=t_per_dataset, language=language)
    
    @staticmethod
    def _create_aggregation_producer_session(name,
                                         exe,
                                         exe_args,
                                         run_args,
                                         exp,
                                         nodes,
                                         tasks,
                                         db_nodes,
                                         db_cpus,
                                         iterations,
                                         bytes_,
                                         t_per_dataset,
                                         language):
        """Create a Model to run a producer for the aggregation scaling session
        :param name: The name of the model being created
        :type name: str
        :param exe: The path to the executable used by the model
        :type exe: str
        :param exe_args: The arguments passed to the executable
        :type exe_args: list[str]
        :param run_args: The arguments passed to the settings
        :type run_args: dict[str,str]
        :param exp: Experiment object for this test
        :type exp: Experiment
        :param nodes: number of nodes for the synthetic aggregation application
        :type nodes: int
        :param tasks: number of tasks per node for the aggregation application
        :type tasks: int
        :param db_nodes: number of database nodes
        :type db_nodes: int
        :param db_cpus: number of cpus used on each database node
        :type db_cpus: int
        :param iterations: number of append/retrieve loops by the application
        :type iterations: int
        :param bytes_: size in bytes of tensors to use for aggregation scaling
        :type bytes_: int
        :param t_per_dataset: the number of tensors per dataset
        :type t_per_dataset: int
        :return: Model reference to the aggregation session to launch
        :rtype: Model
        """
        settings = exp.create_run_settings(exe, exe_args, run_args=run_args)
        settings.set_tasks(nodes * tasks)
        settings.set_tasks_per_node(tasks)
        settings.update_env({
            "SS_ITERATIONS": str(iterations)
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(nodes)

        name = "-".join((
            name,
            language,
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
            "ITER"+str(iterations),
            "TB"+str(bytes_),
            "TPD"+str(t_per_dataset),
            get_uuid()
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
                        tensor_bytes=bytes_,
                        t_per_dataset=t_per_dataset,
                        language=language)
        return model

    @classmethod
    def _create_aggregation_consumer_session_cpp(cls, exp, cons_node_feature, nodes, tasks, db_nodes, db_cpus,
                                            iterations, bytes_, t_per_dataset,
                                            c_threads, cpu_hyperthreads, language):
        return cls._create_aggregation_consumer_session(
            name="aggregate-sess-cons",
            exe=f"./{language}-data-aggregation/build/aggregation_consumer",
            exe_args=[str(nodes*tasks)], run_args=cons_node_feature, exp=exp, nodes=nodes, tasks=tasks,
            db_nodes=db_nodes, db_cpus=db_cpus, iterations=iterations, bytes_=bytes_,
            t_per_dataset=t_per_dataset, c_threads=c_threads, cpu_hyperthreads=cpu_hyperthreads, language=language)

    @staticmethod
    def _create_aggregation_consumer_session(name, 
                                            exe,
                                            exe_args,
                                            run_args,
                                            exp,
                                            nodes,
                                            tasks,
                                            db_nodes,
                                            db_cpus,
                                            iterations,
                                            bytes_,
                                            t_per_dataset,
                                            c_threads,
                                            cpu_hyperthreads,
                                            language):
        """Create a Model to run a consumer for the aggregation scaling session
        :param name: The name of the model being created
        :type name: str
        :param exe: The path to the executable used by the model
        :type exe: str
        :param exe_args: The arguments passed to the executable
        :type exe_args: list[str]
        :param run_args: The arguments passed to the settings
        :type run_args: dict[str,str]
        :param exp: Experiment object for this test
        :type exp: Experiment
        :param nodes: number of nodes for the synthetic aggregation application
        :type nodes: int
        :param tasks: number of tasks per node for the aggregation application
        :type tasks: int
        :param db_nodes: number of database nodes
        :type db_nodes: int
        :param db_cpus: number of cpus used on each database node
        :type db_cpus: int
        :param iterations: number of append/retrieve loops by the application
        :type iterations: int
        :param bytes_: size in bytes of tensors to use for aggregation scaling
        :type bytes_: int
        :param t_per_dataset: the number of tensors per dataset
        :type t_per_dataset: int
        :param c_threads:  the number of client threads to use inside of SR client
        :type c_threads: int
        :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
                                to request that the consumer application utilizes
                                all physical cores for each client thread.
        :type cpu_hyperthreads: int, optional
        :return: Model reference to the aggregation session to launch
        :rtype: Model
        """
        settings = exp.create_run_settings(exe, exe_args, run_args=run_args)
        #settings.set_tasks(1)
        settings.set_tasks_per_node(1)
        settings.set_cpus_per_task(c_threads * cpu_hyperthreads)
        #hmmm noting
        settings.run_args['cpu-bind'] = 'v'
        settings.update_env({
            "SS_ITERATIONS": str(iterations),
            "SR_THREAD_COUNT": str(c_threads)
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(1)



        name = "-".join((
            name,
            language,
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
            "ITER"+str(iterations),
            "TB"+str(bytes_),
            "TPD"+str(t_per_dataset),
            "CT"+str(c_threads),
            get_uuid()
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
                    tensor_bytes=bytes_,
                    t_per_dataset=t_per_dataset,
                    client_threads=c_threads,
                    language=language)
        return model

    def aggregation_scaling_python(self,
                                exp_name="aggregation-standard-scaling-py",
                                launcher="auto",
                                run_db_as_batch=True,
                                db_node_feature = {},
                                prod_node_feature = {},
                                cons_node_feature = {},
                                db_hosts=[],
                                db_nodes=[4,8,16],
                                db_cpus=32,
                                db_port=6780,
                                net_ifname="ipogif0",
                                clients_per_node=[32],
                                client_nodes=[4,8,16,32,64,128],
                                iterations=100,
                                tensor_bytes=[1024],
                                tensors_per_dataset=[4],
                                client_threads=[32],
                                cpu_hyperthreads=2,
                                languages=["cpp"],
                                wall_time="05:00:00",
                                plot="database_nodes"):
            """Run the data aggregation scaling tests with python consumer.
            Permutations of the test include client_nodes, clients_per_node, 
            tensor_bytes, tensors_per_dataset, and client_threads.
            :param exp_name: name of output dir
            :type exp_name: str, optional
            :param launcher: workload manager i.e. "slurm", "pbs"
            :type launcher: str, optional
            :param run_db_as_batch: run database as separate batch submission
                                    each iteration
            :type run_db_as_batch: bool, optional
            :param db_node_feature: dict of runsettings for db
            :type db_node_feature: dict[int,int], optional
            :param prod_node_feature: dict of runsettings for the producer
            :type prod_node_feature: dict[str,str], optional
            :param cons_node_feature: dict of runsettings for the consumer
            :type cons_node_feature: dict[str,str], optional
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
            :param clients_per_node: client tasks per compute node for the aggregation
                                    producer app
            :type clients_per_node: list[int], optional
            :param client_nodes: number of compute nodes to use for the aggregation
                                producer app
            :type client_nodes: list[int], optional
            :param iterations: number of append/retrieve loops run by the applications
            :type iterations: int
            :param tensor_bytes: list of tensor sizes in bytes
            :type tensor_bytes: list[int], optional
            :param tensors_per_dataset: list of number of tensors per dataset
            :type tensor_bytes: list[int], optional
            :param client_threads: list of the number of client threads used for data
                                aggregation
            :type client_threads: list[int], optional
            :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
                                    to request that the consumer application utilizes
                                    all physical cores for each client thread.
            :type cpu_hyperthreads: int, optional
            :param languages: list of languages to use for the tester "cpp" and/or "fortran"
            :type languages: str
            :param wall_time: allotted time for database launcher to run
            :type wall_time: str
            :param plot: flag to plot against in process results
            :type plot: str
            """
            logger.info("Starting dataset aggregation scaling with python tests")
            check_node_allocation(client_nodes, db_nodes)
            logger.info("Experiment allocation passed check")
            exp, result_path = create_experiment_and_dir(exp_name, launcher)
            write_run_config(result_path,
                    colocated=0,
                    client_per_node=clients_per_node,
                    client_nodes=client_nodes,
                    db_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=tensor_bytes,
                    tensors_per_dataset=tensors_per_dataset,
                    client_threads=client_threads,
                    cpu_hyperthreads=cpu_hyperthreads,
                    language=languages,
                    wall_time=wall_time)
            print_yml_file(Path(result_path) / "run.cfg", logger)

            for i, db_node_count in enumerate(db_nodes, start=1):
                logger.info(f"Running permutation {i} of {len(db_nodes)}")

                # start the database only once per value in db_nodes so all permutations
                # are executed with the same database size without bringin down the database
                db = start_database(exp,
                                    db_node_feature,
                                    db_port,
                                    db_node_count,
                                    db_cpus,
                                    None, # not setting threads per queue in throughput tests
                                    net_ifname,
                                    run_db_as_batch,
                                    db_hosts,
                                    wall_time)

                for c_nodes, cpn, bytes_, t_per_dataset, c_threads, language in product(
                    client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset, client_threads, languages
                ):
                    logger.info(f"Running with threads: {c_threads}")
                    # setup a an instance of the C++ producer and start it
                    aggregation_producer_sessions = \
                        self._create_aggregation_producer_session_python(exp, prod_node_feature, c_nodes, cpn,
                                                                db_node_count,
                                                                db_cpus, iterations,
                                                                bytes_, t_per_dataset, language)
                    logger.debug("Producer session created")
                    # setup a an instance of the python consumer and start it
                    aggregation_consumer_sessions = \
                        self._create_aggregation_consumer_session_python(exp, cons_node_feature, c_nodes, cpn,
                                                                db_node_count, db_cpus,
                                                                iterations, bytes_,
                                                                t_per_dataset, c_threads,
                                                                cpu_hyperthreads, language)
                    logger.debug("Consumer session created")
                    exp.start(aggregation_producer_sessions,
                            aggregation_consumer_sessions,
                            summary=True)

                    # confirm scaling test run successfully
                    stat = exp.get_status(aggregation_producer_sessions)
                    if stat[0] != status.STATUS_COMPLETED:
                        logger.error(f"ERROR: One of the scaling tests failed \
                                    {aggregation_producer_sessions.name}")
                    stat = exp.get_status(aggregation_consumer_sessions)
                    if stat[0] != status.STATUS_COMPLETED:
                        logger.error(f"ERROR: One of the scaling tests failed \
                                    {aggregation_consumer_sessions.name}")

                # stop database after this set of permutations have finished
                exp.stop(db)
                #Added to clean up db folder bc of issue with exp.stop()
                #time.sleep(5)
                #check_database_folder(result_path, logger)
            #self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)

    @classmethod
    def _create_aggregation_producer_session_python(cls, exp, prod_node_feature, nodes, tasks, db_nodes, db_cpus,
                                               iterations, _bytes, t_per_dataset, language):
        return cls._create_aggregation_producer_session(
            name="aggregate-sess-prod-for-python-consumer",
            exe=f"./{language}-py-data-aggregation/db/build/aggregation_producer",
            exe_args=[str(_bytes), str(t_per_dataset)], run_args=prod_node_feature,
            exp=exp, nodes=nodes, tasks=tasks, db_nodes=db_nodes, db_cpus=db_cpus,
            iterations=iterations, bytes_=_bytes, t_per_dataset=t_per_dataset, language=language)
    
    @classmethod
    def _create_aggregation_consumer_session_python(cls, exp, cons_node_feature, nodes, tasks, db_nodes, db_cpus,
                                               iterations, bytes_, t_per_dataset,
                                               c_threads, cpu_hyperthreads, language):
        py_script_dir = f"./{language}-py-data-aggregation/db/"
        py_script = "aggregation_consumer.py"
        model = cls._create_aggregation_consumer_session(
            name="aggregate-sess-cons-python",
            exe=sys.executable,
            exe_args=[py_script, str(nodes*tasks)],
            run_args=cons_node_feature,
            exp=exp, nodes=nodes, tasks=tasks, db_nodes=db_nodes, db_cpus=db_cpus,
            iterations=iterations, bytes_=bytes_, t_per_dataset=t_per_dataset,
            c_threads=c_threads, cpu_hyperthreads=cpu_hyperthreads, language=language)
        model.attach_generator_files(to_copy=[py_script_dir + py_script])
        exp.generate(model, overwrite=True)
        write_run_config(model.path,
                        client_total=tasks*nodes,
                        client_per_node=tasks,
                        client_nodes=nodes,
                        database_nodes=db_nodes,
                        database_cpus=db_cpus,
                        iterations=iterations,
                        tensor_bytes=bytes_,
                        t_per_dataset=t_per_dataset,
                        client_threads=c_threads,
                        language=language)
        return model
    
    def aggregation_scaling_python_fs(self,
                            exp_name="aggregation-standard-scaling-py-fs",
                            launcher="auto",
                            prod_node_feature = {},
                            cons_node_feature = {},
                            clients_per_node=[32],
                            client_nodes=[4,8,16,32,64,128],
                            iterations=100,
                            tensor_bytes=[1024],
                            tensors_per_dataset=[4],
                            client_threads=[32],
                            cpu_hyperthreads=2,
                            languages=["cpp"],
                            plot="clients_per_node"):
        """Run the data aggregation scaling tests with python consumer using the
        file system in place of the orchastrator. Permutations of the test include 
        client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset,
        and client_threads.
        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param prod_node_feature: dict of runsettings for the producer
        :type prod_node_feature: dict[str,str], optional
        :param cons_node_feature: dict of runsettings for the consumer
        :type cons_node_feature: dict[str,str], optional
        :param clients_per_node: client tasks per compute node for the aggregation
                                 producer app
        :type clients_per_node: list[int], optional
        :param client_nodes: number of compute nodes to use for the aggregation
                             producer app
        :type client_nodes: list[int], optional
        :param iterations: number of append/retrieve loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        :param tensors_per_dataset: list of number of tensors per dataset
        :type tensors_per_dataset: list[int], optional
        :param client_threads: list of the number of client threads used for data
                               aggregation
        :type client_threads: list[int], optional
        :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
                                 to request that the consumer application utilizes
                                 all physical cores for each client thread.
        :type cpu_hyperthreads: int, optional
        :param languages: list of languages to use for the tester "cpp" and/or "fortran"
        :type languages: str
        :param plot: flag to plot against in process results
        :type plot: str
        """
        logger.info("Starting dataset aggregation scaling with python on file system tests")
        logger.info(f"Running with database backend: None (data to file system)")
        check_node_allocation(client_nodes, [0])
        logger.info("Experiment allocation passed check")
        exp, result_path = create_experiment_and_dir(exp_name, launcher)
        
        write_run_config(result_path,
                    colocated=0,
                    clients_per_node=clients_per_node,
                    client_nodes=client_nodes,
                    iterations=iterations,
                    tensor_bytes=tensor_bytes,
                    tensors_per_dataset=tensors_per_dataset,
                    client_threads=client_threads,
                    cpu_hyperthreads=cpu_hyperthreads,
                    language=languages)
        print_yml_file(Path(result_path) / "run.cfg", logger)

        for c_nodes, cpn, bytes_, t_per_dataset, c_threads, language in product(
            client_nodes, clients_per_node, tensor_bytes, tensors_per_dataset, client_threads, languages
        ):
            logger.info(f"Running with processes: {c_threads}")
            # setup a an instance of the C++ producer and start it
            aggregation_producer_sessions = \
                self._create_aggregation_producer_session_python_fs(exp, prod_node_feature, c_nodes, cpn,
                                                              iterations, bytes_,
                                                              t_per_dataset, language)
            logger.debug("Producer session created")
            # setup a an instance of the python consumer and start it
            aggregation_consumer_sessions = \
                self._create_aggregation_consumer_session_python_fs(exp, cons_node_feature, c_nodes, cpn,
                                                              iterations, bytes_,
                                                              t_per_dataset, c_threads,
                                                              cpu_hyperthreads, language)
            logger.debug("Consumer session created")
            # Bad SmartSim access to set up env vars
            # so that producer writes files to same location
            # that the consumer reads files
            shared_scratch_dir = os.path.join(exp.exp_path, "scratch")
            if os.path.exists(shared_scratch_dir):
                shutil.rmtree(shared_scratch_dir)
            os.mkdir(shared_scratch_dir)
            aggregation_producer_sessions.run_settings.env_vars[
                "WRITE_TO_DIR"
            ] = aggregation_consumer_sessions.run_settings.env_vars[
                "READ_FROM_DIR"
            ] = shared_scratch_dir

            exp.start(aggregation_producer_sessions,
                      aggregation_consumer_sessions,
                      summary=True)

            # confirm scaling test run successfully
            stat = exp.get_status(aggregation_producer_sessions)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"ERROR: One of the scaling tests failed \
                                {aggregation_producer_sessions.name}")
            stat = exp.get_status(aggregation_consumer_sessions)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"ERROR: One of the scaling tests failed \
                                {aggregation_consumer_sessions.name}")
        #self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)

    @classmethod
    def _create_aggregation_producer_session_python_fs(cls, exp, prod_node_feature, nodes, tasks, iterations,
                                                  bytes_, t_per_dataset, language):
        return cls._create_aggregation_producer_session(
            name="aggregate-sess-prod-for-python-consumer-file-system",
            exe=f"./{language}-py-data-aggregation/fs/build/aggregation_producer",
            exe_args=[str(bytes_), str(t_per_dataset)],
            run_args=prod_node_feature,
            exp=exp, nodes=nodes, tasks=tasks, db_nodes=0, db_cpus=0,
            iterations=iterations, bytes_=bytes_, t_per_dataset=t_per_dataset, language=language)
    
    @classmethod
    def _create_aggregation_consumer_session_python_fs(cls, exp, cons_node_feature, nodes, tasks, iterations,
                                                  bytes_, t_per_dataset, c_threads,
                                                  cpu_hyperthreads, language):
        py_script_dir = f"./{language}-py-data-aggregation/fs/"
        py_script = "aggregation_consumer.py"
        model = cls._create_aggregation_consumer_session(
            name="aggregate-sess-cons-python",
            exe=sys.executable,
            exe_args=[py_script, str(nodes*tasks)],
            run_args=cons_node_feature,
            exp=exp, nodes=nodes, tasks=tasks, db_nodes=0, db_cpus=0,
            iterations=iterations, bytes_=bytes_, t_per_dataset=t_per_dataset,
            c_threads=c_threads, cpu_hyperthreads=cpu_hyperthreads, language=language)
        model.attach_generator_files(to_copy=[py_script_dir + py_script])
        exp.generate(model, overwrite=True)
        write_run_config(model.path,
                        client_total=tasks*nodes,
                        client_per_node=tasks,
                        client_nodes=nodes,
                        database_nodes=0,
                        database_cpus=0,
                        iterations=iterations,
                        tensor_bytes=bytes_,
                        t_per_dataset=t_per_dataset,
                        client_threads=c_threads,
                        language=language)
        return model