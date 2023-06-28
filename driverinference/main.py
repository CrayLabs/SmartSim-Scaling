if __name__ == "__main__":
    """ Takes the pwd, then navigates to the root to append packages.
    Python is then able to find our *.py files in that directory.
    """
    sys.path.append("..")

from utils import *
from driverprocessresults.main import *
import time

if __name__ == "__main__":
    """The file may run directly without invoking driver.py and the scaling
    study can still be run.
    """
    import fire
    fire.Fire(Inference())

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


class Inference:

    def inference_standard(self,
                           exp_name="inference-standard-scaling",
                           launcher="auto",
                           run_db_as_batch=True,
                           db_node_feature = {"constraint": "P100"},
                           node_feature = {},
                           db_hosts=[],
                           db_nodes=[4,8],
                           db_cpus=[8],
                           db_tpq=[1],
                           db_port=6780,
                           batch_size=[1000], #bad default min_batch_time_out
                           device="GPU",
                           num_devices=1,
                           net_ifname="ipogif0",
                           clients_per_node=[24, 48],
                           client_nodes=[1],
                           rebuild_model=False,
                           iterations=2,
                           languages=["cpp", "fortran"],
                           wall_time="05:00:00",
                           plot="database_nodes"):
        """Run ResNet50 inference tests with standard Orchestrator deployment
        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param run_db_as_batch: run database as separate batch submission each iteration
        :type run_db_as_batch: bool, optional
        :param db_node_feature: dict of runsettings for the database
        :type db_node_feature: dict[str,str], optional
        :param node_feature: dict of runsettings for the app
        :type node_feature: dict[str,str], optional
        :param db_hosts: optionally supply hosts to launch the database on
        :type db_hosts: list[str], optional
        :param db_nodes: number of compute hosts to use for the database
        :type db_nodes: list[int], optional
        :param db_cpus: number of cpus per compute host for the database
        :type db_cpus: list[int], optional
        :param db_tpq: number of device threads to use for the database
        :type db_tpq: list[int], optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param batch_size: batch size to set Resnet50 model with
        :type batch_size: list[int], optional
        :param device: device used to run the models in the database
        :type device: str, optional
        :param num_devices: number of devices per compute node to use to run ResNet
        :type num_devices: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list[int], optional
        :param client_nodes: number of compute nodes to use for the synthetic scaling app
        :type client_nodes: list[int], optional
        :param rebuild_model: force rebuild of PyTorch model even if it is available
        :type rebuild_model: bool
        :param iterations: number of put/get loops run by the applications
        :type iterations: int
        :param languages: list of languages to use for the tester "cpp" and/or "fortran"
        :type languages: str
        :param wall_time: allotted time for database launcher to run
        :type wall_time: str
        :param plot: flag to plot against in process results
        :type plot: str
        """
        logger.info("Starting inference standard scaling tests")
        check_node_allocation(client_nodes, db_nodes)
        logger.info("Experiment allocation passed check")

        exp, result_path = create_experiment_and_dir(exp_name, launcher)
        logger.debug("Experiment and Results folder created")
        write_run_config(result_path,
                        colocated=0,
                        clients_per_node=clients_per_node,
                        client_nodes=client_nodes,
                        database_hosts=db_hosts,
                        database_nodes=db_nodes,
                        database_cpus=db_cpus,
                        database_port=db_port,
                        batch_size=batch_size,
                        device=device,
                        num_devices=num_devices,
                        iterations=iterations,
                        language=languages,
                        db_node_feature=db_node_feature,
                        node_feature=node_feature,
                        wall_time=wall_time)
        print_yml_file(Path(result_path) / "run.cfg", logger)
        
        perms = list(product(client_nodes, clients_per_node, db_nodes, db_cpus, db_tpq, batch_size, languages))
        logger.info(f"Executing {len(perms)} permutations")
        for i, perm in enumerate(perms, start=1):
            c_nodes, cpn, dbn, dbc, dbtpq, batch, language = perm
            logger.info(f"Running permutation {i} of {len(perms)}")
            print(perm)

            db = start_database(exp,
                                db_node_feature,
                                db_port,
                                dbn,
                                dbc,
                                dbtpq,
                                net_ifname,
                                run_db_as_batch,
                                db_hosts,
                                wall_time)
            # setup a an instance of the synthetic C++ app and start it
            infer_session, resnet_model = self._create_inference_session(exp,
                                                     node_feature,
                                                     c_nodes,
                                                     cpn,
                                                     dbn,
                                                     dbc,
                                                     dbtpq,
                                                     batch,
                                                     device,
                                                     num_devices,
                                                     rebuild_model,
                                                     iterations,
                                                     language)
            logger.debug("Inference session created")
            address = db.get_address()[0]
            setup_resnet(resnet_model,
                        device,
                        num_devices,
                        batch,
                        address,
                        cluster=dbn>1)
            logger.debug("Resnet model set")

            exp.start(infer_session, block=True, summary=True)
            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"One of the scaling tests failed {infer_session.name}")
            exp.stop(db)
            #Added to clean up db folder bc of issue with exp.stop()
            time.sleep(10)
            rdb_folders = os.listdir(Path(result_path) / "database")
            for fold in rdb_folders:
                if '.rdb' in fold:
                    print(fold)
                    os.remove(Path(result_path) / "database" / fold)
            print(f"langauge:{language} exp.stop={i}")
        self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)
  
    def inference_colocated(self,
                            exp_name="inference-colocated-scaling",
                            node_feature={"constraint": "P100"},
                            launcher="auto",
                            nodes=[1],
                            clients_per_node=[12,24,36,60,96],
                            db_cpus=[12],
                            db_tpq=[1],
                            db_port=6780,
                            pin_app_cpus=[False],
                            batch_size=[96],
                            device="GPU",
                            num_devices=1,
                            net_type="uds",
                            net_ifname="lo",
                            rebuild_model=False,
                            iterations=100,
                            languages=["cpp","fortran"],
                            plot="database_cpus"
                            ):
        """Run ResNet50 inference tests with colocated Orchestrator deployment
        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param node_feature: dict of runsettings for the db and app
        :type node_feature: dict[str,str], optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param nodes: compute nodes to use for synthetic scaling app with
                      a co-located orchestrator database
        :type nodes: list[int], optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list, optional
        :param db_cpus: number of cpus per compute host for the database
        :type db_cpus: list[int], optional
        :param db_tpq: number of device threads to use for the database
        :type db_tpq: list[int], optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param pin_app_cpus: pin the threads of the application to 0-(n-db_cpus)
        :type pin_app_cpus: list[bool], optional
        :param batch_size: batch size to set Resnet50 model with
        :type batch_size: list, optional
        :param device: device used to run the models in the database
        :type device: str, optional
        :param num_devices: number of devices per compute node to use to run ResNet
        :type num_devices: int, optional
        :param net_type: type of connection to use ("tcp" or "uds")
        :type net_type: str, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param rebuild_model: force rebuild of PyTorch model even if it is available
        :type rebuild_model: bool
        :param languages: which language to use for the tester "cpp" or "fortran"
        :type languages: str
        :param plot: flag to plot against in process results
        :type plot: str
        """
        logger.info("Starting inference colocated scaling tests")
        
        check_model(device, force_rebuild=rebuild_model)
        
        check_node_allocation(nodes, [0])
        logger.info("Experiment allocation passed check")

        exp, result_path = create_experiment_and_dir(exp_name, launcher)
        logger.debug("Experiment and Results folder created")
        write_run_config(result_path,
                        colocated=1,
                        node_feature=node_feature,
                        experiment_name=exp_name,
                        launcher=launcher,
                        nodes=nodes,
                        clients_per_node=clients_per_node,
                        database_cpus=db_cpus,
                        database_threads_per_queue=db_tpq,
                        database_port=db_port,
                        pin_app_cpus=pin_app_cpus,
                        batch_size=batch_size,
                        device=device,
                        num_devices=num_devices,
                        net_type=net_type,
                        net_ifname=net_ifname,
                        rebuild_model=rebuild_model,
                        iterations=iterations,
                        language=languages,
                        plot=plot
                        )
        print_yml_file(Path(result_path) / "run.cfg", logger)
        perms = list(product(nodes, clients_per_node, db_cpus, db_tpq, batch_size, pin_app_cpus, languages))
        for i, perm in enumerate(perms, start=1):
            c_nodes, cpn, dbc, dbtpq, batch, pin_app, language = perm
            logger.info(f"Running permutation {i} of {len(perms)}")

            infer_session = self._create_colocated_inference_session(exp,
                                                               node_feature,
                                                               c_nodes,
                                                               cpn,
                                                               pin_app,
                                                               net_type,
                                                               net_ifname,
                                                               dbc,
                                                               dbtpq,
                                                               db_port,
                                                               batch,
                                                               device,
                                                               num_devices,
                                                               rebuild_model,
                                                               iterations,
                                                               language)
            logger.debug("Inference session created")

            exp.start(infer_session, block=True, summary=True)

            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"One of the scaling tests failed {infer_session.name}")
        self.process_scaling_results(scaling_results_dir=exp_name, plot_type=plot)


    @staticmethod
    def _set_resnet_model(device="GPU", force_rebuild=False):
            resnet_model = f"./imagenet/resnet50.{device}.pt"
            if not Path(resnet_model).exists() or force_rebuild:
                logger.info(f"AI Model {resnet_model} does not exist or rebuild was asked, it will be created")
                try:
                    save_model(device)
                except:
                    logger.error(f"Could not save {resnet_model} for {device}.")
                    sys.exit(1)

            logger.info(f"Using model {resnet_model}")
            return resnet_model

    @classmethod
    def _create_inference_session(cls,
                                exp,
                                node_feature,
                                nodes,
                                tasks,
                                db_nodes,
                                db_cpus,
                                db_tpq,
                                batch_size,
                                device,
                                num_devices,
                                rebuild_model,
                                iterations,
                                language
                                ):
        resnet_model = cls._set_resnet_model(device, force_rebuild=rebuild_model) #the resnet file name does not store full length of node name
        cluster = 1 if db_nodes > 1 else 0
        run_settings = exp.create_run_settings(f"./{language}-inference/build/run_resnet_inference", run_args=node_feature)
        run_settings.set_nodes(nodes)
        run_settings.set_tasks_per_node(tasks)
        run_settings.set_tasks(tasks*nodes)
        # tell scaling application not to set the model from the application
        # as we will do that from the driver in non-converged deployments
        run_settings.update_env({
            "SS_SET_MODEL": "0",
            "SS_ITERATIONS": str(iterations),
            "SS_COLOCATED": "0",
            "SS_CLUSTER": cluster,
            "SS_NUM_DEVICES": str(num_devices),
            "SS_BATCH_SIZE": str(batch_size),
            "SS_DEVICE": device,
            "SS_CLIENT_COUNT": str(tasks),
            "SR_LOG_FILE": "srlog.out",
            "SR_LOG_LEVEL": "INFO",
            "SR_CONN_TIMEOUT": 1000
        })

        name = "-".join((
            "infer-sess",
            language,
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
            "DBCPU"+str(db_cpus),
            "ITER"+str(iterations),
            "DBTPQ"+str(db_tpq),
            get_uuid()
            ))

        model = exp.create_model(name, run_settings)
        model.attach_generator_files(to_copy=["./imagenet/cat.raw",
                                            resnet_model,
                                            "./imagenet/data_processing_script.txt"])
        exp.generate(model, overwrite=True)
        write_run_config(model.path,
                        colocated=0,
                        client_total=tasks*nodes,
                        client_per_node=tasks,
                        client_nodes=nodes,
                        database_nodes=db_nodes,
                        database_cpus=db_cpus,
                        database_threads_per_queue=db_tpq,
                        batch_size=batch_size,
                        device=device,
                        num_devices=num_devices,
                        language=language,
                        iterations=iterations,
                        node_feature=node_feature)

        return model, resnet_model

    @classmethod
    def _create_colocated_inference_session(cls,
                                       exp,
                                       node_feature,
                                       nodes,
                                       tasks,
                                       pin_app_cpus,
                                       net_type,
                                       net_ifname,
                                       db_cpus,
                                       db_tpq,
                                       db_port,
                                       batch_size,
                                       device,
                                       num_devices,
                                       rebuild_model,
                                       iterations,
                                       language):
        resnet_model = cls._set_resnet_model(device, force_rebuild=rebuild_model)
        # feature = db_node_feature.split( )
        run_settings = exp.create_run_settings(f"./{language}-inference/build/run_resnet_inference", run_args=node_feature)
        run_settings.set_nodes(nodes)
        run_settings.set_tasks(nodes*tasks)
        run_settings.set_tasks_per_node(tasks)
        run_settings.update_env({
            "SS_SET_MODEL": "1",
            "SS_ITERATIONS": str(iterations),
            "SS_COLOCATED": "1",
            "SS_CLUSTER": "0",
            "SS_BATCH_SIZE": str(batch_size),
            "SS_DEVICE": device,
            "SS_CLIENT_COUNT": str(tasks),
            "SS_NUM_DEVICES": str(num_devices),
            "SR_LOG_FILE": "srlog.out",
            "SR_LOG_LEVEL": "info"
        })

        name = "-".join((
            "infer-sess-colo",
            language,
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(nodes),
            "DBCPU"+str(db_cpus),
            "ITER"+str(iterations),
            "DBTPQ"+str(db_tpq),
            get_uuid()
            ))
        model = exp.create_model(name, run_settings)
        model.attach_generator_files(to_copy=["./imagenet/cat.raw",
                                            resnet_model,
                                            "./imagenet/data_processing_script.txt"])

        db_opts = dict(
            db_cpus=db_cpus,
            limit_app_cpus=pin_app_cpus,
            threads_per_queue=db_tpq,
            # turning this to true can result in performance loss
            # on networked file systems(many writes to db log file)
            debug=True,
            loglevel="notice"
        )


        # add co-located database
        if net_type.lower() == "uds":
            model.colocate_db_uds(**db_opts)
        elif net_type.lower() == "tcp":
            model.colocate_db_tcp(
                port=db_port,
                ifname=net_ifname,
                **db_opts
            )
        exp.generate(model, overwrite=True)
        write_run_config(
            model.path,
            colocated=1,
            nodes=nodes,
            client_total=tasks*nodes,
            clients_per_node=tasks,
            database_cpus=db_cpus,
            database_threads_per_queue=db_tpq,
            database_port=db_port,
            pin_app_cpus=pin_app_cpus,
            batch_size=batch_size,
            device=device,
            num_devices=num_devices,
            net_type=net_type,
            net_ifname=net_ifname,
            rebuild_model=rebuild_model,
            iterations=iterations,
            language=language
        )
        return model