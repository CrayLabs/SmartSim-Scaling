import fire
from utils import *
from utils import _get_db_backend
from utils import _check_model
from utils import _get_uuid

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


class Inference:
 
    def inference_standard(self,
                           exp_name="inference-standard-scaling",
                           launcher="auto",
                           run_db_as_batch=True,
                           node_feature = [],
                           db_hosts=[],
                           db_nodes=[12],
                           db_cpus=[2],
                           db_tpq=[1],
                           db_port=6780,
                           batch_size=[1000],
                           device="GPU",
                           num_devices=1,
                           net_ifname="ipogif0",
                           clients_per_node=[48],
                           client_nodes=[12],
                           rebuild_model=False):
        """Run ResNet50 inference tests with standard Orchestrator deployment
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
        :param db_tpq: number of device threads to use for the database
        :type db_tpq: list, optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param batch_size: batch size to set Resnet50 model with
        :type batch_size: list, optional
        :param device: device used to run the models in the database
        :type device: str, optional
        :param num_devices: number of devices per compute node to use to run ResNet
        :type num_devices: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list, optional
        :param client_nodes: number of compute nodes to use for the synthetic scaling app
        :type client_nodes: list, optional
        :param rebuild_model: force rebuild of PyTorch model even if it is available
        :type rebuild_model: bool
        """
        logger.info("Starting inference scaling tests")
        logger.info(f"Running with database backend: {_get_db_backend()}")

        _check_model(device, force_rebuild=rebuild_model)

        exp = create_folder(exp_name, launcher)

        # create permutations of each input list and run each of the permutations
        # as a single inference scaling test
        perms = list(product(client_nodes, clients_per_node, db_nodes, db_cpus, db_tpq, batch_size))
        logger.info(f"Executing {len(perms)} permutations")
        for perm in perms:
            c_nodes, cpn, dbn, dbc, dbtpq, batch = perm

            db = start_database(exp,
                                node_feature,
                                db_port,
                                dbn,
                                dbc,
                                dbtpq,
                                net_ifname,
                                run_db_as_batch,
                                db_hosts)

            # setup a an instance of the synthetic C++ app and start it
            infer_session, resnet_model = self.create_inference_session(exp,
                                                     c_nodes,
                                                     cpn,
                                                     dbn,
                                                     dbc,
                                                     dbtpq,
                                                     batch,
                                                     device,
                                                     num_devices,
                                                     rebuild_model)

            # only need 1 address to set model
            address = db.get_address()[0]
            setup_resnet(resnet_model,
                        device,
                        num_devices,
                        batch,
                        address,
                        cluster=dbn>1)


            exp.start(infer_session, block=True, summary=True)

            # kill the database each time so we get a fresh start
            exp.stop(db)

            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"One of the scaling tests failed {infer_session.name}")
    
    def inference_colocated(self,
                            exp_name="inference-colocated-scaling",
                            db_node_feature={},
                            launcher="auto",
                            nodes=[12],
                            clients_per_node=[18],
                            db_cpus=[2],
                            db_tpq=[1],
                            db_port=6780,
                            pin_app_cpus=[False], #CPU architecutre, GPU architecture 
                            batch_size=[1],
                            device="GPU",
                            num_devices=1,
                            net_ifname="lo",
                            rebuild_model=False
                            ):
        """Run ResNet50 inference tests with colocated Orchestrator deployment
        :param exp_name: name of output dir, defaults to "inference-scaling"
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param nodes: compute nodes to use for synthetic scaling app with
                      a co-located orchestrator database
        :type clients_per_node: list, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list, optional
        :param db_hosts: optionally supply hosts to launch the database on
        :type db_hosts: list, optional
        :param db_nodes: number of compute hosts to use for the database
        :type db_nodes: list, optional
        :param db_cpus: number of cpus per compute host for the database
        :type db_cpus: list, optional
        :param db_tpq: number of device threads to use for the database
        :type db_tpq: list, optional
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param pin_app_cpus: pin the threads of the application to 0-(n-db_cpus)
        :type pin_app_cpus: int, optional
        :param batch_size: batch size to set Resnet50 model with
        :type batch_size: list, optional
        :param device: device used to run the models in the database
        :type device: str, optional
        :param num_devices: number of devices per compute node to use to run ResNet
        :type num_devices: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param rebuild_model: force rebuild of PyTorch model even if it is available
        :type rebuild_model: bool
        """
        logger.info("Starting colocated inference scaling tests")
        logger.info(f"Running with database backend: {_get_db_backend()}")

        _check_model(device, force_rebuild=rebuild_model)
        
        exp = create_folder(exp_name, launcher)

        # create permutations of each input list and run each of the permutations
        # as a single inference scaling test
        perms = list(product(nodes, clients_per_node, db_cpus, db_tpq, batch_size, pin_app_cpus))
        for perm in perms:
            c_nodes, cpn, dbc, dbtpq, batch, pin_app = perm

            infer_session = self.create_colocated_inference_session(exp,
                                                               db_node_feature,
                                                               c_nodes,
                                                               cpn,
                                                               pin_app,
                                                               net_ifname,
                                                               dbc,
                                                               dbtpq,
                                                               db_port,
                                                               batch,
                                                               device,
                                                               num_devices,
                                                               rebuild_model)

            exp.start(infer_session, block=True, summary=True)

            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"One of the scaling tests failed {infer_session.name}")

                
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
    def create_inference_session(cls,
                                exp,
                                nodes,
                                tasks,
                                db_nodes,
                                db_cpus,
                                db_tpq,
                                batch_size,
                                device,
                                num_devices,
                                rebuild_model
                                ):
        resnet_model = cls._set_resnet_model(device, force_rebuild=rebuild_model)

        cluster = 1 if db_nodes > 1 else 0
        run_settings = exp.create_run_settings("./cpp-inference/build/run_resnet_inference")
        run_settings.set_nodes(nodes)
        run_settings.set_tasks_per_node(tasks)
        run_settings.set_tasks(tasks*nodes)
        # tell scaling application not to set the model from the application
        # as we will do that from the driver in non-converged deployments
        run_settings.update_env({
            "SS_SET_MODEL": 0,
            "SS_CLUSTER": cluster,
            "SS_NUM_DEVICES": str(num_devices),
            "SS_BATCH_SIZE": str(batch_size),
            "SS_DEVICE": device,
            "SS_CLIENT_COUNT": str(tasks),
            "SR_LOG_FILE": "srlog.out",
            "SR_LOG_LEVEL": "INFO"
        })
        
        name = "-".join((
            "infer-sess",
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
            "DBC"+str(db_cpus),
            "DBTPQ"+str(db_tpq),
            _get_uuid()
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
                        num_devices=num_devices)

        return model, resnet_model
    
    @classmethod
    def create_colocated_inference_session(cls,
                                       exp,
                                       db_node_feature,
                                       nodes,
                                       tasks,
                                       pin_app_cpus,
                                       net_ifname,
                                       db_cpus,
                                       db_tpq,
                                       db_port,
                                       batch_size,
                                       device,
                                       num_devices,
                                       rebuild_model):
        resnet_model = cls._set_resnet_model(device, force_rebuild=rebuild_model)
        # feature = db_node_feature.split( )
        feature = {
            "constraint" : db_node_feature
            }
        run_settings = exp.create_run_settings("./cpp-inference/build/run_resnet_inference", run_args=feature)
        run_settings.set_nodes(nodes)
        run_settings.set_tasks(nodes*tasks)
        run_settings.set_tasks_per_node(tasks)
        run_settings.update_env({
            "SS_SET_MODEL": "1",  # set the model from the scaling application
            "SS_CLUSTER": "0",     # never cluster for colocated db
            "SS_BATCH_SIZE": str(batch_size),
            "SS_DEVICE": device,
            "SS_CLIENT_COUNT": str(tasks),
            "SS_NUM_DEVICES": str(num_devices),
            "SR_LOG_FILE": "srlog.out",
            "SR_LOG_LEVEL": "info"
        })

        name = "-".join((
            "infer-sess-colo",
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(nodes),
            "DBC"+str(db_cpus),
            "DBTPQ"+str(db_tpq),
            _get_uuid()
            ))
        model = exp.create_model(name, run_settings)
        model.attach_generator_files(to_copy=["./imagenet/cat.raw",
                                            resnet_model,
                                            "./imagenet/data_processing_script.txt"])

        # add co-located database
        model.colocate_db(port=db_port,
                        db_cpus=db_cpus,
                        limit_app_cpus=pin_app_cpus,
                        ifname=net_ifname,
                        threads_per_queue=db_tpq,
                        # turning this to true can result in performance loss
                        # on networked file systems(many writes to db log file)
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
                        database_threads_per_queue=db_tpq,
                        batch_size=batch_size,
                        device=device,
                        num_devices=num_devices)
        return model
      
if __name__ == "__main__":
    import sys
    sys.path.append('..')

if __name__ == "__main__":
    import fire
    fire.Fire(Inference())