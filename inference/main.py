import fire
from utils import *
from utils import _get_db_backend
from utils import _check_model

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


class Inference():
    def inference_standard(self,
                            exp_name="standard-inference-scaling",
                            launcher="auto",
                            run_db_as_batch=True,
                            batch_args={},
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
                                db_port,
                                dbn,
                                dbc,
                                dbtpq,
                                net_ifname,
                                run_db_as_batch,
                                batch_args,
                                db_hosts)

            # setup a an instance of the synthetic C++ app and start it
            infer_session, resnet_model = create_inference_session(exp,
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


        # def inference_colocated(self,
        #                         exp_name="colocated-inference-scaling",
        #                         launcher="auto",
        #                         nodes=[12],
        #                         clients_per_node=[18],
        #                         db_cpus=[2],
        #                         db_tpq=[1],
        #                         db_port=6780,
        #                         pin_app_cpus=[False],
        #                         batch_size=[1],
        #                         device="GPU",
        #                         num_devices=1,
        #                         net_ifname="ipogif0",
        #                         rebuild_model=False
        #                         ):
        #     """Run ResNet50 inference tests with colocated Orchestrator deployment

        #     :param exp_name: name of output dir, defaults to "inference-scaling"
        #     :type exp_name: str, optional
        #     :param launcher: workload manager i.e. "slurm", "pbs"
        #     :type launcher: str, optional
        #     :param nodes: compute nodes to use for synthetic scaling app with
        #                   a co-located orchestrator database
        #     :type clients_per_node: list, optional
        #     :param clients_per_node: client tasks per compute node for the synthetic scaling app
        #     :type clients_per_node: list, optional
        #     :param db_hosts: optionally supply hosts to launch the database on
        #     :type db_hosts: list, optional
        #     :param db_nodes: number of compute hosts to use for the database
        #     :type db_nodes: list, optional
        #     :param db_cpus: number of cpus per compute host for the database
        #     :type db_cpus: list, optional
        #     :param db_tpq: number of device threads to use for the database
        #     :type db_tpq: list, optional
        #     :param db_port: port to use for the database
        #     :type db_port: int, optional
        #     :param pin_app_cpus: pin the threads of the application to 0-(n-db_cpus)
        #     :type pin_app_cpus: int, optional
        #     :param batch_size: batch size to set Resnet50 model with
        #     :type batch_size: list, optional
        #     :param device: device used to run the models in the database
        #     :type device: str, optional
        #     :param num_devices: number of devices per compute node to use to run ResNet
        #     :type num_devices: int, optional
        #     :param net_ifname: network interface to use i.e. "ib0" for infiniband or
        #                        "ipogif0" aries networks
        #     :type net_ifname: str, optional
        #     :param rebuild_model: force rebuild of PyTorch model even if it is available
        #     :type rebuild_model: bool
        #     """
        #     logger.info("Starting colocated inference scaling tests")
        #     logger.info(f"Running with database backend: {_get_db_backend()}")

        #     self._check_model(device, force_rebuild=rebuild_model)
            
        #     exp = create_folder(self, exp_name, launcher)

        #     # create permutations of each input list and run each of the permutations
        #     # as a single inference scaling test
        #     perms = list(product(nodes, clients_per_node, db_cpus, db_tpq, batch_size, pin_app_cpus))
        #     for perm in perms:
        #         c_nodes, cpn, dbc, dbtpq, batch, pin_app = perm

        #         infer_session = create_colocated_inference_session(self,
        #                                                            exp,
        #                                                            c_nodes,
        #                                                            cpn,
        #                                                            pin_app,
        #                                                            net_ifname,
        #                                                            dbc,
        #                                                            dbtpq,
        #                                                            db_port,
        #                                                            batch,
        #                                                            device,
        #                                                            num_devices,
        #                                                            rebuild_model)

        #         exp.start(infer_session, block=True, summary=True)

        #         # confirm scaling test run successfully
        #         stat = exp.get_status(infer_session)
        #         if stat[0] != status.STATUS_COMPLETED:
        #             logger.error(f"One of the scaling tests failed {infer_session.name}")

                
if __name__ == "__main__":
    import sys
    sys.path.append('..')

if __name__ == "__main__":
    fire.Fire(Inference())