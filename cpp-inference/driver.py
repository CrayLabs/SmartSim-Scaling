import os
import configparser
import datetime
from pathlib import Path
from itertools import product
from tqdm import tqdm

import smartsim
from smartsim import Experiment, status
from smartredis import Client

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


class SmartSimScalingTests:

    def __init__(self):
        self.resnet_model = "../imagenet/resnet50.pt"
        self.date = str(datetime.datetime.now().strftime("%Y-%m-%d"))

    def resnet_standard(self,
                        exp_name="inference-scaling",
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
                        client_nodes=[12]):
        """Run ResNet50 inference tests with standard Orchestrator deployment

        :param exp_name: name of output dir
        :type exp_name: str, optional
        :param launcher: workload manager i.e. "slurm", "pbs"
        :type launcher: str, optional
        :param run_db_as_batch: run database as seperate batch submission each iteration
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
        """
        logger.info("Starting inference scaling tests")

        exp = Experiment(name=exp_name, launcher=launcher)
        exp.generate()
        log_to_file(f"{exp.exp_path}/scaling-{self.date}.log")

        # create permutations of each input list and run each of the permutations
        # as a single inference scaling test
        perms = list(product(client_nodes, clients_per_node, db_nodes, db_cpus, db_tpq, batch_size))
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
            infer_session = create_inference_session(exp,
                                                     c_nodes,
                                                     cpn,
                                                     dbn,
                                                     dbc,
                                                     dbtpq,
                                                     batch,
                                                     device,
                                                     num_devices)

            # only need 1 address to set model
            address = db.get_address()[0]
            setup_resnet(self.resnet_model,
                         device,
                         num_devices,
                         batch,
                         address,
                         cluster=bool(dbn>1))

            exp.start(infer_session, block=True, summary=True)

            # kill the database each time so we get a fresh start
            exp.stop(db)

            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"One of the scaling tests failed {infer_session.name}")


    def resnet_colocated(self,
                         exp_name="inference-scaling",
                         launcher="auto",
                         nodes=[12],
                         clients_per_node=[18],
                         db_cpus=[2],
                         db_tpq=[1],
                         db_port=6780,
                         pin_app_cpus=[False],
                         batch_size=[1],
                         device="GPU",
                         num_devices=1,
                         net_ifname="ipogif0",
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
        """
        logger.info("Starting colocated inference scaling tests")

        exp = Experiment(name=exp_name, launcher=launcher)
        exp.generate()
        log_to_file(f"{exp.exp_path}/scaling-{self.date}.log")

        # create permutations of each input list and run each of the permutations
        # as a single inference scaling test
        perms = list(product(nodes, clients_per_node, db_cpus, db_tpq, batch_size, pin_app_cpus))
        for perm in perms:
            c_nodes, cpn, dbc, dbtpq, batch, pin_app = perm

            infer_session = create_colocated_inference_session(exp,
                                                               c_nodes,
                                                               cpn,
                                                               pin_app,
                                                               net_ifname,
                                                               dbc,
                                                               dbtpq,
                                                               db_port,
                                                               batch,
                                                               device,
                                                               num_devices)

            exp.start(infer_session, block=True, summary=True)

            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"One of the scaling tests failed {infer_session.name}")


    def process_scaling_results(self, scaling_dir="inference-scaling", overwrite=True):
        """Create a results directory with performance data and plots

        With the overwrite flag turned off, this function can be used
        to build up a single csv with the results of runs over a long
        period of time.

        :param scaling_dir: directory to create results from
        :type scaling_dir: str, optional
        :param overwrite: overwrite any existing results
        :type overwrite: bool, optional
        """

        import pandas as pd
        from process_results import create_run_csv

        dataframes = []
        result_dir = Path(scaling_dir) / "results"
        runs = [d for d in os.listdir(scaling_dir) if d.startswith("infer-sess")]

        try:
            # write csv each so this function is impodent
            # csv's will not be written if they are already created
            for run in tqdm(runs, desc="Processing scaling results...", ncols=80):
                try:
                    run_path = Path(scaling_dir) / run
                    create_run_csv(run_path, delete_previous=overwrite)
                # want to catch all exceptions and skip runs that may
                # not have completed or finished b/c some reason i.e. node failure
                except Exception as e:
                    logger.warning(f"Skipping {run} could not process results")
                    logger.error(e)
                    continue

            # collect all written csv into dataframes to concat
            for run in tqdm(runs, desc="Collecting scaling results...", ncols=80):
                try:
                    results_path = os.path.join(result_dir, run, run + ".csv")
                    run_df = pd.read_csv(str(results_path))
                    dataframes.append(run_df)
                # catch all and skip for reason listed above
                except Exception as e:
                    logger.warning(f"Skipping {run} could not read results csv")
                    logger.error(e)
                    continue

            final_df = pd.concat(dataframes, join="outer")
            exp_name = os.path.basename(scaling_dir)
            csv_path = result_dir / f"{exp_name}-{self.date}.csv"
            final_df.to_csv(str(csv_path))

        except Exception:
            logger.error("Could not preprocess results")
            raise


def start_database(exp, port, nodes, cpus, tpq, net_ifname, run_as_batch, batch_args, hosts):
    """Create and start the Orchestrator

    This function is only called if the scaling tests are
    being executed in the standard deployment mode where
    seperate computational resources are used for the app
    and the database.

    :param port: port number of database
    :type port: int
    :param nodes: number of database nodes
    :type nodes: int
    :param cpus: number of cpus per node
    :type cpus: int
    :param tpq: number of threads per queue
    :type tpq: int
    :param net_ifname: network interface name
    :type net_ifname: str
    :return: orchestrator instance
    :rtype: Orchestrator
    """
    db = exp.create_database(port=port,
                            db_nodes=nodes,
                            batch=run_as_batch,
                            interface=net_ifname,
                            threads_per_queue=tpq,
                            single_cmd=True)
    if run_as_batch:
        db.set_walltime("00:10:00")
        for k, v in batch_args.items():
            db.set_batch_arg(k, v)
    if hosts:
        db.set_hosts(hosts)

    db.set_cpus(cpus)
    exp.generate(db)
    exp.start(db)
    logger.info("Orchestrator Database created and running")
    return db

def setup_resnet(model, device, num_devices, batch_size, address, cluster=True):
    """Set and configure the PyTorch resnet50 model for inference

    :param model: path to serialized resnet model
    :type model: str
    :param device: CPU or GPU
    :type device: str
    :param batch_size: batch size for the Orchestrator (batches of batches)
    :type batch_size: int
    :param cluster: true if using a cluster orchestrator
    :type cluster: bool
    """
    devices = []
    if num_devices > 0:
        devices = [f"{device.upper()}:{str(i)}" for i in range(num_devices)]
    else:
        devices = [device.upper()]
    for i, dev in enumerate(devices):
        client = Client(address=address, cluster=cluster)
        client.set_model_from_file(f"resnet_model_{str(i)}",
                                    model,
                                    "TORCH",
                                    dev,
                                    batch_size)
        client.set_script_from_file(f"resnet_script_{str(i)}",
                                    "../imagenet/data_processing_script.txt",
                                    dev)
        logger.info(f"Resnet Model and Script in Orchestrator on device {dev}")


def create_inference_session(exp,
                             nodes,
                             tasks,
                             db_nodes,
                             db_cpus,
                             db_tpq,
                             batch_size,
                             device,
                             num_devices
                             ):
    cluster = 1 if db_nodes > 1 else 0
    run_settings = exp.create_run_settings("./build/run_resnet_inference")
    run_settings.set_nodes(nodes)
    run_settings.set_tasks_per_node(tasks)
    # tell scaling application not to set the model from the application
    # as we will do that from the driver in non-converged deployments
    run_settings.update_env({
        "SS_SET_MODEL": 0,
        "SS_CLUSTER": cluster
    })

    name = "-".join((
        "infer-sess",
        "N"+str(nodes),
        "T"+str(tasks),
        "DBN"+str(db_nodes),
        "DBC"+str(db_cpus),
        "DBTPQ"+str(db_tpq)
        ))

    model = exp.create_model(name, run_settings)
    model.attach_generator_files(to_copy=["../imagenet/cat.raw",
                                          "../imagenet/resnet50.pt",
                                          "../imagenet/data_processing_script.txt"])
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

    return model


def create_colocated_inference_session(exp,
                                       nodes,
                                       tasks,
                                       pin_app_cpus,
                                       net_ifname,
                                       db_cpus,
                                       db_tpq,
                                       db_port,
                                       batch_size,
                                       device,
                                       num_devices):
    run_settings = exp.create_run_settings("./build/run_resnet_inference")
    run_settings.set_nodes(nodes)
    run_settings.set_tasks_per_node(tasks)
    run_settings.update_env({
        "SS_SET_MODEL": "1",  # set the model from the scaling application
        "SS_CLUSTER": "0",     # never cluster for colocated db
        "SS_BATCH_SIZE": str(batch_size),
        "SS_DEVICE": device,
        "SS_CLIENT_COUNT": str(tasks),
        "SS_NUM_DEVICES": str(num_devices)
    })

    name = "-".join((
        "infer-sess-colo",
        "N"+str(nodes),
        "T"+str(tasks),
        "DBN"+str(nodes),
        "DBC"+str(db_cpus),
        "DBTPQ"+str(db_tpq)
        ))
    model = exp.create_model(name, run_settings)
    model.attach_generator_files(to_copy=["../imagenet/cat.raw",
                                          "../imagenet/resnet50.pt",
                                          "../imagenet/data_processing_script.txt"])

    # add co-located database
    model.colocate_db(port=db_port,
                      db_cpus=db_cpus,
                      limit_app_cpus=pin_app_cpus,
                      ifname=net_ifname,
                      threads_per_queue=db_tpq,
                      # turning this to true can result in performance loss
                      # on networked file systems(many writes to db log file)
                      debug=False,
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


def write_run_config(path, **kwargs):
    name = os.path.basename(path)
    config = configparser.ConfigParser()
    config["run"] = {
        "name": name,
        "path": path,
        "smartsim_version": smartsim.__version__,
        "smartredis_version": "0.3.0", # put in smartredis __version__
        "date": str(datetime.datetime.now().strftime("%Y-%m-%d"))
    }
    config["attributes"] = kwargs

    config_path = Path(path) / "run.cfg"
    with open(config_path, "w") as f:
        config.write(f)


if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
