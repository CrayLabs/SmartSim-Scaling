import os
import shutil
import sys
import configparser
import datetime
from pathlib import Path
from itertools import product
from tqdm import tqdm
from uuid import uuid4
import pandas as pd
from process_results import create_run_csv
from imagenet.model_saver import save_model


import smartsim
from smartsim import Experiment, status
from smartredis import Client


from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")

def get_date():
    date = str(datetime.datetime.now().strftime("%Y-%m-%d"))
    return date

def _check_model(device, force_rebuild=False):
    if device.startswith("GPU") and (force_rebuild or not Path("./imagenet/resnet50.GPU.pt").exists()):
        from torch.cuda import is_available
        if not is_available():
            message = "resnet50.GPU.pt model missing in ./imagenet directory. \n"
            message += "Since CUDA is not available on this node, the model cannot be created. \n"
            message += "Please run 'python imagenet/model_saver.py --device GPU' on a node with an available CUDA device."
            logger.error(message)
            sys.exit(1)

def create_folder(exp_name, launcher): 
    i = 0
    while os.path.exists("results/"+exp_name+"/run" + str(i)):
        i += 1
    path2 = os.path.join(os.getcwd(), "results/"+exp_name+"/run" + str(i)) #autoincrement
    os.makedirs(path2)
    exp = Experiment(name="results/"+exp_name+"/run" + str(i), launcher=launcher)
    exp.generate()
    log_to_file(f"{exp.exp_path}/scaling-{get_date()}.log")
    return exp

def start_database(exp, port, nodes, cpus, tpq, net_ifname, run_as_batch, batch_args, hosts):
    """Create and start the Orchestrator

    This function is only called if the scaling tests are
    being executed in the standard deployment mode where
    separate computational resources are used for the app
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
                            single_cmd=True,
                            hosts=hosts)
    if run_as_batch:
        db.set_walltime("48:00:00")
        for k, v in batch_args.items():
            db.set_batch_arg(k, v)

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
    client = Client(address=address, cluster=cluster)
    device = device.upper()
    if (device == "GPU") and (num_devices > 1):
        client.set_model_from_file_multigpu("resnet_model_0",
                                            model,
                                            "TORCH",
                                            0, num_devices,
                                            batch_size)
        client.set_script_from_file_multigpu("resnet_script_0",
                                             "./imagenet/data_processing_script.txt",
                                             0, num_devices)
        logger.info(f"Resnet Model and Script in Orchestrator on {num_devices} GPUs")
    else:
        # Redis does not accept CPU:<n>. We are either
        # setting (possibly multiple copies of) the model and script on CPU, or one
        # copy of them (resnet_model_0, resnet_script_0) on ONE GPU.
        for i in range (num_devices):
            client.set_model_from_file(f"resnet_model_{i}",
                                       model,
                                       "TORCH",
                                       device,
                                       batch_size)
            client.set_script_from_file(f"resnet_script_{i}",
                                        "./imagenet/data_processing_script.txt",
                                        device)
            logger.info(f"Resnet Model and Script in Orchestrator on device {device}:{i}")


def create_aggregation_producer_session(exp, nodes, tasks, db_nodes, db_cpus,
                                        iterations, _bytes, t_per_dataset):
    return _create_aggregation_producer_session(
        name="aggregate-sess-prod",
        exe="./cpp-data-aggregation/build/aggregation_producer",
        exe_args=[str(_bytes), str(t_per_dataset)],
        exp=exp, nodes=nodes, tasks=tasks, db_nodes=db_nodes, db_cpus=db_cpus,
        iterations=iterations, bytes_=_bytes, t_per_dataset=t_per_dataset)


def create_aggregation_producer_session_python(exp, nodes, tasks, db_nodes, db_cpus,
                                               iterations, _bytes, t_per_dataset):
    return _create_aggregation_producer_session(
        name="aggregate-sess-prod-for-python-consumer",
        exe="./py-data-aggregation/db/build/aggregation_producer",
        exe_args=[str(_bytes), str(t_per_dataset)],
        exp=exp, nodes=nodes, tasks=tasks, db_nodes=db_nodes, db_cpus=db_cpus,
        iterations=iterations, bytes_=_bytes, t_per_dataset=t_per_dataset)


def create_aggregation_producer_session_python_fs(exp, nodes, tasks, iterations,
                                                  bytes_, t_per_dataset):
    return _create_aggregation_producer_session(
        name="aggregate-sess-prod-for-python-consumer-file-system",
        exe="./py-data-aggregation/fs/build/aggregation_producer",
        exe_args=[str(bytes_), str(t_per_dataset)],
        exp=exp, nodes=nodes, tasks=tasks, db_nodes=0, db_cpus=0,
        iterations=iterations, bytes_=bytes_, t_per_dataset=t_per_dataset)


def _create_aggregation_producer_session(name,
                                         exe,
                                         exe_args,
                                         exp,
                                         nodes,
                                         tasks,
                                         db_nodes,
                                         db_cpus,
                                         iterations,
                                         bytes_,
                                         t_per_dataset):
    """Create a Model to run a producer for the aggregation scaling session

    :param name: The name of the model being created
    :type name: str
    :param exe: The path to the executable used by the model
    :type exe: str
    :param exe_args: The arguments passed to the executable
    :type exe_args: list[str]
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
    :param _bytes: size in bytes of tensors to use for aggregation scaling
    :type _bytes: int
    :param t_per_dataset: the number of tensors per dataset
    :type t_per_dataset: int
    :return: Model reference to the aggregation session to launch
    :rtype: Model
    """
    settings = exp.create_run_settings(exe, exe_args)
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
        "N"+str(nodes),
        "T"+str(tasks),
        "DBN"+str(db_nodes),
        "ITER"+str(iterations),
        "TB"+str(bytes_),
        "TPD"+str(t_per_dataset),
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
                     tensor_bytes=bytes_,
                     t_per_dataset=t_per_dataset)
    return model

def create_aggregation_consumer_session(exp, nodes, tasks, db_nodes, db_cpus,
                                        iterations, bytes_, t_per_dataset,
                                        c_threads, cpu_hyperthreads):
    return _create_aggregation_consumer_session(
        name="aggregate-sess-cons",
        exe="./cpp-data-aggregation/build/aggregation_consumer",
        exe_args=[str(nodes*tasks)], exp=exp, nodes=nodes, tasks=tasks,
        db_nodes=db_nodes, db_cpus=db_cpus, iterations=iterations, bytes_=bytes_,
        t_per_dataset=t_per_dataset, c_threads=c_threads, cpu_hyperthreads=cpu_hyperthreads)

def create_aggregation_consumer_session_python(exp, nodes, tasks, db_nodes, db_cpus,
                                               iterations, bytes_, t_per_dataset,
                                               c_threads, cpu_hyperthreads):
    py_script_dir = "./py-data-aggregation/db/"
    py_script = "aggregation_consumer.py"
    model = _create_aggregation_consumer_session(
        name="aggregate-sess-cons-python",
        exe=sys.executable,
        exe_args=[py_script, str(nodes*tasks)],
        exp=exp, nodes=nodes, tasks=tasks, db_nodes=db_nodes, db_cpus=db_cpus,
        iterations=iterations, bytes_=bytes_, t_per_dataset=t_per_dataset,
        c_threads=c_threads, cpu_hyperthreads=cpu_hyperthreads)
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
                     client_threads=c_threads)
    return model

def create_aggregation_consumer_session_python_fs(exp, nodes, tasks, iterations,
                                                  bytes_, t_per_dataset, c_threads,
                                                  cpu_hyperthreads):
    py_script_dir = "./py-data-aggregation/fs/"
    py_script = "aggregation_consumer.py"
    model = _create_aggregation_consumer_session(
        name="aggregate-sess-cons-python",
        exe=sys.executable,
        exe_args=[py_script, str(nodes*tasks)],
        exp=exp, nodes=nodes, tasks=tasks, db_nodes=0, db_cpus=0,
        iterations=iterations, bytes_=bytes_, t_per_dataset=t_per_dataset,
        c_threads=c_threads, cpu_hyperthreads=cpu_hyperthreads)
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
                     client_threads=c_threads)
    return model
    
def _create_aggregation_consumer_session(name, 
                                         exe,
                                         exe_args,
                                         exp,
                                         nodes,
                                         tasks,
                                         db_nodes,
                                         db_cpus,
                                         iterations,
                                         bytes_,
                                         t_per_dataset,
                                         c_threads,
                                         cpu_hyperthreads):
    """Create a Model to run a consumer for the aggregation scaling session

    :param name: The name of the model being created
    :type name: str
    :param exe: The path to the executable used by the model
    :type exe: str
    :param exe_args: The arguments passed to the executable
    :type exe_args: list[str]
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
    :rtype c_threads: int
    :type t_per_dataset: int
    :rtype: Model
    :param cpu_hyperthreads: the number of hyperthreads per cpu.  This is done
                             to request that the consumer application utilizes
                             all physical cores for each client thread.
    :type cpu_hyperthreads: int, optional
    """
    settings = exp.create_run_settings(exe, exe_args)
    #settings.set_tasks(1)
    settings.set_tasks_per_node(1)
    settings.set_cpus_per_task(c_threads * cpu_hyperthreads)
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
        "N"+str(nodes),
        "T"+str(tasks),
        "DBN"+str(db_nodes),
        "ITER"+str(iterations),
        "TB"+str(bytes_),
        "TPD"+str(t_per_dataset),
        "CT"+str(c_threads),
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
                tensor_bytes=bytes_,
                t_per_dataset=t_per_dataset,
                client_threads=c_threads)
    return model

def write_run_config(path, **kwargs):
    name = os.path.basename(path)
    config = configparser.ConfigParser()
    config["run"] = {
        "name": name,
        "path": path,
        "smartsim_version": smartsim.__version__,
        "smartredis_version": "0.3.1", # TODO put in smartredis __version__
        "db": _get_db_backend(),
        "date": str(datetime.datetime.now().strftime("%Y-%m-%d"))
    }
    config["attributes"] = kwargs

    config_path = Path(path) / "run.cfg"
    with open(config_path, "w") as f:
        config.write(f)

def _get_uuid():
    uid = str(uuid4())
    return uid[:4]

def _get_db_backend():
    db_backend_path = smartsim._core.config.CONFIG.database_exe
    return os.path.basename(db_backend_path)
