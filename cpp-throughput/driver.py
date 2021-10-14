import pandas as pd
import os.path as osp
from itertools import product

from smartsim.settings import AprunSettings, SrunSettings, JsrunSettings
from smartsim.database import PBSOrchestrator, SlurmOrchestrator, LSFOrchestrator
from smartsim import Experiment, constants

from smartsim.utils.log import get_logger
logger = get_logger("Scaling Tests")

WLM="lsf"
exp = Experiment(name="throughput-scaling-tests", launcher=WLM)

class ThroughputScaling:
    """Create a battery of throughput scaling tests to launch
    on a slurm, LSF, or PBS system, launch them, and record the
    performance results.

    Note: You must obtain an interactive allocation with
    at least max(client_nodes) nodes and max(clients_per_node)
    cores per node.

    Note: There must be max(db_nodes) nodes available with at least
    max(db_cpus) cores on each node.
    """

    def __init__(self):
        logger.info("Starting Scaling Tests")

    def throughput(self,
                   db_nodes=[16, 32, 64],
                   db_cpus=32,
                   db_port=6780,
                   clients_per_node=[32],
                   client_nodes=[128, 256, 512],
                   tensor_bytes=[1024, 8192, 16384, 32769, 65538, 131076,
                                 262152, 524304, 1024000, 2048000, 4096000],
                   batch=False):

        """Run the throughput scaling tests

        The lists of clients_per_node, tensor_bytes and client_nodes will
        be permuted into 3-tuples and run as an individual test.

        The number of tests will be client_nodes * clients_per_node * tensor_bytes

        if multiple db sizes are selected, a database cluster will be spun
        up for all permutations for each database size.

        Each run, the database will launch as a batch job that will wait until
        its running (e.g. not queued) before running the client driver.

        Resource constraints listed in this module are specific to in house
        systems and will need to be changed for your system.

        :param db_nodes: list of db node sizes
        :type db_nodes: list[int], optional
        :param db_cpus: number of cpus per db shard
        :type db_cpus: int, optional
        :param db_port: database port
        :type db_port: int, optional
        :param clients_per_node: list of ranks per node
        :type clients_per_node: list[int], optional
        :param client_nodes: list of client node counts
        :type client_nodes: list[int], optional
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        :param batch: whether the DB should be launched through a separate
                      batch job
        :type batch: bool
        """


        data_locations = []
        for db_node_count in db_nodes:

            db = start_database(db_port, int(db_node_count), db_cpus, batch=batch)
            logger.info("Orchestrator Database created and running")

            perms = list(product(client_nodes, clients_per_node, tensor_bytes))
            for perm in perms:

                # setup a an instance of the C++ driver and start it
                scale_session = create_throughput_session(perm[0],   # nodes
                                                        perm[1],     # tasks
                                                        db_node_count,
                                                        perm[2]      # tensor bytes
                                                        )
                exp.start(scale_session, summary=True)

                # confirm scaling test run successfully
                status = exp.get_status(scale_session)
                if status[0] != constants.STATUS_COMPLETED:
                    logger.error(f"ERROR: One of the scaling tests failed {scale_session.name}")
                    break

                # get the statistics from the run
                post = create_post_process(scale_session.path,
                                        scale_session.name)
                data_locations.append((scale_session.path, scale_session.name, perm))
                exp.start(post)


            try:
                # get the statistics from post processing
                # and add to the experiment summary
                stats_df = get_stats(data_locations, db_node_count)
                summary_df = exp.summary()
                final_df = pd.merge(summary_df, stats_df, on="Name")

                # save experiment info
                print(final_df)
                final_df.to_csv(exp.name + f"_{str(db_node_count)}_" + ".csv")


            except Exception:
                print("Could not preprocess results")

            exp.stop(db)

def start_database(port, nodes, cpus, batch):
    """Create and start the Redis database for the scaling test

    This function launches the redis database instances. If
    ``batch==True``, the database is launched through a
    Sbatch, Bsub, or Qsub script, otherwise it will use nodes
    from the allocation in which this driver is running.

    :param port: port number of database
    :type port: int
    :param nodes: number of database nodes
    :type nodes: int
    :param cpus: number of cpus per node
    :type cpus: int
    :return: orchestrator instance
    :rtype: Orchestrator
    """
    if WLM == "pbs":
        db = PBSOrchestrator(port=port,
                            db_nodes=nodes,
                            batch=batch)
    elif WLM == "slurm":
        db = SlurmOrchestrator(port=port,
                               db_nodes=nodes,
                               batch=batch)
    elif WLM == "lsf":
        db_per_host = 42//cpus  # Summit specific
        db = LSFOrchestrator(port=port,
                             db_per_host=db_per_host,
                             db_nodes=nodes*db_per_host,
                             batch=batch,
                             cpus_per_shard=cpus,
                             gpus_per_shard=6//db_per_host,
                             project="GEN150_SMARTSIM")
    else:
        raise Exception("WLM not supported")


    if batch:
        if WLM=="lsf":
            db.set_walltime("03:00")
        else:
            db.set_walltime("3:00:00")
    exp.generate(db)
    exp.start(db)
    return db

def create_throughput_session(nodes, tasks, db, _bytes):
    """Create a scaling session using the C++ driver with the SmartRedis client

    :param nodes: number of nodes for the client driver
    :type nodes: int
    :param tasks: number of tasks for each client driver
    :type tasks: int
    :param db: number of database nodes
    :type db: int
    :param bytes: the tensor size in bytes
    :type bytes: int
    :return: created Model instance
    :rtype: Model
    """
    if WLM == "pbs":
        run = AprunSettings("./build/throughput", str(_bytes))
        run.set_tasks(nodes * tasks)
    elif WLM == "slurm":
        run = SrunSettings("./build/throughput", str(_bytes))
        run.set_tasks_per_node(tasks)
    elif WLM == "lsf":
        run_args = dict()
        run_args["b"] = "packed:1"
        run = JsrunSettings("./build/throughput",
                            str(_bytes),
                            run_args=run_args)
        run.update_env({"OMP_NUM_THREADS": "1"})
        run.set_rs_per_host(tasks)
        run.set_cpus_per_rs(42//tasks)  # Summit specific
        run.set_num_rs(nodes*tasks)
    else:
        raise Exception("WLM not supported")

    name = "-".join(("throughput-sess", str(nodes), str(tasks), str(_bytes), str(db)))
    model = exp.create_model(name, run)
    model.attach_generator_files(to_copy=["./process_results.py"])
    exp.generate(model, overwrite=True)
    return model

def create_post_process(model_path, name):
    """Create a Model to post process the throughput results

    :param model_path: path to model output data
    :type model_path: str
    :param name: name of the model
    :type name: str
    :return: created post processing model
    :rtype: Model
    """

    exe_args = f"process_results.py --path={model_path} --name={name}"
    if WLM == "pbs":
        run = AprunSettings("python", exe_args=exe_args)
        run.set_tasks(1)
    elif WLM == "slurm":
        run = SrunSettings("python", exe_args=exe_args)
        run.set_tasks(1)
    elif WLM == "lsf":
        run = JsrunSettings("python", exe_args=exe_args)
        run.set_tasks(1)
    else:
        raise Exception("WLM not supported")


    pp_name = "-".join(("post", name))
    post_process = exp.create_model(pp_name, run, path=model_path)
    return post_process

def get_stats(data_locations, db_node_count):
    """Compile throughput results into pandas dataframe

    :param data_locations: path, name, (nodes, tasks, db nodes)
    :type data_locations: tuple(str, str, tuple(int, int, int))
    :param db_node_count: number of database nodes for this run
    :type db_node_count: int
    :return: dataframe
    :rtype: pd.DataFrame
    """
    all_data = None
    for data_path, name, job_info in data_locations:
        data_path = osp.join(data_path, name + ".csv")
        data = pd.read_csv(data_path)
        data = data.drop("Unnamed: 0", axis=1)

        # add node and task information
        data["db"] = db_node_count
        data["nodes"] = job_info[0]
        data["tasks"] = job_info[1]
        data["bytes"] = job_info[2]

        if not isinstance(all_data, pd.DataFrame):
            all_data = data
        else:
            all_data = pd.concat([all_data, data])
    return all_data

if __name__ == "__main__":
    import fire
    fire.Fire(ThroughputScaling())
