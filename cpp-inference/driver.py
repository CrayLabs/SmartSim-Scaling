import pandas as pd
import os.path as osp
from itertools import product

from smartsim.settings import SrunSettings
from smartsim.database import SlurmOrchestrator
from smartsim import Experiment, slurm, constants
from smartredis import Client

from smartsim.utils.log import get_logger
logger = get_logger("Scaling Tests")

exp = Experiment(name="inference-scaling-tests", launcher="slurm")

class SlurmScalingTests:
    """Create a battery of scaling tests to launch
    on a slurm system, launch them, and record the
    performance results.
    """

    def __init__(self):
        logger.info("Starting Scaling Tests")

    def resnet(self,
               db_nodes=[4,8,16],
               db_cpus=36,
               db_tpq=4,
               db_port=6780,
               batch_size=1000,
               device="GPU",
               model="../imagenet/resnet50-1.7.0.pt",
               clients_per_node=[48],
               client_nodes=[20, 40, 60, 80, 100, 120, 140, 160]):
        """Run the resnet50 inference tests.

        The lists of clients_per_node, db_nodes, and client_nodes will
        be permuted into 3-tuples and run as an individual test.

        The number of tests will be client_nodes * clients_per_node * db_nodes

        An allocation will be obtained of size max(client_nodes) and will be
        used for each run of the client driver

        Each run, the database will launch as a batch job that will wait until
        its running (e.g. not queued) before running the client driver.

        Resource constraints listed in this module are specific to in house
        systems and will need to be changed for your system.

        :param db_nodes: list of db node sizes
        :type db_nodes: list[int], optional
        :param db_cpus: number of cpus per db shard
        :type db_cpus: int, optional
        :param db_tpq: device threads per database shard
        :type db_tpq: int, optional
        :param db_port: database port
        :type db_port: int, optional
        :param batch_size: batch size for inference
        :type batch_size: int, optional
        :param device: CPU or GPU
        :type device: str, optional
        :param model: path to serialized model
        :type model: str, optional
        :param clients_per_node: list of ranks per node
        :type clients_per_node: list[int], optional
        :param client_nodes: list of client node counts
        :type client_nodes: list[int], optional
        """

        # obtain allocation for the inference client program
        allocation = slurm.get_allocation(nodes=max(client_nodes),
                                          time="10:00:00",
                                          options={"exclusive": None,
                                                   "constraint": '[SK48*107&SK56*53]'})


        data_locations = []
        # create permutations of each input list and run each of the permutations
        # as a single inference scaling test
        perms = list(product(client_nodes, clients_per_node, db_nodes))
        for perm in perms:

            # start a new database each time
            db_node_count = perm[2]
            db = start_database(db_port, db_node_count, db_cpus, db_tpq)
            address = ":".join((db.hosts[0], str(db.ports[0])))
            logger.info("Orchestrator Database created and running")

            # set models and script
            setup_resnet(model, device, batch_size, address, cluster=bool(db_node_count>1))
            logger.info("PyTorch Model and Script in Orchestrator")


            # setup a an instance of the C++ driver and start it
            infer_session = create_resnet_inference_session(perm[0], # nodes
                                                            perm[1], # tasks
                                                            perm[2], # db nodes
                                                            allocation
                                                            )
            exp.start(infer_session, summary=True)

            # confirm scaling test run successfully
            status = exp.get_status(infer_session)
            if status[0] != constants.STATUS_COMPLETED:
                logger.error(f"ERROR: One of the scaling tests failed {infer_session.name}")
                break

            # get the statistics from the run
            post = create_post_process(infer_session.path,
                                       infer_session.name,
                                       allocation)
            data_locations.append((infer_session.path, infer_session.name, perm))
            exp.start(post)

            # release DB batch job
            exp.stop(db)

        try:
            # get the statistics from post processing
            # and add to the experiment summary
            stats_df = get_stats(data_locations)
            summary_df = exp.summary()
            final_df = pd.merge(summary_df, stats_df, on="Name")

            # save experiment info
            print(final_df)
            final_df.to_csv(exp.name + ".csv")

        except Exception:
            print("Could not preprocess results")

        slurm.release_allocation(allocation)


def start_database(port, nodes, cpus, tpq):
    """Create and start the Redis database for the scaling test

    This function launches the redis database instances as a
    Sbatch script.

    :param port: port number of database
    :type port: int
    :param nodes: number of database nodes
    :type nodes: int
    :param cpus: number of cpus per node
    :type cpus: int
    :param tpq: number of threads per queue
    :type tpq: int
    :return: orchestrator instance
    :rtype: Orchestrator
    """
    db = SlurmOrchestrator(port=port,
                            db_nodes=nodes,
                            batch=True,
                            threads_per_queue=tpq)
    db.set_cpus(cpus)
    db.set_walltime("1:00:00")
    db.set_batch_arg("exclusive", None)
    db.set_batch_arg("C", "P100") # specific to our testing machines; request GPU nodes
    exp.generate(db)
    exp.start(db)
    return db

def setup_resnet(model, device, batch_size, address, cluster=True):
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
    client.set_model_from_file("resnet_model",
                                model,
                                "TORCH",
                                device,
                                batch_size)
    client.set_script_from_file("resnet_script",
                                "../imagenet/data_processing_script.txt",
                                device)

def create_resnet_inference_session(nodes, tasks, db, allocation):
    """Create a inference session using the C++ driver with the SmartRedis client

    :param nodes: number of nodes for the client driver
    :type nodes: int
    :param tasks: number of tasks for each client
    :type tasks: int
    :param db: number of database nodes
    :type db: int
    :param allocation: client allocation id
    :type allocation: str
    :return: created Model instance
    :rtype: Model
    """
    srun = SrunSettings("./build/run_resnet_inference", alloc=allocation)
    srun.set_nodes(nodes)
    srun.set_tasks_per_node(tasks)

    name = "-".join(("infer-sess", str(nodes), str(tasks), str(db)))
    model = exp.create_model(name, srun)
    model.attach_generator_files(to_copy=["../imagenet/cat.raw", "./process_results.py"])
    exp.generate(model, overwrite=True)
    return model

def create_post_process(model_path, name, allocation):
    """Create a Model to post process the inference results

    :param model_path: path to model output data
    :type model_path: str
    :param name: name of the model
    :type name: str
    :param allocation: client allocation
    :type allocation: str
    :return: created post processing model
    :rtype: Model
    """

    exe_args = f"process_results.py --path={model_path} --name={name}"
    srun = SrunSettings("python", exe_args=exe_args, alloc=allocation)
    srun.set_nodes(1)
    srun.set_tasks(1)

    pp_name = "-".join(("post", name))
    post_process = exp.create_model(pp_name, srun, path=model_path)
    return post_process

def get_stats(data_locations):
    """Compile inference results into pandas dataframe

    :param data_locations: path, name, (nodes, tasks, db nodes)
    :type data_locations: tuple(str, str, tuple(int, int, int))
    :return: dataframe
    :rtype: pd.DataFrame
    """
    all_data = None
    for data_path, name, job_info in data_locations:
        data_path = osp.join(data_path, name + ".csv")
        data = pd.read_csv(data_path)
        data = data.drop("Unnamed: 0", axis=1)

        # add node and task information
        data["nodes"] = job_info[0]
        data["tasks"] = job_info[1]
        data["db_size"] = job_info[2]

        if not isinstance(all_data, pd.DataFrame):
            all_data = data
        else:
            all_data = pd.concat([all_data, data])
    return all_data


if __name__ == "__main__":
    import fire
    fire.Fire(SlurmScalingTests())
