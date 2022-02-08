import pandas as pd
import os.path as osp
from itertools import product


from smartsim import Experiment, status
from smartredis import Client

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")

log_to_file("scaling.log")

exp = Experiment(name="inference-scaling-tests", launcher="auto")

class SmartSimScalingTests:
    """Create a battery of scaling tests to launch
    on a slurm system, launch them, and record the
    performance results.
    """

    def __init__(self):
        logger.info("Starting Scaling Tests")

    def resnet(self,
               db_nodes=[16],
               db_cpus=[2, 4],
               db_tpq=[1],
               db_port=6780,
               batch_size=16,
               device="GPU",
               colocated=True,
               model="../imagenet/resnet50.pt",
               clients_per_node=[14, 16, 18],
               client_nodes=[16]):
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

        data_locations = []

        # create permutations of each input list and run each of the permutations
        # as a single inference scaling test
        perms = list(product(client_nodes, clients_per_node, db_nodes, db_cpus, db_tpq))
        for perm in perms:

            # set batch size to client count
            batch_size = perm[1]

            # start a new database each time
            if not colocated:
                db_node_count = perm[2]
                db = start_database(db_port,
                                    db_node_count,
                                    perm[3], # db_cpus
                                    perm[4]) # db_tpq
                logger.info("Orchestrator Database created and running")


                # setup a an instance of the C++ driver and start it
                infer_session = create_inference_session(perm[0], # nodes
                                                         perm[1], # tasks
                                                         perm[2], # db nodes
                                                         perm[3], # db cpus
                                                         perm[4]) # db tpq

                # only need 1 address to set model
                address = db.get_address()[0]
                setup_resnet(model, device, batch_size, address, cluster=bool(db_node_count>1))
                logger.info("PyTorch Model and Script in Orchestrator")

                exp.start(infer_session, block=True, summary=True)

                # kill the database each time so we get a fresh start
                exp.stop(db)

            else:
                infer_session = create_colocated_inference_session(perm[0], # nodes
                                                                   perm[1], # tasks
                                                                   perm[2],  # db nodes
                                                                   perm[3],  # db cpus
                                                                   perm[4],  # db tpq
                                                                   db_port,
                                                                   batch_size,
                                                                   device
                                                                   )
                exp.start(infer_session, block=True, summary=True)

            # confirm scaling test run successfully
            stat = exp.get_status(infer_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"ERROR: One of the scaling tests failed {infer_session.name}")
                break

            # run process_results.py to collect the results for this
            # single run into a CSV
            post = create_post_process(infer_session.path,
                                       infer_session.name)
            data_locations.append((infer_session.path, infer_session.name, perm))
            exp.start(post)

        try:
            # get the statistics from all post processing jobs
            # and add to the experiment summary
            stats_df = get_stats(data_locations)
            summary_df = get_run_df()
            final_df = pd.merge(summary_df, stats_df, on="Name")

            # save experiment info
            print(final_df)
            final_df.to_csv(exp.name + ".csv")

        except Exception as e:
            print("Could not preprocess results")
            print(str(e))


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
    db = exp.create_database(port=port,
                            db_nodes=nodes,
                            batch=True,
                            interface="ipogif0",
                            threads_per_queue=tpq)
    db.set_batch_arg("exclusive", None)
    db.set_batch_arg("C", "P100")
    db.set_cpus(cpus)
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

def create_inference_session(nodes, tasks, db_nodes, db_cpus=1, db_tpq=1):
    """Create a inference session using the C++ driver with the SmartRedis client

    :param nodes: number of nodes for the client driver
    :type nodes: int
    :param tasks: number of tasks for each client
    :type tasks: int
    :param db: number of database nodes
    :type db: int
    :return: created Model instance
    :rtype: Model
    """
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
                                          "./process_results.py",
                                          "../imagenet/resnet50.pt",
                                          "../imagenet/data_processing_script.txt"])
    exp.generate(model, overwrite=True)
    return model


def create_colocated_inference_session(nodes,
                                       tasks,
                                       db_nodes,
                                       db_cpus,
                                       db_tpq,
                                       db_port,
                                       batch_size,
                                       device):
    run_settings = exp.create_run_settings("./build/run_resnet_inference")
    run_settings.set_nodes(nodes)
    run_settings.set_tasks_per_node(tasks)
    run_settings.update_env({
        "SS_SET_MODEL": "1",  # set the model from the scaling application
        "SS_CLUSTER": "0",     # never cluster for colocated db
        "SS_BATCH_SIZE": str(batch_size),
        "SS_DEVICE": device,
        "SS_CLIENT_COUNT": str(tasks)
    })

    name = "-".join((
        "infer-sess-colo",
        "N"+str(nodes),
        "T"+str(tasks),
        "DBN"+str(db_nodes),
        "DBC"+str(db_cpus),
        "DBTPQ"+str(db_tpq)
        ))
    model = exp.create_model(name, run_settings)
    model.attach_generator_files(to_copy=["../imagenet/cat.raw",
                                          "./process_results.py",
                                          "../imagenet/resnet50.pt",
                                          "../imagenet/data_processing_script.txt"])

    # add co-located database
    model.colocate_db(port=db_port,
                      db_cpus=db_cpus,
                      limit_app_cpus=True,
                      ifname="ipogif0",
                      threads_per_queue=db_tpq,
                      # turning this to true can result in performance loss
                      # on networked file systems(many writes to db log file)
                      debug=False,
                      loglevel="notice")
    exp.generate(model, overwrite=True)
    return model


def create_post_process(model_path, name):
    """Create a Model to post process the inference results

    :param model_path: path to model output data
    :type model_path: str
    :param name: name of the model
    :type name: str
    :return: created post processing model
    :rtype: Model
    """

    exe_args = f"process_results.py --path={model_path} --name={name}"
    run_settings = exp.create_run_settings("python", exe_args=exe_args)
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    pp_name = "-".join(("post", name))
    post_process = exp.create_model(pp_name, run_settings, path=model_path)
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


def get_run_df():
    index = 0
    df = pd.DataFrame(
        columns=[
            "Name",
            "Entity-Type",
            "JobID",
            "RunID",
            "Time",
            "Status",
            "Returncode",
        ]
    )
    for job in exp._control.get_jobs().values():
        for run in range(job.history.runs + 1):
            df.loc[index] = [
                job.entity.name,
                job.entity.type,
                job.history.jids[run],
                run,
                job.history.job_times[run],
                job.history.statuses[run],
                job.history.returns[run],
            ]
            index += 1
    return df


if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
