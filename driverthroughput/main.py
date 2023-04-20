import fire
from utils import *

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")

class Throughput:
    #just throwing data in to db and requesting it back
    def throughput_standard(self,
                           exp_name="throughput-standard-scaling",
                           launcher="auto",
                           run_db_as_batch=True,
                           node_feature={}, #dont need GPU
                           db_hosts=[],
                           db_nodes=[12], #shards
                           db_cpus=36,
                           db_port=6780,
                           net_ifname="ipogif0",
                           clients_per_node=[32],
                           client_nodes=[128, 256, 512],
                           iterations=100,
                           tensor_bytes=[1024, 8192, 16384, 32768, 65536, 131072,
                                         262144, 524288, 1024000, 2048000, 4096000]):

        """Run the throughput scaling tests

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
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list, optional
        :param client_nodes: number of compute nodes to use for the synthetic scaling app
        :type client_nodes: list, optional
        :param iterations: number of put/get loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        """
        logger.info("Starting throughput scaling tests")
        logger.info(f"Running with database backend: {get_db_backend()}")
        logger.info(f"Running with launcher: {launcher}")

        exp = create_folder(exp_name, launcher)

        for db_node_count in db_nodes:

            # start the database only once per value in db_nodes so all permutations
            # are executed with the same database size without bringin down the database
            db = start_database(exp,
                                node_feature,
                                db_port,
                                db_node_count,
                                db_cpus,
                                None, # not setting threads per queue in throughput tests
                                net_ifname,
                                run_db_as_batch,
                                db_hosts)


            perms = list(product(client_nodes, clients_per_node, tensor_bytes))
            for perm in perms:
                c_nodes, cpn, _bytes = perm

                # setup a an instance of the C++ driver and start it
                throughput_session = self.create_throughput_session(exp,
                                                               c_nodes,
                                                               cpn,
                                                               db_node_count,
                                                               db_cpus,
                                                               iterations,
                                                               _bytes)
                exp.start(throughput_session, summary=True)

                # confirm scaling test run successfully
                stat = exp.get_status(throughput_session)
                if stat[0] != status.STATUS_COMPLETED: # might need to add error check to inference tests
                    logger.error(f"ERROR: One of the scaling tests failed {throughput_session.name}")

            # stop database after this set of permutations have finished
            exp.stop(db)
    
    @classmethod
    def create_throughput_session(cls,
                              exp,
                              nodes,
                              tasks,
                              db_nodes,
                              db_cpus,
                              iterations,
                              _bytes):
        """Create a Model to run a throughput scaling session

        :param exp: Experiment object for this test
        :type exp: Experiment
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
        settings = exp.create_run_settings("./cpp-throughput/build/throughput", str(_bytes))
        cluster = 1 if db_nodes > 1 else 0
        settings.set_tasks(nodes * tasks)
        settings.set_tasks_per_node(tasks)
        settings.update_env({
            "SS_ITERATIONS": str(iterations),
            "SS_CLUSTER": cluster
            #need to update
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(nodes)

        name = "-".join((
            "throughput-sess",
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(db_nodes),
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
                    tensor_bytes=_bytes)
        return model
    
    def throughput_colocated(self,
                           exp_name="throughput-colocated-scaling",
                           launcher="auto",
                           node_feature={}, #dont need GPU
                           nodes=[4], 
                           #no db_tpq? it dont matter
                           #no device? always will be GPU
                           #no num_devices? NO bc its used for resnet
                           #no pin app cpus? YES ADD IT : launch this program on these cores and dont let it move
                           #48 compute cores (compute core on the processor which is on the node), launching 24 instances of the program, 
                           db_cpus=[2], #how many cpus each db is getting
                           db_port=6780,
                           net_ifname="lo", #is this the correct netif name
                           clients_per_node=[3], #how many cpus the app is getting per node
                           pin_app_cpus=[False],
                           #client_nodes=[128, 256, 512] -> used for both through and inference std, its in inf colo but not an arg
                           iterations=100, #iterations is not a flag for inference but it is for throughput
                           tensor_bytes=[1024, 8192, 16384, 32768, 65536, 131072,
                                         262144, 524288, 1024000, 2048000, 4096000]): #size of the tensor
# 48 clients OR 48 apps
# 48 cores OR 48 cpus
# clients are consuming a full core
# db is also tyeing to consume a full 48 cores
# why do we need to specify both db_cpus and clients_per_node for colo


        """Run the throughput scaling tests

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
        :param db_port: port to use for the database
        :type db_port: int, optional
        :param net_ifname: network interface to use i.e. "ib0" for infiniband or
                           "ipogif0" aries networks
        :type net_ifname: str, optional
        :param clients_per_node: client tasks per compute node for the synthetic scaling app
        :type clients_per_node: list, optional
        :param client_nodes: number of compute nodes to use for the synthetic scaling app
        :type client_nodes: list, optional
        :param iterations: number of put/get loops run by the applications
        :type iterations: int
        :param tensor_bytes: list of tensor sizes in bytes
        :type tensor_bytes: list[int], optional
        """
        logger.info("Starting colocated throughput scaling tests") #check punctuation of colocated
        logger.info(f"Running with database backend: {get_db_backend()}")
        logger.info(f"Running with launcher: {launcher}") # do we keep launcher? or add launcher to inference?

        #why not check model here?
        exp = create_folder(exp_name, launcher)

        perms = list(product(nodes, clients_per_node, db_cpus, tensor_bytes, pin_app_cpus)) #client_nodes
        for perm in perms:
            c_nodes, cpn, dbc, _bytes, pin_app = perm #might need to add

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
                                                            net_ifname)#network interface that the db will be listing over
            exp.start(throughput_session, summary=True) #block = true?

            # confirm scaling test run successfully
            stat = exp.get_status(throughput_session)
            if stat[0] != status.STATUS_COMPLETED:
                logger.error(f"ERROR: One of the scaling tests failed {throughput_session.name}")

        # stop database after this set of permutations have finished
        exp.stop(db)
    
    @classmethod
    def _create_colocated_throughput_session(cls,
                              exp,
                              node_feature,
                              nodes,
                              tasks, #client per node
                              db_cpus,
                              db_port,
                              iterations,
                              _bytes,
                              pin_app_cpus,
                              net_ifname):
        """Create a Model to run a throughput scaling session

        :param exp: Experiment object for this test
        :type exp: Experiment
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
        settings = exp.create_run_settings("./cpp-throughput/build/throughput", str(_bytes), run_args=node_feature)
        settings.set_tasks(nodes * tasks)
        settings.set_tasks_per_node(tasks)
        settings.update_env({
            "SS_ITERATIONS": str(iterations),
            "SS_CLUSTER": "0"
            #need to update
            #will need to look at the cpp throughput 
        })
        # TODO replace with settings.set("nodes", condition==exp.launcher=="slurm")
        if exp._launcher == "slurm":
            settings.set_nodes(nodes)

        name = "-".join((
            "throughput-sess-colo", #possibly change name here
            "N"+str(nodes),
            "T"+str(tasks),
            "DBN"+str(nodes),
            "ITER"+str(iterations),
            "TB"+str(_bytes),
            get_uuid()
            ))
        # 12 nodes for the app
        # 24 clients p N
        # 16 nodes given to db
        #shard: over multiple nodes, each shard is a portion of one db
        model = exp.create_model(name, settings)
        # add co-located database
        
        #model launched on multiple nodes - redis database setup to speak with the mdoel on each node
        #much faster bc not going off the network
        model.colocate_db(port=db_port, #this is the model, colocating the db with the model: the db belongs to the model object and not the entire exp
                        db_cpus=db_cpus,
                        ifname=net_ifname,
                        limit_app_cpus=pin_app_cpus,
                        #threads_per_queue=db_tpq,
                        # turning this to true can result in performance loss
                        # on networked file systems(many writes to db log file)
                        debug=True,
                        loglevel="notice")
        exp.generate(model, overwrite=True)#creates the run directory, on the file system
        write_run_config(model.path,
                    colocated=1,
                    pin_app_cpus=int(pin_app_cpus),
                    client_total=tasks*nodes,
                    client_per_node=tasks,
                    client_nodes=nodes,
                    database_nodes=nodes,
                    database_cpus=db_cpus,
                    iterations=iterations,
                    tensor_bytes=_bytes)
        return model