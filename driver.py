import sys
from inference.main import Inference



class SmartSimScalingTests(Inference):
    def driver():
        if sys.argv[1] == "inference_standard":
            return inference_standard()


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
    #     return 0


    # def throughput_scaling(self,
    #                        exp_name="standard-throughput-scaling",
    #                        launcher="auto",
    #                        run_db_as_batch=True,
    #                        batch_args={},
    #                        db_hosts=[],
    #                        db_nodes=[12],
    #                        db_cpus=36,
    #                        db_port=6780,
    #                        net_ifname="ipogif0",
    #                        clients_per_node=[32],
    #                        client_nodes=[128, 256, 512],
    #                        iterations=100,
    #                        tensor_bytes=[1024, 8192, 16384, 32768, 65536, 131072,
    #                                      262144, 524288, 1024000, 2048000, 4096000]):

    #     return 0

    # def aggregation_scaling(self,
    #                         exp_name="aggregation-scaling",
    #                         launcher="auto",
    #                         run_db_as_batch=True,
    #                         batch_args={},
    #                         db_hosts=[],
    #                         db_nodes=[12],
    #                         db_cpus=36,
    #                         db_port=6780,
    #                         net_ifname="ipogif0",
    #                         clients_per_node=[32],
    #                         client_nodes=[128, 256, 512],
    #                         iterations=20,
    #                         tensor_bytes=[1024, 8192, 16384, 32769, 65538,
    #                                       131076, 262152, 524304, 1024000,
    #                                       2048000],
    #                         tensors_per_dataset=[1,2,4],
    #                         client_threads=[1,2,4,8,16,32],
    #                         cpu_hyperthreads=2):

    #     return 0

    # def aggregation_scaling_python(self,
    #                         exp_name="aggregation-scaling-py-db",
    #                         launcher="auto",
    #                         run_db_as_batch=True,
    #                         batch_args={},
    #                         db_hosts=[],
    #                         db_nodes=[12],
    #                         db_cpus=36,
    #                         db_port=6780,
    #                         net_ifname="ipogif0",
    #                         clients_per_node=[32],
    #                         client_nodes=[128, 256, 512],
    #                         iterations=20,
    #                         tensor_bytes=[1024, 8192, 16384, 32769, 65538,
    #                                       131076, 262152, 524304, 1024000,
    #                                       2048000],
    #                         tensors_per_dataset=[1,2,4],
    #                         client_threads=[1,2,4,8,16,32],
    #                         cpu_hyperthreads=2):
    #     return 0

    # def aggregation_scaling_python_fs(self,
    #                         exp_name="aggregation-scaling-py-fs",
    #                         launcher="auto",
    #                         clients_per_node=[32],
    #                         client_nodes=[128, 256, 512],
    #                         iterations=20,
    #                         tensor_bytes=[1024, 8192, 16384, 32769, 65538,
    #                                       131076, 262152, 524304, 1024000,
    #                                       2048000],
    #                         tensors_per_dataset=[1,2,4],
    #                         client_threads=[1,2,4,8,16,32],
    #                         cpu_hyperthreads=2):
    #     return 0


    # def process_scaling_results(self, scaling_dir="inference-scaling", overwrite=True):
    #     """Create a results directory with performance data and plots

    #     With the overwrite flag turned off, this function can be used
    #     to build up a single csv with the results of runs over a long
    #     period of time.

    #     :param scaling_dir: directory to create results from
    #     :type scaling_dir: str, optional
    #     :param overwrite: overwrite any existing results
    #     :type overwrite: bool, optional
    #     """

    #     dataframes = []
    #     result_dir = Path(scaling_dir) / "results"
    #     runs = [d for d in os.listdir(scaling_dir) if "sess" in d]

    #     try:
    #         # write csv each so this function is idempotent
    #         # csv's will not be written if they are already created
    #         for run in tqdm(runs, desc="Processing scaling results...", ncols=80):
    #             try:
    #                 run_path = Path(scaling_dir) / run
    #                 create_run_csv(run_path, delete_previous=overwrite)
    #             # want to catch all exceptions and skip runs that may
    #             # not have completed or finished b/c some reason i.e. node failure
    #             except Exception as e:
    #                 logger.warning(f"Skipping {run} could not process results")
    #                 logger.error(e)
    #                 continue

    #         # collect all written csv into dataframes to concat
    #         for run in tqdm(runs, desc="Collecting scaling results...", ncols=80):
    #             try:
    #                 results_path = os.path.join(result_dir, run, run + ".csv")
    #                 run_df = pd.read_csv(str(results_path))
    #                 dataframes.append(run_df)
    #             # catch all and skip for reason listed above
    #             except Exception as e:
    #                 logger.warning(f"Skipping {run} could not read results csv")
    #                 logger.error(e)
    #                 continue

    #         final_df = pd.concat(dataframes, join="outer")
    #         exp_name = os.path.basename(scaling_dir)
    #         csv_path = result_dir / f"{exp_name}-{self.date}.csv"
    #         final_df.to_csv(str(csv_path))

    #     except Exception:
    #         logger.error("Could not preprocess results")
    #         raise

if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
