import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from driverprocessresults.scaling_plotter import *
import sys

from pathlib import Path
from statistics import median
from configparser import ConfigParser

from smartsim.log import get_logger, log_to_file
logger = get_logger("Scaling Tests")


class ProcessResults:
    def process_scaling_results(self, 
                                scaling_results_dir="throughput-standard-scaling", 
                                plot_type="database_nodes",
                                overwrite=True):
            """Create a results directory with performance data and plots
            With the overwrite flag turned off, this function can be used
            to build up a single csv with the results of runs over a long
            period of time.
            :param scaling_results_dir: directory to create results from
            :type scaling_results_dir: str, optional
            :param plot: flag to plot against in process results
            :type plot: str
            :param overwrite: overwrite any existing results
            :type overwrite: bool, optional
            """
            dataframes = []
            final_stat_dir = "results" / Path(scaling_results_dir) / "stats"
            run_folders = os.listdir("results" / Path(scaling_results_dir))
            logger.debug(f"Running with final_stat_dir: {final_stat_dir}")
            logger.debug(f"Running with run_folders: {run_folders}")
            session_folders = []
            run_list = []
            for run_folder in run_folders:
                if 'run' in run_folder:
                    session_folders += ["results/" + scaling_results_dir + "/" + run_folder + 
                                        "/" + d for d in os.listdir("results/" + scaling_results_dir + "/" + run_folder) if "sess" in d]
                    run_list += [Path("results") / scaling_results_dir / run_folder]
                    logger.debug(f"Running with run_list: {run_list} for run_folder: {run_folder}")
            try:
                # write csv each so this function is idempotent
                # csv's will not be written if they are already created
                #tqdm creates a smart progress bar for the loops
                # for session_folder in tqdm(session_folders, desc="Processing scaling results...", ncols=80):
                #     try:
                #         self._create_run_csv(session_folder, delete_previous=overwrite)
                #         logger.debug(f"csv created for session_folder: {session_folder}")
                #     # want to catch all exceptions and skip runs that may
                #     # not have completed or finished b/c some reason i.e. node failure
                #     except Exception as e:
                #         logger.warning(f"Skipping {session_folder} could not process results")
                #         logger.error(e)
                #         continue
                #collect all written csv into dataframes to concat
                for run in tqdm(run_list, desc="Creating scaling plots...", ncols=80):
                    try:
                        scaling_plotter(run, scaling_results_dir, plot_type)
                        logger.debug(f"Data read and saved for: {run}")
                    # want to catch all exceptions and skip runs that may
                    # not have completed or finished b/c some reason i.e. node failure
                    except Exception as e:
                        logger.warning(f"Skipping {run} in {scaling_results_dir}: could not process results")
                        logger.error(e)
                        continue
                # for session in tqdm(session_folders, desc="Collecting scaling results...", ncols=80):
                #     try:
                #         session_name = os.path.basename(session)
                #         split = os.path.dirname(session)
                #         run_name = os.path.basename(split)
                #         stats_path = os.path.join(final_stat_dir, run_name, session_name, session_name + ".csv")
                #         run_df = pd.read_csv(str(stats_path))
                #         dataframes.append(run_df)
                #         logger.debug(f"Results collected for session: {session}")
                    
                #     # catch all and skip for reason listed above
                #     except Exception as e:
                #         logger.warning(f"Skipping path {session}: could not read results csv")
                #         logger.error(e)
                #         continue
                # final_df = pd.concat(dataframes, join="outer")
                # exp_name = os.path.basename(scaling_results_dir)
                # csv_path = final_stat_dir / f"{exp_name}-{get_date()}.csv"
                # final_df.to_csv(str(csv_path))

            except Exception:
                logger.error(f"Could not preprocess results for {scaling_results_dir}")
                raise

    @classmethod
    def _create_run_csv(cls, session_path, delete_previous=False, verbose=False):
        session_name = os.path.basename(session_path)
        split = os.path.dirname(session_path)
        run_name = os.path.basename(split)
        all_stats_dir = cls._create_stats_dir(session_path)
        session_stats_dir = all_stats_dir / run_name / session_name
        logger.debug(f"Running with session_name: {session_name}")
        logger.debug(f"Running with run_name: {run_name}")
        logger.debug(f"Running with session stats dir: {session_stats_dir}")
        logger.debug(f"Running with all stats dir: {all_stats_dir}")
        if delete_previous and session_stats_dir.is_dir():
            shutil.rmtree(session_stats_dir)
        if not session_stats_dir.is_dir():
            os.makedirs(session_stats_dir)
            function_times = {}
            files = os.listdir(Path(session_path))
            logger.debug("Beginning to loop through files")
            for file in files: 
                if '.csv' in file:
                    fp = os.path.join(session_path, file)
                    #opening csv file
                    with open(fp) as f:
                        #scan over all lines in file
                        for i, line in enumerate(f):
                            vals = line.split(',')
                            if vals[1] in function_times.keys():
                                function_times[vals[1]].append(float(vals[2]))
                            else:
                                function_times[vals[1]] = [float(vals[2])]
                else:
                    if verbose:
                        print(file)
            logger.debug("Finished to loop through files")
            if verbose:
                print('Min {0}'.format(min(function_times['client()'])))
                print('Max {0}'.format(max(function_times['client()'])))
            try:
                if "run_model" in function_times:
                    logger.debug("Run model started")
                    if verbose:
                        num_run = len(function_times['run_model'])
                        print(f'there are {num_run} values in the run_model entries')
                        #is it neccessary to print all run_model values?
                    cls._make_hist_plot(function_times['run_script'], 'run_script()', 'run_script.pdf', session_stats_dir)
                    cls._make_hist_plot(function_times['run_model'], 'run_model()', 'run_model.pdf', session_stats_dir)
                    logger.debug("Run model completed")
                function_types = ["put_tensor", "unpack_tensor", "get_list", "main()"] 
                for function in function_types:
                    if function in function_times:
                        logger.debug(f"{function} started")
                        cls._make_hist_plot(function_times[function], function, function + ".pdf", session_stats_dir)
                        logger.debug(f"{function} completed")

            except KeyError as e:
                raise KeyError(f'{e} not found in function_times for run {session_name}')
            data = cls._make_stats(session_path, function_times)
            data_df = pd.DataFrame(data, index=[0])
            file_name = session_stats_dir / ".".join((session_name, "csv"))
            data_df.to_csv(file_name)
    
    @staticmethod
    def _make_hist_plot(data, title, fname, session_stats_dir):
        x = plt.hist(data, color = 'blue', edgecolor = 'black', bins = 500)
        plt.title(title)                                   
        plt.xlabel('Time (s)')
        plt.ylabel('MPI Ranks')
        med = median(data)
        min_ylim, max_ylim = plt.ylim()
        plt.axvline(med, color='red', linestyle='dashed', linewidth=1)
        plt.text(med, max_ylim*0.9, '   Median: {:.2f}'.format(med))
        # save the figure in the result dir
        file_path = Path(session_stats_dir) / fname
        plt.savefig(file_path)
        plt.clf() #Clear the current figure

    @staticmethod
    def _make_stats(session_path, timing_dict):
        data = ProcessResults._read_run_config(session_path)
        for k, v in timing_dict.items():
            if len(v) > 1:
                array = np.array(v)
                arr_min = np.min(array)
                arr_max = np.max(array)
                arr_mean = np.mean(array)

                data["_".join((k, "min"))] = arr_min
                data["_".join((k, "mean"))] = arr_mean
                data["_".join((k, "max"))] = arr_max
            else:
                data[k] = float(v[0])
        return data

    @staticmethod
    def _create_stats_dir(session_path):
        stats_dir = Path(session_path).parent.parent / "stats"
        if not stats_dir.is_dir():
            stats_dir.mkdir()
        return stats_dir

    @staticmethod
    def _read_run_config(run_path):
        run_path = Path(run_path) / "run.cfg"
        config = ConfigParser()
        config.read(run_path)
        data = {
            "name": config["run"]["name"],
            "date": config["run"]["date"],
            "db": config["run"]["db"],
            "smartsim": config["run"]["smartsim_version"],
            "smartredis": config["run"]["smartredis_version"]
        }
        for k, v in config["attributes"].items():
            data[k] = v
        return data