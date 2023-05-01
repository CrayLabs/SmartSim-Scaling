import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from statistics import median
from configparser import ConfigParser


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

        # a data structure that organizes data into a 2-dimensional table of rows and columns
        dataframes = []
        # create folder path driver-scaling-test/results and assigns to var
        result_dir = Path(scaling_dir) / "results" #CHANGE PATH TO NEW RESULTS HOLDER
        # scans over file names w/ "sess" within the driver/results 
        runs = [d for d in os.listdir(scaling_dir) if "sess" in d]

        try:
            # write csv each so this function is idempotent
            # csv's will not be written if they are already created
            #tqdm creates a smart progress bar for the loops
            for run in tqdm(runs, desc="Processing scaling results...", ncols=80): #QUESTION: ncols: hardcoded?
                try:
                    #QUESTION: run var? Where is it assigned?
                    run_path = Path(scaling_dir) / run
                    #create_run_csv is called 80 time per sess file
                    create_run_csv(run_path, delete_previous=overwrite)
                # want to catch all exceptions and skip runs that may
                # not have completed or finished b/c some reason i.e. node failure
                except Exception as e:
                    logger.warning(f"Skipping {run} could not process results")
                    logger.error(e)
                    continue

            # collect all written csv into dataframes to concat
            for run in tqdm(runs, desc="Collecting scaling results...", ncols=80): #QUESTION: ncols: hardcoded?
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


def create_run_csv(run_path, delete_previous=True, verbose=False):

    #grabbing the sess file name that was passed in
    run_name = os.path.basename(run_path)
    #creates a folder named results within the scaling folder and assigns to all_results_dir
    all_results_dir = _create_results_dir(run_path)
    #assigns scaling/results/sess to result_dir
    result_dir = all_results_dir / run_name

    #QUESTION: this is checking to see if basically the sess file has already been accounted for? Bc someone can recall process_results multiple times and add sess
    #but they have to specify if they want the previous deleted and redone or skipped via delete_previous bool
    if delete_previous and result_dir.is_dir():
        shutil.rmtree(result_dir)

    #if the file DNE, then we created scaling/results/sess
    #if file exists, we skip over if statement
    if not result_dir.is_dir():
        result_dir.mkdir()

        #dict
        function_times = {}
        #listing all the files in the sess folder
        files = os.listdir(str(run_path))

        #loops through all files in sess folder
        for file in files: 
            if '.csv' in file:
                #creating fp=sess/file.csv
                fp = os.path.join(run_path, file)
                #opening csv file
                with open(fp) as f:
                    #scan over all lines in file
                    for i, line in enumerate(f):
                        #QUESTION: ask for an explanation
                        #does this created 2 keys and assigns all values to the 2 keys?
                        vals = line.split(',')
                        if vals[1] in function_times.keys():
                            function_times[vals[1]].append(float(vals[2]))
                        else:
                            function_times[vals[1]] = [float(vals[2])]
            else:
                if verbose: #QUESTION: what is the meaning? right now verbose will always be false?
                    print(file) #why do we print file()?

        #QUESTION: explanation?
        if verbose:
            print('Min {0}'.format(min(function_times['client()'])))
            print('Max {0}'.format(max(function_times['client()'])))
        try:
            # throughput tests do not have a 'run_model` or `run_script` timing
            # scans whole func for "run_model"
            if "run_model" in function_times:
                #will always be false and therefore skip this?
                if verbose:
                    #Question: explanation?
                    num_run = len(function_times['run_model'])
                    print(f'there are {num_run} values in the run_model entries')
                _make_hist_plot(function_times['run_script'], 'run_script()', 'run_script.pdf', result_dir)
                _make_hist_plot(function_times['run_model'], 'run_model()', 'run_model.pdf', result_dir)

            _make_hist_plot(function_times['client()'], 'client()', 'client_constructor_dist.pdf', result_dir)

            if "put_tensor" in function_times:
                _make_hist_plot(function_times['put_tensor'], 'put_tensor()', 'put_tensor.pdf', result_dir)

            if "unpack_tensor" in function_times:
                _make_hist_plot(function_times['unpack_tensor'], 'unpack_tensor()', 'unpack_tensor.pdf', result_dir)

            if "get_list" in function_times:
                _make_hist_plot(function_times['get_list'], 'get_list()', 'get_list.pdf', result_dir)

            _make_hist_plot(function_times['main()'], 'main()', 'main.pdf', result_dir)
        except KeyError as e:
            raise KeyError(f'{e} not found in function_times for run {run_name}')

        # get stats
        data = _make_stats(run_path, function_times)    # get stats
        data_df = pd.DataFrame(data, index=[0])
        file_name = result_dir / ".".join((run_name, "csv"))
        data_df.to_csv(file_name)

#QUESTION: is this fn making a hist per dict key?
def _make_hist_plot(data, title, fname, result_dir):
    #creating histogram, assigning to var x
    x = plt.hist(data, color = 'blue', edgecolor = 'black', bins = 500)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('MPI Ranks')
    med = median(data)
    min_ylim, max_ylim = plt.ylim() # get or set the y-limits of the current axes
    plt.axvline(med, color='red', linestyle='dashed', linewidth=1) #used to generate vertical lines along the plot's dimensions
    plt.text(med, max_ylim*0.9, '   Median: {:.2f}'.format(med)) #add a text to the axes at location x, y in data coordinates

    # save the figure in the result dir
    file_path = Path(result_dir) / fname #add the hist as a pdf into the results folder
    plt.savefig(file_path)
    plt.clf() #Clear the current figure

def _make_stats(run_path, timing_dict):
    data = read_run_config(run_path)
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
            data[k] = np.float(v[0])
    return data

def _create_results_dir(run_path):

    results_dir = Path(run_path).parent / "results"
    if not results_dir.is_dir():
        results_dir.mkdir()
    return results_dir

def read_run_config(run_path):
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

