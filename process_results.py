import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from statistics import median
from configparser import ConfigParser


def create_run_csv(run_path, delete_previous=True, verbose=False):

    run_name = os.path.basename(run_path)
    all_results_dir = _create_results_dir(run_path)
    result_dir = all_results_dir / run_name

    if delete_previous and result_dir.is_dir():
        shutil.rmtree(result_dir)

    if not result_dir.is_dir():
        result_dir.mkdir()

        function_times = {}
        files = os.listdir(str(run_path))

        for file in files:
            if '.csv' in file:
                fp = os.path.join(run_path, file)
                with open(fp) as f:
                    for i, line in enumerate(f):
                        vals = line.split(',')
                        if vals[1] in function_times.keys():
                            function_times[vals[1]].append(float(vals[2]))
                        else:
                            function_times[vals[1]] = [float(vals[2])]
            else:
                if verbose:
                    print(file)

        if verbose:
            print('Min {0}'.format(min(function_times['client()'])))
            print('Max {0}'.format(max(function_times['client()'])))
        try:
            # throughput tests do not have a 'run_model` or `run_script` timing
            if "run_model" in function_times:
                if verbose:
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


def _make_hist_plot(data, title, fname, result_dir):
    x = plt.hist(data, color = 'blue', edgecolor = 'black', bins = 500)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('MPI Ranks')
    med = median(data)
    min_ylim, max_ylim = plt.ylim()
    plt.axvline(med, color='red', linestyle='dashed', linewidth=1)
    plt.text(med, max_ylim*0.9, '   Median: {:.2f}'.format(med))

    # save the figure in the result dir
    file_path = Path(result_dir) / fname
    plt.savefig(file_path)
    plt.clf()

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

