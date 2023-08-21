import os
import matplotlib.pyplot as plt
import matplotlib
import gzip
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import numpy as np
from glob import glob
from tqdm.auto import tqdm
import matplotlib.patches as mpatches
import time
import sys
import traceback # give me the traceback
import configparser
import json
from pathlib import Path
import asyncio
from joblib import Parallel, delayed
from multiprocessing import Manager
from itertools import chain
import pprint
import math

from smartsim.log import get_logger, log_to_file
logger = get_logger("Plotter")
configs = []

class PlotResults:
    def _fast_flatten(cls, input_list):
        """Define a function to flatten large 2D lists quickly.
        """
        return list(chain.from_iterable(input_list))
    
    def _readCSV(cls, timing_file, config, frames):
        """Read in the Data as Pandas DataFrames
        """
        # NOTE: can't use "engine="pyarrow" because not all features are there
        tmp_df = pd.read_csv(timing_file, header=0, names=["rank", "function", "time"])
        for key, value in config._sections['attributes'].items():
            tmp_df[key] = value
        frames.append(tmp_df)

    def scaling_read_data(self, run_cfg_path, scaling_test_name):
        """Read performance results and create a dataframe.
        To mitigate performance runtime, outside code from 
        https://gist.github.com/TariqAHassan/fc77c00efef4897241f49
        e61ddbede9e?permalink_comment_id=2987243
        is implemented.
        :param run_cfg_path: directory to create plots from
        :type run_cfg_path: str
        :param scaling_test_name: name of scaling test your are plotting
        :type scaling_test_name: str
        """
        logger.debug("Entered plotter method")
        try:
            # creating a list that can be shared across memory
            frames = list()
            # read run.cfg to create columns in list
            for run_cfg in Path(run_cfg_path).rglob('run.cfg'):
                config = configparser.ConfigParser()
                config.read(run_cfg)
                configs.append(config)
            for config in tqdm(configs, desc="Processing configs...", ncols=80):
                timing_files = Path(config['run']['path']).glob('rank*.csv')
                # NOTE: setting n_jobs to -1 makes it use all available cpus
                timingFiles = tqdm(timing_files, desc="Processing timing files...", ncols=80)
                # reading timing files in parallel
                Parallel(n_jobs=-1, prefer="threads")(delayed(self._readCSV)(timing_file, config, frames) for timing_file in timingFiles)
            #construct a dictionary using the column names from one of the dataframes
            COLUMN_NAMES = frames[0].columns
            # construct a dictionary from the column names
            df_dict = dict.fromkeys(COLUMN_NAMES, [])
            logger.debug(f"columns were {COLUMN_NAMES}")
            #Iterate through the columns
            for col in COLUMN_NAMES:
                extracted = (frame[col] for frame in frames if col in frame.columns.tolist())
                df_dict[col] = self._fast_flatten(extracted)
            #produce the combined DataFrame
            df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
            logger.debug(f"df: {df}")
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_tb(e.__traceback__)
            traceback.print_exception(*exc_info)
        # write dataframe to file
        df.to_csv(Path("results/" + scaling_test_name + "/stats") / os.path.basename(run_cfg_path) / "dataframe.csv.gz", chunksize=100000, encoding='utf-8', index=False, compression='gzip')

    def scaling_plotter(run_cfg_path, scaling_test_name, var_input):
        """Create violin plots with performance data.
        :param run_cfg_path: directory to create plots from
        :type run_cfg_path: str
        :param scaling_test_name: name of scaling test your are plotting
        :type scaling_test_name: str
        :param var_input: plot on a specific flag
        :type var_input: str
        """
        df = pd.read_csv(Path("results/" + scaling_test_name + "/stats") / os.path.basename(run_cfg_path) / "dataframe.csv.gz")
        try:
            #save the results of the processing to a python pickle file
            #pass in the saved dataframe to the plotting part
            #palette = sns.set_palette("colorblind", color_codes=True)
            font = {'family' : 'sans',
                    'weight' : 'normal',
                    'size'   : 14}
            matplotlib.rc('font', **font)
            logger.debug("Dataframe created")
            plt.style.use('default') #plt.style.use("dark_background")
            client_total = [int(x) for x in df['client_total'].unique()]
            client_per_n = [int(x) for x in df['client_per_node'].unique()]
            if 'colo' in scaling_test_name:
                database_nodes = sorted([int(x) for x in df['client_nodes'].unique()])
            else:
                database_nodes = sorted([int(x) for x in df['database_nodes'].unique()])
            database_cpus = [int(x) for x in df['database_cpus'].unique()]
            client_nodes = [int(x) for x in df['client_nodes'].unique()]
            grid_spacing = np.min(np.diff(client_nodes))*(client_per_n[0])
            logger.debug(f"grid_spacing: {grid_spacing}")
            ordered_client_total = sorted(df['client_total'].unique())
            start = 48
            stop = ordered_client_total[len(ordered_client_total) - 1]
            logger.debug(f"Ordered client total: {ordered_client_total}")
            step = math.ceil((stop-start) / (len(ordered_client_total)))
            xticks = list(range(start, stop, step))
            logger.debug(f"xticks: {xticks}")
            function_names = df['function'].unique()
            languages = df['language'].unique()
            legend_entries = []
            var_list = sorted(df[var_input].unique())
            violin_opts = dict(     
                    showmeans = True, #will display mean  
                    showextrema = True, #will display extrema  
                    widths= grid_spacing/(len(database_nodes)*5)
                )
            #ranks = [node*threads for node in nnodes]
            for function_name in tqdm(function_names, desc="Processing function name...", ncols=80):
                #declare a figure and figsize is width, height in inches
                fig = plt.figure(figsize=[16,5]) #keep it constant since it is just plotting it - everything else is relative to the data
                #nrows = 1 ncols= 
                axs = fig.subplots(1,2,sharey=True)
                
                for lang_idx, language in tqdm(enumerate(languages), desc="Processing languages...", ncols=80):
                    language_df = df.groupby('language').get_group(language)
                    for idx, var in tqdm(enumerate(var_list), desc="Processing vars...", ncols=80):
                        #group by database number
                        var_df = language_df.groupby(var_input).get_group(var)
                        step2 = math.ceil((stop-start) / (len(ordered_client_total)))
                        logger.debug(f"step2: {step2}")
                        #group by function - take client total and time
                        function_df = var_df.groupby('function').get_group(function_name)[ ['client_total','time'] ]
                        #loop through client_total - assign times in data list
                        data = [function_df.groupby('client_total').get_group(client)['time'] for client in ordered_client_total]
                        new_xticks = []
                        # what we're doing here is offsetting xticks by 250 relative to idx
                        # (this prevents the graphs from stacking on top of one another)
                        #
                        for aidx, val in enumerate(xticks):
                            if len(var_list) > 1:
                                new_xticks.append(val + (200*idx) - 200)
                            else:
                                new_xticks.append(val)
                        plot = axs[lang_idx].violinplot(data, positions=new_xticks, **violin_opts)
                        [col.set_alpha(0.3) for col in plot["bodies"]]
                        props_dict = dict(color=plot["cbars"].get_color().flatten())
                        entry = plot["cbars"]
                        legend_entries.append(entry)    
                        means = [np.mean(function_df.groupby('client_total').get_group(client)['time']) for client in ordered_client_total]
                        logger.debug(f"MEANS: {means}\n")
                        axs[lang_idx].plot(new_xticks, means, ':', color=props_dict['color'], alpha=0.5) 

                    data_labels = [f"{var} {var_input}" for var in var_list]
                    axs[lang_idx].legend(legend_entries, data_labels, loc='upper left')
                    axs[lang_idx].set_xlabel('Number of Clients')
                    axs[lang_idx].set_title(language)
                    axs[lang_idx].set_xticks(xticks, labels=ordered_client_total, minor=False)
                    axs[lang_idx].set_ylabel(f'{function_name}\nTime (s)')
                    axs[lang_idx].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.3f'))
                    axs[lang_idx].yaxis.set_minor_locator(AutoMinorLocator())
                    plt.tight_layout()
                    plt.draw()
                png_file = Path("results/" + scaling_test_name + "/stats") / os.path.basename(run_cfg_path) / f"{function_name}.png"
                plt.savefig(png_file)
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_tb(e.__traceback__)
            traceback.print_exception(*exc_info)