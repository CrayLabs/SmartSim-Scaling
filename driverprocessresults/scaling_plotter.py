import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
from tqdm.auto import tqdm
import matplotlib.patches as mpatches

import sys
import configparser
import json
from pathlib import Path

from smartsim.log import get_logger, log_to_file
logger = get_logger("Plotter")


def scaling_plotter(run_cfg_path, scaling_test_name, var_input):
    logger.debug("Entered plotter method")
    palette = sns.set_palette("colorblind", color_codes=True)

    font = {'family' : 'sans',
            'weight' : 'normal',
            'size'   : 14}
    matplotlib.rc('font', **font)

    configs = []

    for run_cfg in Path(run_cfg_path).rglob('run.cfg'):
        config = configparser.ConfigParser()
        config.read(run_cfg)
        configs.append(config)
    df_list = []
    for config in tqdm(configs, desc="Processing configs...", ncols=80):
        timing_files = Path(config['run']['path']).glob('rank*.csv')
        for timing_file in tqdm(timing_files, desc="Processing timing files...", ncols=80):
            tmp_df = pd.read_csv(timing_file, header=0, names=["rank", "function", "time"])
            for key, value in config._sections['attributes'].items():
                tmp_df[key] = value
            df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)
    #save the results of the processing to a python pickle file
    #pass in the saved dataframe to the plotting part
    logger.debug("Dataframe created")
    violin_opts = dict(        
            showmeans = True,
            showextrema = True,        
        )
    plt.style.use('default')
    client_total = [int(x) for x in df['client_total'].unique()]
    client_per_n = [int(x) for x in df['client_per_node'].unique()] #not used
    database_nodes = [int(x) for x in df['database_nodes'].unique()]
    database_cpus = [int(x) for x in df['database_cpus'].unique()]
    client_nodes = [int(x) for x in df['client_nodes'].unique()] #not used
    grid_spacing = np.min(np.diff(database_nodes))*(client_per_n[0]) #change database nodes to client_nodes
    #ranks = [node*client_per_n[0] for node in client_nodes]
    spacing = grid_spacing/3.5
    widths = grid_spacing/5
    ordered_client_total = sorted(df['client_total'].unique())
    function_names = df['function'].unique()
    languages = df['language'].unique()
    legend_entries = []
    var_list = df[var_input].unique()
    #ranks = [node*threads for node in nnodes]
    for function_name in tqdm(function_names, desc="Processing function name...", ncols=80):
        fig = plt.figure(figsize=[12,4])
        axs = fig.subplots(1,2,sharey=True)
        
        for lang_idx, language in tqdm(enumerate(languages), desc="Processing languages...", ncols=80):
            language_df = df.groupby('language').get_group(language)
            
            for idx, var in tqdm(enumerate(var_list), desc="Processing vars...", ncols=80):
                var_df = language_df.groupby(var_input).get_group(var)
                function_df = var_df.groupby('function').get_group(function_name)[ ['client_total','time'] ]
                data = [function_df.groupby('client_total').get_group(client)['time'] for client in ordered_client_total]
                #having the -idx edits the clients total but removing puts graphs on top
                pos = [int(client)+spacing*(idx-(len(database_nodes)-1)/2) for client in ordered_client_total]
                x_marks = [int(client) for client in ordered_client_total]
                plot = axs[lang_idx].violinplot(data, positions=pos, **violin_opts, widths=grid_spacing/2.5)
                [col.set_alpha(0.3) for col in plot["bodies"]]
                props_dict = dict(color=plot["cbars"].get_color().flatten())
                entry = plot["cbars"]
                legend_entries.append(entry)
                means = [np.mean(function_df.groupby('client_total').get_group(client)) for client in ordered_client_total]
                axs[lang_idx].plot(pos, means, ':', color=props_dict['color'], alpha=0.5)     

            data_labels = [f"{var} DB nodes" for var in var_list]
            axs[lang_idx].legend(legend_entries, data_labels, loc='upper left')
            axs[lang_idx].set_xlabel('Number of Clients')
            axs[lang_idx].set_title(language)
            axs[lang_idx].set_xticks(client_total, minor=False)
            axs[lang_idx].set_xticklabels([client for client in client_total], fontdict={'fontsize': 10})
            axs[lang_idx].set_ylabel(f'{function_name}\nTime (s)')
            axs[lang_idx].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.2f'))
            plt.tight_layout()
            plt.draw()
        png_file = Path("results/" + scaling_test_name + "/stats") / os.path.basename(run_cfg_path) / f"{function_name}.png"
        plt.savefig(png_file)