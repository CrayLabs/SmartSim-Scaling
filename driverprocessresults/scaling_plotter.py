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
    for config in configs:
        timing_files = Path(config['run']['path']).glob('rank*.csv')
        for timing_file in timing_files:
            tmp_df = pd.read_csv(timing_file, header=0, names=["rank", "function", "time"])
            for key, value in config._sections['attributes'].items():
                tmp_df[key] = value
            df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)
    logger.debug("Dataframe created")
    violin_opts = dict(        
            showmeans = True,
            showextrema = True,        
        )
    plt.style.use('default')

    ordered_client_total = sorted(df['client_total'].unique())

    function_names = df['function'].unique()
    languages = df['language'].unique()
    legend_entries = []
    var_list = df[var_input].unique()
    logger.debug("Values initialized")
    for function_name in function_names:
        fig = plt.figure(figsize=[12,4])
        logger.debug(f"Looping through function name: {function_name}")
        for lang_idx, language in enumerate(languages):
            logger.debug(f"Looping through language: {language}")
            axs = fig.subplots(1,2,sharey=True)
            language_df = df.groupby('language').get_group(language)
            for idx, var in enumerate(var_list):
                logger.debug("Looping through var: {var}")
                var_df = language_df.groupby(var_input).get_group(var)
                function_df = var_df.groupby('function').get_group(function_name)[ ['client_total','time'] ]
                data = [function_df.groupby('client_total').get_group(client)['time'] for client in ordered_client_total]
                pos = [int(client)-idx*36 for client in ordered_client_total]
                plot = axs[lang_idx].violinplot(data, pos, **violin_opts, widths=24)
                [col.set_alpha(0.3) for col in plot["bodies"]]
                props_dict = dict(color=plot["cbars"].get_color().flatten())
                entry = plot["cbars"]
                legend_entries.append(entry)     
            data_labels = [f"{var} DB nodes" for var in var_list]
            axs[lang_idx].legend(legend_entries, data_labels, loc='upper left')
            axs[lang_idx].set_xlabel('Number of Clients')
            axs[lang_idx].set_title(language)
            axs[lang_idx].set_xticks(pos)
        axs[0].set_ylabel(f'{function_name}\nTime (s)')
        png_file = Path("results/" + scaling_test_name + "/stats") / os.path.basename(run_cfg_path) / f"{function_name}.png"
        plt.savefig(png_file)
        logger.debug(f"Plot created and saved for function name: {function_name} and saved to path: {png_file}")
    logger.debug(f"Plotting complete")