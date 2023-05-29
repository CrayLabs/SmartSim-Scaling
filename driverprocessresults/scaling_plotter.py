import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
from tqdm.auto import tqdm

import sys
import configparser
import json
from pathlib import Path


def scaling_plotter(run_cfg_path, scaling_test_name, var_input):
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
        print("1")
        timing_files = Path(config['run']['path']).glob('rank*.csv')
        print("2")
        for timing_file in timing_files:
            print("4")
            tmp_df = pd.read_csv(timing_file, header=0, names=["rank", "function", "time"])
            print("5")
            for key, value in config._sections['attributes'].items():
                print("6")
                tmp_df[key] = value
                print("7")
            df_list.append(tmp_df)
            print("8")
    
    df = pd.concat(df_list, ignore_index=True)
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
    for function_name in function_names:
        fig = plt.figure(figsize=[12,4])
        for lang_idx, language in enumerate(languages):
            axs = fig.subplots(1,2,sharey=True)
            language_df = df.groupby('language').get_group(language)
            for idx, var in enumerate(var_list):
                var_df = language_df.groupby(var_input).get_group(var)
                function_df = var_df.groupby('function').get_group(function_name)[ ['client_total','time'] ]
                #var_df = function_df.groupby(var_input).get_group(var)[ ['client_total','time'] ]
                # print(var_df)
                #sys.exit()
                data = [function_df.groupby('client_total').get_group(client)['time'] for client in ordered_client_total]
                # print(data)
                pos = [int(client)-idx*25 for client in ordered_client_total]
                plot = axs[lang_idx].violinplot(data, pos, **violin_opts, widths=24)
            axs[lang_idx].set_xlabel('Number of Clients')
            axs[lang_idx].set_title(language)
            axs[lang_idx].set_xticks(pos)
            entry = plot["cbars"]
            #legend_entries.append(entry)
            # data_labels = [f"{var} {var_input}" for var in var_list]
            #axs[lang_idx].legend(loc='upper left')
        axs[0].set_ylabel(f'{function_name}\nTime (s)')
        png_file = Path("results/" + scaling_test_name + "/stats") / os.path.basename(run_cfg_path) / f"{function_name}.png"
        plt.savefig(png_file)
        print(png_file)
    sys.exit()