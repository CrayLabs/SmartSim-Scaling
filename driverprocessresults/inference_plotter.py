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

def inference_plotter_standard(run_cfg_path):
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

    function_names = ['put_tensor', 'unpack_tensor']
    languages = ['cpp']

    for function_name in function_names:
        fig = plt.figure(figsize=[12,4])
        axs = fig.subplots(1,2,sharey=True)
        for i, language in enumerate(languages):
            language_df = df.groupby('language').get_group(language)
            function_df = language_df.groupby('function').get_group(function_name)[ ['client_total','time'] ]
            data = [function_df.groupby('client_total').get_group(client)['time'] for client in ordered_client_total]
            pos = [int(client) for client in ordered_client_total]
            axs[i].violinplot(data, pos, **violin_opts, widths=24)
            axs[i].set_xlabel('Number of Clients')
            axs[i].set_title(language)
            axs[i].set_xticks(pos)
        axs[0].set_ylabel(f'{function_name}\nTime (s)')
    # plt.box(put_tensor_df['client_total'], put_tensor_df['time'])
        png_file = Path("results/inference-standard-scaling/stats") / os.path.basename(run_cfg_path) / f"{function_name}.png"
        plt.savefig(png_file)
        print(png_file)
    sys.exit()


def inference_plotter_colocated(run_cfg_path):
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

    function_names = ['put_tensor', 'unpack_tensor']
    languages = ['cpp']

    for function_name in function_names:
        fig = plt.figure(figsize=[12,4])
        axs = fig.subplots(1,2,sharey=True)
        for i, language in enumerate(languages):
            language_df = df.groupby('language').get_group(language)
            function_df = language_df.groupby('function').get_group(function_name)[ ['client_total','time'] ]
            data = [function_df.groupby('client_total').get_group(client)['time'] for client in ordered_client_total]
            pos = [int(client) for client in ordered_client_total]
            axs[i].violinplot(data, pos, **violin_opts, widths=24)
            axs[i].set_xlabel('Number of Clients')
            axs[i].set_title(language)
            axs[i].set_xticks(pos)
        axs[0].set_ylabel(f'{function_name}\nTime (s)')
    # plt.box(put_tensor_df['client_total'], put_tensor_df['time'])
        png_file = Path("results/inference-colocated-scaling/stats") / os.path.basename(run_cfg_path) / f"{function_name}.png"
        plt.savefig(png_file)
        print(png_file)
    sys.exit()
