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
    config = configparser.RawConfigParser()
    config.read(run_cfg_path + "/run.cfg")
    attributes_dict = dict(config.items('attributes'))
    run_dict = dict(config.items('run'))
    # print("\nattributes dict: ", attributes_dict)
    # print("\nrun_dict: ", run_dict)
    # sys.exit()
    palette = sns.set_palette("colorblind", color_codes=True)

    font = {'family' : 'sans',
            'weight' : 'normal',
            'size'   : 14}
    matplotlib.rc('font', **font)

    nnodes = json.loads(attributes_dict['client_nodes'])
    DB_nodes = json.loads(attributes_dict['database_nodes'])
    DB_cpus = json.loads(attributes_dict['database_cpus'])[0]
    threads = json.loads(attributes_dict['client_per_node'])[0]
    db_tpq = json.loads(attributes_dict['database_threads_per_queue'])[0]

    aggregate = False

    df_dbs = dict()
    base_path = run_cfg_path
    print(base_path)
    sys.exit()

    functions = ['put_tensor', 'run_script', 'run_model', 'unpack_tensor']
    # print("path: ")
    # sys.exit()
    for DB_node in tqdm(DB_nodes, desc=base_path):
    
        dfs = dict()

        for node in tqdm(nnodes, desc=f"{DB_node} DB nodes", leave=False):
            path_root = os.path.join(base_path, f'infer-sess-colo-N{node}-T{threads}-DBN{DB_node}-DBCPU{DB_cpus}-DBTPQ{db_tpq}-*')
            path = glob(path_root)[0]
            # print(path)
            # sys.exit()
            files = os.listdir(path)
            # print(files)
            # sys.exit()
            
            function_times = {}

            for file in tqdm(files, desc=f"{node} client nodes", leave=False):
                if '.csv' in file and 'rank_' in file:
                    fp = os.path.join(path, file)
                    function_rank_times = {}
                    with open(fp) as f:
                        for i, line in enumerate(f):
                            vals = line.split(',')
                            if vals[1] not in functions:
                                continue
                            if not aggregate:
                                if vals[1] in function_times.keys():
                                    function_times[vals[1]].append(float(vals[2]))
                                else:
                                    function_times[vals[1]] = [float(vals[2])]
                            else:
                                if vals[1] in function_rank_times.keys():
                                    function_rank_times[vals[1]] += float(vals[2])
                                else:
                                    function_rank_times[vals[1]] = float(vals[2])
                                
                    for k,v in function_rank_times.items():
                        if k in function_times:
                            function_times[k].append(v)
                        else:
                            function_times[k] = [v]
                
            data_df = pd.DataFrame(function_times)
            dfs[node] = data_df

            # print(f"Completed {node} nodes for {DB_node} DB nodes")

        df_dbs[DB_node] = dfs
    save = True
    all_in_one = False
    labels = ["put_tensor", "unpack_tensor", "run_model", "run_script"]
    palette = sns.set_palette("colorblind", color_codes=True)
    
    
    for style in tqdm(["light", "dark"], desc="Plotting"):
        if style == "light":
            plt.style.use("default")
        else:
            plt.style.use("dark_background")
        # print("nnodes", nnodes[0])
        # print("threads", threads)
        # sys.exit()
        grid_spacing = np.min(np.diff(nnodes))*threads #THIS IS NOT WORKING
        sys.exit()
        legend_entries = []
        ranks = [node*threads for node in nnodes]
        
        widths = grid_spacing/5
        spacing = grid_spacing/3.5
        color_short = "brgmy"

        aggregate_suffix = "_agg" if aggregate else ""
        plot_type = "violin"

        # Set subplot_index to None to plot to separate files, to 1 to have all plots in one
        subplot_index = 1 if all_in_one else None
        if subplot_index:
            plt.figure(figsize=(8*2,5*2+3))

        for label in tqdm(labels, desc=f"{style} style"):
            if subplot_index:
                ax = plt.subplot(2,2,subplot_index)
            else:
                fig, ax = plt.subplots(figsize=(8,5))

            for i, DB_node in enumerate(tqdm(DB_nodes, desc=label, leave=False)):
                dfs = df_dbs[DB_node]
                positions = ranks+spacing*(i-(len(DB_nodes)-1)/2)
                
                data_list = [dfs[node][label] for node in nnodes]
                
                if plot_type=="violin":
                    plot = ax.violinplot(data_list, positions=positions,
                                        widths=grid_spacing/2.5, showextrema=True)
                    [col.set_alpha(0.3) for col in plot["bodies"]]
                    props_dict = dict(color=plot["cbars"].get_color().flatten())
                    entry = plot["cbars"]
                    legend_entries.append(entry)
                else:
                    props_dict = dict(color=color_short[i])
                    plot = ax.boxplot(data_list, showfliers=True, positions=positions, whis=1e9, 
                                boxprops=props_dict, whiskerprops=props_dict, medianprops=props_dict, capprops=props_dict, widths=widths)
                    legend_entries.append(plot["whiskers"][0])
                means = [np.mean(dfs[node][label]) for node in nnodes]
                ax.plot(positions, means, ':', color=props_dict['color'], alpha=0.5)

            
            data_labels = [f"{db_node} DB nodes" for db_node in DB_nodes]
            ax.legend(legend_entries, data_labels, loc='upper left')
            
            ax.set_xticks(ranks, minor=False)
            ax.set_xticklabels([rank for rank in ranks], fontdict={'fontsize': 12})

            plt.title(label)
            plt.xlabel("MPI Ranks")
            plt.ylabel("Time [s]")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.2f'))

            plt.tight_layout()
            plt.draw()

            
            if not subplot_index:
                if save:
                    plt.savefig(f"{label}_{plot_type}{aggregate_suffix}_{style}.pdf")
                    plt.savefig(f"{label}_{plot_type}{aggregate_suffix}_{style}.png")
            else:
                subplot_index += 1

        if subplot_index and save:
            plt.savefig(f'all_in_one_{plot_type}{aggregate_suffix}_{style}.pdf')
            plt.savefig(f'all_in_one_{plot_type}{aggregate_suffix}_{style}.png')
            
        
    print("done")
    sys.exit()