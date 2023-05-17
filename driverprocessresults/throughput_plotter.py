import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
from itertools import product
from tqdm.auto import tqdm
import configparser
import json
import sys
from pathlib import Path
from pprint import pprint

# def throughput_plotter_standard(run_cfg_path):
#     config = configparser.RawConfigParser()
#     config.read(run_cfg_path + "/run.cfg")
#     attributes_dict = dict(config.items('attributes'))
#     run_dict = dict(config.items('run'))
#     palette = sns.set_palette("colorblind", color_codes=True)

#     backends = [run_dict['db']]
#     nnodes_all = json.loads(attributes_dict['client_nodes']) #make sure this is correct, naming is misleading
#     #Question: so its possible to have two backends in a run.cfg file?
#     for backend, nnodes in tqdm(product(backends, nnodes_all), total=len(backends)*len(nnodes_all), desc="Product loop"):
#         DB_nodes = json.loads(attributes_dict['database_nodes'])
#         sizes = json.loads(attributes_dict['tensor_bytes'])
#         threads = json.loads(attributes_dict['client_per_node'])[0] #should iterate through the list maybe, might need to change [0] to [i]
#         loop_iters = 100 #will need to add this to run.cfg file for standard
#         DB_cpu = json.loads(attributes_dict['database_cpus'])[0] #will need to change this for colocated, implement list support for cpus via standard
#         pin = "False" #this is for colocated

#         df_dbs = dict() #maybe change the name of this

#         for DB_node in tqdm(DB_nodes, leave=False, desc=f"{backend}-{nnodes}", ncols=80): #check my understanding of this going through once
#             dfs = dict() #ask Al what this stands for

#             for size in tqdm(sizes, leave=False, desc=f"{DB_node} DB nodes", ncols=80): #change size to tensor_size later
                
#                 #print(nnodes)
#                 if json.loads(attributes_dict['colocated']) == 1: #this might be bad to have here for runtime issues
#                     path_root = os.path.join(run_cfg_path, f'throughput-sess-colo-N{nnodes}-T{threads}-DBCPU{DB_cpu}-PIN{pin}-ITER{loop_iters}-TB{size}-*')
#                 else:
#                     path_root = os.path.join(run_cfg_path, f'throughput-sess-N{nnodes}-T{threads}-DBN{DB_node}-DBCPU{DB_cpu}-ITER{loop_iters}-TB{size}-*')
#                 try:
#                     globbed = glob(path_root)           
#                     path = globbed[0] #ask Al why its searching for index 0
#                     files = os.listdir(path)

#                     function_times = {'loop_time': []}

#                     for file in tqdm(files, leave=False, desc=f"Size {size}", ncols=80):
#                         if '.csv' in file and 'rank_' in file:
#                             fp = os.path.join(path, file)
#                             with open(fp) as f:
#                                 for i, line in enumerate(f):
#                                     vals = line.split(',')
#                                     if vals[1] in function_times.keys():
#                                         speed = size*loop_iters/float(vals[2])/1e9
#                                         function_times[vals[1]].append(speed)

#                     speed = function_times['loop_time']

#                     speed = function_times['loop_time']
#                     data_df = pd.DataFrame(function_times)
#                     dfs[size] = data_df #this is a dict of dataframes stored by tensor size
#                 except:
#                     print("WARNING, MISSING PATH:", path_root)

#             df_dbs[DB_node] = dfs #this is a dict that stores a dict of dataframes by db_node
        
#         # Set to false if this code is run inside a notebook
#         save = True #create this into a flag to pass in

#         #leave in progress bar makes it clear after done
#         for dark in tqdm([True, False], leave=False, desc="Plot style loop", ncols=80): #figure out what the true,false val is
#             if dark: #ask about dark, when will it go to white
#                 plt.style.use("dark_background")
#                 plot_color="dark"
#             else:
#                 plt.style.use("default")
#                 plot_color="light"

                        
#             labels = ["loop_time"] #could make this customizable

#             legend_entries = []

#             ranks = np.asarray(sizes)
#             whiskers = 1e9
#             color_short = "rgbmy"
#             plot_type = "boxplot" #need to make this into a flag for violin and agg

#             rank_pos = np.log(ranks/ranks[0])+1

#             distance = np.min(np.diff(rank_pos))
#             widths = distance/(len(DB_nodes))
#             spacing = distance/(len(DB_nodes)+0.5)

#             quantiles = [[0.25, 0.75] for _ in ranks]

#             for label in tqdm(labels, desc=f"Dark plot: {dark}", leave=False, ncols=80):

#                 fig, ax = plt.subplots(figsize=(8,5))

#                 if plot_type != "agg":
#                     ax2 = ax.twinx()
                
#                 for i, DB_node in enumerate(tqdm(DB_nodes, leave=False, desc="DB node plot loop", ncols=80)):
#                     dfs = df_dbs[DB_node]
#                     data_list = [dfs[size][label] for size in sizes]
#                     props_dict = {"color": sns.color_palette()[i]}
                    
#                     positions = rank_pos if plot_type == "agg" else rank_pos+spacing*(i-(len(DB_nodes)-1)/2)
#                     means = [np.sum(dfs[size][label]) for size in sizes]
#                     ax.plot(positions, means, '.-', color=props_dict['color'], alpha=0.75)
#                     if plot_type != "agg":
#                         if plot_type=="violin":
#                             plot = ax2.violinplot(data_list, positions=positions,
#                                                 widths=widths, showextrema=True)
#                             [col.set_alpha(0.3) for col in plot["bodies"]]
#                             entry = plot["cbars"]
#                             legend_entries.append(entry)
#                         elif plot_type=="boxplot":
#                             plot = ax2.boxplot(data_list, showfliers=True, positions=positions, whis=whiskers, labels=['']*len(ranks),
#                                             boxprops=props_dict, whiskerprops=props_dict, medianprops=props_dict, capprops=props_dict, widths=widths/2)
#                             legend_entries.append(plot["whiskers"][0])
#                         else:
#                             raise ValueError("Only boxplot, violin, and agg are valid plot types")


#                     ax.set_ylim([0, 200])
#                     if plot_type != "agg":
#                         ax2.set_ylim([0, 200/(threads*nnodes)])
#                     ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.0f'))
            
#             ax.set_xlim([rank_pos[0]-distance/2, rank_pos[-1]+distance/2])
#             ax.set_xticks(rank_pos, minor=False)

#             if plot_type != means:
#                 x_minor_ticks = []
#                 for i, pos in enumerate(rank_pos[:-1]):
#                     if i and pos-rank_pos[i-1] > distance*1.5:
#                         x_minor_ticks.append(pos-distance/2)
#                     x_minor_ticks.append(pos+distance/2)

#                 ax.set_xticks(x_minor_ticks, minor=True)

#             labels = ["1", "8", "16", "32", "64", "128", "256", "512", "1000", "2000", "4000"]
#             ax.set_xticklabels(labels, fontdict={'fontsize': 10})

#             if plot_type != "agg":
#                 ax.grid(True, which="minor", axis="x", ls=":", markevery=rank_pos[:-1]+distance/2)

#             if plot_type == means:
#                 ax2.legend(legend_entries,  [f'{db_node} DB nodes' for db_node in DB_nodes],
#                         loc='upper left')
#             else:
#                 ax.legend([f'{db_node} DB nodes' for db_node in DB_nodes],
#                         loc='upper left')

#             plt.title(f"{nnodes} client nodes, {threads} clients per node - {backend} backend")
#             plt.xlabel("Message size [kiB]")
#             if plot_type != "agg":
#                 ax2.set_ylabel("Single client throughput distribution [GB/s]")
#             ax.set_ylabel("Throughput [GB/s]")
#             plt.tick_params(
#                 axis='x',          # changes apply to the x-axis
#                 which='minor',     # both major and minor ticks are affected
#                 bottom=False,      # ticks along the bottom edge are off
#                 top=False,         # ticks along the top edge are off
#                 labelbottom=True)


#             plt.tight_layout()
#             plt.draw()
#             # print("got here")
#             # sys.exit()
#             if save:
#                 png_file = Path("results/throughput-standard-scaling/stats") / os.path.basename(run_cfg_path) / f"{plot_type}-{label}-{nnodes}-{backend.lower()}_{plot_color}.png"
#                 #Path(run_cfg_path) / f"{label}-{nnodes}-{backend.lower()}_{plot_color}.png"
#                 plt.savefig(png_file)


def throughput_plotter_colocated(run_cfg_path):
    config = configparser.RawConfigParser()
    config.read(run_cfg_path + "/run.cfg")
    attributes_dict = dict(config.items('attributes'))
    run_dict = dict(config.items('run'))
    palette = sns.set_palette("colorblind", color_codes=True)

    backends = [run_dict['db']]
    nnodes_all = json.loads(attributes_dict['client_nodes']) #make sure this is correct, naming is misleading
    #Question: so its possible to have two backends in a run.cfg file?
    df_dbn = dict()
    for backend, nnodes in tqdm(product(backends, nnodes_all), total=len(backends)*len(nnodes_all), desc="Product loop"):
        #DB_nodes = json.loads(attributes_dict['database_nodes'])
        sizes = json.loads(attributes_dict['tensor_bytes'])
        threads = json.loads(attributes_dict['client_per_node']) #should iterate through the list maybe, might need to change [0] to [i]
        loop_iters = 100 #will need to add this to run.cfg file for standard
        DB_cpu = json.loads(attributes_dict['database_cpus'])[0] #will need to change this for colocated, implement list support for cpus via standard
        pin = "False" #this is for colocated

        df_dbs = dict() #maybe change the name of this

        for thread in tqdm(threads, leave=False, desc=f"{backend}-{nnodes}", ncols=80): #check my understanding of this going through once
            dfs = dict() #ask Al what this stands for

            for size in tqdm(sizes, leave=False, desc=f"{thread} DB nodes", ncols=80): #change size to tensor_size later
                #print(nnodes)
                if json.loads(attributes_dict['colocated']) == 1: #this might be bad to have here for runtime issues
                    path_root = os.path.join(run_cfg_path, f'throughput-sess-colo-N{nnodes}-T{thread}-DBCPU{DB_cpu}-PIN{pin}-ITER{loop_iters}-TB{size}-*')
                else:
                    path_root = os.path.join(run_cfg_path, f'throughput-sess-N{nnodes}-T{thread}-DBN{thread}-DBCPU{DB_cpu}-ITER{loop_iters}-TB{size}-*')
                try:
                    globbed = glob(path_root)           
                    path = globbed[0] #ask Al why its searching for index 0
                    files = os.listdir(path)

                    function_times = {'loop_time': []}

                    for file in tqdm(files, leave=False, desc=f"Size {size}", ncols=80):
                        if '.csv' in file and 'rank_' in file:
                            fp = os.path.join(path, file)
                            with open(fp) as f:
                                for i, line in enumerate(f):
                                    vals = line.split(',')
                                    if vals[1] in function_times.keys():
                                        speed = size*loop_iters/float(vals[2])/1e9
                                        function_times[vals[1]].append(speed)

                    speed = function_times['loop_time']

                    speed = function_times['loop_time']
                    data_df = pd.DataFrame(function_times)
                    dfs[size] = data_df #this is a dict of dataframes stored by tensor size
                except:
                    print("WARNING, MISSING PATH:", path_root)
            df_dbs[thread] = dfs #this is a dict that stores a dict of dataframes by db_node
        df_dbn[nnodes] = df_dbs
        # Set to false if this code is run inside a notebook
        save = True #create this into a flag to pass in
    
        #leave in progress bar makes it clear after done
    for dark in tqdm([True, False], leave=False, desc="Plot style loop", ncols=80): #figure out what the true,false val is
        if dark: #ask about dark, when will it go to white
            plt.style.use("dark_background")
            plot_color="dark"
        else:
            plt.style.use("default")
            plot_color="light"
        
        labels = ["loop_time"] #could make this customizable

        legend_entries = []

        ranks = np.asarray(sizes)
        whiskers = 1e9
        color_short = "rgbmy"
        plot_type = "boxplot" #need to make this into a flag for violin and agg

        rank_pos = np.log(ranks/ranks[0])+1

        distance = np.min(np.diff(rank_pos))
        widths = distance/(len(threads))
        spacing = distance/(len(threads)+0.5)

        #quantiles = [[0.25, 0.75] for _ in ranks]
        pprint(df_dbn)
        sys.exit()
        for label in tqdm(labels, desc=f"Dark plot: {dark}", leave=False, ncols=80):
            
            fig, ax = plt.subplots(figsize=(8,5))
            if plot_type != "agg":
                ax2 = ax.twinx()
            for j, nnodes in enumerate(tqdm(nnodes_all, leave=False, desc="nnodes", ncols=80)):
                df_dbs = df_dbn[nnodes]
                for i, thread in enumerate(tqdm(threads, leave=False, desc="cpn", ncols=80)):
                    dfs = df_dbs[thread]
                    data_list = [dfs[size][label] for size in sizes] #divide this by the number of client nodes
                    props_dict = {"color": sns.color_palette()[i]}
                    
                    positions = rank_pos if plot_type == "agg" else rank_pos+spacing*(i-(len(threads)-1)/2)
                    means = [np.sum(dfs[size][label]) for size in sizes]
                    ax.plot(positions, means, '.-', color=props_dict['color'], alpha=0.75)
                    if plot_type != "agg":
                        if plot_type=="violin":
                            plot = ax2.violinplot(data_list, positions=positions,
                                                widths=widths, showextrema=True)
                            [col.set_alpha(0.3) for col in plot["bodies"]]
                            entry = plot["cbars"]
                            legend_entries.append(entry)
                        elif plot_type=="boxplot":
                            plot = ax2.boxplot(data_list, showfliers=True, positions=positions, whis=whiskers, labels=['']*len(ranks),
                                            boxprops=props_dict, whiskerprops=props_dict, medianprops=props_dict, capprops=props_dict, widths=widths/2)
                            legend_entries.append(plot["whiskers"][0])
                        else:
                            raise ValueError("Only boxplot, violin, and agg are valid plot types")
                    ax.set_ylim([0, 200])
                    if plot_type != "agg":
                        ax2.set_ylim([0, 200/(thread*nnodes)])#ask about this
                    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.0f'))
        
        ax.set_xlim([rank_pos[0]-distance/2, rank_pos[-1]+distance/2])
        ax.set_xticks(rank_pos, minor=False)
        # print("test")
        # sys.exit()

        if plot_type != means:
            x_minor_ticks = []
            for i, pos in enumerate(rank_pos[:-1]):
                if i and pos-rank_pos[i-1] > distance*1.5:
                    x_minor_ticks.append(pos-distance/2)
                x_minor_ticks.append(pos+distance/2)

            ax.set_xticks(x_minor_ticks, minor=True)

        labels = ["1", "8", "16", "32", "64", "128", "256", "512", "1000", "2000", "4000"]
        ax.set_xticklabels(labels, fontdict={'fontsize': 10})

        if plot_type != "agg":
            ax.grid(True, which="minor", axis="x", ls=":", markevery=rank_pos[:-1]+distance/2)

        if plot_type == means:
            ax2.legend(legend_entries,  [f'{thread} clients per node' for thread in threads],
                    loc='upper left')
        else:
            ax.legend([f'{thread} clients per node' for thread in threads],
                    loc='upper left')

        plt.title(f"{nnodes} client nodes, {threads} clients per node - {backend} backend")
        plt.xlabel("Message size [kiB]")
        if plot_type != "agg":
            ax2.set_ylabel("Single client colocated throughput distribution [GB/s]")
        ax.set_ylabel("Colocated Throughput [GB/s]")
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='minor',     # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True)


        plt.tight_layout()
        plt.draw()
        # print("got here")
        # sys.exit()
        if save:
            png_file = Path("results/throughput-colocated-scaling/stats") / os.path.basename(run_cfg_path) / f"{plot_type}-{label}-{backend.lower()}_{plot_color}.png"
            #Path(run_cfg_path) / f"{label}-{nnodes}-{backend.lower()}_{plot_color}.png"
            plt.savefig(png_file)
    print("test")
    sys.exit()