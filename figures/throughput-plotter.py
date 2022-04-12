import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
from itertools import product
from tqdm.auto import tqdm


palette = sns.set_palette("colorblind", color_codes=True)

backends = ["Redis","KeyDB"]
nnodes_all = [128,256,512]


for backend, nnodes in tqdm(product(backends, nnodes_all), total=len(backends)*len(nnodes_all), desc="Product loop"):

    # Adapt to your setup
    base_path = f"../throughput-scaling-{backend.lower()}"

    DB_nodes = [16,32,64]
    sizes = [1024, 1024000, 131072, 16384, 2048000, 262144, 32768, 4096000, 524288, 65536, 8192]
    threads = 36
    loop_iters = 100
    sizes.sort()

    df_dbs = dict()

    for DB_node in tqdm(DB_nodes, leave=False, desc=f"{backend}-{nnodes}"):

        dfs = dict()

        for size in tqdm(sizes, leave=False, desc=f"{DB_node} DB nodes"):
            path_root = os.path.join(base_path, f'throughput-sess-N{nnodes}-T{threads}-DBN{DB_node}-ITER{loop_iters}-TB{size}-*')
            try:
                globbed = glob(path_root)            
                path = globbed[0]
                
                files = os.listdir(path)

                function_times = {'loop_time': []}

                for file in tqdm(files, leave=False, desc=f"Size {size}"):
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
                dfs[size] = data_df

            except:
                print("WARNING, MISSING PATH:", path_root)
                

        df_dbs[DB_node] = dfs

    # Set to false if this code is run inside a notebook
    save = True

    for dark in tqdm([True, False], leave=False, desc="Plot style loop"):
        if dark:
            plt.style.use("dark_background")
            plot_color="dark"
        else:
            plt.style.use("default")
            plot_color="light"

                    
        labels = ["loop_time"]

        legend_entries = []

        ranks = np.asarray(sizes)
        whiskers = 1e9
        color_short = "rgbmy"
        plot_type = "agg"

        rank_pos = np.log(ranks/ranks[0])+1

        distance = np.min(np.diff(rank_pos))
        widths = distance/(len(DB_nodes))
        spacing = distance/(len(DB_nodes)+0.5)

        quantiles = [[0.25, 0.75] for _ in ranks]

        for label in tqdm(labels, desc=f"Dark plot: {dark}", leave=False):

            fig, ax = plt.subplots(figsize=(8,5))

            if plot_type != "agg":
                ax2 = ax.twinx()
            
            for i, DB_node in enumerate(tqdm(DB_nodes, leave=False, desc="DB node plot loop")):
                dfs = df_dbs[DB_node]
                data_list = [dfs[size][label] for size in sizes]
                props_dict = {"color": sns.color_palette()[i]}
                
                positions = rank_pos if plot_type == "agg" else rank_pos+spacing*(i-(len(DB_nodes)-1)/2)
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
                    ax2.set_ylim([0, 200/(threads*nnodes)])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%2.0f'))
        
        ax.set_xlim([rank_pos[0]-distance/2, rank_pos[-1]+distance/2])
        ax.set_xticks(rank_pos, minor=False)

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
            ax2.legend(legend_entries,  [f'{db_node} DB nodes' for db_node in DB_nodes],
                    loc='upper left')
        else:
            ax.legend([f'{db_node} DB nodes' for db_node in DB_nodes],
                    loc='upper left')

        plt.title(f"{nnodes} client nodes, {threads} clients per node - {backend} backend")
        plt.xlabel("Message size [kiB]")
        if plot_type != "agg":
            ax2.set_ylabel("Single client throughput distribution [GB/s]")
        ax.set_ylabel("Throughput [GB/s]")
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='minor',     # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True)


        plt.tight_layout()
        plt.draw()

        if save:
            plt.savefig(f"{label}-{nnodes}-{backend.lower()}_{plot_color}.png")



