# Process Results

SmartSim-Scaling offers support to plot numeric data produced by
a complete scalability test. Currently, we support one statistical graphing
method: violin plot. The violin plots give a data summary and density of
each client function associated with the respective scaling test.

The following client functions per scaling tests are listed below:

#### Inference
  1) ``put_tensor``     (send image to database)
  2) ``run_script``     (preprocess image)
  3) ``run_model``      (run resnet50 on the image)
  4) ``unpack_tensor``  (retrieve the inference result)


#### Throughput
  1) ``put_tensor``     (send image to database)
  2) ``unpack_tensor``  (retrieve the data)


#### Data Aggregation
  1) ``append_to_list`` (add dataset to the aggregation list)
  2) ``poll_list_length``         (check when the next aggregation list is ready)
  3) ``get_datasets_from_list``   (retrieve the data from the aggregation list)

> Note that the process results function is called after a completed scaling test
> meaning the graphs will automatically be produced.


## Collecting Performance Results

The ``process_scaling_results`` function will collect the produced timings 
from ``results/SCALING-TEST-NAME/RUN#`` and make a series of plots for each client function. 
The function will make a collective csv of timings per each run. These
artifacts will be in a ``results/SCALING-TEST-NAME/stats/RUN`` folder inside 
the directory where the function was pointed to the scaling data 
with the ``scaling_dir`` flag shown below. This function is 
automatically called after a scaling test has completed. 

Below you will find the options for process results execution.

```text
NAME
    driver.py process_scaling_results - Create a results directory with performance data and plots

SYNOPSIS
    driver.py process_scaling_results <flags>

DESCRIPTION
    With the overwrite flag turned off, this function can be used
    to build up a single csv with the results of runs over a long
    period of time.

FLAGS
    --scaling_dir=SCALING_DIR
        Default: 'inference-standard-scaling'
        directory to create results from
    --plot_type=PLOT_TYPE
        Default: 'database_nodes'
        directory to create results from
    --overwrite=OVERWRITE
        Default: True
        overwrite any existing results
```

For example for the inference standard tests (if you don't change the output dir name)
you can run:

```bash
python driver.py process_scaling_results
```

If you would like to rather run the `throughput-colocated-scaling`:

```bash
python driver.py process_scaling_results --scaling_dir="throughput-colocated-scaling"
```

## Plot Performance Results

The ``scaling_read_data`` function will collect the produced timings 
from ``results/SCALING-TEST-NAME/RUN#`` and create a pandas dataframe
to use within the ``scaling_plotter`` function. The dataframe is stored
in a compressed csv.gz file within ``results/SCALING-TEST-NAME/stats/RUN#``. 
This function is useful when debugging to avoid the timely cost
of reprocessing your data if you need to reproduce the violin plots.

Below you will find the options for scaling read data execution.

```text
NAME
    driver.py scaling_read_data - Create a dataframe to store in a compressed file

SYNOPSIS
    driver.py scaling_read_data <flags>

DESCRIPTION
    This function produces a dataframe and stores it into a compressed file.

FLAGS
    --run_cfg_path=RUN_CFG_PATH
        Default: No Default
        path to a specific run file 
        Example: results/throughput-standard-scaling/run-2023-07-05-21:26:18
    --scaling_test_name=SCALING_TEST_NAME
        Default: No Default
        directory to create dataframe from
        Example: throughput-standard-scaling
```

For example for the inference standard tests you can run:

```bash
python driver.py scaling_read_data --scaling_dir="inference-standard-scaling" --run_cfg_path="results/inference-standard-scaling/run-2023-07-05-21:26:18"
```

## Read and Store Results

The ``scaling_plotter`` function will plot the performance data. Using the 
dataframe produced by ``scaling_read_data``, the function will create
graphs per client function associated with the scaling test. The graphs are
saved to ``results/SCALING-TEST-NAME/stats/RUN#`` as a png file.

Below you will find the options for scaling plotter execution.

```text
NAME
    driver.py scaling_plotter - Create performance plots

SYNOPSIS
    driver.py scaling_plotter <flags>

DESCRIPTION
    This function will plot your results using the stored dataframe.

FLAGS
    --run_cfg_path=RUN_CFG_PATH
        Default: No Default
        path to a specific run file 
        Example: results/throughput-standard-scaling/run-2023-07-05-21:26:18
    --scaling_test_name=SCALING_TEST_NAME
        Default: No Default
        directory to create dataframe from
        Example: throughput-standard-scaling
    --var_input=VAR_INPUT
        Default: No Default
        permutation to plot on
        Example: database_nodes
```

For example for the inference standard tests you can run:

```bash
python driver.py scaling_plotter --scaling_dir="inference-standard-scaling" --run_cfg_path="results/inference-standard-scaling/run-2023-07-05-21:26:18" --var_input="database_nodes"
```