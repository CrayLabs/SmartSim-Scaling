import sys
from driverinference.main import Inference
from driverthroughput.main import Throughput
from driverdataaggregation.main import DataAggregation
from driverprocessresults.main import ProcessResults
from driverprocessresults.scaling_plotter import PlotResults

class SmartSimScalingTests(Inference, Throughput, DataAggregation, ProcessResults, PlotResults):
    ...

if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
