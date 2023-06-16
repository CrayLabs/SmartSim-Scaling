import sys
from driverinference.main import Inference
from driverthroughput.main import Throughput
from driverdataaggregation.main import DataAggregation
from driverprocessresults.main import ProcessResults

class SmartSimScalingTests(Inference, Throughput, DataAggregation, ProcessResults):
    ...

if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
