import sys
from driverinference.main import Inference
from driverthroughput.main import Throughput
from driverdataaggregation.main import DataAggregation

class SmartSimScalingTests(Inference, Throughput, DataAggregation):
    ...

if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
