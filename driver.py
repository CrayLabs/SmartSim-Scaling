import sys
from inference.main import Inference
from throughput.main import Throughput
from dataaggregation.main import DataAggregation



class SmartSimScalingTests(Inference, Throughput, DataAggregation):
    def driver():
        if sys.argv[1] == "inference_standard":
            return inference_standard()
        
        if sys.argv[1] == "inference_colocated":
            return inference_colocated()
        
        if sys.argv[1] == "throughput_standard":
            return throughput_standard()
        
        if sys.argv[1] == "data_aggregation":
            return aggregation_scaling()
        
        if sys.argv[1] == "aggregation_scaling_python":
            return aggregation_scaling_python()
        
        if sys.argv[1] == "aggregation_scaling_python_fs":
            return aggregation_scaling_python_fs()

if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
