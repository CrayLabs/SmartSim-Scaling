import sys
from inference.main import Inference
from throughput.main import Throughput



class SmartSimScalingTests(Inference, Throughput):
    def driver():
        if sys.argv[1] == "inference_standard":
            return inference_standard()
        
        if sys.argv[1] == "inference_colocated":
            return inference_colocated()
        
        if sys.argv[1] == "throughput-standard":
            return throughput_standard()

if __name__ == "__main__":
    import fire
    fire.Fire(SmartSimScalingTests())
