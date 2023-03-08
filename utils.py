import os
import datetime
from smartsim import Experiment
from smartsim.log import get_logger, log_to_file



def create_folder(self, exp_name, launcher): 
    i = 0
    while os.path.exists("results/"+exp_name+"/run" + str(i)):
        i += 1
    path2 = os.path.join(os.getcwd(), "results/"+exp_name+"/run" + str(i)) #autoincrement
    os.makedirs(path2)
    exp = Experiment(name="results/"+exp_name+"/run" + str(i), launcher=launcher)
    exp.generate()
    log_to_file(f"{exp.exp_path}/scaling-{self.date}.log")
    return exp