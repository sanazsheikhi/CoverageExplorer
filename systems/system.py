import os
import sys
import importlib


explorer_path = os.path.dirname(os.getcwd())
sampling_path = explorer_path + '/sampling'
utility_path = explorer_path  + '/utility'
benchmark_path = explorer_path  + '/systems'
sys.path.append(utility_path)
sys.path.append(sampling_path)
sys.path.append(benchmark_path + '/kinematic_car/')
sys.path.append(benchmark_path + '/ACASXU/acasxu_dubins/')
sys.path.append(benchmark_path + '/automatic_transmission/')

class System:

    def __init__(self, benchmark):
        self.system_module = importlib.import_module(f'{benchmark}')

    def preprocessing(self):
        self.systemObj = self.system_module.pre_processing()


    def step(self, state, args_cmd):
        return self.systemObj.step(state, args_cmd)
