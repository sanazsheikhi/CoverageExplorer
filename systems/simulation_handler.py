import sys
import os
import yaml
import math
import numpy as np
from copy import copy



explorer_path = os.path.dirname(os.getcwd())
sampling_path = explorer_path + '/sampling'
utility_path = explorer_path  + '/utility'
systems_path = explorer_path  + '/systems'
config_path = explorer_path + '/config'
sys.path.append(config_path)
sys.path.append(utility_path)
sys.path.append(sampling_path)
sys.path.append(systems_path + '/kinematic_car/')
sys.path.append(systems_path + '/ACASXU/acasxu_dubins/')
sys.path.append(systems_path + '/automatic_transmission/')




'import case studies'
import sampling as rs  # rs for rejection sampling
import system as systemHandler


class Simulation_Handler:

    mpc = None
    rsampler = None
    curr_run = 0
    benchmark = ''

    def __init__(self, benchmark=None):

        self.benchmark = benchmark

        self.system_handler = systemHandler.System(benchmark)

        configFile = config_path + '/' + benchmark + '_config.yml'
        with open(configFile) as yaml_file:
            config_ = yaml.safe_load(yaml_file)

        config = config_.get(benchmark, {})
        if config['inputs_lower_bound']: self.inputs_lower_bound = config['inputs_lower_bound']
        if config['inputs_upper_bound']: self.inputs_upper_bound = config['inputs_upper_bound']
        if config['init_state']: self.init_state = config['init_state']

    def update_mpc(self, mpc_):
        self.mpc = mpc_


    def update_sampler(self, trajectories):
        self.rsampler.update_kdt(trajectories)


    def make_sampler(self, lower_bound=None, upper_bound=None, trajectories=None, convexHull=None):
        # print(f"sim_Handler::update_sampler")
        'Make the sampler object either from traiing data to grow the koopman model or get it from the last determined boxbound'
        if trajectories:
            self.rsampler = rs.Sampling(benchmark=self.benchmark, trajectories=trajectories)
        elif convexHull:
            self.rsampler = rs.Sampling(benchmark=self.benchmark, convexHull=convexHull)
        elif lower_bound and upper_bound:
            self.rsampler = rs.Sampling(benchmark=self.benchmark, lower_bound=lower_bound, upper_bound=upper_bound)
        else:
            'make sampler from the box bound of the last koopman training data'
            lower_bound_, upper_bound_ = self.get_boxbound()
            self.rsampler = rs.Sampling(benchmark=self.benchmark, lower_bound= lower_bound_, upper_bound=upper_bound_)


    def get_boxbound(self):

        'make sampler from the box bound of the last koopman training data'
        if self.rsampler:
            return self.rsampler.get_boxbound()

        return None, None


    def get_training_data(self, sim_count, steps):

        print("SimHandler::get_training_data")
        train_states, train_inputs = [], []
        dim = len(self.inputs_lower_bound)

        for i in range(sim_count):

            states = [self.init_state]

            inputs = []
            samples = np.random.uniform(low=np.array(self.inputs_lower_bound),
                                        high=np.array(self.inputs_upper_bound),
                                        size=(steps, dim))

            np.random.seed(i)
            self.system_handler.preprocessing()

            for i in range(steps):
                next_state = self.system_handler.step(states[-1], samples[i])
                inputs = inputs + [samples[i]]
                states.append(next_state)

            train_states.append(np.array(states))
            train_inputs.append(np.array(inputs))

        return train_states, train_inputs



    def run(self, sim_count, steps):

        print("simHandler::run")

        mpc = self.mpc
        rsampler = self.rsampler
        all_traj_states = []
        all_traj_inputs = []
        all_test_cases = []

        # This is to generate new seed, loop counter doesn't solve the problem
        internal_counter = 0

        for i in range(sim_count):

            states_ = copy(self.init_state)

            # to form traj from steps to store
            tmp_states, tmp_inputs = [states_], []

            'Finding a target state'
            target_ = rsampler.rejection_sampling(seed=internal_counter)
            target = np.transpose(mpc.normalize_state(np.array([target_])))
            internal_counter += 1

            print(f"target_ {target_}")

            np.random.seed(i)
            self.system_handler.preprocessing()

            for j in range(steps):
                # no transpose() for new obs_function
                # print(f"i {i}  j {j}")
                curr_state = mpc.normalize_state(np.array([states_]))
                action, _ = mpc.predict(curr_state, target)

                'No solution found'
                if action is None:  break

                #states_ =  self.system_module.step(states_, action)
                states_ = copy(self.system_handler.step(states_, action))

                tmp_states.append(states_)  # to form traj to store
                tmp_inputs.append(action)  # to form traj to store

            # We need trajectories of the same length for kmeans clustering
            if len(tmp_states) == steps + 1:
                'Omit no solution found traces'
                all_traj_states.append(np.array(tmp_states))
                all_traj_inputs.append(np.array(tmp_inputs))
                all_test_cases.append([self.init_state, tmp_inputs])

                'Add to kdt-tree for rejection sampling'
                rsampler.update_kdt([np.array(tmp_states)])
                i += 1


        return all_traj_states, all_traj_inputs, all_test_cases




