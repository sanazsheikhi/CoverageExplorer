import os
import sys
import yaml
import pickle
import mpc as mpc_
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


explorer_path = os.path.dirname(os.getcwd())
systems_path = explorer_path + '/systems'
sampling_path = explorer_path + '/sampling'
coverage_path = explorer_path + '/coverage'
utility_path = explorer_path  + '/utility'
config_path = explorer_path + '/config'
result_path = utility_path + '/results'
sys.path.append(explorer_path)
sys.path.append(sampling_path)
sys.path.append(systems_path)
sys.path.append(coverage_path)
sys.path.append(utility_path)
sys.path.append(config_path)
sys.path.append(result_path)


import simulation_handler as sh
import clustering as cl
import ConvexHullSampling as cvx


class Explorer:

    curr_traindata_states = [] # updated training data
    curr_traindata_inputs = [] # updated training data
    curr_run = 0
    all_states = [] # all trajectories during all runs
    all_inputs = [] # all trajectories during all runs
    error_vectors = []


    def __init__(self, benchmark=''):

        if benchmark == '':
            print("Please define a benchmark:")
            benchmark = input()

        self.benchmark = benchmark

        configFile = config_path + '/' + benchmark + '_config.yml'
        with open(configFile) as yaml_file:
            config_ = yaml.safe_load(yaml_file)

        config = config_.get(benchmark, {})
        if config['dt'] :   self.dt = config['dt']
        if config['steps']: self.steps = config['steps']
        if config['train_sim_count']:   self.train_sim_count = config['train_sim_count']
        if config['run_sim_count']: self.run_sim_count = config['run_sim_count']
        if config['cluster_count']: self.cluster_count = config['cluster_count']
        if config['sim_count_per_train']: self.sim_count_per_train = config['sim_count_per_train']
        if config['traj_per_cluster']:  self.traj_per_cluster = config['traj_per_cluster']


        self.clustering_obj = cl.Clustering()
        self.sim_handler = sh.Simulation_Handler(benchmark)

        self.get_traindata(clusters = self.cluster_count,
                           traj_per_cluster= self.traj_per_cluster, iteration=0)
        self.update_mpc()

        self.make_convex()
        self.sim_handler.make_sampler(convexHull = self.convexhull)

        self.sim_handler.update_mpc(self.mpc)


    def run_tests(self):

        'reseting the sampler object'
        all_traj_states, _ , all_test_cases = self.sim_handler.run(self.run_sim_count, self.steps)

        fname = utility_path + '/results/' + self.benchmark  + '_trajectories.pkl'
        with open(fname, 'wb') as fhandle:
            pickle.dump(all_traj_states, fhandle)

        fname = utility_path + '/results/' + self.benchmark  + '_testCases.pkl'
        with open(fname, 'wb') as fhandle:
            pickle.dump(all_test_cases, fhandle)


    def make_convex(self, data=None):
        self.convexhull = cvx.Convex()
        self.convexhull.make_exact_convex_2(self.clustered_points,
                                            self.cluster_count)

    def update_mpc(self):
        states = deepcopy(self.curr_traindata_states)
        inputs = deepcopy(self.curr_traindata_inputs)
        self.mpc = mpc_.MPC(self.benchmark, states, inputs,
                            self.steps, self.dt)


    def get_traindata(self, traj_states = None, traj_inputs=None,
                      iteration=None, clusters=5, traj_per_cluster=3):
        print("Train data processing")

        if traj_states is None:
            self.curr_traindata_states, self.curr_traindata_inputs = (
                self.sim_handler.get_training_data(self.sim_count_per_train, self.steps))
            tmp_states, tmp_inputs = (deepcopy(self.curr_traindata_states),
                                      deepcopy(self.curr_traindata_inputs))
        else:
            tmp_states = self.curr_traindata_states + traj_states
            tmp_inputs = self.curr_traindata_inputs + traj_inputs

        'selected trajectory list and clustered'
        self.clustered_points, lst = \
        (self.clustering_obj.select_trajectories_Kmeans(tmp_states,
                             k=clusters, percent=traj_per_cluster))
        res = 0
        if lst:
            """
            We should clear the "curr_traindata_states"; O.W. the error at 
            different iterations does not go down; So do NOT comment out the 
            following clear() lines. Another consequence is the number of 
            training trajetories accumalate up to 20 K!
            """
            self.curr_traindata_states.clear()
            self.curr_traindata_inputs.clear()

            for i in lst:
                self.curr_traindata_states.append(tmp_states[i])
                self.curr_traindata_inputs.append(tmp_inputs[i])
        else:
            res = -1

        return res


    def AdaptiveAutoKoop(self):
        'Per iteration we do train_sim_count simulations to update the model'
        count = 0
        iter = 1 # training iteration
        while count < self.train_sim_count:
            traj_states, traj_inputs, _ = self.sim_handler.run(self.sim_count_per_train, self.steps)

            self.get_traindata(traj_states=traj_states, traj_inputs=traj_inputs,
                               iteration=iter, clusters=self.cluster_count,
                               traj_per_cluster=self.traj_per_cluster)
            iter += 1
            self.update_mpc()
            self.make_convex()
            self.sim_handler.make_sampler(convexHull=self.convexhull)
            self.sim_handler.update_mpc(self.mpc)
            count += self.sim_count_per_train
            if self.cluster_count < self.sim_count_per_train:
                self.cluster_count +=  1



def main(benchmark=None):

    if benchmark is None:
        benchmark = sys.argv[1]

    explorer = Explorer(benchmark)
    explorer.AdaptiveAutoKoop()
    explorer.run_tests()



if __name__ == "__main__":

    main()