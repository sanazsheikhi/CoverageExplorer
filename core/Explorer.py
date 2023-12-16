import os
import sys
import pickle
import mpc as mpc_
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


# explorer_path = '/home/sanaz/Documents/Projects/CoverageGuidedExplorer'
explorer_path = os.path.dirname(os.getcwd())
benchmark_path = explorer_path + '/benchmarks'
sampling_path = explorer_path + '/sampling'
coverage_path = explorer_path + '/coverage'
utility_path = explorer_path  + '/utility'
result_path = utility_path + '/results'
sys.path.append(explorer_path)
sys.path.append(sampling_path)
sys.path.append(benchmark_path)
sys.path.append(coverage_path)
sys.path.append(utility_path)
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

    def __init__(self, benchmark='kinematic_car',
                       train_sim_count=100, sim_count_per_train=100,
                       run_sim_count=100 , steps=100, dt=0.1, cluster_count=3,
                       traj_per_cluster=0.5, sd=[10,10]):
        self.sd = sd
        self.dt = dt
        self.steps = steps
        self.benchmark = benchmark
        self.train_sim_count = train_sim_count
        self.run_sim_count = run_sim_count
        self.cluster_count = cluster_count
        self.sim_count_per_train = sim_count_per_train
        self.traj_per_cluster = traj_per_cluster

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

        all_traj_states, _  = self.sim_handler.run(self.benchmark,
                              [self.run_sim_count, self.steps, self.dt])

        fname = utility_path + '/results/' + self.benchmark  + '_explorer.pkl'
        with open(fname, 'wb') as fhandle:
            pickle.dump(all_traj_states, fhandle)



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
        if traj_states is None:
            self.curr_traindata_states, self.curr_traindata_inputs = (
                self.sim_handler.get_training_data(self.benchmark,
                                                   self.sim_count_per_train,
                                                   self.steps, self.dt))
            print(f"self.curr_traindata_states: {self.curr_traindata_states}")
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
            traj_states, traj_inputs = self.sim_handler.run(self.benchmark,
                          [self.sim_count_per_train, self.steps, self.dt])

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


    def plot_train_data(self):
        plt.title = "TrainData"
        for s in self.curr_traindata_states:
            states = np.array(s)
            if self.benchmark == 'kinematic_car' or \
                    self.benchmark == "automatic_transmission":
                plt.plot(states[:,0], states[:,1])
            elif self.benchmark == 'ACASXU':
                plt.plot(states[:, 3], states[:, 4]) # acas-ownship
        plt.show()


def main(benchmark=None):

    if benchmark is None:
        benchmark = sys.argv[1]

        """
        run_sim_count: num of simulations for test runs
        train_sim_count: total num of simulations for model training  
        sim_count_per_train: num of simulations per training iteration
        traj_per_cluster: percent of trajectories selected from each cluster
        steps: num of time steps per simulation
        sd: standard deviation for cps coverage score
        """

    if benchmark == 'kinematic_car':
        # print(f"kinematic_car benchmark")
        run_sim_count = 100
        train_sim_count = 1000
        sim_count_per_train = 100
        traj_per_cluster = 0.5
        cluster_count =  10
        steps = 100
        sd = [5, 5]
        dt=0.1

    elif benchmark == 'ACASXU':
        # print(f"ACASXU benchmark")
        run_sim_count = 10
        train_sim_count = 50
        sim_count_per_train = 100
        cluster_count = 10
        traj_per_cluster = 0.5
        steps = 200000
        dt= 1.0
        sd = [1000, 1000]

    elif benchmark == 'automatic_transmission':
        # print(f"automatic_transmission benchmark")
        run_sim_count = 100
        train_sim_count = 200
        sim_count_per_train = 50
        traj_per_cluster = 1.0
        cluster_count = 5
        steps = 100
        sd = [1, 50]
        dt= 0.01

    elif benchmark == 'point_mass':
        # print(f"3D_point_mass benchmark")
        """ 
        For this case study we do not train a model using Koopman. 
        Because the system dynamics is available and the system is 
        considered white-box. However, we use mpc and rejection sampling.
        """
        # ToDo: consistent with rest of the systems!:
        sim_handler = sh.Simulation_Handler()
        all_traj_states, _  = sim_handler.run(benchmark)
        fname = utility_path + '/results/' + benchmark  + '_explorer.pkl'
        with open(fname, 'wb') as fhandle:
            pickle.dump(all_traj_states, fhandle)

        return



    explorer = Explorer(benchmark, train_sim_count, sim_count_per_train,
         run_sim_count, steps, dt, cluster_count, traj_per_cluster, sd)
    explorer.AdaptiveAutoKoop()
    explorer.run_tests()



if __name__ == "__main__":
    main('point_mass')

