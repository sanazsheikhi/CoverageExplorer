
import numpy as np
from scipy.stats import multivariate_normal
from kdtree_2 import kdtree as kdt


class KDT_builder:

    def __init__(self, states):
        self.states = states
        self.tree = kdt.create(states)
        # self.tree = kdt.create([states[0]]) # making tree out of 1 state


    def insert(self, node_list):
        # print(f"insert node_list {node_list}")
        for i in range(len(node_list)):
            n = node_list[i]
            # if self.tree.search(n) == None:
            # self.tree.insert(n)
            self.tree.add(n)


    def nearest_neighbor(self, point):

        result = self.tree.search_knn(point, 1)
        if result is None:
            return None
        return result[0][0].data

    def get_tree(self):
        return self.tree


class Sampling:

    sampled = 0

    def __init__(self, lower_bound=None, upper_bound=None):

        size = 1
        self.states = [np.random.uniform(lower_bound, upper_bound, (size, len(lower_bound)))[0]] # it is already in the form of a list
        self.kdt = KDT_builder(self.states)
        self.max_distribution = self.f(state=self.states[0], mean=self.states[0])

        'For Sampling from max/min range of states'
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def update_kdt(self, traj_list):
        states = []
        for traj in traj_list:
            for i in range(traj.shape[0]):
                states.append(traj[i])

        self.kdt.insert(states)


    def f(self, state, mean=None, cov_=1):

        if state is []:
            return 0

        'cov=[[100, 0.0], ..., [0.0, 100]]'
        cov = []
        dimension = len(state)
        for i in range(dimension):
            tmp = [0.0]*dimension
            tmp[i] = cov_
            cov.append(tmp)

        # print("mean ",mean)
        if mean is None:
            'closest point'
            mean = self.kdt.nearest_neighbor(state)
            if mean is None:
                return 0

        rv = multivariate_normal(mean, cov, allow_singular=True)

        return rv.pdf(state)


    def g(self, state, f_state=None):

        'max pdf at the mean of one of the distributions, no matter which one'
        if f_state is None:
            f_state = self.f(state)

        return self.max_distribution - f_state


    def rejection_sampling(self, seed=None):

        out = None
        state = None
        factor = 1
        dimension = len(self.states[0])
        loop_conter = 0 # to avoid infiinte loop in case it can not find proper case via rejection sampling

        while out is None:

            'To avoid long/infinite loops'
            loop_conter += 1
            if loop_conter > 3:
                return state

            """we need to update seed here in case the sampled state gets rejected
               we sample with a new seed to avoid duplicates and infinite loops"""
            if seed:
                np.random.seed(seed)
                seed += 1


            state = np.random.uniform(self.lower_bound, self.upper_bound, (1, dimension))[0] # main


            r = self.f(state)  # envelop
            res_g = self.g(state, f_state=r)

            R = np.random.uniform(0, r * factor)
            if R < res_g:
                out = state
                # self.sampled += 1
                'Adding the sampled node to the tree'
                self.kdt.insert([np.array(out)])

        #print(f"rejection_sampling out {out}")

        return out




