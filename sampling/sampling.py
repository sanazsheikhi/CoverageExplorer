import os
import sys
import numpy as np
import BoxBoundSampling as bbs
from scipy.stats import multivariate_normal


explorer_path = os.path.dirname(os.getcwd())
utility_path = explorer_path + '/utility'
sys.path.append(utility_path)


from kdtree_2 import kdtree as kdt

class KDT_builder:

    def __init__(self, states):
        self.states = states
        self.tree = kdt.create(states)


    def insert(self, node_list):
        for i in range(len(node_list)):
            n = node_list[i]
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
    convexhull = None
    bbsampler = None

    def __init__(self, benchmark, lower_bound=None, upper_bound=None,
                 convexHull=None, boxboundsampler=None, trajectories=None):
        size = 1
        self.benchmark = benchmark

        if boxboundsampler:
            'Sampling from box-bound'
            self.bbsampler = boxboundsampler
            self.lower_bound, self.upper_bound = self.bbsampler.get_bounds()
            self.states = [self.bbsampler.sample_point()]

        elif convexHull:
            'Sampling from convexhull'
            self.convexhull = convexHull
            self.lower_bound, self.upper_bound = self.convexhull.get_bounds()
            self.states = [self.convexhull.sample()[0]]

        elif lower_bound and upper_bound:
            'For Sampling from max/min range of states'
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.states = [np.random.uniform(lower_bound, upper_bound,
                                        (size, len(lower_bound)))[0]]

        elif trajectories:
            self.bbsampler = bbs.BoxBoundSampling(self.benchmark, trajectories)
            self.lower_bound, self.upper_bound = self.bbsampler.get_bounds()
            self.states = [self.bbsampler.sample_point()]

        self.kdt = KDT_builder(self.states)
        self.max_distribution = self.f(state=self.states[0], mean=self.states[0])


    def get_boxbound(self):
        return self.lower_bound, self.upper_bound


    def update_convexhull(self, convex):
        self.convexhull = convex


    def update_bounds(self, lower_bound=None, upper_bound=None):
        if lower_bound:
            self.lower_bound = lower_bound
        if upper_bound:
            self.upper_bound = upper_bound


    def update_kdt(self, traj_list):
        states = []
        for traj in traj_list:
            for i in range(traj.shape[0]):
                states.append(np.array([traj[i,0], traj[i,1]]))
        self.kdt.insert(states)


    def f(self, state, mean=None, cov_=100):
        if state is []: return 0
        cov = []
        dimension = len(state)
        for i in range(dimension):
            tmp = [0.0]*dimension
            tmp[i] = cov_
            cov.append(tmp)

        if mean is None:
            'closest point'
            mean = self.kdt.nearest_neighbor(state)
            if mean is None:
                return 0

        rv = multivariate_normal(mean, cov, allow_singular=True)

        return rv.pdf(state)


    def g(self, state, f_state=None):
        'max pdf at the mean of one of the distributions'
        if f_state is None:
            f_state = self.f(state)

        return self.max_distribution - f_state


    def rejection_sampling(self, seed=None):
        seed_ = seed
        out = None
        state = None
        factor = 1
        dimension = len(self.states[0])
        loop_conter = 0

        while out is None:
            'to avoid infinte loop in case it can not find proper case'
            loop_conter += 1
            if loop_conter > 3:
                return state

            """ we need to update seed here in case the sampled state gets 
               rejected we sample with a new seed to avoid duplicates """
            if seed_:
                np.random.seed(seed_)
                seed_ += 1

            if self.convexhull:
                state = self.convexhull.sample()[0]
            elif self.bbsampler:
                state = self.bbsampler.sample_point()
            else:
                n_points = 1
                dim = 2
                state = np.random.uniform(low=self.lower_bound,
                                          high=self.upper_bound,
                                          size=(n_points, dim))[0]
            r = self.f(state)  # envelop
            res_g = self.g(state, f_state=r)

            R = np.random.uniform(0, r * factor)
            if R < res_g:
                out = state
                self.kdt.insert([np.array(out)])

        return out




