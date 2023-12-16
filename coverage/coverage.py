import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.stats import multivariate_normal
from copy import copy, deepcopy
from scipy.spatial import KDTree


class Coverage():

    'This class relates to computing CPS coverage score'

    def __init__(self, file_name=None, data_list=None, sd=None,
                       lower_bound=None, upper_bound=None):

        if file_name:
            self.states = self.read_file(file_name)
        elif data_list:
            self.states = self.process_data(data_list)

        self.sd = sd
        self.dimensions = 2

        if lower_bound and upper_bound:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
        else:
            self.get_bound()

        self.tree = KDTree(self.states)  # scipy kdtree
        self.set_param()


    def set_param(self):
        self.cov = []
        self.ranges = []
        for i in range(self.dimensions):
            tmp = [0.0] * self.dimensions
            tmp[i] = self.sd[i] # for making the SD diagnal matrix
            self.cov.append(copy(tmp))
            tmp.clear()
            self.ranges.append([self.lower_bound[i], self.upper_bound[i]])


    def  _pdf(self, a=None, b=None, c=None, d=None, e=None, f=None):
        '_multi_variate_pdf supporting up to 6 dimensions'
        state = []
        if a is not None: state.append(a)
        if b is not None: state.append(b)
        if c is not None: state.append(c)
        if d is not None: state.append(d)
        if e is not None: state.append(e)
        if f is not None: state.append(f)

        if state is []:
            return 0

        'find the closest point'
        _, ii = self.tree.query(state, k=1)

        mean_ = self.states[ii]
        s = np.array(state)
        mean = np.array(mean_)
        out = multivariate_normal.pdf(s, mean, self.cov, allow_singular=True)

        return out


    def get_coverage(self):
        'integrates all the Gaussian scores obtained by each simulation'
        options = {'limit': 100}
        cov = nquad(self._pdf, self.ranges, opts=[options]*self.dimensions)[0]
        print(f"Coverage {cov}")

        return cov


    def coverage_progress(self, size):
        'Computing coverage progress through time'
        result = []
        batch_size = len(self.states) // size
        original_states_size = len(self.states)
        original_states = deepcopy(self.states)
        # print(f"len(states) {len(self.states)} size {size} "
        #       f" batch_size {batch_size}")
        for i in range(size+1):
            index = (i+1) * batch_size

            'Capturing the last batch'
            if index >= original_states_size:
                index = original_states_size

            'For the new chunck of data'
            self.states = deepcopy(original_states[0:index])
            self.get_bound()
            self.tree = KDTree(self.states)

            cov = self.get_coverage()
            result.append(cov)
            print(f"coverage at round {i}:  {cov}")
        print(f"Coverage improvement {result}")

        return cov


    def process_data(self, list_trajectories):
        trajectories = []
        for traj in list_trajectories:
            trajectories.append(traj[:, 0:self.dimensions])
        data = np.concatenate(trajectories, axis=0)

        return data


    def read_file(self, fname):
        trajectories = []
        with open(fname, 'rb') as fhandle:
            trajectories_ = np.array(pickle.load(fhandle))

        for traj in trajectories_:
            trajectories.append(traj[:, 0:self.dimensions])

        data = np.concatenate(trajectories, axis=0)

        return data


    def get_bound(self):
        lower_bound, upper_bound = [], []
        for i in range(self.dimensions):
            lower_bound.append(min(self.states[:,i]))
            upper_bound.append(max(self.states[:,i]))

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound



def plot(fname):

    fig, axes = plt.subplots(figsize=(5, 5))
    trajectories = []
    with open(fname, 'rb') as fhandle:
        trajectories_ = np.array(pickle.load(fhandle))

    for traj in trajectories_:
        trajectories.append(traj[:, 0:2])
        axes.plot(traj[:, 0], traj[:, 1], color='green')


    axes.set_xlabel('X ', fontweight='bold', fontsize=16)
    axes.set_ylabel('Y ', fontweight='bold', fontsize=16)

    # Make X  and Yaxis tick labels bold
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontweight('bold')

    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontweight('bold')

    'set ticks fontsize'
    axes.tick_params(axis='both', labelsize=10)

    # Decrease the space between the axis label and the ticks
    axes.xaxis.labelpad = -3
    axes.yaxis.labelpad = -10

    'Limit the axis'
    axes.set_xticks([-80000, -40000, 0, 30000, 60000]) # ACASXU
    axes.set_xticklabels([-80, -40, 0, 30, 40])
    axes.set_yticks([0, 25000, 50000, 75000, 100000]) # ACASXU
    axes.set_yticklabels([0, 25, 50, 75, 100])

    # axes.set_xticks([-400, -200, 0, 300, 600]) # kinematic
    # axes.set_yticks([-500, -250, 0, 200, 400]) # kinematic

    # Rotate axis ticks by 45 degrees
    # axes.tick_params(axis='x', rotation=45)
    # axes.tick_params(axis='y', rotation=45)

    plt.show()


def main():

    fname =  input("fname: ")
    plot(fname)

    # print("computing coverage")
    # sd1 = input('please insert sd1:')
    # sd2 = input('please insert sd2:')
    # sd_list = [int(sd1), int(sd2)]
    # cov_obj = Coverage(file_name=fname, sd=sd_list)
    # cov_obj.get_coverage()
    'DO NOT COMMENT THIS LINE IN ANY SITUATION'
    # print(f" Coverage: {cov}")
    # print("computing coverage progress")
    # # cov_obj.coverage_progress(size=2000)


if __name__ == "__main__":
    main()


