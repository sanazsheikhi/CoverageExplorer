""" This version of coverage plotting program mostly works with pickle files. It has been changed to work with pickle files """

import os
import sys
import scipy.io
import time
import math
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from scipy.integrate import nquad, dblquad
from scipy.stats import multivariate_normal
import ast
from scipy.spatial import KDTree
from scipy.stats import norm





def plot_2D(points, lowerBound, upperBound, str_title=""):

    x, y = [], []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    'Dont need to scale numbers ust the numbers on ticks'
    # plt.figure(figsize=(20, 10), dpi=100)
    # ax = plt.gca()
    # plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
    # ax.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    plt.locator_params(nbins=4)
    plt.xlabel('X', fontsize=35, fontweight='bold')
    plt.ylabel('Y', fontsize=35, fontweight='bold')
    # plt.xlim([-20000, 20000])
    # plt.ylim([0, 100000])

    # xticks, _ = plt.xticks()
    # yticks, _ = plt.yticks()
    # xticks = [i/1000 for i in xticks]
    # yticks = [j/1000 for j in yticks]
    plt.xticks(fontsize=35, fontweight='bold')
    plt.yticks(fontsize=35, fontweight='bold')
    # ax.set_xticklabels(xticks)
    # ax.set_yticklabels(yticks)
    # plt.figure(figsize=(9.6, 7.2))  # for two fig in one column
    # plt.xticks(rotation=45)
    # plt.title(str_title)
    plt.xlim([lowerBound[0], upperBound[0]])
    plt.ylim([lowerBound[1], upperBound[1]])
    plt.scatter(np.array(x), np.array(y), s=5, color='blue')
    plt.show()


def plot_3D(points,  lowerBound, upperBound, str_title=""):

    x, y, z = [], [], []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
        z.append(points[i][2])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, z, 50, cmap='binary')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(60, 35)
    ax.set_xlim([lowerBound[0], upperBound[0]])
    ax.set_ylim([lowerBound[1], upperBound[1]])
    ax.set_zlim([lowerBound[2], upperBound[2]])
    fig






class Coverage():

    'This class relates to computing Gaussian, kernel pdf, coverage metric'

    def __init__(self):
        pass

    def set_param(self,fname, min_dimensions, max_dimensions, SD=[1,100]):

        'Eliminate empty elements'
        # self.positions = []
        # for p in positions:
        #     if p != []:
        #         self.positions.append(p)
        self.positions = read_file(fname)
        # print(f"get_coverage: positions {self.positions}")


        self.tree = KDTree(self.positions) # scipy kdtree
        self.cov = []
        self.ranges = []

        self.dimensions = len(self.positions[0])
        for i in range(self.dimensions):
            tmp = [0.0] * self.dimensions
            tmp[i] = SD[i] # for making the SD diagnal matrix
            self.cov.append(copy.copy(tmp))
            tmp.clear()
            tmp = [min_dimensions[i], max_dimensions[i]]
            self.ranges.append(tmp)


        print(f"set_param self.dimensions  {self.dimensions }")
        print(f"min_dimensions {min_dimensions}")
        print(f"max_dimensions {max_dimensions}")
        print(f"set_param self.cov {self.cov}")
        print(f"set_param self.ranges {self.ranges}")


    def  _pdf(self, a=None, b=None, c=None, d=None, e=None, f=None):

        '_multi_variate_pdf supporting up to 6 dimensions'
        state = []
        if a is not None: state.append(a)
        if b is not None: state.append(b)
        if c is not None: state.append(c)
        if d is not None: state.append(d)
        if e is not None: state.append(e)
        if f is not None: state.append(f)

        if state is []: return 0

        'find the closest point'
        _, ii = self.tree.query(state, k=1)
        mean_ = self.positions[ii]

        s = np.array(state)
        mean = np.array(mean_)
        out = multivariate_normal.pdf(s, mean, self.cov, allow_singular=True)

        # print(f"PDF: state {state} mean {mean_}  out {out}")

        return out



    def _get_coverage(self):

        'This function integrates all the Gaussian scores obtained by each simulation'
        # Index out of .. error for nquad
        options = {'limit': 100}
        # out = nquad(self._pdf, [[self.min_o, self.max_o],[self.min_i, self.max_i]], opts=[options, options])[0]
        out = nquad(self._pdf, self.ranges, opts=[options]*self.dimensions)[0]

        return out


def read_file(fname):

    'the file is either a pickle (python) or mat (matlab)'
    if '.mat' in fname:
        d = scipy.io.loadmat(fname)['out']
        data = d.tolist()
    elif 'pkl' in fname:
        with open(fname, 'rb') as fhandle:
            data = pickle.load(fhandle)

    # print(f"{data}")
    print(f"size {len(data)}")

    return data


def get_coverage(fname):

    print("Calculating coverage ....")
    min_dimensions = [0.0, 0.0, 1.0] # [speed, RPM, gear]
    max_dimensions = [120.0, 4500.0, 4.0]
    SD = [1, 100, 0.5] # [sd_speed, sd_RPM] two dimensssions have different scope
    coverage = Coverage()
    coverage.set_param(fname, min_dimensions, max_dimensions, SD)
    # plot_contour(coverage) # It definately should sit after sit param because of _pdf parameters
    cov = coverage._get_coverage()

    return cov




def main():

    fname = sys.argv[1]


    cov = get_coverage(fname)

    'DO NOT COMMENT THIS LINE IN ANY SITUATION'
    print(f"****************** coverage: {cov}")




if __name__ == "__main__":
    main()
