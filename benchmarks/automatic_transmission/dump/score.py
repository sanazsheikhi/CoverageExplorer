import pathlib
import copy
import sys

import scipy.integrate as integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
from argparse import Namespace


class GaussianDistribution:
    _max_pdfs = []

    def __init__(self, fname):
        self._crash_percents = self.read_data(fname)

    def read_data(self, fn):

        with open(fn, 'rb') as fhandle:
            percents = pickle.load(fhandle)

        return percents


    def _get_crash_count(self):
        return len(self._crash_percents)

    def pdfs(self, x):

        if len(self._crash_percents) == 0:
            print("No crash data received")
            return 0

        closest = self._crash_percents[
            min(range(len(self._crash_percents)), key=lambda i: abs(self._crash_percents[i] - x))]

        result = norm.pdf(x, closest, 1)

        # print(f"result {result}")
        return result


    def pdf_integral(self, l=0):

        score = integrate.quad(self.pdfs, 0, 100, limit=100)
        return score


    def _incremental_scores(self):
        'Plots the scores measured each 10k frames by the fuzzer'

        if len(self.incremental_scores) == 1 or len(self.incremental_scores) == 0:
            print("No incremental score received.")
            return 0

        time = list(range(0, (len(self.incremental_scores)) * 10000, 10000))

        plt.plot(time, self.incremental_scores)
        plt.xlabel('Time frames (10k unit)')
        plt.ylabel('Coverage score score')
        plt.title(f"Coverage Score improvement of {self._tool}")
        plt.savefig(f"plots/{self._tool}_coverage_improve.png")


    def _max_pdf(self):
        ''
        if len(self._max_pdfs) == 0:
            for i in range(0, 101):
                self._max_pdfs.append(self.pdfs(i))

        plt.plot(range(0, 101), self._max_pdfs)
        plt.xlabel('Projection of State Space')
        plt.ylabel("Max Score")
        plt.title('Maximum Coverage Score ')
        plt.savefig(f"plots/{self._tool}_max_coverage_score.png")


def main():

    gd = GaussianDistribution(sys.argv[1])
    print(f"Coverage Score : {gd.pdf_integral()[0]} \n")
    # gd._incremental_scores()


if __name__ == "__main__":
    main()


