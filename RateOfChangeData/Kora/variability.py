
import numpy as np
import pandas as pd


def load_ints():
    return np.array([l.strip('\n').split(';') for l in open('tunings.txt', 'r')], float)


def pairwise():
    ints_i, ints_j = load_ints()
    std = 0.
    count = 0
    diff_list = []
    for i0, j0 in zip(ints_i, ints_j):
        std += (2 * ((i0 - j0) / 2)**2)
        count += 2
        diff_list.extend([abs(0.5*(i0 - j0))]*2)
    np.savetxt('deviations_from_mean.txt', diff_list)
    return (std / count)**0.5



if __name__ == "__main__":

    print(pairwise())




