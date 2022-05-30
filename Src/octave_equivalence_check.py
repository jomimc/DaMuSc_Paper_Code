import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def how_close(ints, cut=100):
    count = []
    for j, i in enumerate(ints[:-1]):
        scale = np.cumsum(ints[j:])
        di = np.min(np.abs(scale - 1200))
        if di <= cut:
            count.append(di)
    return count



def real_vs_random(ints, n_rep=50):
    ints = np.array(ints)
    real = how_close(ints)
    random = []
    for i in range(n_rep):
        np.random.shuffle(ints)
        random.extend(how_close(ints))
#   print(np.mean(random), np.mean(real))
#   print(mannwhitneyu(random, real))
    return real, random


def check_tunings_for_octave_equiv(int_list):
    df = pd.DataFrame(columns=["real_mean", "rand_mean", "mw_stat", "pval"])
    real_all = []
    rand_all = []
    for ints in int_list:
        real, random = real_vs_random(ints)
        real_all.extend(real)
        rand_all.extend(random)
        real_mean = np.mean(real)
        rand_mean = np.mean(random)
        mannwhit = mannwhitneyu(random, real)
        df.loc[len(df)] = [real_mean, rand_mean, mannwhit[0], mannwhit[1]]
    return df, real_all, rand_all


