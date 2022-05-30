
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_tunings():
    return pd.read_csv('tunings.csv')


def get_cents_from_ratio(ratio):
    return 1200.*np.log10(ratio)/np.log10(2)


def get_ints(df, i):
    freq = df.loc[i][1:].values[::-1]
    return np.array([get_cents_from_ratio(f2/f1) for f1, f2 in zip(freq[:-1], freq[1:])])


def pairwise(df, i, j):
    std = 0.
    count = 0
    ints_i = get_ints(df, i)
    ints_j = get_ints(df, j)
    for i0, j0 in zip(ints_i, ints_j):
        std += (2 * ((i0 - j0) / 2)**2)
        count += 2
    return (std / count)**0.5


def all_pairwise(df):
    N = len(df)
    std = np.zeros((N,N), float)
    for i in range(N-1):
        for j in range(i+1, N):
            s = pairwise(df, i, j)
            std[i,j] = s
            std[j,i] = s
    return std


def overall(df, plot=False):
    diff_list = []
    std = 0.
    count = 0
    ints = np.array([get_ints(df, i) for i in range(5)])
    scales = np.cumsum(ints, axis=1)
    for X, xcat in zip([ints, scales], ['steps', 'scales']):
        for i in range(X.shape[1]):
            diff = (X[:,i] - np.mean(X[:,i]))**2
            std += np.sum(diff)
            count += X.shape[0]
            diff_list.extend(list(diff))
        if plot:
            sns.distplot(np.sqrt(diff_list))
            np.savetxt(f'deviations_from_mean_{xcat}.txt', np.sqrt(diff_list))
            plt.show()
        print(xcat)
        print((std / count)**0.5)



if __name__ == "__main__":

    df = load_tunings()
    overall(df, True)


