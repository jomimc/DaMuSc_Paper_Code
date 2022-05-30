import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def get_cents_from_ratio(ratio):
    return 1200.*np.log10(ratio)/np.log10(2)


def convert_freq_scale_to_ints(scale):
    return np.array([get_cents_from_ratio(f2/f1) for f1, f2 in zip(scale[:-1], scale[1:])])


def extract_intervals(data):
    ints = []
    freq = []
    for i in range(2, data.shape[0]):
        for j in range(1, data.shape[1]):
            try:
                f1 = int(data[i,j-1])
                f2 =int(data[i,j])
                ints.append(get_cents_from_ratio(f2/f1))
                freq.append(f2)
            except ValueError:
                pass
    return ints, freq

def load_slendro():
    df_s = pd.read_csv('new_slendro.csv')
    cols = np.array([1,2,3,4,5,1.1], dtype=str)
    for i in range(len(cols)-1):
        tmp_col = np.roll(cols, -i)
        new_col = '-'.join(tmp_col[:2])
        df_s[new_col] = get_cents_from_ratio(df_s[tmp_col[1]] / df_s[tmp_col[0]])

    intervals = np.array(df_s.loc[df_s.notnull().all(axis=1), df_s.columns[-5:]])
    scales    = np.array([ [0] + list(np.cumsum(intervals[i])) for i in range(len(intervals))])
    return intervals, scales

def load_pelog():
    df_s = pd.read_csv('new_pelog.csv')
    cols = np.array([1,2,3,4,5,6,7,1.1], dtype=str)
    for i in range(len(cols)-1):
        tmp_col = np.roll(cols, -i)
        new_col = '-'.join(tmp_col[:2])
        df_s[new_col] = get_cents_from_ratio(df_s[tmp_col[1]] / df_s[tmp_col[0]])

    intervals = np.array(df_s.loc[df_s.notnull().all(axis=1),df_s.columns[-7:]])
    scales    = np.array([ [0] + list(np.cumsum(intervals[i])) for i in range(len(intervals))])
    return intervals, scales



def get_standard_deviation(freq, ysamp='ints'):
    npairs = int(len(freq)/2)
    std = 0.
    count = 0
    for i in range(npairs):
        c1 = convert_freq_scale_to_ints(freq[i*2])
        c2 = convert_freq_scale_to_ints(freq[i*2+1])
        if ysamp == 'scale':
            c1 = np.cumsum(c1)
            c2 = np.cumsum(c2)
        for x, y in zip(c1, c2):
            if np.isfinite(x) and np.isfinite(y):
                std += (2 * ((x - y) / 2)**2)
                count += 2
    return (std / count)**0.5


def kunst_comparison():
    pelog = pd.read_csv('kunst_comparison_pelog.csv')
    slendro = pd.read_csv('kunst_comparison_slendro.csv')

    for xsamp in ['ints', 'scale']:
        print(xsamp)
        print(get_standard_deviation(pelog.values, xsamp))
        print(get_standard_deviation(slendro.values, xsamp))


if __name__ == "__main__":

    kunst_comparison()




