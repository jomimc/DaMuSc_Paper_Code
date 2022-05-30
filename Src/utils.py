from collections import Counter, defaultdict
import json
import re
import sys
import time

import matplotlib.pyplot as plt
from itertools import permutations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import lognorm
import seaborn as sns
from sklearn.cluster import DBSCAN
import statsmodels.nonparametric.api as smnp


OCT_CUT = 50



#############################################################################
### Functions to be used in reformatting the data

def get_cents_from_ratio(ratio):
    return 1200.*np.log10(ratio)/np.log10(2)


def str_to_ints(st, delim=';'):
    return [int(s) for s in st.split(delim) if len(s)]


def ints_to_str(i):
    return ';'.join([str(x) for x in i])



#############################################################################
### Functions for extracting and reformatting the raw data


def get_all_intervals(step_ints):
    return np.array([i for j in range(len(step_ints)) for i in np.cumsum(np.roll(step_ints, j))])


def get_all_variants(scale):
    step_ints = np.diff(scale).astype(int)
    for i in range(len(step_ints)):
        yield np.cumsum(np.append([0], np.roll(step_ints, -i)))


def process_scale(scale):
    step_ints = np.diff(scale).astype(int)
    N = len(step_ints)
    tonic_ints = scale[1:] - scale[0]
    all_ints = get_all_intervals(step_ints)
    return N, step_ints, tonic_ints, all_ints


def extract_scale_using_tonic(ints, tonic, oct_cut):
    # If in str or list format, there are explicit instructions
    # for each interval
    # Otherwise, there is simply a starting note, and it should
    # not go beyond a single octave
    try:
        tonic = eval(tonic)
    except Exception as e:
        pass

    if isinstance(tonic, str):
        tonic = np.array(str_to_ints(tonic))
        tmin, tmax = min(tonic), max(tonic)

    elif isinstance(tonic, (list, np.ndarray)):
        tmin, tmax = min(tonic), max(tonic)

    elif isinstance(tonic, (int, float)):
        i_tonic = int(tonic) - 1
        tonic = np.zeros(len(ints)+1)
        tonic[i_tonic] = 1
        tonic[-1] = 2
        tmin, tmax = 1, 2

    scale = []
    for i, t1, t2 in zip(ints, tonic[:-1], tonic[1:]):
        if t1 == tmin:
            if len(scale):
                yield np.array(scale)
            scale = [0, i]

        elif len(scale):
            scale.append(i + scale[-1])

    if scale[-1] > (1200 - OCT_CUT):
        yield np.array(scale)


def extract_specific_variants(ints, tonic, variants):
    if isinstance(tonic, str):
        tonic = np.array(str_to_ints(tonic), int)
    for v in variants.split(','):
        v = str_to_ints(v)
        extra = 0
        scale = []
        for i, t in zip(ints, tonic[:-1]):
            if t == v[0]:
                if len(scale):
                    if scale[-1] > (1200 - OCT_CUT):
                        yield np.array(scale)
                scale = [0, i]
            elif len(scale) and t in v:
                scale.append(scale[-1] + i)
            elif len(scale):
                scale[-1] = scale[-1] + i
                
    if scale[-1] > (1200 - OCT_CUT):
        yield np.array(scale)


def eval_tonic(tonic):
    if isinstance(tonic, str):
        return tonic != 'N/A'
    elif isinstance(tonic, (int, float)):
        return not np.isnan(tonic)


def extract_scale_from_measurement(row, oct_cut=OCT_CUT, use_specified_variants=True, use_all_variants=False):
    ints = np.array(row.Intervals)

    # This column exists only for this instruction;
    # If 'Y', then add the final interval needed for the scale
    # to add up to an octave;
    # This exists because some the final interval is sometimes not
    # reported in papers simply because it is redundant if your analysis assumes the octave.
    # The column should only equal 'Y' if the source indicates that
    # the octave is actually used in the relevant source.
    if row.Octave_modified == 'Y':
        final_int = 1200 - sum(ints)
        yield np.array([0.] + list(np.cumsum(list(ints) + [final_int])))
        # There can only be one possible scale in this case
        return


    # STILL CONFUSION OVER THE TERM MODE!!!
    # Some sources provide an instrument tuning, and specify in which
    # ways subsets of the notes are used as scales ('variants').
    # In this case, the information is available under the column 'Variants',
    # and multiple scales can be extracted from a single tuning.
    if use_specified_variants:
        # If row.Variants is not null, this should produce some scales
        try:
            for scale in extract_specific_variants(ints, row.Tonic, row.Variants):
                yield scale
            # If not extracting all possible variants, then we can exit now
            if not use_all_variants:
                return
        except AttributeError:
            pass

    # If the entry includes information on tonality, and if
    # not using all possible variants, follow the instructions given.
    # This avoids double-counting in case use_all_variant == True
    if not use_all_variants:
        if eval_tonic(row.Tonic):
            for scale in extract_scale_using_tonic(ints, row.Tonic, oct_cut):
                if abs(1200 - scale[-1]) <= oct_cut:
                    yield scale
            return


    if sum(ints) >= (1200 - oct_cut):
        start_from = 0
        for i in range(len(ints)):
            if i < start_from:
                continue
            sum_ints = np.cumsum(ints[i:], dtype=int)
            # If the total sum of ints is less than the cutoff, ignore this entry
            if sum_ints[-1] < (1200 - OCT_CUT):
                break
            # Find the scale degree by finding the note closest to 1200
            idx_oct = np.argmin(np.abs(sum_ints-1200))
            oct_val = sum_ints[idx_oct]
            # If the total sum of ints is greater than the cutoff, move
            # on to the next potential scale
            if abs(oct_val - 1200) > OCT_CUT:
                continue
            
            # If all variants are not being used (i.e., if each interval is only
            # allowed to be counted in a scale once) then start looking
            # for new scales from this index 
            if not use_all_variants:
                start_from = idx_oct + i + 1

            yield np.array([0.] + list(sum_ints[:idx_oct+1]))


#############################################################################
### OLD CODE :: TO BE CLEANED



#############################################################################
### Clusting the scales by the distance between interval sets


def find_min_pair_int_dist(b, c):
    dist = 0.0
    for i in range(len(b)):
        dist += np.min(np.abs(c-b[i]))
    return dist


def step_int_distance(pair_ints):
    pair_dist = np.zeros((len(pair_ints), len(pair_ints)), dtype=float)
    for i in range(len(pair_ints)):
        for j in range(len(pair_ints)):
            dist1 = find_min_pair_int_dist(pair_ints[i], pair_ints[j])
            dist2 = find_min_pair_int_dist(pair_ints[j], pair_ints[i])
            pair_dist[i,j] = (dist1 + dist2) * 0.5
    return pair_dist


def cluster_pair_ints(df, n_clusters):
    step_ints = np.array([np.array([float(x) for x in y.split(';')]) for y in df.step_intervals])
    dist = step_int_distance(pair_ints)
    li = linkage(pdist(pair_dist), 'ward')
    return fcluster(li, li[-n_clusters,2], criterion='distance')


def label_scales_by_cluster(df, n=16):
    nc = cluster_pair_ints(df, n)
    df[f"cl_{n:02d}"] = nc
    return df



def distribution_statistics(X, xhi=0, N=1000):
    X = X[np.isfinite(X)]
    if xhi:
        bins = np.linspace(0, xhi, N)
    else:
        bins = np.linspace(0, np.max(X), N)
    hist = np.histogram(X, bins=bins)[0]

    bin_mid = bins[:-1] + 0.5 * np.diff(bins[:2])
    mode = bin_mid[np.argmax(hist)]
    median = np.median(X)
    mean = np.mean(X)

    shape, loc, scale = lognorm.fit(X)
    return mean, median, mode, shape, loc, scale


#############################################################################
### New from here...



def sample_df_index(df, xsamp='SocID', s=5):
    out = []
    for c in df[xsamp].unique():
        idx = df.loc[df[xsamp]==c].index
        out.extend(list(np.random.choice(idx, replace=False, size=min(s, len(idx)))))
    return np.array(out)


def sample_df_value(df, ysamp='scale', xsamp='SocID', s=5):
    out = []
    for c in df[xsamp].unique():
        if ysamp == 'scale':
            Y = df.loc[df[xsamp]==c, ysamp]
            out.extend([x for y in np.random.choice(Y, replace=False, size=min(s, len(Y))) for x in y])
        elif ysamp == 'step_intervals':
            Y = df.loc[df[xsamp]==c, 'scale']
            out.extend([x for y in np.random.choice(Y, replace=False, size=min(s, len(Y))) for x in np.diff(y)])
    return np.array(out)


def sample_shuffled_scales(df, xsamp='SocID', s=5, inc_last=False):
    out = []
    for c in df[xsamp].unique():
        int_list = df.loc[df[xsamp]==c, 'Intervals']
        for ints in np.random.choice(int_list, replace=False, size=min(s, len(int_list))):
            ints = np.array(ints)
            np.random.shuffle(ints)
            if inc_last:
                out.extend(list(np.cumsum(ints)))
            else:
                out.extend(list(np.cumsum(ints[:-1])))
    return np.array(out)


def create_new_scales(df, n_rep=10):
    ints = [x for y in df.Intervals for x in y]
    n_notes = df.scale.apply(len).values
    df_list = []

    for i in range(n_rep):
        new_ints = [np.random.choice(ints, replace=True, size=n) for n in n_notes]
        new_df = df.copy()
        new_df.Intervals = new_ints
        new_df['scale'] = new_df.Intervals.apply(np.cumsum)
        df_list.append(new_df)

    return df_list


def gini_coef(X):
    count = np.array(sorted(Counter(X).values()))
    X = np.arange(count.size) / count.size
    Y = np.cumsum(count / count.sum())
    return 0.5 - np.trapz(Y, X)
    


