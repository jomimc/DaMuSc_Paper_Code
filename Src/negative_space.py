from collections import Counter, defaultdict
from itertools import product, permutations
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from multiprocessing import Pool
import numpy as np
from palettable.scientific.diverging import Vik_10
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
from sklearn.cluster import DBSCAN

N_PROC = 20


def sum_to_n(n, size, limit=None, nMin=1):
    """Produce all lists of `size` positive integers in decreasing order
    that add up to `n`."""
    if size == 1:
        yield [n] 
        return
    if limit is None:
        limit = n 
    # The minimim size to start from is
    # at least as big as 'nMin'
    # and if there is a lot of space left to fill it is
    # as least as big as (n + size - 1)
    start = max((n + size - 1) // size, nMin)

    # The maximum size is given by 'limit'
    # or else if there is not enough space left, it is
    # less than or equal to (n - (size - 1) * nMin)
    stop = min(limit, n - (size - 1) * nMin) + 1 

    for i in range(start, stop):
        for tail in sum_to_n(n - i, size - 1, i, nMin=nMin):
            yield [i] + tail



### This function enumerates all possible sets of integers of a
#   fixed set size (i) so that they sum to a fixed total (nI).
#   The function does not care about the order in which intervals
#   are presented, so you get each set once;
#   i.e., once you get [1, 1, 1, 2], you cannot get [1, 1, 2, 1]
# Inputs:
#       i       ::  number of intervals to pick
#       nI      ::  number of divisions in the octave (grid size given by 1200 / nI)
#       iLimit  ::  size of largest interval (size in cents given by iLimit * 1200 / nI)
#       nMin    ::  size of smallest interval (size in cents given by nMin * 1200 / nI)
def get_all_interval_sets(i, nI=240, iLimit=80, nMin=4):
    ints = []
    timeS = time.time()
    print(i)
    print(nI, i, iLimit, nMin)
    for partition in sum_to_n(nI, i, limit=iLimit, nMin=nMin):
        ints.append([float(x)*(1200./float(nI)) for x in partition])
    print(len(ints), ' scales found after ...')
    print((time.time()-timeS)/60., ' minutes')
    return np.array(ints)


def possible_scales(d, lo, hi, N):
    X = list(np.linspace(lo, hi, N))
    return np.array(list(set(permutations(X))))


### This function expands the interval sets obtained previously
#   to create a set of unique scales
def expand_set(scales):
    all_scales = set()
    for s in scales:
        all_scales = all_scales.union(set(permutations(s)))
    return np.array(list(all_scales))



def create_sets_of_scales():
    imin = 60
    imax = 320
    di = 20

    i = 7
    nI = int(1200 / di)
    iLimit = int(imax / di)
    nMin = int(imin / di)

    ints = get_all_interval_sets(i, nI, iLimit, nMin)
    print(len(ints))
    all_ints = expand_set(ints)
    print(len(all_ints))
    np.save(f'../PossibleScales/possible_{i}_{di}_{imin}_{imax}.npy', all_ints)


def extract_distances(df, n, di, imin, imax, close=50, far=100):
    poss = np.load(f'../PossibleScales/possible_{n}_{di}_{imin}_{imax}.npy')
    poss = np.cumsum(poss, axis=1)[:,:n-1]
    scale = np.array([x for x in df.loc[df.n_notes==n, 'scale']])[:,1:n]
    dist = cdist(scale, poss)
    dmin = dist.min(axis=0)
    np.save(f"../PossibleScales/possible_{n}_{di}_{imin}_{imax}_md1.npy", dmin)
    np.save(f"../PossibleScales/possible_{n}_{di}_{imin}_{imax}_close{close}.npy", poss[dmin<=close])
    np.save(f"../PossibleScales/possible_{n}_{di}_{imin}_{imax}_far{far}.npy", poss[dmin>=far])


def d_by_c(df7, dist):
    cont = df7.Region
    cuniq = np.unique(cont)
    ckey = {c:i for i, c in enumerate(cuniq)}
    cidx = np.array([ckey[c] for c in cont])
    cdist = defaultdict(dict)
    for i in range(len(cuniq)):
        for j in range(len(cuniq)):
            cdist[i][j] = dist[cidx==i][:,cidx==j]
        
    return ckey, cdist


def continent_overlap(df):
    cont = df.Region
    cuniq = np.unique(cont)

    cuniq = ['Western', 'East Asia', 'South Asia', 'Middle East', 'Oceania', 'Latin America',
       'Africa', 'South East Asia']

    overlap = np.zeros((len(cuniq), len(cuniq)), float)
    ofrac = np.zeros((len(cuniq), len(cuniq)), float)
    onorm = np.zeros((len(cuniq), len(cuniq)), float)
    onorm2 = np.zeros((len(cuniq), len(cuniq)), float)
    for i in range(len(cuniq)):
        for j in range(len(cuniq)):
            if i == j:
                continue
            u1 = set(df.loc[(df.Region==cuniq[i]), 'disc'])
            u2 = set(df.loc[(df.Region==cuniq[j]), 'disc'])
            l1 = len(u1)
            l2 = len(u1.intersection(u2))

            overlap[i,j] = l2
            ofrac[i,j] = float(l2 / l1)
            onorm[i,j] = float(l2 / (l1 * len(u2)**0.5))
            onorm2[i,j] = float(l2 / (l1 + len(u2)))

    return cuniq, overlap, ofrac, onorm, onorm2


def plot_overlap(o, l, fig='', ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    im = ax.imshow(o)

    vmin, vmax = o.min(), o.max()
    vmin, vmax = -0.25, 0.25
    cmap = ListedColormap(Vik_10.hex_colors)
    bounds = np.linspace(vmin, vmax, cmap.N + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(o, cmap=cmap, norm=norm)
    
#   im = ax.imshow(o, vmin=-0.3, vmax=0.3)
#   fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, cmap=cmap, norm=norm, boundaries=bounds)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(l)))
    ax.set_xticklabels(l, rotation=60)
    ax.set_yticks(np.arange(len(l)))
    ax.set_yticklabels(l, rotation=0)


def scales_to_grid_dist(df2):
    poss = np.load('../possible7.npy')
    poss_scale = np.cumsum(poss, axis=1)
    poss = None
    df7 = df2.loc[df2.n_notes==7].reset_index(drop=True)
    s7 = np.array([[float(x) for x in y] for y in df7.scale])
    dist = cdist(poss_scale, s7[:,1:])
    min_dist = dist.min(axis=1)
    disc = dist.argmin(axis=0)
    return min_dist, disc



def group_similarity(df7, eps=100, min_s=3):
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    sdist = cdist(s7, s7)
    clust = DBSCAN(eps=eps, min_samples=min_s, metric='precomputed').fit(sdist)
    l = clust.labels_

    groups = defaultdict(int)
    for k, v in Counter(l).items():
        if k != -1:
            cont = df7.loc[l==k, 'Region'].values
            for i in range(len(cont)-1):
                for j in range(i+1, len(cont)):
                    k2 = ', '.join(sorted(set([cont[i], cont[j]])))
                    groups[k2] += 1
    return groups


def mean_distance_between_groups(df7):
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    sdist = cdist(s7, s7)
    cont = df7.Region.unique()
    dist = np.zeros((len(cont), len(cont)), float)
    np.fill_diagonal(dist, np.nan)
    for i in range(len(cont)):
        idx1 = df7.Region==cont[i]
        for j in range(len(cont)):
            if i == j:
                continue
            idx2 = df7.Region==cont[j]
            d = sdist[idx1][:,idx2].mean() / sdist[idx1].mean()
            dist[i,j] = d
    return dist, cont


def cond_prob_group(df, nc):
    cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa']
    ncont = np.array([Counter(df.Region)[c] for c in cont])
    cont_idx = {c:i for i, c in enumerate(cont)}
    dist = np.zeros((len(cont), len(cont)), float)
    
    for c, v in Counter(nc).items():
        count = Counter(df.loc[nc==c, 'Region'])
        for c1 in cont:
            i = cont_idx[c1]
            for c2 in cont:
                j = cont_idx[c2]
                if count.get(c2, 0) > 0:
                    dist[i,j] = dist[i,j] + count.get(c1, 0)

    for i, n in enumerate(ncont.astype(float)):
        dist[i] = dist[i] / n
    return cont, dist


def cont_group_similarity_2(df7):
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    sdist = cdist(s7, s7)
    li = linkage(s7, method='ward')
    all_dist = []
    for i in range(1, len(li)-1)[::-1]:
        nc = fcluster(li, li[-i, 2], criterion='distance')
        cont, dist = cond_prob_group(df7, nc)
        all_dist.append(dist)
    return cont, np.array(all_dist)


def get_dist_(inputs):
    df, nc = inputs
    np.random.seed(int(str(time.time()).split('.')[1]))
    np.random.shuffle(nc)
    cont, dist = cond_prob_group(df, nc)
    return dist


def simulate_prob(df7, idx):
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    sdist = cdist(s7, s7)
    li = linkage(s7, method='ward')
    all_dist = []
    for i in idx:
        nc = fcluster(li, li[i, 2], criterion='distance')
        with Pool(N_PROC) as pool:
            all_dist.append(np.array(list(pool.imap_unordered(get_dist_, [(df7, nc)]*1000, 5))))
    return np.array(all_dist)
        

def joint_prob(df):
    cont, ncont = np.array([[k,v] for k, v in df.Region.value_counts().items()]).T
    ncont = ncont.astype(float)
    prob = ncont / np.sum(ncont)
    return prob, np.outer(prob, prob)


def prob_given_cluster(df, nc):
    cont, ncont = np.array([[k,v] for k, v in df.Region.value_counts().items()]).T
    ncont = ncont.astype(int)
    prob = ncont / np.sum(ncont)
    pi, pj = np.meshgrid(prob, prob)
    base_prob = np.zeros(pi.shape, float)
    for i, j in product(*[range(ncont.size)]*2):
        for c, ntot in Counter(nc).items():
            imax = min(ntot, ncont[i]+1)
            base_prob[i,j] += sum([prob[j]**n / imax for m in range(1, imax) for n in range(1, min(ntot, ncont[j]+1))])
    return base_prob


def plot_cluster6(df):
    cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa']
    cols = sns.color_palette()[:len(cont)]
    fig, ax = plt.subplots(1,6)
    X = np.arange(len(cont))
    for i, c in enumerate(df.nc6.unique()):
        count = df.loc[df.nc6==c, 'Region'].value_counts()
        Y = [count.get(c2,0) for c2 in cont]
        ax[i].barh(X, Y, color=cols)
    ax[0].set_yticks(X)
    ax[0].set_yticklabels(cont)


if __name__ == "__main__":
    
    create_sets_of_scales()


    

