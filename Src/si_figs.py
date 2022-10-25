from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import pickle
import time

from alphashape import alphashape
from descartes import PolygonPatch
import geopandas
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from multiprocessing import Pool
import numpy as np
from palettable.colorbrewer.qualitative import Paired_12, Set2_8, Dark2_8, Pastel2_8, Pastel1_9
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, set_link_color_palette
from scipy.spatial.distance import pdist, cdist, jensenshannon
from scipy.stats import mannwhitneyu, pearsonr, linregress, lognorm, norm, binom, entropy
import seaborn as sns
from shapely.geometry.point import Point
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

import octave as OC
import main_figs
import process_csv
import utils

N_PROC = 8

PATH_FIG = Path("../Figures")
PATH_DATA = Path("../Figures/Data")




#####################################################################
### SI 3


def si3(df):
    g = sns.displot(df.octave_dev, binwidth=5)
    g.ax.set_xlabel('Octave deviation / cents')
    g.savefig(PATH_FIG.joinpath("si3.pdf"))



#####################################################################
### SI 4


def plot_sampling_curve(df, ax, xsamp='SocID', nmax=33):
    nsamp = np.arange(1, nmax)
    tot_len = []
    gini = []
    for n in nsamp:
        soc = OC.get_inst_subsample(df, xsamp, n)[xsamp].values
        count = np.array(list(Counter(soc).values()))
        gini.append(utils.gini_coef(soc))
        tot_len.append(len(soc))
    ax[0].plot(nsamp, gini, '-', c='k', label='Gini')
    ax[1].plot(nsamp, tot_len, '--', c=sns.color_palette()[1], label=r'$N_{samp}$')
    ax[0].set_xlabel(r"Max $N_{samp}$ per category")
    ax[0].set_ylabel('Gini coefficient')
    ax[1].set_ylabel(r'$N_{samp}$')
    ax[0].legend(bbox_to_anchor=(0.7, 0.3), frameon=False)
    ax[1].legend(bbox_to_anchor=(0.7, 0.2), frameon=False)


def si4(df):
    fig, ax = plt.subplots(1,2,figsize=(10,4))
    fig.subplots_adjust(wspace=0.5)
    for i, xsamp in enumerate(['SocID', 'Region']):
        nmax = max(df[xsamp].value_counts().values)
        plot_sampling_curve(df, (ax[i], ax[i].twinx()), xsamp, nmax)
        ax[i].set_title(xsamp)
    fig.savefig(PATH_FIG.joinpath("si4.pdf"))


#####################################################################
### SI 5


def si5(df):
    df = df.loc[df.Reduced_scale=='N']
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(5,3, height_ratios=[1, .3, 1, .4, 1.5], width_ratios=[1,.3,1])
    ax = [fig.add_subplot(gs[i*2,:]) for i in range(2)] + \
         [fig.add_subplot(gs[4,i]) for i in [0,2]]
#   fig.subplots_adjust(hspace=0.0, wspace=0.3)
#   fig, ax = plt.subplots(2,1,figsize=(12,5))
#   fig.subplots_adjust(hspace=0.5)

    count = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))[0,1,2,:,0]
    Narr = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))[0,1,2,:,1]
    prob = np.mean(count / Narr, axis=0)

    data = np.load(PATH_DATA.joinpath("int_prob_lognorm_shuffle_nonequidistant.npy"))
    count_shuf = data[0,1,2,:,0]
    Narr_shuf = data[0,1,2,:,1]
    prob_shuf = np.mean(count_shuf / Narr_shuf, axis=0)
    prob_less = np.array([binom.cdf(count[i], Narr[i], prob_shuf) for i in range(len(count))])
    prob_obs = np.min([prob_less, 1 - prob_less], axis=0)
    
    col = sns.color_palette()
    bins = np.arange(15, 5000, 30)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.

    ax[0].plot(X, prob, lw=1.2, c=col[0], label='Original')
    ax[0].plot(X, prob_shuf, lw=1.2, c=col[1], label='Shuffled')
    ax[0].legend(loc='upper right', frameon=False)

    is_less = prob_less.mean(axis=0) < 0.5
    p_m = prob_obs.mean(axis=0)
    ax[1].plot(X[is_less], p_m[is_less], 'o', c='k', label='Infrequent', fillstyle='full', ms=4)
    ax[1].plot(X[is_less==False], p_m[is_less==False], 'o', c='k', label='Frequent', fillstyle='none')
    ax[1].set_yscale('log')
    ax[1].plot(X, [0.05/X.size]*X.size, ':k')
    ax[1].legend(loc='lower right', frameon=False)

    ci = [0.025, 0.975]
    ax[1].fill_between(X, *np.quantile(prob_obs, ci, axis=0), color=col[1], alpha=0.5)

    for a in ax:
        a.set_xlim(0, 3000)
        main_figs.set_xticks(a, 600, 200, '%d')
        a.grid()
        a.set_ylabel('Probability')
        a.set_xlabel('Interval size / cents')
    
    ax[2].bar(X, np.histogram(df.largest_int, bins=bins, density=True)[0], dx, color=sns.color_palette()[2], ec='k', lw=1)
    ax[3].bar(X, np.histogram(df.second_largest_int, bins=bins, density=True)[0], dx, color=sns.color_palette()[3], ec='k', lw=1)

    for a in ax[2:]:
        a.set_ylabel('Density')
    ax[2].set_xlabel('Largest interval size / cents')
    ax[3].set_xlabel('Second largest interval size / cents')

    fs = 14
    x = [-0.06, -0.06] + [-0.15]*3
    for i, b in zip([0,1,2,3], 'ABCD'):
        ax[i].text(x[i], 1.05, b, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si5.pdf"))


#####################################################################
### SI 6


def si6():
    fig, ax = plt.subplots(7,1,figsize=(7,13))
    fig.subplots_adjust(hspace=0.6)
    ints = np.arange(200, 2605, 5)
    col = sns.color_palette()

    w1_list = [50, 75, 100, 125, 150, 175, 200]
    w2_list = [5, 10, 15, 20, 30, 40]
    for i, w1 in enumerate(w1_list):
        w2 = 10
        m1, lo1, hi1, m2, lo2, hi2 = main_figs.load_interval_data([f"../IntStats/0_w1{w1}_w2{w2}_I{i:04d}.npy" for i in ints])
        ax[i].plot(ints, m1, '-', label='Support', color=col[0])
        ax[i].fill_between(ints, lo1, hi1, color='grey')
        ax[i].plot(ints, m2, ':', label='Against', color=col[1])
        ax[i].fill_between(ints, lo2, hi2, color='grey')

        ax[i].set_ylim(0, 0.42)
        ax[i].set_title(r"$w$ = {0:d} cents".format(w1), loc='left')
    ax[-1].set_xlabel("Interval size / cents")
    ax[3].set_ylabel("Fraction of significant results")

    fig.savefig(PATH_FIG.joinpath("si6.pdf"), bbox_inches='tight')



#####################################################################
### SI 7


def si7(df):
    fig, ax = plt.subplots(6,2,figsize=(12,12))
    fig.subplots_adjust(hspace=0.5)
    multiple_dist(ax[:,0], 'step_intervals')
    multiple_dist(ax[:,1], 'scale')
    ax[0,0].legend(loc='upper right', bbox_to_anchor=(1.65, 1.6), frameon=False, ncol=4)
#   ax[0,1].legend(loc='upper right', bbox_to_anchor=(1.1, 1.3), frameon=False, ncol=4)
    ax[5,0].set_xlabel('Step / cents')
    ax[5,1].set_xlabel('Note / cents')
    for i in range(6):
        ax[i,0].set_ylabel("Density")
        main_figs.set_xticks(ax[i,0], 100, 50, '%d')#0.005, 0.00125, '%4.3f')
        main_figs.set_xticks(ax[i,1], 200, 100, '%d')#0.001, 0.00025, '%5.3f')

        lo, hi = ax[i,0].get_ylim()
        ax[i,0].set_ylim(0, hi)
        for x in np.arange(100, 500, 100):
            ax[i,0].plot([x]*2, [lo, hi], ':k', alpha=0.3)

        lo, hi = ax[i,1].get_ylim()
        ax[i,1].set_ylim(0, hi)
        for x in np.arange(100, 1200, 100):
            ax[i,1].plot([x]*2, [lo, hi], ':k', alpha=0.3)
        ax[i,1].set_xlim(-5, 1205)

    fig.savefig(PATH_FIG.joinpath("si7.pdf"), bbox_inches='tight')


def multiple_dist(ax, stem='scale'):
    cols = Paired_12.hex_colors

    data = pickle.load(open(PATH_DATA.joinpath(f"{stem}.pickle"), 'rb'))
    X = data['X']
    for i, n in enumerate(range(4, 10)):
        for j, ysamp in enumerate(['Region', 'SocID', 'Theory', 'Measured']):
            m, lo, hi = data[f"{ysamp}_{n}"]
            ax[i].plot(X, m, '-', label=ysamp, c=sns.color_palette()[j])
            ax[i].fill_between(X, lo, hi, alpha=0.5, color=sns.color_palette()[j])
#           set_ticks(ax[i], 200, 100, '%d', 0.001, 0.00025, '%5.3f')
#           ax[i].legend(loc='upper left', frameon=False)
        ax[i].set_title(f"N={n}", loc='left')

#   for i in [2,5]:
#   ax[j].set_xlim(0, xlim[j])
#   for i in range(3):
#       ax[i].set_ylabel("Density")

#   lo, hi = ax[0].get_ylim()
#   ax[0].set_ylim(0, hi)
#   for x in np.arange(100, 400, 100):
#       ax[0].plot([x]*2, [lo, hi], ':k', alpha=0.3)

#   lo, hi = ax[1].get_ylim()
#   ax[1].set_ylim(0, hi)
#   for x in np.arange(100, 1200, 100):
#       ax[1].plot([x]*2, [lo, hi], ':k', alpha=0.3)



#####################################################################
### SI 8


def annotate_scale(ax, xy, s, r=3, c='k', fs=10):
    theta = np.random.rand() * np.pi * 2
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    if x < 0:
        r *= 2
    xyt = xy + np.array([x, y])
#   xyt = xy + np.array(xyt)
    ax.annotate(s, xy=xy, xytext=xyt, arrowprops={'arrowstyle':'->'}, color=c, fontsize=fs)
    
 

def si8(df, n=5, eps=2, min_samp=5, n_ex=6, seed=8158817):
    if seed == 0:
        seed = int(str(time.time()).split('.')[1])
    np.random.seed(seed)
    print(seed)

    Reg = ['South East Asia', 'Africa', 'Oceania', 'South Asia', 'Western',
            'Latin America', 'Middle East', 'East Asia']
    reg_val = df.loc[df.n_notes==n, 'Region'].values
    col2 = sns.color_palette()
    col = np.array(Dark2_8.hex_colors)
    col_key = {r:c for r, c in zip(Reg, col)}
    colors = [col_key[r] for r in reg_val]

    scale = np.array([x for x in df.loc[df.n_notes==n, 'scale']])[:,1:-1]
    tsne = TSNE(perplexity=20).fit(scale)
    X, Y = tsne.embedding_.T

    fig = plt.figure(figsize=(16,12))
    gs = GridSpec(11,4,width_ratios=[1,1,1,1])
    ax = fig.add_subplot(gs[:6,:2])
    ax2 = np.array([[fig.add_subplot(gs[i,j]) for j in [2,3]] for i in range(6)])
    ax3 = [fig.add_subplot(gs[7:,i]) for i in range(4)]
    fig.subplots_adjust(wspace=0.3, hspace=1.0)



    clust = DBSCAN(eps=eps, min_samples=min_samp).fit(tsne.embedding_).labels_
    N = np.max(clust) + 1
    bins = np.arange(15, 1200, 30)
    xgrid = bins[:-1] + 15
    xbar = np.arange(len(Reg))
    bar_col = list(col_key.values())

    for i, (c, nc) in enumerate(sorted(Counter(clust[clust>=0]).items(), key=lambda x: x[1], reverse=True)):
        if i >= n_ex:
            continue
        print(c, nc)

        alpha_shape = alphashape(tsne.embedding_[clust==c], 0.2)
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4, color=col2[i%8]))

        sns.distplot(scale[clust==c].ravel(), kde=False, norm_hist=True, bins=bins, color=col2[i%8], ax=ax2[i,0])
#       ax2[i,0].set_title(f"C = {c}; N = {nc}")
        ax2[i,0].set_yticks([])
        ylo, yhi = ax2[i,0].get_ylim()
        scale_mean = scale[clust==c].mean(axis=0)
        for j in range(1, n):
#           ax2[i,0].plot([j*1200/n]*2, [0, yhi], ':', c='grey')
            ax2[i,0].plot([scale_mean[j-1]]*2, [0, yhi], '-k')
            ax2[i,0].fill_between([j*1200/n-30, j*1200/n+30], [0,0], [yhi,yhi], color='grey', alpha=0.5)
        ax2[i,0].set_xticks(scale_mean)
        ax2[i,0].set_xticklabels(np.round(scale_mean, 0).astype(int), rotation=30)
        sns.distplot(scale[clust==c].ravel(), kde=False, norm_hist=True, bins=bins, color=col2[i%8], ax=ax2[i,0])

        reg_count = [np.sum(reg_val[clust==c]==r) for r in Reg]
        ax2[i,1].bar(xbar, reg_count, 0.5, color=bar_col)
        ax2[i,1].set_xticks([])
        ax2[i,1].set_ylabel("Count")
    ax2[5,0].set_xlabel("Scale note / cents")
    ax2[5,1].set_xlabel("Region")
    ax2[5,1].set_xticks(xbar)
    ax2[5,1].set_xticklabels(Reg, rotation=60)

    ax.scatter(X, Y, s=20, c=colors)

    equi = np.mean(np.abs(scale - np.arange(1,n)*1200/n), axis=1)
    is_equi = equi <= 30
    ax3[0].scatter(X[is_equi], Y[is_equi], s=20, c='#4FD0E0', alpha=0.8, label='Equidistant')
    ax3[0].scatter(X[is_equi==False], Y[is_equi==False], s=20, c='grey', alpha=0.5, label='Non-equidistant')
    ax3[0].legend(loc='best', frameon=False)
    
    is_theory = df.loc[df.n_notes==n, 'Theory'] == 'Y'
    ax3[1].scatter(X[is_theory], Y[is_theory], s=20, c=col2[2], alpha=0.5, label='Theory')
    ax3[1].scatter(X[is_theory==False], Y[is_theory==False], s=20, c=col2[4], alpha=0.5, label='Measured')
    ax3[1].legend(loc='best', frameon=False)

    soc_lbls = ['Japan', 'Myanmar']
    for i, (soc, j) in enumerate(zip(['Jap5', 'Mya29'], [3,6])):
        is_soc = df.loc[df.n_notes==n, 'SocID'].values == soc
        ax3[i+2].scatter(X[is_soc==False], Y[is_soc==False], s=10, c='grey', alpha=0.5)
        ax3[i+2].scatter(X[is_soc], Y[is_soc], s=30, c=col[j], alpha=0.8, label=soc_lbls[i])
        ax3[i+2].legend(loc='best', frameon=False)


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for a in ax2.ravel():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    for a in ax3:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel("tSNE Dimension 1")
        a.set_ylabel("tSNE Dimension 2")
    for a in ax2[:,0]:
        a.spines['left'].set_visible(False)
    ax.set_xlabel("tSNE Dimension 1")
    ax.set_ylabel("tSNE Dimension 2")

    handles = [Line2D([], [], ls='', marker='o', color=c) for c in col_key.values()]
    ax.legend(handles, Reg, loc='best', frameon=False)

    fs = 14
    x = [-0.05, -0.14, -0.24] + [-0.10] * 4
    for i, (a, b) in enumerate(zip([ax, ax2[0,0], ax2[0,1]]+ax3, 'ABCDEFG')):
        a.text(x[i], 1.02, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"si8.pdf"), bbox_inches='tight')
#   return clust, tsne.embedding_


#####################################################################
### SI 9

def si9(df):
    fig, ax = plt.subplots(2,2,figsize=(8,8))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    col = ['grey', '#4DA5E8', '#D14B50']
    bins = np.arange(0, 520, 10)
    X = bins[:-1] + 0.5 * np.diff(bins[:2])

    def cumsum(X):
        return np.cumsum(X / X.sum())

    for i, (n, soc_ex, l) in enumerate(zip([5,7], ['Gam17', 'Tha44'], ['Gamelan', 'Thai'])):
        ints = np.array([x for x in df.loc[df.n_notes==n, 'step_intervals']])
        equi_ints1 = np.max(ints, axis=1) - np.min(ints, axis=1)
        equi_ints2 = np.mean(np.abs(ints - 1200/n), axis=1)
        is_soc = df.loc[df.n_notes==n, 'SocID'].values == soc_ex

        for j, equi_ints in enumerate([equi_ints1, equi_ints2]):
            ax[j,i].plot(X, cumsum(np.histogram(equi_ints, bins=bins, density=True)[0]), c=col[1], label='Real')
            ax[j,i].plot(X, cumsum(np.histogram(equi_ints[is_soc], bins=bins, density=True)[0]), c=col[2], label=l)

        iparams = {5:"20_60_420", 7:"20_60_320"}[n]
        path = f"possible_{n}_{iparams}"
        data = np.load(f"../PossibleScales/{path}.npy")
        equi_grid1 = np.max(data, axis=1) - np.min(data, axis=1)
        equi_grid2 = np.mean(np.abs(data - 1200/n), axis=1)
        for j, equi_grid in enumerate([equi_grid1, equi_grid2]):
            hist = np.histogram(equi_grid, bins=bins, density=True)[0]
            ax[j,i].plot(X, cumsum(hist), c=col[0], label='Grid')
            ax[j,i].set_ylabel("Cumulative probability")
            ax[j,i].set_title(f"N = {n}")
        ax[0,i].set_xlabel("Step size range / cents")
        ax[1,i].set_xlabel("Mean step size deviation from\nequidistant step size / cents")
        ax[0,i].set_xlim(0, 400)
        ax[1,i].set_xlim(0, 150)

    fig.savefig(PATH_FIG.joinpath(f"si9.pdf"), bbox_inches='tight')


#####################################################################
### SI 10

def tsne_real_poss(ax, n=7):
    far = np.load(PATH_DATA.joinpath(f"tsne_grid_far_{n}.npy"))
    close = np.load(PATH_DATA.joinpath(f"tsne_grid_close_{n}.npy"))
    col = ['grey', '#4FD0E0', 'k']

    ax.scatter(*far.T, color=col[0], alpha=0.1, s=3, rasterized=True, label='Grid')
    ax.scatter(*close.T, color=col[1], alpha=0.1, s=3, rasterized=True, label='Real')
    ax.plot([0], [0], 'o', c=col[2], fillstyle='none', ms=6, label='Equidistant')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("tSNE dimension 1")
    ax.set_ylabel("tSNE dimension 2")
    handles = [Line2D([], [], lw=0, marker='o', color=c) for c in col[:2]] + \
              [Line2D([], [], lw=0, marker='o', fillstyle='none', color=col[2])]
    lbls = ['Grid', 'Real', 'Equidistant']
    ax.legend(handles, lbls, loc='upper center', bbox_to_anchor=(0.5, 1.20), frameon=False, ncol=2)


def real_poss_dist(s7, ax, n=7):
    col = np.array(Set2_8.hex_colors)[[1,0]]
    sdist = cdist(s7, s7, metric='cityblock') / (n-1)
    np.fill_diagonal(sdist, sdist.max())
    iparams = {5:"20_60_420", 7:"20_60_320"}[n]
    min_dist = np.load(f"../PossibleScales/possible_{n}_{iparams}_md1.npy")
    bins = np.arange(0, 120, 1)
#   sns.distplot(sdist.min(axis=0), kde=False, norm_hist=True, label='Real-Real', ax=ax, color=col[0], bins=bins)
#   sns.distplot(min_dist, kde=False, norm_hist=True, label='Real-Grid', ax=ax, color=col[1], bins=bins)
    sns.histplot(sdist.min(axis=0), stat='density', label='Real-Real', ax=ax, color=col[0], bins=bins, cumulative=True, fill=False, element='step')
    sns.histplot(min_dist, stat='density', label='Real-Grid', ax=ax, color=col[1], bins=bins, cumulative=True, fill=False, element='step')
    yhi = ax.get_ylim()[1]
#   ax.plot([50]*2, [0, yhi*0.8], '--k', alpha=0.5)
#   ax.plot([100]*2, [0, yhi*0.8], '--k', alpha=0.5)


### Input: ascending list of notes in scale, minus the tonic and octave (0 and 1200 cents)
def get_mean_equid(scale):
    d = 0.
    n = len(scale) + 1
    for i, s in enumerate(scale):
        d += abs(s - (i + 1) * 1200 / n)
    return d / (n - 1)


def dist_equi(scales, n):
    equi = np.arange(1, n) * 1200. / n
    return cdist(scales, equi.reshape(1,n-1), metric='cityblock').ravel() / (n-1)


def equidistance_diff(soc, ax, n=7):
    col = ['grey', '#4DA5E8', '#D14B50']
    bins = np.arange(-0.5, 241, 1)
    X = bins[:-1] + np.diff(bins[:2])
    d7 = np.load(PATH_DATA.joinpath(f"real_equi_{n}.npy"))
    
    hist = np.histogram(d7, bins=bins, density=True)[0]
    ax.plot(X, np.cumsum(hist/hist.sum()), label='Real', c=col[1])

    iparams = {5:"20_60_420", 7:"20_60_320"}[n]
    path = f"possible_{n}_{iparams}"
    path_hist = PATH_DATA.joinpath(f"{path}_hist.npy")
    if path_hist.exists() and 0:
        hist = np.load(path_hist)
    else:
        data = np.cumsum(np.load(f"../PossibleScales/{path}.npy"), axis=1)[:,:-1]
        d = dist_equi(data, n)
#       print(np.quantile(d, [0.05, 0.1, 0.15, 0.2, 0.25]))
        hist = np.histogram(d, bins=bins, density=True)[0]
        np.save(path_hist, hist)

    ax.plot(X, np.cumsum(hist/hist.sum()), label="Grid", c=col[0])

    hist = np.histogram(d7[soc=='Gam17'], bins=bins, density=True)[0]
    ax.plot(X, np.cumsum(hist/hist.sum()), label='Gamelan', c=col[2])
    ax.plot([43,43], [0,1], ':k')
    ax.set_ylim(0, 1.05)


def create_maximally_ordered_scales(scales):
    N, nn = scales.shape
    tonic = np.zeros(N).reshape(N,1)
    supertonic = np.ones(N).reshape(N,1) * 1200
    scales = np.append(tonic, np.append(scales, supertonic, axis=1), axis=1)
    ints = np.diff(scales, axis=1)
    np.matrix.sort(ints)
    new_scales = []
    for i in range(nn):
        new_scales.extend(list(np.cumsum(np.roll(ints, i, axis=1), axis=1)))
    return np.array(new_scales)[:,:-1]


def create_shuffled_scales(scales, nrep=50):
    N, nn = scales.shape
    tonic = np.zeros(N).reshape(N,1)
    supertonic = np.ones(N).reshape(N,1) * 1200
    scales = np.append(tonic, np.append(scales, supertonic, axis=1), axis=1)
    ints = np.diff(scales, axis=1)
    new_scales = []
    for i in range(N):
        for j in range(nrep):
            new_scales.append(np.cumsum(np.random.choice(ints[i], nn + 1, replace=False)))
    return np.array(new_scales)[:,:-1]


def si10(df, dx=20, n=5):
    fig = plt.figure(figsize=(13,6))
    s1 = 4
    w1 = 6 * s1
    s2 = 2 * s1
    w2 = 35
    gs = GridSpec(3,4*w2+3*s2, height_ratios=[1, 0.2, 1])
    ax = [fig.add_subplot(gs[2,j*(29+16):(j+1)*29+j*16]) for j in range(4)] + [None] + \
         [fig.add_subplot(gs[0,j*(w2+s2):(j+1)*w2+j*s2]) for j in [1,2,0,3]]
    fig.subplots_adjust(wspace=0.0, hspace=0.2)
#   elif n == 5:
#       gs = GridSpec(3,12, height_ratios=[1, 0.2, 1])
#       ax = [fig.add_subplot(gs[2,j*3:(j+1)*3]) for j in range(4)] + [[],[]] + \
#            [fig.add_subplot(gs[0,i*4:(i+1)*4]) for i in [0,2,1]]

#   fig, ax = plt.subplots(1,6, figsize=(16,3))
    df7 = df.loc[df.n_notes==n].reset_index(drop=True)
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    s7_sorted = create_maximally_ordered_scales(s7)
    s7_shuffled = create_shuffled_scales(s7)
    soc = df7.SocID.values

    # tSNE embedding of grid scales
    tsne_real_poss(ax[7])


    # Distance between equidistant scales, and  real / grid scales
    equidistance_diff(soc, ax[5], n=n)
#   ax[6].set_xlim(-20, 330)


    # Distributions for real scales, similar scales, and different scales
    iparams = {5:"20_60_420", 7:"20_60_320"}[n]
    all_grid = np.cumsum(np.load(f'../PossibleScales/possible_{n}_{iparams}.npy')[:,:-1], axis=1)

    lbls = ['Real', 'Sorted', 'Shuffled', 'Grid']
#   col = ['#50AEF5', sns.color_palette()[2], sns.color_palette()[4], 'grey']
    col = ['#4DA5E8', '#E08051', '#3ABA40', 'grey']
    all_entropy = []
    for i, (d, l) in enumerate(zip([s7, s7_sorted, s7_shuffled, all_grid], lbls)):

        # Distributions of step intervals,
        if l not in ['Sorted', 'Shuffled']:
            ints = np.diff(d, axis=1)
            xlo, xhi = np.min(ints), np.max(ints)
            xlo = dx * ((xlo // dx) - 1) - int(dx/2)
            xhi = dx * (3 + xhi // dx)
            bins = np.arange(xlo, xhi, dx)
            X = bins[:-1] + np.diff(bins[:2])
            ax[6].plot(X, np.histogram(ints, bins=bins, density=True)[0], label=lbls[i], c=col[i])

        # Distributions of each scale note,
        for j, s in enumerate(d.T):
            xlo, xhi = np.min(s), np.max(s)
            xhi += dx
            bins = np.arange(dx * ((xlo // dx) - 1), dx * (2 + xhi // dx), dx)
            X = bins[:-1] + np.diff(bins[:2])
            hist = np.histogram(s, bins=bins, density=True)[0]
#           print(l, j, entropy(hist))
            all_entropy.append(entropy(hist))
            if l not in ['Sorted', 'Shuffled']:
                ax[j].plot(X, hist, label=lbls[i], c=col[i])
                ax[j].plot([(j+1)*1200/n]*2, [0, hist.max()], ':k', alpha=0.5)

                ax[j].set_xlabel(f"Note {j+2} / cents")
                ax[j].set_ylim(0, ax[j].get_ylim()[1])

    all_entropy = np.array(all_entropy).reshape(4,n-1)
    X = np.arange(2, n+1)
    for i in range(4):
        ax[8].plot(X, all_entropy[i], '-o', label=lbls[i], color=col[i], lw=0.5)
    ax[8].set_xticks(X)

    for a in [ax[0],  ax[6]]:
        a.set_ylabel("Density")
    ax[5].set_ylabel("Cumulative Probability", labelpad=-1)
#   ax[6].set_ylabel("Probability")
    ax[8].set_ylabel("Entropy")

#   ax[6].set_xlabel("Distance to nearest scale / cents")
    ax[5].set_xlabel("Mean Note Distance from\nEquiheptatonic Scale / cents")
    ax[6].set_xlabel("Step size / cents")
    ax[8].set_xlabel("Note position")

    ax[3].legend(bbox_to_anchor=(0.5, 1.0), frameon=False)
#   ax[6].legend(loc='lower right', frameon=False)
    ax[5].legend(loc='lower right', frameon=False)
    ax[6].legend(loc='upper right', frameon=False)
    ax[8].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), frameon=False, ncol=2)

    ax[5].set_xlim(0, 220)

    for a in ax[1:n-1]:
        a.spines['left'].set_visible(False)

    xmaj = [200, 200, 200, 200, 50, 100]
    for x, a in zip(xmaj, ax[:n-1] + ax[5:]):
        main_figs.set_xticks(a, x, x/2, '%d')

#   ax[2].set_xticklabels(['', 250, 500, 750, ''])

    for a in ax[:]:
        try:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_yticks([])
        except:
            pass
    ax[5].set_yticks(np.arange(0, 1.2, .2))

    fs = 14
    x = [-0.075, -0.072, -0.08, -0.15, -0.15, -0.08]
    for i, (j, b) in enumerate(zip([7,5,6,0,8], 'ABCDEF')):
        ax[j].text(x[i], 1.02, b, transform=ax[j].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"si10.pdf"), bbox_inches='tight')


#####################################################################
### SI 11-12


def plot_histogram(mean_dist, xlbls, ylbls, fig='', ax='', rot=90):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    im = ax.imshow(mean_dist)
    ax.set_xticks(range(len(mean_dist)))
    ax.set_yticks(range(len(mean_dist)))
    ax.invert_yaxis()
    ax.set_xticklabels(xlbls, rotation=rot)
    ax.set_yticklabels(ylbls)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Mean distance / cents', rotation=270)


def si11_12(df, n=7, min_samp=5, norm=False, rot=90):
    soc = df.loc[df.n_notes==n, 'SocID'].values
    societies = np.array([k for k, v in df.loc[df.n_notes==n, 'SocID'].value_counts().items() if v >= min_samp])
    df_soc = pd.read_csv(process_csv.PATH_RAW.joinpath('Metadata/societies.csv'))
    soc_name = np.array([df_soc.loc[df_soc.ID==s, 'Name.x'].values[0] for s in societies])

    scale = np.array([x for x in df.loc[df.n_notes==n, 'scale']])[:,1:-1]
    dist = cdist(scale, scale, metric='cityblock') / (n-1)
    mean_dist = np.array([[np.mean(dist[soc==s1][:,soc==s2]) for s1 in societies] for s2 in societies])
    min_dist = np.array([[np.mean(np.min(dist[soc==s1][:,soc==s2], axis=0)) for s1 in societies] for s2 in societies])
    equidist = np.mean(np.abs(scale - np.arange(1,n)*1200/n), axis=1)
    theory_frac = np.array([np.mean(df.loc[df.SocID==s, "Theory"]=='Y') for s in societies])

    if norm:
        mean_diag = np.diag(mean_dist)
        norm_factor = np.mean(np.meshgrid(mean_diag, mean_diag), axis=0)
        mean_dist = mean_dist / norm_factor

    fig = plt.figure(figsize=(12,14))
    gs = GridSpec(5,2, height_ratios=[1,0.2,0.3,0.35,0.3])
    ax = [fig.add_subplot(gs[j*2,i]) for j in [0,1,2] for i in [0,1]]
#   ax = [fig.add_subplot(gs[1,i]) for i in [0,1]] + [fig.add_subplot(gs[0,:])]
    fig.subplots_adjust(wspace=0.5, hspace=0.0)

    li = linkage(mean_dist[np.triu_indices(len(mean_dist), 1)], method='ward')
    idx = np.argsort(fcluster(li, li[0,2], criterion='distance'))
    plot_histogram(mean_dist[idx][:,idx], soc_name[idx], societies[idx], fig, ax[0], rot=rot)

    X = np.arange(len(societies))
    ax[2].plot(X, np.array([np.mean(equidist[soc==s1]) for s1 in societies[idx]]))
    ax[4].bar(X, theory_frac[idx], 0.5, ec='k', lw=0.5)
    for a in [ax[2], ax[4]]:
        a.set_xticks(X)
        a.set_xticklabels(soc_name[idx], rotation=rot)


    i, j = np.triu_indices(len(min_dist), 1)
    md_sum = min_dist[i,j] + min_dist[j,i]
    li = linkage(md_sum, method='ward')
    idx = np.argsort(fcluster(li, li[0,2], criterion='distance'))
    plot_histogram(min_dist[idx][:,idx], soc_name[idx], societies[idx], fig, ax[1])

#   ax[3].plot(X, np.array([np.sum(soc==s1) for s1 in societies[idx]]))
    ax[3].plot(X, np.array([np.mean(equidist[soc==s1]) for s1 in societies[idx]]))
    ax[5].bar(X, theory_frac[idx], 0.5, ec='k', lw=0.5)
    for a in [ax[3], ax[5]]:
        a.set_xticks(X)
        a.set_xticklabels(soc_name[idx], rotation=rot)


    ttls = ["Cluster by mean distance between all scales", "Cluster by mean distance between closest scales"]
    ylbl = ["SocID"] * 2 + ["Mean distance from\nequidistant scale / cents"] * 2 + ["Fraction theory scales"] * 2
    for i, a in enumerate(ax):
        a.set_xlabel("Society")
        if i < 2:
            a.set_title(ttls[i])
        a.set_ylabel(ylbl[i])

    if n == 5:
        fig.savefig(PATH_FIG.joinpath("si11.pdf"), bbox_inches='tight')
    elif n == 7:
        fig.savefig(PATH_FIG.joinpath("si12.pdf"), bbox_inches='tight')



#####################################################################
### SI 13-14


def si13_14(df):
    tuning_variability_database(df)
    tuning_variability_extra()


def tuning_variability_database(df):
    fig, ax = plt.subplots(4,2,figsize=(8,10))
    fig.subplots_adjust(wspace=0.3, hspace=0.7)
    idx_list = [(df.n_notes==5)&(df.SocID=='Gam17'),
                (df.n_notes==7)&(df.SocID=='Gam17'),
                (df.n_notes==7)&(df.SocID=='Gam17')&(df.RefID==12),
                (df.n_notes==7)&(df.SocID=='Tha44')]
    tlbls = ["Gamelan, Slendro", "Gamelan, Pelog", "Gamelan, Pelog (Surjodiningrat et al.)", "Thai"]
    xlbls = ["Deviation of scale note from mean / cents", "Deviation of step size from mean / cents"]
    for i, idx in enumerate(idx_list):
        for j, ysamp in enumerate(['scale', 'step_intervals']):
            ints = np.array([y for y in df.loc[idx, ysamp]])
            if ysamp == 'scale':
                ints = ints[:,1:]

            err = np.concatenate([ints[:,i] - ints[:,i].mean() for i in range(ints.shape[1])])
            sns.histplot(err, ax=ax[i,j])
            ax[i,j].set_title(tlbls[i])
            ax[i,j].set_xlabel(xlbls[j])
            ax[i,j].set_ylabel("Count")

            sigma = np.std(err)
            ax[i,j].text(0.7, 0.8, r"$\sigma$ = {0:4.1f}".format(sigma), transform=ax[i,j].transAxes)
            
    fig.savefig(PATH_FIG.joinpath("si13.pdf"), bbox_inches='tight')


def tuning_variability_extra():
    base = process_csv.PATH_BASE.joinpath("TuningVariabilityData")
    path_list = [base.joinpath("Zar/melodic_scale_note_deviation_from_mean.txt"),
                 base.joinpath("Zar/harmonic_scale_note_deviation_from_mean.txt"),
                 base.joinpath("Ney/scale_note_deviation_from_mean.txt"),
                 base.joinpath("Ney/step_interval_deviation_from_mean.txt"),
                 base.joinpath("SchneiderCarillon/interval_deviation_from_mean.txt"),
                 base.joinpath("SchneiderCarillon/equiv_interval_deviation_from_mean.txt")]

    fig, ax = plt.subplots(3,2,figsize=(8,10))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    ax = ax.reshape(ax.size)

    tlbls = ["Georgian Zar", "Georgian Zar", "Turkish Ney", "Turkish Ney", "Belgian Carillon", "Belgian Carillon"]
    xlbls = [f"Deviation of {a} from mean / cents" for a in ["melodic scale note", "harmonic scale note",
                                                             "scale note", "step interval", "interval", "equivalent interval"]]

    for i, path in enumerate(path_list):
        data = np.loadtxt(path).ravel()
        sns.histplot(data, ax=ax[i])
        ax[i].set_title(tlbls[i])
        ax[i].set_xlabel(xlbls[i])
        ax[i].set_ylabel("Count")
    
        sigma = np.nanstd(data)
        ax[i].text(0.74, 0.8, r"$\sigma$ = {0:4.1f}".format(sigma), transform=ax[i].transAxes)

    fig.savefig(PATH_FIG.joinpath("si14.pdf"), bbox_inches='tight')


#####################################################################
### SI 15


def si15(df, inst):
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(5,4, height_ratios=[1, 1, .3, 1, 1])
    ax = np.array([fig.add_subplot(gs[i//4+i//8,i%4]) for i in range(16)]).reshape(2,8)
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    fig.delaxes(ax[1,6])
    fig.delaxes(ax[1,7])

    bins = np.arange(0, 710, 10)
    X = bins[:-1] + 5
    R = df.Region.unique()
    for i in range(R.size):
#       for (j, d, xc) in zip([0,1], [inst, df], ['Intervals', 'step_intervals']):
        for (j, d, xc) in zip([0,1], [inst, df.loc[df.Theory=='Y']], ['Intervals', 'step_intervals']):
            if ((j == 1) & (i >= 6)):
                continue
            hist = np.histogram([x for y in d.loc[d.Region==R[i], xc] for x in y], bins=bins)[0]
            hist = hist / hist.sum()
            for k in range(R.size):
                if ((j == 1) & (k >= 6)):
                    continue
                ax[j,k].plot(X, hist, '-k', alpha=0.3)
            
    for i in range(R.size):
        for (j, d, xc) in zip([0,1], [inst, df.loc[df.Theory=='Y']], ['Intervals', 'step_intervals']):
            if ((j == 1) & (i >= 6)):
                continue
            hist = np.histogram([x for y in d.loc[d.Region==R[i], xc] for x in y], bins=bins)[0]
            hist = hist / hist.sum()
            ax[j,i].plot(X, hist, '-', c=sns.color_palette('husl', 8)[i], label=R[i])
            ax[j,i].legend(loc='best', frameon=False)
            if i >= 4:
                ax[j,i].set_xlabel("Step interval / cents")
            ax[j,i].set_ylabel("Frequency")
            ax[j,i].set_xlim(-5, 620)
            ax[j,i].spines['right'].set_visible(False)
            ax[j,i].spines['top'].set_visible(False)
    ax[0,0].set_title("Measured scales")
    ax[1,0].set_title("Theory scales")

    fs = 14
    for i, b in zip([0,1], 'AB'):
        ax[i,0].text(-0.2, 1.05, b, transform=ax[i,0].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath("si15.pdf"), bbox_inches='tight')


def get_distances(df, scale, Reg):
    sdist = cdist(scale, scale, metric='cityblock') / (scale.shape[1] - 1)
    idx = {r:df.Region == r for r in Reg}
    out = defaultdict(dict)
    for i in range(len(Reg)):
        for j in range(i, len(Reg)):
            sd = sdist[idx[Reg[i]]][:, idx[Reg[j]]]
            sd = sd[sd>0].ravel()
            out[Reg[i]][Reg[j]] = sd 
            out[Reg[j]][Reg[i]] = sd
    return out


def si16(df, n=7, eps=2, min_samp=5, n_ex=4, seed=52542, annot=True, embed_alg='tsne'):
    if seed == 0:
        seed = int(str(time.time()).split('.')[1])
    np.random.seed(seed)
    print(seed)

    Reg = ['South East Asia', 'Africa', 'South Asia', 'Western',
            'Latin America', 'Middle East', 'East Asia', 'Oceania']
    reg_val = df.loc[df.n_notes==n, 'Region'].values
    col2 = sns.color_palette()
    col = np.array(Dark2_8.hex_colors)
    col_key = {r:c for r, c in zip(Reg, col)}
    colors = np.array([col_key[r] for r in reg_val])

    scale = np.array([x for x in df.loc[df.n_notes==n, 'scale']])[:,1:-1]
    if embed_alg == 'tsne':
        embedding = TSNE(perplexity=20).fit(scale).embedding_
    elif embed_alg == 'umap':
        embedding = get_umap_embedding(scale)
    X, Y = embedding.T

#   fig, ax = plt.subplots(4,4,figsize=(10,10))
#   ax = ax.reshape(ax.size)
    fig = plt.figure(figsize=(16,16))
    fig.subplots_adjust(wspace=0.7, hspace=0.8)
    gs = GridSpec(5,8, height_ratios=[1, 1, .1, .7, .7])
    ax = [fig.add_subplot(gs[0,i*2:(i+1)*2]) for i in range(4)] + \
         [fig.add_subplot(gs[1,i*2+1:(i+1)*2+1]) for i in range(3)] + \
         [fig.add_subplot(gs[3,i*2:(i+1)*2]) for i in range(4)] + \
         [fig.add_subplot(gs[4,i*2+1:(i+1)*2+1]) for i in range(3)]

    Reg = ['South East Asia', 'Africa', 'South Asia', 'Western',
            'Latin America', 'Middle East', 'East Asia']
    sdist = get_distances(df.loc[df.n_notes==n], scale, Reg)
    for i, r in enumerate(Reg):
        ax[i].scatter(X, Y, s=30, c='grey', alpha=0.3)
        idx = reg_val == r
        ax[i].scatter(X[idx], Y[idx], s=30, c=colors[idx], alpha=0.9)

        lbl = 'between-regions'
        for j in range(len(Reg)):
            if i == j:
                sns.kdeplot(sdist[r][Reg[j]], color=col_key[r], ax=ax[7+i], label="within-region")
            else:
                sns.kdeplot(sdist[r][Reg[j]], color='k', ax=ax[7+i], alpha=0.3, label=lbl)
                lbl = ''
        ax[7+i].legend(loc='best', frameon=False)

        ax[i].set_title(f"{r}", loc='left')
        ax[7+i].set_title(f"{r}", loc='left')
        ax[i].set_xlabel("tSNE dimension 1")
        ax[i].set_ylabel("tSNE dimension 1")
        ax[7+i].set_xlabel("Distance between scales / cents")

    fs = 16
    for i, b in zip([0,7], 'AB'):
        ax[i].text(-0.2, 1.05, b, transform=ax[i].transAxes, fontsize=fs)
    fig.savefig(PATH_FIG.joinpath("si16.pdf"), bbox_inches='tight')


