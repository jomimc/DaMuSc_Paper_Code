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
from sklearn.preprocessing import StandardScaler
import umap

import octave as OC
import utils

N_PROC = 8

PATH_FIG = Path("../Figures")
PATH_DATA = Path("../Figures/Data")


def set_ticks(ax, xMaj, xMin, xForm, yMaj, yMin, yForm):
    ax.xaxis.set_major_locator(MultipleLocator(xMaj))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xForm))
    ax.xaxis.set_minor_locator(MultipleLocator(xMin))

    ax.yaxis.set_major_locator(MultipleLocator(yMaj))
    ax.yaxis.set_major_formatter(FormatStrFormatter(yForm))
    ax.yaxis.set_minor_locator(MultipleLocator(yMin))

def set_xticks(ax, xMaj, xMin, xForm):
    ax.xaxis.set_major_locator(MultipleLocator(xMaj))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xForm))
    ax.xaxis.set_minor_locator(MultipleLocator(xMin))


def major_ticks( ax ):
    ax.tick_params(axis='both', which='major', right='on', top='on', \
                  labelsize=12, length=6, width=2, pad=8)


def minor_ticks( ax ):
    ax.tick_params(axis='both', which='minor', right='on', top='on', \
                  labelsize=12, length=3, width=1, pad=8)



#####################################################################
### FIG 1

def scale_diagram():
    fig, ax = plt.subplots(1,2, figsize=(5,6))
    fig.subplots_adjust(wspace=0.0)

    ratio_str = ["1:1", "9:8", "5:4", "4:3", "3:2", "5:3", "15:8", "2:1"]
    ratio = np.array([1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2])
    ints = np.log2(ratio) * 1200
#   ints = np.cumsum(np.array([0, 200, 200, 100, 200, 200, 200, 100], float))
    freq = 261.626 * 2**(ints/1200)

    for i in range(ints.size):
        ax[0].plot([0,1.5], [freq[i]]*2, '-k')
        ax[1].plot([0,1.5], [ints[i]]*2, '-k')
        ax[0].text(0.05, freq[i]+3, f"{int(round(freq[i]))} Hz", fontsize=10)
        ax[1].text(0.05, ints[i]+15, f"{int(round(ints[i]))} cents", fontsize=10)
        ax[1].text(2.50, ints[i]+15, f"{ratio_str[i]}", fontsize=10)


    for a in ax:
        a.set_xlim(0, 3)
        a.set_xticks([])
        a.set_yticks([])
        for side in ['top', 'bottom', 'right', 'left']:
            a.spines[side].set_visible(False)

    fig.savefig(PATH_FIG.joinpath(f"scale_example.svg"), bbox_inches='tight')



#####################################################################
### FIG 2


def fig2(df):
    df = df.loc[(df.n_notes>3)&(df.n_notes<10)].reset_index(drop=True)
#   df.loc[df.Country=='Laos', 'Country'] = "Lao PDR"
    df.loc[df.Country=='Singapore', 'Country'] = "Malaysia"
    df.loc[df.Country=='Korea', 'Country'] = "South Korea"

    counts = df.loc[(df.Theory=='N')&(df.Country.str.len()>0), 'Country'].value_counts()
    countries = counts.keys()
    co = counts.values

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world['cent_col'] = world.centroid.values

    coord = [world.loc[world.name==c, 'cent_col'].values[0] for c in countries]
    gdf = geopandas.GeoDataFrame(pd.DataFrame(data={'Country':countries, 'count':co, 'coord':coord}), geometry='coord')
    

    Cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', 'Africa', 'Oceania', 'Latin America']
    theory = [len(df.loc[(df.Theory=='Y')&(df.Region==c)]) for c in Cont]
    inst   = [len(df.loc[(df.Theory=='N')&(df.Region==c)]) for c in Cont]

    cont_coord = [Point(*x) for x in [[17, 48], [32, 33], [79, 24], [110, 32], [107, 12], [18, 8], [150, -20], [-70, -10]]]

    cont_df = geopandas.GeoDataFrame(pd.DataFrame(data={'Cont':Cont, 'count':theory, 'coord':cont_coord}), geometry='coord')

    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(2,3, width_ratios=[1.0, 7.0, 1.0], height_ratios=[1,0.6])
    gs.update(wspace=0.1 ,hspace=0.10)
    ax = [fig.add_subplot(gs[0,:]), fig.add_subplot(gs[1,1])]
    col = np.array(Set2_8.mpl_colors)[[1,0]]
    col = [Set2_8.mpl_colors[1], "#4DA5E8"]
    ft1 = 12

    ec = 0.7
    world.plot(ax=ax[0], color=(1.0, 1.0, 1.0), edgecolor=(ec, ec, ec), lw=0.5)
    world.loc[world.name.apply(lambda x: x in countries)].plot(ax=ax[0], color=(0.6, 0.6, 0.6), edgecolor=(1.0,1.0,1.0), lw=0.2)
    gdf.plot(color=col[0], ax=ax[0], markersize=gdf['count'].values*0.8, alpha=1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim(-185, 185)
    ax[0].set_ylim(-60, 88)

    width = 0.4
    X = np.arange(len(Cont))
    ax[1].bar(X - width/2, theory, width, label='Theory', color=col[1], alpha=0.8)
    ax[1].bar(X + width/2, inst, width, label='Measured', color=np.array([col[0]])*0.9, alpha=0.8)
    xtra = {1: 0.1, 2:0.05, 3:-0.05}
    for i in range(len(theory)):
        ax[1].annotate(str(theory[i]), (X[i] - 0.4 + xtra[len(str(theory[i]))], theory[i]+5), fontsize=ft1)
        ax[1].annotate(str(inst[i]), (X[i] + xtra[len(str(inst[i]))], inst[i]+5), fontsize=ft1)

    ax[1].set_xticks(X)
    [tick.label.set_fontsize(ft1) for tick in ax[1].xaxis.get_major_ticks()]
    [tick.label.set_fontsize(ft1) for tick in ax[1].yaxis.get_major_ticks()]
    Cont = ['Western', 'Middle East', 'South Asia', 'East Asia', 'South East Asia', '     Africa    ', 'Oceania', 'Latin America']
    ax[1].set_xticklabels(Cont, rotation=28, fontsize=ft1)
    ax[1].legend(loc='upper right', frameon=False, fontsize=ft1)
    ax[1].set_ylabel('Number of scales', fontsize=ft1+2)
    ax[1].set_ylim(0, 220)

    for direction in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[direction].set_visible(False)
    for direction in ['top', 'right']:
        ax[1].spines[direction].set_visible(False)

    fig.savefig(PATH_FIG.joinpath(f"world_map.pdf"), bbox_inches='tight')
    

#####################################################################
### FIG 3

def fig3(df, ysamp='scale', xsamp='SocID', s=5):
    df = df.loc[df.Reduced_scale=='N']
    fig = plt.figure(figsize=(13,8))
    gs = GridSpec(7,2, height_ratios=[1.5, .6, 1, .3, 1, .3, 1], width_ratios=[2,1])
    ax = [fig.add_subplot(gs[0,i]) for i in range(2)] + \
         [fig.add_subplot(gs[i*2,:]) for i in range(1,4)]
    fig.subplots_adjust(hspace=0.0, wspace=0.3)

    count = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))[0,1,2,:,0]
    Narr = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))[0,1,2,:,1]
    prob = count / Narr
    prob_less = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))[0,1,2,:,2]

    count_t1 = np.load(PATH_DATA.joinpath("int_prob_lognorm_shuffle.npy"))[0,1,2,:,0]
    Narr_t1 = np.load(PATH_DATA.joinpath("int_prob_lognorm_shuffle.npy"))[0,1,2,:,1]
    prob_t1 = np.mean(count_t1 / Narr_t1, axis=0)

    count_t2 = np.array([np.load(PATH_DATA.joinpath(f"int_prob_lognorm_alt{i}.npy"))[0,1,2,:,0] for i in range(10)]).mean(axis=0)
    Narr_t2 = np.array([np.load(PATH_DATA.joinpath(f"int_prob_lognorm_alt{i}.npy"))[0,1,2,:,1] for i in range(10)]).mean(axis=0)
    prob_t2 = np.mean(count_t2 / Narr_t2, axis=0)

    prob_less_t1 = np.array([binom.cdf(count[i], Narr[i], prob_t1) for i in range(len(count))])
    prob_less_t2 = np.array([binom.cdf(count[i], Narr[i], prob_t2) for i in range(len(count))])

    prob_obs = np.min([prob_less, 1 - prob_less], axis=0)
    prob_obs_t1 = np.min([prob_less_t1, 1 - prob_less_t1], axis=0)
    prob_obs_t2 = np.min([prob_less_t2, 1 - prob_less_t2], axis=0)

    bins = np.arange(15, 5000, 30)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.

    step_intervals(ax[1], df, n_rep=10, dx=10, xhi=1000)

    col = sns.color_palette()
    ci = [0.025, 0.975]
    ax[-3].fill_between(X, *np.quantile(prob_obs, ci, axis=0), color=col[0], alpha=0.5)
    ax[-2].fill_between(X, *np.quantile(prob_obs_t1, ci, axis=0), color=col[1], alpha=0.5)
    ax[-1].fill_between(X, *np.quantile(prob_obs_t2, ci, axis=0), color=col[2], alpha=0.5)

    Y = utils.sample_df_value(df, ysamp, xsamp, s)
    shape, loc, scale = [0.93, -45.9, 605.4]
    params = lognorm.fit(Y, loc=loc, scale=scale)
    bin_prob = np.diff(lognorm.cdf(bins, *params))

    count = np.histogram(Y, bins=bins)[0]
    N = count.sum()

    ax[0].plot(X, prob.mean(axis=0)*100, '-', c=col[0], lw=1.2, label='Original')
    ax[0].plot(X, bin_prob*100, '--k', label='Lognorm')
    ax[0].plot(X, prob_t1*100, '-', c=col[1], lw=1.2, label='Shuffle')
    ax[0].plot(X, prob_t2*100, '-', c=col[2], lw=1.2, label='Resample')
    ax[0].legend(loc='upper right', bbox_to_anchor=(1.02, 0.965), frameon=False)

    for i, (pl, po) in enumerate(zip([prob_less, prob_less_t1, prob_less_t2], [prob_obs, prob_obs_t1, prob_obs_t2])):
        is_less = pl.mean(axis=0) < 0.5
        p_m = po.mean(axis=0)
        ax[i-3].plot(X[is_less], p_m[is_less], 'o', c='k', label='Infrequent', fillstyle='full', ms=4)
        ax[i-3].plot(X[is_less==False], p_m[is_less==False], 'o', c='k', label='Frequent', fillstyle='none')
        ax[i-3].set_yscale('log')
        ax[i-3].plot(X, [0.05/X.size]*X.size, ':k')
        ax[i-3].legend(loc='lower right', bbox_to_anchor=(0.95, 0.13), frameon=False)

    for a in ax:
        a.set_xlim(0, 3000)
        a.grid()
    for a in ax[:2]:
        a.set_ylabel('Probability / %')
    ax[0].set_xlabel(r'Scale note / cents')
    ax[1].set_xlabel('Step size / cents')
    ax[-1].set_xlabel('Scale note / cents')

    for a in ax[2:]:
        set_xticks(a, 600, 200, '%d')
        a.set_ylabel('Probability')
        a.set_yticks([10.**x for x in [-15, -10, -5]])
        ylo, yhi = a.get_ylim()
        for x in [200, 700]:
            a.plot([x]*2, [ylo, yhi], '--', c='grey')
        a.set_ylim(ylo, yhi)


    fs = 14
    x = [-0.12, -0.24] + [-0.07]*3
    for i, b in zip([0,1,2,3,4], 'ABCDE'):
        ax[i].text(x[i], 1.05, b, transform=ax[i].transAxes, fontsize=fs)

    set_ticks(ax[0], 600, 200, '%d', 1, 0.5, '%d')
    set_ticks(ax[1], 200,  50, '%d', 2, 1, '%d')
    ax[1].set_xlim(0, 1000)

    fig.savefig(PATH_FIG.joinpath(f"fig3.pdf"), bbox_inches='tight')


def step_intervals(ax, df, n_rep=10, dx=10, xhi=1000, xmax=5000):
    df = df.loc[utils.sample_df_index(df, 'SocID', 5)]
    Y = [x for y in df.scale for x in y]
    shape, loc, scale = [0.93, -45.9, 605.4]
    params = lognorm.fit(Y, loc=loc, scale=scale)
    new_ints = [x for i in range(n_rep) for y in df.Intervals for x in np.diff(sorted(lognorm.rvs(*params, len(y))))]
    bins = np.arange(0, xmax, dx)
    X = bins[:-1] + dx / 2
    hist_lognorm = np.histogram(new_ints, bins=bins, density=True)[0]
    ax.plot(X, hist_lognorm/hist_lognorm.sum()*100, '--k', label='Lognorm')

    X = bins[:-1] + np.sum(bins[:2]) / 2
    data = np.load(PATH_DATA.joinpath(f"step_int_density.npy"))
    Y = data.mean(axis=0)
    Ynorm = Y.sum() / 100
    col = sns.color_palette()[0]
    ax.plot(X, Y/Ynorm, label='Original', c=col)
    ylo, yhi = np.quantile(data/Ynorm, [0.025, 0.975], axis=0)
    ax.fill_between(X, ylo, yhi, color=col, alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.75), frameon=False)
    ax.set_xlim(0, xhi)


#####################################################################
### FIG 4

def fig4(res):
    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(3,2, width_ratios=[1.5, 1], height_ratios=[1,.6,1.5])
    ax = [fig.add_subplot(gs[2,:]), None, fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])]
    fig.subplots_adjust(hspace=0.05, wspace=0.3)
    ints = np.arange(200, 2605, 5)
    col = [Set2_8.mpl_colors[1], "#4DA5E8"]

    lbls = ['Greater freq than chance', 'Less freq than chance']

    m1, lo1, hi1, m2, lo2, hi2 = np.array([load_interval_data([f"../IntStats/{j}_w1100_w220_I{i:04d}.npy" for i in ints]) for j in range(1,4)]).mean(axis=0)
    ax[0].fill_between(ints, lo1, hi1, color='grey', alpha=0.5, label='Null')
    ax[0].fill_between(ints, lo2, hi2, color='grey', alpha=0.5)

    m1, lo1, hi1, m2, lo2, hi2 = load_sampled_interval_data(ints, 'cult', 5)
    ax[0].plot(ints, m1, '-', label='Close', color=col[0])
    ax[0].fill_between(ints, lo1, hi1, color=col[0], alpha=0.5)
    ax[0].plot(ints, m2, '-', label='Far', color=col[1])
    ax[0].fill_between(ints, lo2, hi2, color=col[1], alpha=0.5)


    ax[0].set_ylabel("Fraction of significant results")
    ax[0].set_xticks(range(0, 2700, 200))
    ax[0].xaxis.set_tick_params(which='minor', bottom=True)
    ax[0].set_xlabel("Interval size / cents")

    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.80), frameon=False, ncol=2)
    ax[0].grid()

    col2 = ['grey', 'k'] + list(np.array(sns.color_palette('dark'))[[2,4]])
    lbls = ['NS', ' ', r"$p<0.05$", r"$p<0.005$"]
    for i, (sig, ms, l) in enumerate(zip(['NS', '', '*', '**'], [4, 0, 5, 5], lbls)):
        X, Y = res.loc[res.sig==sig, ['mean_real', 'mean_shuf']].values.T
        print(i, ms, l)
        ax[2].plot(X, Y, 'o', c=col2[i], alpha=0.5, label=l, ms=ms)
    mx = max(res.mean_real.max(), res.mean_shuf.max())
    ax[2].plot([0, mx], [0, mx], '-k')
    ax[2].set_xlabel("Deviation of original intervals\nfrom the octave")
    ax[2].set_ylabel("Deviation of shuffled intervals\nfrom the octave")
    ax[2].legend(bbox_to_anchor=(1.0,0.45), frameon=False, handletextpad=0, ncol=2)

    octave_sig(ax[3])

    ax[2].annotate('A', (-0.18, 1.05), xycoords='axes fraction', fontsize=16)
    ax[0].annotate('B', (-0.08, 1.05), xycoords='axes fraction', fontsize=16)
    ax[3].annotate('C', (-0.25, 1.05), xycoords='axes fraction', fontsize=16)

    for i, a in enumerate(ax[:2]):
        if i == 1:
            continue
        a.set_xlim(0, 2620)
        a.set_ylim(0, 0.47)
        set_ticks(a, 400, 100, '%d', 0.2, 0.1, '%3.1f')
        a.tick_params(axis='both', which='major', direction='in', length=6, width=2)
        a.tick_params(axis='both', which='minor', direction='in', length=4, width=1)

    fig.savefig(PATH_FIG.joinpath("fig4.pdf"), bbox_inches='tight')


def load_sampled_interval_data(ints, xsamp, n):
    data = np.array([[np.load(f"../IntStats/{xsamp}samp{j}_w1100_w220_I{i:04d}.npy") for i in ints] for j in range(n)])
    data = np.concatenate([d for d in data], axis=2)
    total = data.sum(axis=1)
    Y1 = data[:,0,:] / total
    Y2 = data[:,1,:] / total
    out = []
    for y in [Y1, Y2]:
        out.append(np.mean(y, axis=1))
        out.append(np.quantile(y, 0.025, axis=1))
        out.append(np.quantile(y, 0.975, axis=1))
    return out


def load_interval_data(path_list):
    data = np.array([np.load(path) for path in path_list])
    total = data.sum(axis=1)
    Y1 = data[:,0,:] / total
    Y2 = data[:,1,:] / total
    out = []
    for y in [Y1, Y2]:
        out.append(np.mean(y, axis=1))
        out.append(np.quantile(y, 0.025, axis=1))
        out.append(np.quantile(y, 0.975, axis=1))
    return out


def octave_sig(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    sigma = np.arange(0, 55, 5)
    sigma = list(range(20)) + list(range(20,55,5))
    data = np.array([np.load(f"../IntStats/sigma{s}_w1100_w220_I1200.npy") for s in sigma])
    lbls = ['Close', 'Far']
    col = [Set2_8.mpl_colors[1], "#4DA5E8"]
    for i in range(2):
        frac = data[:,i] / data.sum(axis=1)
        m = np.mean(frac, axis=1)
        lo = np.quantile(frac, 0.025, axis=1)
        hi = np.quantile(frac, 0.975, axis=1)
        ax.plot(sigma, m, label=lbls[i], c=col[i])
        ax.fill_between(sigma, hi, lo, color='grey', alpha=0.5)
        if i == 0:
            x0 = sigma[np.argmin(np.abs(m - 0.3477))]

    ax.plot([x0]*2, [0, 1], ':k')
    ax.set_ylim(0, 1)

    ax.set_xlabel(r"Tuning deviation, $\sigma$ / cents")
    ax.set_ylabel("Fraction of\nsignificant results")
    ax.legend(loc='best', frameon=False)



#####################################################################
### FIG 5


def fig5():
    fig = plt.figure(figsize=(6,7.5))
    gs = GridSpec(3,1)
    ax = [fig.add_subplot(gs[i,0]) for i in range(3)]
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    scale_degree(ax[0])
    multiple_dist(ax[1:])

    fs = 14
    x = [-0.12, -0.24] + [-0.07]*3
    for i, b in zip([0,1,2,3,4], 'ABC'):
        ax[i].text(-0.1, 1.05, b, transform=ax[i].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"fig5.pdf"), bbox_inches='tight')


def scale_degree(ax):
    data = pickle.load(open(PATH_DATA.joinpath("scale_degree.pickle"), 'rb'))
    X = data['X']
    width = 0.15
    lbls = ['Region', 'SocID', 'Theory', 'Measured']
    cols = sns.color_palette()
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        err = np.array([m - lo, hi - m])
        ax.bar(X + (i-2)*width, m, width, yerr=err, color=cols[i], label=l, ec='k', alpha=0.7)

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Number of Notes")
    ax.set_ylabel("Probability")


def multiple_dist(ax):
    lbl = ['Region', 'SocID', 'Theory', 'Measured']
#   lbl = ['Theory', 'Measured']
    path_stem = ['step_intervals', 'scale']
    xlbls = ['Step / cents', 'Note / cents']
    xlim = [530, 1270]
    cols = Paired_12.hex_colors

    for j, stem in enumerate(path_stem):
        data = pickle.load(open(PATH_DATA.joinpath(f"{stem}.pickle"), 'rb'))
        X = data['X']
        for k, l in enumerate(lbl):
            m, lo, hi = data[l]
            ax[j].plot(X, m, '-', label=l)
            ax[j].fill_between(X, lo, hi, alpha=0.5)

        ax[j].set_xlabel(xlbls[j])
        if j == 0:
            set_ticks(ax[j], 100, 50, '%d', 0.005, 0.00125, '%4.3f')
        else:
            set_ticks(ax[j], 200, 100, '%d', 0.001, 0.00025, '%5.3f')
            set_xticks(ax[j], 200, 100, '%d')
        ax[j].set_xlim(0, xlim[j])
        ax[j].set_ylabel("Density")

    lo, hi = ax[0].get_ylim()
    ax[0].set_ylim(0, hi)
    for x in np.arange(100, 400, 100):
        ax[0].plot([x]*2, [lo, hi], ':k', alpha=0.3)

    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(0, hi)
    for x in np.arange(100, 1200, 100):
        ax[1].plot([x]*2, [lo, hi], ':k', alpha=0.3)

    ax[0].legend(loc='upper right', frameon=False)




#####################################################################
### FIG 6


def annotate_scale(ax, xy, s, r=3, c='k', fs=10):
    theta = np.random.rand() * np.pi * 2
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    if x < 0:
        r *= 2
    xyt = xy + np.array([x, y])
    ax.annotate(s, xy=xy, xytext=xyt, arrowprops={'arrowstyle':'->'}, color=c, fontsize=fs)
    
 
def get_umap_embedding(X):
    reducer = umap.UMAP(n_components=2, n_neighbors=20)
    return reducer.fit_transform(StandardScaler().fit_transform(X))


def fig6(df, n=7, eps=2, min_samp=5, n_ex=4, seed=52542, annot=True, embed_alg='tsne'):
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
    if embed_alg == 'tsne':
        embedding = TSNE(perplexity=20).fit(scale).embedding_
    elif embed_alg == 'umap':
        embedding = get_umap_embedding(scale)
    X, Y = embedding.T

    fig = plt.figure(figsize=(15,11))
    gs = GridSpec(8,8)
    ax = fig.add_subplot(gs[:4,:3])
    ax2 = np.array([[fig.add_subplot(gs[i,3:6]), fig.add_subplot(gs[i,6:])] for i in range(4)])
    ax3 = [fig.add_subplot(gs[5:,i*2:(i+1)*2]) for i in range(4)]
    fig.subplots_adjust(wspace=0.5, hspace=0.6)

    if annot:
        diatonic_modes = ['Ionian', 'Aeolian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Locrian']
        dia_xy_vec = [(-6,7), (6,0), (-16,-1), (8,-1), (-12,-5), (-2,7), (2,6)]

        thaats = ['Asavari', 'Bhairav', 'Bhairavi', 'Bilawal', 'Kalyan', 'Khafi', 'Khamaj', 'Marwa', 'Purvi', 'Todi']
        tha_xy_vec = [(5,-7), (9,-2), (9,-4), (-7,-14), (1,-6), (-12,2), (-10,4), (-2.5,-7.2), (17,-2), (5,-8)]

#       other_scales = ["Nev'eser Makam", "Maqam Sultani Yakah", "Mela Chitrambari", "Tsinganikos",
#                       "Mela Pavani", "Dastgah-e Homayun", "Maqam Nakriz", "Mela Kantamani", "Huzzam Makam", "Maqam Saba"]
#       melakarta = df.loc[(df.n_notes==n)&(df.SocID=='Car9'), 'Name'].unique()
        melakarta = ['Mela Salagam']

        for s, xy in zip(thaats, tha_xy_vec):
            i = np.where(df.loc[df.n_notes==n, 'Name']==s)[0][0]
            ax.annotate(s, (X[i], Y[i]), (X[i] + xy[0], Y[i] + xy[1]), arrowprops={'arrowstyle':'->'}, color='r')

        for s, xy in zip(diatonic_modes, dia_xy_vec):
            i = np.where(df.loc[df.n_notes==n, 'Name']==s)[0][0]
            ax.annotate(s, (X[i], Y[i]), (X[i] + xy[0], Y[i] + xy[1]), arrowprops={'arrowstyle':'->'}, color='k')

#       for s in other_scales:
        for s in melakarta:
            i = np.where(df.loc[df.n_notes==n, 'Name']==s)[0][0]
            ax3[2].annotate(s, (X[i], Y[i]), (X[i] - 23, Y[i] - 18), arrowprops={'arrowstyle':'->'}, color='k')



    clust = DBSCAN(eps=eps, min_samples=min_samp).fit(embedding).labels_
    N = np.max(clust) + 1
    bins = np.arange(15, 1200, 30)
    xgrid = bins[:-1] + 15
    xbar = np.arange(len(Reg))
    bar_col = list(col_key.values())

    for i, (c, nc) in enumerate(sorted(Counter(clust[clust>0]).items(), key=lambda x: x[1], reverse=True)):
        if i >= n_ex:
            continue
        print(i, c, nc)

        alpha_shape = alphashape(embedding[clust==c], 0.2)
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4, color=col2[i%8]))

        sns.distplot(scale[clust==c].ravel(), kde=False, norm_hist=True, bins=bins, color=col2[i%8], ax=ax2[i,0])
        ax2[i,0].set_yticks([])
        ylo, yhi = ax2[i,0].get_ylim()
        scale_mean = scale[clust==c].mean(axis=0)
        for j in range(1, n):
            ax2[i,0].fill_between([j*1200/n-30, j*1200/n+30], [0,0], [yhi,yhi], color='grey', alpha=0.5)
            ax2[i,0].plot([scale_mean[j-1]]*2, [0, yhi], '-k')
        ax2[i,0].set_xticks(scale_mean)
        ax2[i,0].set_xticklabels(np.round(scale_mean, 0).astype(int), rotation=30)
        sns.distplot(scale[clust==c].ravel(), kde=False, norm_hist=True, bins=bins, color=col2[i%8], ax=ax2[i,0])

        reg_count = [np.sum(reg_val[clust==c]==r) for r in Reg]
        ax2[i,1].bar(xbar, reg_count, 0.5, color=bar_col)
        ax2[i,1].set_xticks([])
        ax2[i,1].set_ylabel("Count")
    ax2[3,0].set_xlabel("Scale note / cents")
    ax2[3,1].set_xlabel("")
    ax2[3,1].set_xticks(xbar)
    Reg = ['South East Asia', '    Africa    ', 'Oceania', 'South Asia', 'Western',
            'Latin America', 'Middle East', 'East Asia']
    ax2[3,1].set_xticklabels(Reg, rotation=60)

    ax.scatter(X, Y, s=20, c='grey', alpha=0.5)

    equi = np.mean(np.abs(scale - np.arange(1,n)*1200/n), axis=1)
    is_equi = equi <= 30
    ax3[0].scatter(X[is_equi==False], Y[is_equi==False], s=20, c='grey', alpha=0.3, label='Non-equidistant')
    ax3[0].scatter(X[is_equi], Y[is_equi], s=20, c='#4FD0E0', alpha=0.8, label='Equidistant')

    is_theory = df.loc[df.n_notes==n, 'Theory'] == 'Y'
    ax3[1].scatter(X[is_theory==False], Y[is_theory==False], s=20, c=col2[2], alpha=0.3, label='Measured')
    ax3[1].scatter(X[is_theory], Y[is_theory], s=20, c=col2[4], alpha=0.3, label='Theory')
    
    soc_lbls = ['Carnatic', 'Thai']
    for i, (soc, j) in enumerate(zip(['Car9', 'Tha44'], [3,0])):
        is_soc = df.loc[df.n_notes==n, 'SocID'].values == soc
        ax3[i+2].scatter(X[is_soc==False], Y[is_soc==False], s=10, c='grey', alpha=0.3)
        ax3[i+2].scatter(X[is_soc], Y[is_soc], s=30, c=col[j], alpha=0.65, label=soc_lbls[i])

    for a in ax3:
        a.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False, ncol=2, columnspacing=0.6, handletextpad=0.3)


    for a in ax2.ravel():
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    for a in ax3 + [ax]:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_xlabel("tSNE dimension 1")
        a.set_ylabel("tSNE dimension 2")
        a.set_xticks([])
        a.set_yticks([])
    for a in ax2[:,0]:
        a.set_ylabel("Density")
        a.set_yticks([])
#       a.spines['left'].set_visible(False)

    fs = 14
    x = [-0.05, -0.14, -0.24] + [-0.10] * 4
    for i, (a, b) in enumerate(zip([ax, ax2[0,0], ax2[0,1]]+ax3, 'ABCDEFG')):
        a.text(x[i], 1.02, b, transform=a.transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"fig6.pdf"), bbox_inches='tight')
    return clust, embedding



#####################################################################
### FIG 7


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
    sns.histplot(sdist.min(axis=0), stat='density', label='Real-Real', ax=ax, color=col[0], bins=bins, cumulative=True, fill=False, element='step')
    sns.histplot(min_dist, stat='density', label='Real-Grid', ax=ax, color=col[1], bins=bins, cumulative=True, fill=False, element='step')
    yhi = ax.get_ylim()[1]


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
#   print('\n', np.quantile(d7, [0.75, 0.8, 0.85, 0.9, 0.95]))
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

    hist = np.histogram(d7[soc=='Tha44'], bins=bins, density=True)[0]
    ax.plot(X, np.cumsum(hist/hist.sum()), label='Thai', c=col[2])
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


def create_maximally_mixed_scales(scales):
    N, nn = scales.shape
    tonic = np.zeros(N).reshape(N,1)
    supertonic = np.ones(N).reshape(N,1) * 1200
    scales = np.append(tonic, np.append(scales, supertonic, axis=1), axis=1)
    ints = np.diff(scales, axis=1)
    np.matrix.sort(ints)
    if ints.shape[1] == 5:
        ints = ints[:,[0,4,1,3,2]]
    elif ints.shape[1] == 7:
        ints = ints[:,[0,6,1,5,2,4,3]]
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


def fig7(df, dx=20, n=7):
    fig = plt.figure(figsize=(13,6))
    s1 = 4
    w1 = 6 * s1
    s2 = 2 * s1
    w2 = 35
    gs = GridSpec(3,4*w2+3*s2, height_ratios=[1, 0.2, 1])
    ax = [fig.add_subplot(gs[2,j*(w1+s1):(j+1)*w1+j*s1]) for j in range(6)] + [None] + \
         [fig.add_subplot(gs[0,j*(w2+s2):(j+1)*w2+j*s2]) for j in [1,2,0,3]]
    fig.subplots_adjust(wspace=0.0, hspace=0.2)

    df7 = df.loc[df.n_notes==n].reset_index(drop=True)
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]
    s7_sorted = create_maximally_ordered_scales(s7)
#   s7_mixed = create_maximally_mixed_scales(s7)
    s7_shuffled = create_shuffled_scales(s7)
    soc = df7.SocID.values

    # tSNE embedding of grid scales
    tsne_real_poss(ax[9])


    # Distance between equidistant scales, and  real / grid scales
    equidistance_diff(soc, ax[7], n=n)


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
            ax[8].plot(X, np.histogram(ints, bins=bins, density=True)[0], label=lbls[i], c=col[i])

        # Distributions of each scale note,
        for j, s in enumerate(d.T):
            xlo, xhi = np.min(s), np.max(s)
            xhi += dx
            bins = np.arange(dx * ((xlo // dx) - 1), dx * (2 + xhi // dx), dx)
            X = bins[:-1] + np.diff(bins[:2])
            hist = np.histogram(s, bins=bins, density=True)[0]
            all_entropy.append(entropy(hist))
            if l not in ['Sorted', 'Shuffled']:
                ax[j].plot(X, hist, label=lbls[i], c=col[i])
                ax[j].plot([(j+1)*1200/n]*2, [0, hist.max()], ':k', alpha=0.5)

                ax[j].set_xlabel(f"Note {j+2} / cents")
                ax[j].set_ylim(0, ax[j].get_ylim()[1])

    all_entropy = np.array(all_entropy).reshape(4,6)
    X = np.arange(2, 8)
    for i in range(4):
        ax[10].plot(X, all_entropy[i], '-o', label=lbls[i], color=col[i], lw=0.5)
    ax[10].set_xticks(X)

    for a in [ax[0], ax[8]]:
        a.set_ylabel("Density")
    ax[7].set_ylabel("Cumulative Probability")
    ax[10].set_ylabel("Entropy")

    ax[7].set_xlabel("Mean Note Distance from\nEquiheptatonic Scale / cents")
    ax[8].set_xlabel("Step size / cents")
    ax[10].set_xlabel("Note position")

    ax[3].legend(bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax[7].legend(loc='lower right', frameon=False)
    ax[8].legend(loc='upper right', frameon=False)
    ax[10].legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), frameon=False, ncol=2)

    ax[7].set_xlim(0, 220)

    for a in ax[1:n-1]:
        a.spines['left'].set_visible(False)

    xmaj = [200, 200, 250, 250, 200, 200, 50, 100]
    for x, a in zip(xmaj, ax[:n-1] + ax[7:]):
        set_xticks(a, x, x/2, '%d')
    ax[2].set_xlim(280, 800)
    ax[3].set_xlim(350, 1000)

    for a in ax[:]:
        try:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_yticks([])
        except:
            pass

    fs = 14
    x = [-0.075, -0.072, -0.08, -0.15, -0.15, -0.08]
    for i, (j, b) in enumerate(zip([9,7,8,0,10], 'ABCDEF')):
        ax[j].text(x[i], 1.02, b, transform=ax[j].transAxes, fontsize=fs)

    fig.savefig(PATH_FIG.joinpath(f"fig7.pdf"), bbox_inches='tight')




def get_int_prob_via_sampling(df, ysamp='scale', xsamp='SocID', s=6, ax='', fa=0.5):
    if xsamp == '':
        Y = [x for y in df[ysamp] for x in y]
    else:
        Y = utils.sample_df_value(df, ysamp, xsamp, s)

    data = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))
    bins = np.arange(15, 5000, 30)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.

    shape, loc, scale = [0.93, -45.9, 605.4]
    params = lognorm.fit(Y, loc=loc, scale=scale)
    bin_prob = np.diff(lognorm.cdf(bins, *params))

    count = np.histogram(Y, bins=bins)[0]
    N = count.sum()

    if isinstance(ax, str):
        fig, ax = plt.subplots(2,1)
    col = np.array(Set2_8.mpl_colors)
    ax[0].plot(X, count / N, '-', c=col[1], lw=0.9)
    ax[0].plot(X, bin_prob, '-k')

    prob_less_than = binom.cdf(count, N, bin_prob)
    prob_obs = np.min([prob_less_than, 1 - prob_less_than], axis=0)
    is_less = prob_less_than < 0.5
    color = ['r' if i else 'k' for i in is_less]
    ax[1].scatter(X, prob_obs, c=color)
    ax[1].set_yscale('log')
    ax[1].plot(X, [0.05/X.size]*X.size, ':k')

    for a in ax:
        a.set_xlim(0, 3000)
        set_xticks(a, 600, 200, '%d')
        a.set_ylabel('Density')



def plot_boot_int_prob_via_sampling(y='scale', x='SocID', n=5, ax=''):
    ysamp = np.array(['scale', 'AllInts'])
    xsamp = np.array(['Region', 'SocID'])
    nsamp = np.array([1, 2, 5, 10])
    if (x not in xsamp) or (y not in ysamp) or (n not in nsamp):
        print("Wrong parameters chosen")
        return
    data = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))
    prob_less = data[np.where(ysamp==y)[0][0], np.where(xsamp==x)[0][0], np.where(nsamp==n)[0][0]]
    prob_obs = np.min([prob_less, 1 - prob_less], axis=0)

    prob_less_mean = np.mean(prob_less, axis=0)
    prob_obs_mean = np.mean(prob_obs, axis=0)

    is_less = prob_less_mean < 0.5
    colors = ['r' if i else 'k' for i in is_less]


    X = np.arange(30, 5000, 30)
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    
    ax.scatter(X, prob_obs_mean, s=20, color=colors)
    ax.fill_between(X, *np.quantile(prob_obs, [0.025, 0.975], axis=0), color='grey', alpha=0.2)
    ax.plot(X, [0.05 / X.size]*X.size, ':k')
    ax.set_yscale('log')
    ax.set_title(f"{x}, max_samp = {n}")
    ax.set_xlabel(y)
    ax.set_xlim(0, 3000)


def plot_boot_int_prob_array(y='scale'):
    xsamp = np.array(['Region', 'SocID'])
    nsamp = np.array([1, 2, 5, 10])
    fig, ax = plt.subplots(4,2)
    fig.subplots_adjust(hspace=0.5)
    for i, x in enumerate(xsamp):
        for j, n in enumerate(nsamp):
            plot_boot_int_prob_via_sampling(y, x, n, ax[j,i])


def regplot(X, Y, ax=[], xy=(), c='k', log=True):
    idx = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[idx], Y[idx]
    X2 = np.linspace(X.min(), X.max(), 1000)

    if log:
        ax.plot(10**X, 10**Y, 'o', fillstyle='full', alpha=0.5, ms=4, c=c)
    else:
        ax.plot(X, Y, 'o', fillstyle='full', alpha=0.5, ms=4, c=c)
    gradient, intercept, r, p = linregress(X, Y)[:4]
    Y2 = intercept + gradient * X2
    if log:
        ax.plot(10**X2, 10**Y2, '-', color=c, lw=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.plot(X2, Y2, '-', color=c, lw=2)

    txt = r"$r=$" + f"{r:4.2f}" + "\n" + "$p = 10^{{{0}}}$".format(int(round(np.log10(p))))
    ax.text(*xy, txt, color=c)


def plot_octave_correlation():
    X = np.arange(30, 5000, 30)
    prob_less = np.load(PATH_DATA.joinpath("int_prob_lognorm.npy"))[0,1,2].mean(axis=0)
    prob_obs = np.min([prob_less, 1 - prob_less], axis=0)
    idx1 = X <= 1200
    idx2 = (1230 <= X) & (X <= 2400)
    idx3 = (2430 <= X) & (X <= 3600)

    fig = plt.figure(figsize=(12,6))
    gs = GridSpec(4,2, height_ratios=[1, 1, .5, 2], width_ratios=[1,1])
    ax = [fig.add_subplot(gs[:2,0]), fig.add_subplot(gs[3,0])] + \
         [fig.add_subplot(gs[i,1]) for i in [0,1,3]]
    fig.subplots_adjust(wspace=0.3, hspace=0.05)

    for i in [idx1, idx2]:
        ax[0].plot(X[idx1], np.log10(prob_obs[i]))
    regplot(np.log10(prob_obs[idx1]), np.log10(prob_obs[idx2]), ax[1], (10**-5, 10**-3), sns.color_palette()[0])

    ints = np.arange(200, 2605, 5)
    m1, lo1, hi1, m2, lo2, hi2 = load_sampled_interval_data(ints, 'cult', 5)
    i1 = (ints>=200)&(ints<=1200)
    i2 = (ints>=1400)&(ints<=2400)
    col = np.array(Set2_8.mpl_colors)[[1,0]]
    for i, (idx, p, l) in enumerate(zip([i1, i2], ['-', '--'], ['First', 'Second'])):
        ax[3].plot(ints[i1], m1[idx], p, c=col[0], label=f"{l}, Support")
        ax[2].plot(ints[i1], m2[idx], p, c=col[1], label=f"{l}, Against")
    ax[2].legend(loc='upper left', ncol=2, frameon=False)
    ax[3].legend(loc='upper left', ncol=2, frameon=False)

    regplot(m1[i1], m1[i2], ax[4], (0.20, 0.20), col[0], False)
    regplot(m2[i1], m2[i2], ax[4], (0.25, 0.10), col[1], False)

    xlbls = ["Scale degree / cents", r"$p$, First octave", "", "Scale degree / cents", "Fraction significant, First octave"]
    ylbls = [r"Significance, $p$", r"$p$, Second octave", "", " "*20+"Fraction signficant", "Fraction significant, Second octave"]
    for i, a in enumerate(ax):
        a.set_xlabel(xlbls[i])
        a.set_ylabel(ylbls[i])
        a.tick_params(axis='both', which='major', direction='in', length=6, width=2)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    





