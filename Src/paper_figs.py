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
from scipy.stats import entropy
import seaborn as sns
from shapely.geometry.point import Point
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

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


def scale_degree():
    fig, ax = plt.subplots(figsize=(6,4))
    data = pickle.load(open(PATH_DATA.joinpath("scale_degree.pickle"), 'rb'))
    X = data['X']
    width = 0.15
    lbls = ['All', 'Theory', 'Measured', 'Region', 'SocID']
    cols = sns.color_palette()
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        err = np.array([m - lo, hi - m])
        ax.bar(X + (i-2)*width, m, width, yerr=err, color=cols[i], label=l, ec='k')

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Scale degree")
    ax.set_ylabel("Normalised frequency")

    fig.savefig(PATH_FIG.joinpath("scale_degree.pdf"), bbox_inches='tight')



def multiple_dist():
    fig, ax = plt.subplots(3,2, figsize=(12,5))
    fig.subplots_adjust(wspace=0, hspace=0)
    lblA = ['All', 'Theory', 'Measured']
    lblB = ['Region', 'SocID']
    lblC = ['Idiophone', 'Aerophone', 'Chordophone']
    path_stem = ['step_intervals', 'scale']#, 'all_intervals']
    xlbls = ['Adjacent Interval / cents', 'Scale note / cents', 'Interval / cents']
    xlim = [570, 1370, 1370]
    cols = Paired_12.hex_colors

    for j, stem in enumerate(path_stem):
        data = pickle.load(open(PATH_DATA.joinpath(f"{stem}.pickle"), 'rb'))
        X = data['X']
        for i, lbl in enumerate([lblA, lblB, lblC]):
            for k, l in enumerate(lbl):
                m, lo, hi = data[l]
                ax[i,j].plot(X, m, '-', c=cols[((i*len(lblA)+k)*2+1)%12], label=l)
                ax[i,j].fill_between(X, lo, hi, color=cols[((i*len(lblA)+k)*2)%12], alpha=0.5)

            if i == 2:
                ax[i,j].set_xlabel(xlbls[j])
            if j == 0:
                set_ticks(ax[i,j], 100, 50, '%d', 1.02, 1.01, '%4.2f')
            else:
                set_ticks(ax[i,j], 200, 100, '%d', 1.002, 1.001, '%5.3f')
            ax[i,j].set_xlim(0, xlim[j])
            ax[i,0].set_ylabel("Density")

    for a in ax[:,0]:
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi)
        for x in np.arange(100, 400, 100):
            a.plot([x]*2, [lo, hi], ':k', alpha=0.3)

    for a in ax[:,1:].ravel():
        lo, hi = a.get_ylim()
        a.set_ylim(lo, hi)
        for x in np.arange(100, 1200, 100):
            a.plot([x]*2, [lo, hi], ':k', alpha=0.3)

    for a in ax[:2,:].ravel():
        a.set_xticks([])
    for a in ax[:,:].ravel():
        a.set_yticks([])
    ax[0,0].legend(loc='upper right', frameon=False)
    ax[1,0].legend(loc='upper right', frameon=False)
    ax[2,0].legend(loc='upper right', frameon=False)

    fig.savefig(PATH_FIG.joinpath("multi_dist.pdf"), bbox_inches='tight')


def scale_dist():
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("scale.pickle"), 'rb'))
    X = data['X']

    lbls = ['All', 'Theory', 'Measured', 'Region', 'SocID']
    cols = Paired_12.hex_colors
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        ax.plot(X, m, '-', c=cols[i*2+1], label=lbls[i])
        ax.fill_between(X, lo, hi, color=cols[i*2], alpha=0.5)

    ax.legend(loc='best', frameon=False)
    ax.set_xlabel("Scale note")
    ax.set_ylabel("Normalised frequency")

    fig.savefig(PATH_FIG.joinpath("scale_dist.pdf"), bbox_inches='tight')


def step_int_dist():
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("step_intervals.pickle"), 'rb'))
    X = data['X']

    lbls = ['All', 'Theory', 'Measured', 'Region', 'SocID']
    cols = Paired_12.hex_colors
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        ax.plot(X, m, '-', c=cols[i*2+1], label=lbls[i])
        ax.fill_between(X, lo, hi, color=cols[i*2], alpha=0.5)

    ax.legend(loc='best', frameon=False)
    fig.savefig(PATH_FIG.joinpath("int_dist.pdf"), bbox_inches='tight')


def all_int_dist():
    fig, ax = plt.subplots()
    data = pickle.load(open(PATH_DATA.joinpath("all_intervals.pickle"), 'rb'))
    X = data['X']

    lbls = ['All', 'Theory', 'Measured', 'Region', 'SocID']
    cols = Paired_12.hex_colors
    for i, l in enumerate(lbls):
        m, lo, hi = data[l]
        ax.plot(X, m, '-', c=cols[i*2+1], label=lbls[i])
        ax.fill_between(X, lo, hi, color=cols[i*2], alpha=0.5)

    ax.legend(loc='best', frameon=False)
    fig.savefig(PATH_FIG.joinpath("all_int_dist.pdf"), bbox_inches='tight')


def octave_equiv(df, octave=1200, n_rep=10):
    res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep)
    fig, ax = plt.subplots(2)
    ax = ax.reshape(ax.size)
    sns.scatterplot(x='f_real', y='f_shuf', data=res, hue='sig', ax=ax[0])
    mx = max(res.f_real.max(), res.f_shuf.max())
    ax[0].plot([0, mx], [0, mx], '-k')

    sns.scatterplot(x='mean_real', y='mean_shuf', data=res, hue='sig', ax=ax[1])
    mx = max(res.mean_real.max(), res.mean_shuf.max())
    ax[1].plot([0, mx], [0, mx], '-k')

    ax[0].set_xlabel("Fraction of all original intervals within w2 of octave")
    ax[1].set_xlabel("Deviation of all original intervals within (w1 of octave) from the octave")
    ax[0].set_ylabel("Fraction of all shuffled intervals within w2 of octave")
    ax[1].set_ylabel("Deviation of all shuffled intervals within (w1 of octave) from the octave")

    ax[0].legend(loc='best', frameon=False)
    ax[1].legend(loc='best', frameon=False)

#   w_arr = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#   n_greater = []
#   n_less = []
#   n_nonsig = []
#   for w in w_arr:
#       for i in range(5):
#           res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep, w2=w)
#           n_greater.append(len(res.loc[(res.MWU<0.05)&(res.f_real>res.f_shuf)]))
#           n_less.append(len(res.loc[(res.MWU<0.05)&(res.f_real<res.f_shuf)]))
#           n_nonsig.append(len(res.loc[(res.MWU>=0.05)]))
#   n_greater = np.array(n_greater).reshape(len(w_arr), 5).mean(axis=1)
#   n_less = np.array(n_less).reshape(len(w_arr), 5).mean(axis=1)
#   n_nonsig = np.array(n_nonsig).reshape(len(w_arr), 5).mean(axis=1)
#   total = np.sum([n_greater, n_less, n_nonsig], axis=0)

#   ax[2].plot(w_arr, n_greater / total, label='support')
#   ax[2].plot(w_arr, n_less / total, label='against')
#   ax[2].plot(w_arr, n_nonsig / total, label='non_sig')

#   ax[2].set_xlabel("window_1")
#   ax[2].set_ylabel("Normalised frequency")
#   ax[2].legend(loc='best', frameon=False)

#   w_arr = [50, 75, 100, 125, 150, 175, 200]
#   n_greater = []
#   n_less = []
#   n_nonsig = []
#   for w in w_arr:
#       print(w)
#       for i in range(5):
#           res = OC.octave_chance_individual(df, octave=octave, n_rep=n_rep, w1=w)
#           n_greater.append(len(res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)]))
#           n_less.append(len(res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)]))
#           n_nonsig.append(len(res.loc[(res.MWU>=0.05)]))
#   n_greater = np.array(n_greater).reshape(len(w_arr), 5).mean(axis=1)
#   n_less = np.array(n_less).reshape(len(w_arr), 5).mean(axis=1)
#   n_nonsig = np.array(n_nonsig).reshape(len(w_arr), 5).mean(axis=1)
#   total = np.sum([n_greater, n_less, n_nonsig], axis=0)

#   ax[3].plot(w_arr, n_greater / total, label='support')
#   ax[3].plot(w_arr, n_less / total, label='against')
#   ax[3].plot(w_arr, n_nonsig / total, label='non_sig')

#   ax[3].set_xlabel("window_2")
#   ax[3].set_ylabel("Normalised frequency")
#   ax[3].legend(loc='best', frameon=False)

#   fig.savefig(PATH_FIG.joinpath("octave_demo.pdf"), bbox_inches='tight')



def get_stats(df, i, n_rep=20, w=20, nrep2=10):
    out = np.zeros((3,nrep2), float)
    for j in range(nrep2):
        res = OC.octave_chance_individual(df, octave=i, n_rep=n_rep, w=20)
        out[0,j] = len(res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)])
        out[1,j] = len(res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)])
        out[2,j] = len(res.loc[(res.MWU>=0.05)])
    return out.mean(axis=1)


def unexpected_intervals(df, n_rep=20):
    fig, ax = plt.subplots(2,1)
    df = df.loc[:, ['Intervals', 'scale']]

    ints = np.arange(200, 2520, 10)
    with Pool(N_PROC) as pool:
        res = np.array(pool.starmap(get_stats, product([df], ints)))
    n_greater = res[:,0]
    n_less = res[:,1]
    n_nonsig = res[:,2]

    n_tot = n_greater + n_less + n_nonsig
    
    ax[0].plot(ints, n_greater / n_tot, label='support')
    ax[0].plot(ints, n_less / n_tot, label='against')
    ax[1].plot(ints, (n_greater - n_less) / n_tot, label='against')
    ax[0].plot(ints, [0]*len(ints), ':k')
    ax[1].plot(ints, [0]*len(ints), ':k')
#   ax.plot(ints, n_nonsig / n_tot, label='non_sig')


def identify_culture(cultures):
    if len(set(cultures)) == 1:
        return cultures[0]
    else:
        return 'Multiple'


### NEED TO FIX INDEXING! REF AND DF ARE NOT MATCHING
def octave_by_source(df, res):
    text = OC.load_text_summary()

    df['RefID'] = df['RefID'].astype(int)
    src_cnt_tot = df.RefID.value_counts()
    df0 = df.loc[df.scale.apply(lambda x: x[-2] >= 1200)]
    src_cnt_use = df0.RefID.value_counts()

    src_support = df0.loc[df0.index[res.loc[(res.MWU<0.05)&(res.mean_real<res.mean_shuf)].index], "RefID"].value_counts()
    src_against = df0.loc[df0.index[res.loc[(res.MWU<0.05)&(res.mean_real>res.mean_shuf)].index], "RefID"].value_counts()
    src_insig   = df0.loc[df0.index[res.loc[(res.MWU>=0.05)].index], 'RefID'].value_counts()

    src_cult = {k:c for k, c in zip(df.RefID, df.Reference)}
    tots = []
    
    res_lbl = ' & '.join(['Total', 'Testable', 'Support', 'Against', 'Non-sig'])
    print(f"Ref & SocID &  {res_lbl} & Text support & SocID support & Overall \\\\")
    for i in text.index:
        k = i + 1
        try:
            name = f"{int(k):4d}  {src_cult[k][:40]:40s}"
            inst_results_list = [d.get(k,0) for d in [src_cnt_tot, src_cnt_use, src_support, src_against, src_insig]]
            inst_results = " & ".join([f"{d:4d}" for d in inst_results_list])
            t1, t2 = text.loc[i]
            culture = identify_culture(df.loc[df.RefID==k, 'SocID'].values)
#           print(src_against.get(k,0), src_support.get(k,0), t1, t2)
            evidence = (src_against.get(k,0) < src_support.get(k,0)) | ('Yes' in t1) | ('Yes' in t2)
            tots.append(inst_results_list +  [('Yes' in t1), ('Yes' in t2), evidence])
            print(f"{name} & {culture[:15]:15s} & {inst_results} & {str(t1):6s} & {str(t2):10s} & {evidence:3b} \\\\")
        except Exception as e:
#           print(e)
            pass
    tots = np.sum(tots, axis=0)
    print(" "*65 + "  ".join([f"{int(t):4d}" for t in tots]))


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


def load_sampled_interval_data(ints, xsamp, n):
    data = np.array([[np.load(f"../IntStats/{xsamp}samp{j}_w1100_w220_I{i:04d}.npy") for i in ints] for j in range(n)])
#   data = data.mean(axis=0)
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


def octave_sig(ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()

    sigma = np.arange(0, 55, 5)
    data = np.array([np.load(f"../IntStats/sigma{s}_w1100_w220_I1200.npy") for s in sigma])
    lbls = ['Greater freq\nthan chance', 'Less freq\nthan chance']
    col = np.array(Set2_8.mpl_colors)[[1,0]]
    for i in range(2):
        frac = data[:,i] / data.sum(axis=1)
        m = np.mean(frac, axis=1)
        lo = np.quantile(frac, 0.025, axis=1)
        hi = np.quantile(frac, 0.975, axis=1)
        ax.plot(sigma, m, label=lbls[i], c=col[i])
        ax.fill_between(sigma, hi, lo, color='grey', alpha=0.5)

    ax.set_xlabel("Tuning deviation (cents)")
    ax.set_ylabel("Fraction of\nsignificant results")
    ax.legend(loc='best', frameon=False)


def interval_sig(res):
    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(12,3, width_ratios=[1, 0.4, 2])
    ax = [fig.add_subplot(gs[i*3:(i+1)*3,2]) for i in range(4)] + \
         [fig.add_subplot(gs[:5,0]), fig.add_subplot(gs[8:,0])]
#   fig, ax = plt.subplots(3,1)
    fig.subplots_adjust(hspace=0, wspace=0)
    ints = np.arange(200, 2605, 5)
    col2 = sns.color_palette()
    col = np.array(Set2_8.mpl_colors)[[1,0]]

    lbls = ['Greater freq than chance', 'Less freq than chance']
    m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/0_w1100_w220_I{i:04d}.npy" for i in ints])
    ax[0].plot(ints, m1, '-', label=lbls[0], color=col[0])
    ax[0].fill_between(ints, lo1, hi1, color='grey')
    ax[0].plot(ints, m2, '-', label=lbls[1], color=col[1])
    ax[0].fill_between(ints, lo2, hi2, color='grey')

    for j in range(1, 3):
        m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/{j}_w1100_w220_I{i:04d}.npy" for i in ints])
        ax[3].plot(ints, m1, '-', label='Support', color=col[0])
        ax[3].fill_between(ints, lo1, hi1, color='grey')
        ax[3].plot(ints, m2, '-', label='Against', color=col[1])
        ax[3].fill_between(ints, lo2, hi2, color='grey')

#   cont = ['South East Asia', 'Africa', 'Oceania', 'South Asia', 'Western',
#           'Latin America', 'East Asia', 'Middle East']
#   for c in cont[:5]:
#       m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/{c}_w1100_w220_I{i:04d}.npy" for i in ints])
#   for j in range(3):
    m1, lo1, hi1, m2, lo2, hi2 = load_sampled_interval_data(ints, 'cont', 3)
    ax[1].plot(ints, m1, '-', label='Support', color=col[0])
    ax[1].fill_between(ints, lo1, hi1, color='grey')
    ax[1].plot(ints, m2, '-', label='Against', color=col[1])
    ax[1].fill_between(ints, lo2, hi2, color='grey')


    m1, lo1, hi1, m2, lo2, hi2 = load_sampled_interval_data(ints, 'cult', 3)
    ax[2].plot(ints, m1, '-', label='Support', color=col[0])
    ax[2].fill_between(ints, lo1, hi1, color='grey')
    ax[2].plot(ints, m2, '-', label='Against', color=col[1])
    ax[2].fill_between(ints, lo2, hi2, color='grey')


    ax[1].set_ylabel("Fraction of significant results")
    ax[3].set_xticks(range(0, 2700, 200))
    ax[3].xaxis.set_tick_params(which='minor', bottom=True)
    ax[3].set_xlabel("Interval size (cents)")

    ax[0].legend(loc='upper left', bbox_to_anchor=(0.00, 1.35), frameon=False, ncol=2)

    sns.scatterplot(x='mean_real', y='mean_shuf', data=res.sort_values(by='sig'), hue='sig', ax=ax[4], alpha=0.5, palette=col2[:4])
    mx = max(res.mean_real.max(), res.mean_shuf.max())
    ax[4].plot([0, mx], [0, mx], '-k')
    ax[4].set_xlabel("Deviation of original intervals\nfrom the octave")
    ax[4].set_ylabel("Deviation of shuffled intervals\nfrom the octave")
    ax[4].legend(bbox_to_anchor=(0.7,0.6), frameon=False, handletextpad=0)

    octave_sig(ax[5])

    ax[4].annotate('A', (-0.1, 1.05), xycoords='axes fraction', fontsize=16)
    ax[0].annotate('B', (-0.1, 1.10), xycoords='axes fraction', fontsize=16)
    ax[5].annotate('C', (-0.1, 1.05), xycoords='axes fraction', fontsize=16)

    txt = ['All', 'Reg-Samp', 'Cult-Samp', 'Null']
    for i, a in enumerate(ax[:4]):
        a.set_xlim(0, 2620)
        a.set_ylim(0, 0.47)
        set_ticks(a, 400, 100, '%d', 0.2, 0.1, '%3.1f')
        a.tick_params(axis='both', which='major', direction='in', length=6, width=2)
        a.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
        a.annotate(txt[i], (0.04, 0.82), xycoords='axes fraction')
    for a in ax[:3]:
        a.set_xticklabels(['']*7)

    fig.savefig(PATH_FIG.joinpath("interval_sig.pdf"), bbox_inches='tight')


def window_size():
    fig, ax = plt.subplots(7,1)
    ints = np.arange(200, 2605, 5)
    col = sns.color_palette()

    w1_list = [50, 75, 100, 125, 150, 175, 200]
    w2_list = [5, 10, 15, 20, 30, 40]
    for i, w1 in enumerate(w1_list):
        for w2 in w2_list:
            try:
                print(w1, w2)
                m1, lo1, hi1, m2, lo2, hi2 = load_interval_data([f"../IntStats/0_w1{w1}_w2{w2}_I{i:04d}.npy" for i in ints])
                ax[i].plot(ints, m1, '-', label='Support', color=col[0])
                ax[i].fill_between(ints, lo1, hi1, color='grey')
                ax[i].plot(ints, m2, ':', label='Against', color=col[1])
                ax[i].fill_between(ints, lo2, hi2, color='grey')
            except:
                print(f"{w1}_{w2} not done yet")

    fig.savefig(PATH_FIG.joinpath("window_effect.pdf"), bbox_inches='tight')


def inst_notes(df, ysamp='scale'):
    df = df.loc[df.Reduced_scale=='N']
    alt_df = OC.create_new_scales(df)[0]

#   fig, ax = plt.subplots(4,2, figsize=(10,4))
    fig = plt.figure(figsize=(10,5))
    gs = GridSpec(5,3, width_ratios=[1, 0.3, 1], height_ratios=[1,1,1,0.8,1])
    ax = [fig.add_subplot(gs[i,j]) for i in range(3) for j in [0,2]] + \
         [fig.add_subplot(gs[4,j]) for j in [0,2]]
    ax = np.array(ax).reshape(4,2)
    fig.subplots_adjust(hspace=0, wspace=0)
    col = np.array(Set2_8.mpl_colors)

    OC.get_int_prob_via_sampling(df, ysamp=ysamp, xsamp='', ax=ax[0,0])
    OC.get_int_prob_via_sampling(alt_df, ysamp=ysamp, xsamp='', s=6, ax=ax[3,0]) 

    labels = ['Data', 'Lognormal fit', '99% CI']
    handles = [Line2D([], [], linestyle=ls, color=m, label=l) for ls, m, l in zip('-:', [col[1], 'k'], labels[:2])] + \
              [Patch(facecolor=col[0], alpha=0.5, label=labels[-1])]
    ax[0,0].legend(handles=handles, bbox_to_anchor=(1.70, 1.50), frameon=False, ncol=3)
    
    for i in range(3):
        OC.get_int_prob_via_sampling(df, ysamp=ysamp, xsamp='Region', s=6, ax=ax[1,0], fa=0.5/3) 
        OC.get_int_prob_via_sampling(df, ysamp=ysamp, xsamp='SocID', s=1, ax=ax[2,0], fa=0.5/3) 
        OC.get_int_prob_via_sampling(alt_df, ysamp=ysamp, xsamp='SocID', s=1, ax=ax[3,1], fa=0.5/3) 


    lbls2 = ['Idiophone', 'Aerophone', 'Chordophone']
    for i, l in enumerate(lbls2):
        OC.get_int_prob_via_sampling(df.loc[df.Inst_type==l], ysamp=ysamp, xsamp='', ax=ax[i,1])
        ax[i,1].annotate(l, (0.70, 0.70), xycoords='axes fraction')

    ax[0,0].annotate('A', (-0.1, 1.05), xycoords='axes fraction', fontsize=16)
    ax[3,0].annotate('B', (-0.1, 1.05), xycoords='axes fraction', fontsize=16)
        
    txt = ["All", "Reg-Samp", "Cult-Samp"]
    for a in ax.ravel():
#       a.set_ylim(0, 0.002)
        a.set_xlim(0, 3000)
        set_ticks(a, 600, 200, '%d', 0.001, 0.0005, '%5.3f')
        a.set_ylabel('Density')
        a.tick_params(axis='both', which='major', direction='in', length=6, width=2)
        a.tick_params(axis='both', which='minor', direction='in', length=4, width=1)
        a.set_yticks([0, 0.001])

    for i, a in enumerate(ax[:3,0]):
        a.annotate(txt[i], (0.70, 0.70), xycoords='axes fraction')

    ax[3,0].annotate(txt[0], (0.70, 0.70), xycoords='axes fraction')
    ax[3,1].annotate(txt[2], (0.70, 0.70), xycoords='axes fraction')

    for a in ax[:2,:].ravel():
        a.set_xticklabels(['']*5)
    ax[2,0].set_xlabel('Interval from the lowest note / cents')
    ax[2,1].set_xlabel('Interval from the lowest note / cents')
    ax[3,0].set_xlabel('Interval from the lowest note / cents')
    ax[3,1].set_xlabel('Interval from the lowest note / cents')

    fig.savefig(PATH_FIG.joinpath(f"inst_notes_{ysamp}.pdf"), bbox_inches='tight')


def clustering(df):
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(6,5, width_ratios=[0.7, 0.7, 1, 0.1, 1])
    ax = [fig.add_subplot(gs[:,0])] + \
         [fig.add_subplot(gs[j,i]) for i in [2, 4] for j in range(6)]


    df7 = df.loc[df.n_notes==7].reset_index(drop=True)
    s7 = np.array([y for y in df7.scale])[:,1:-1]
    li = linkage(s7, method='ward')
    nc = fcluster(li, li[-6,2], criterion='distance')
    df7['nc6'] = nc

    ## Dendrogram
    thresh = li[-5, 2]
    cols = list(np.array(Paired_12.hex_colors)[[5,1,7,3,11,9]])
    set_link_color_palette(cols)
    idx_label = [137, # 1, Mela Salagam
                 375, # 1, Gamelan Swastigitha Pelog
                 22, # 2, Locrian
                 93, # 2, Bhairavi
                 357, # 2, Khong Mon
                 540, # 2, So-na
                 343, # 2, Mbira 3
                 0, # 3, Major
                 209, # 6, Maqam Mahur A
                 255, # 3, In
                 429, # 6, Matape 2
                 493, # 6, Marimba 8
                 267, # 6, Yanyue
                 354, # 6, Ranat T'hong
                 472, # 6, Gamelan 25
                 234, # 4, Dastgad-e Chahargah
                 316, # 4, Hicaz Makam 1
                 2, # 6, Harmonic Minor 
                 420, # 6, Asena 27
                 439, # 6, Guinea Malinke 3
                 135, # 3, Mela Shulini
                 385, # 3, Asena 5
                 206, # 5, Maqam Athar Kurd
                 470, # 5, Kwaiker
                 460] # 5, Ranad thum lek
    for i in idx_label:
        try:
            print(i, df7.loc[i, ['Name', 'nc6']].values)
        except:
            print(i, None)

    x_lbls = [name if i in idx_label else '' for i, name in enumerate(df7.Name)]
    dendrogram(li, labels=x_lbls, color_threshold=thresh, above_threshold_color='k', leaf_rotation=00, ax=ax[0], orientation='left')


    ## Scale and Region dist

    cont = ['South East Asia', 'Africa', 'Oceania', 'South Asia', 'Western',
            'Latin America', 'East Asia', 'Middle East']
    cont_tot = Counter(df7['Region'])
    Y_tot = np.array([cont_tot.get(c2,0) for c2 in cont])

    bins = np.arange(15, 1290, 30)
    X = bins[:-1] + np.diff(bins[:2])
    xbar = np.arange(len(cont))
    width = 0.3

    txt = 'abcdef'
    for i in range(6):
        c = 6 - 2
        scale_arr = np.array([x for x in df7.loc[nc==c, 'scale'].values])
        sns.distplot(scale_arr.ravel(), bins=bins, ax=ax[i+1], kde=False, norm_hist=True, color=cols[c-1])
        count = Counter(df7.loc[nc==c, 'Region'])
        Y = np.array([count.get(c2,0) for c2 in cont])
        ax[i+7].bar(xbar - width/2, Y, width, color=Dark2_8.hex_colors, ec='k')

        print(c, entropy(Y/Y_tot))
        ax.append(ax[i+7].twinx())
        ax[i+13].bar(xbar + width/2, Y/Y_tot, width, color=Pastel2_8.hex_colors, ec='k')
        ax[i+1].annotate(txt[i], (0.05, 0.97), xycoords='axes fraction', fontsize=12)
        ax[i+7].annotate(txt[i], (0.05, 0.97), xycoords='axes fraction', fontsize=12)

        yhi = ax[i+1].get_ylim()[1] * 0.5
        for j in range(1,7):
            ax[i+1].plot([scale_arr[:,j].mean()]*2, [0, yhi], ':k')
        

    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=10)
#   ax[0].spines['bottom'].set_visible(False)
    ax[0].set_xlabel("Distance between scale clusters")
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    for a in ax:
        a.spines['top'].set_visible(False)

    for a in ax[1:]:
        a.tick_params(axis='x', which='major', direction='in', length=6, width=2)
        a.tick_params(axis='x', which='minor', direction='in', length=4, width=1)

    for a in ax[1:7]:
        a.set_yticks([])
        a.set_ylabel("Density")
        a.spines['right'].set_visible(False)
        set_xticks(a, 200, 100, '%d')

    for a in ax[7:13]:
        a.set_ylabel("Frequency")

    for a in ax[1:6] + ax[13:18]:
        a.set_xticklabels([])

    for a in ax[13:]:
        a.set_ylabel("Relative\nFrequency")

    ax[6].set_xlabel("Scale note")
    ax[12].set_xlabel("Region")
    ax[12].set_xticks(xbar)
    ax[12].set_xticklabels(cont, rotation=90)


    ax[0].annotate('A', (0.0, 1.00), xycoords='axes fraction', fontsize=16)
    ax[1].annotate('B', (-0.2, 1.05), xycoords='axes fraction', fontsize=16)
    ax[7].annotate('C', (-0.2, 1.05), xycoords='axes fraction', fontsize=16)

    fig.savefig(PATH_FIG.joinpath(f"cluster7.pdf"), bbox_inches='tight')


def real_poss_dist(s7, ax, n=7):
    col = np.array(Set2_8.hex_colors)[[1,0]]
    sdist = cdist(s7, s7)
    np.fill_diagonal(sdist, sdist.max())
    iparams = {5:"20_60_420", 7:"20_60_320"}[n]
    min_dist = np.load(f"../PossibleScales/possible_{n}_{iparams}_md1.npy")
    sns.distplot(sdist.min(axis=0), kde=False, norm_hist=True, label='Real-Real', ax=ax, color=col[0])
    sns.distplot(min_dist, kde=False, norm_hist=True, label='Real-Grid', ax=ax, color=col[1])
    yhi = ax.get_ylim()[1]
    ax.plot([50]*2, [0, yhi*0.8], '--k', alpha=0.5)
    ax.plot([100]*2, [0, yhi*0.8], '--k', alpha=0.5)


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


def equidistance_diff(s7, ax, n=7):
    bins = np.arange(-5, 240, 10)
    X = bins[:-1] + np.diff(bins[:2])
    d7 = dist_equi(s7, n)
    print('\n', np.quantile(d7, [0.75, 0.8, 0.85, 0.9, 0.95]))
    hist = np.histogram(d7, bins=bins, density=True)[0]
    ax.plot(X, hist, label='Real')

    iparams = {5:"20_60_420", 7:"20_60_320"}[n]
    path_list = [f"possible_{n}_{iparams}{a}" for a in ["_close50", "_far100", ""]]
    lbls = ['Grid-Close', 'Grid-Far', 'Grid-All']
    for i, path in enumerate(path_list):
#       print(i, path)
        path_hist = PATH_DATA.joinpath(f"{path}_hist.npy")
        if path_hist.exists() and 1:
            hist = np.load(path_hist)
        else:
            if i <= 1:
                data = np.load(f"../PossibleScales/{path}.npy")
            else:
                data = np.cumsum(np.load(f"../PossibleScales/{path}.npy"), axis=1)[:,:-1]
            d = dist_equi(data, n)
            if i == 2:
                print(np.quantile(d, [0.05, 0.1, 0.15, 0.2, 0.25]))
            hist = np.histogram(d, bins=bins, density=True)[0]
            np.save(path_hist, hist)

        ax.plot(X, hist, label=lbls[i])


def non_scales_diff(df, dx=20, n=7):
    fig = plt.figure(figsize=(12,5))
    if n == 7:
        gs = GridSpec(3,6, height_ratios=[1, 0.2, 1])
        ax = [fig.add_subplot(gs[2,j]) for j in range(6)] + \
             [fig.add_subplot(gs[0,i*2:(i+1)*2]) for i in [0,2,1]]
    elif n == 5:
        gs = GridSpec(3,12, height_ratios=[1, 0.2, 1])
        ax = [fig.add_subplot(gs[2,j*3:(j+1)*3]) for j in range(4)] + [[],[]] + \
             [fig.add_subplot(gs[0,i*4:(i+1)*4]) for i in [0,2,1]]

#   fig, ax = plt.subplots(1,6, figsize=(16,3))
    df7 = df.loc[df.n_notes==n].reset_index(drop=True)
    s7 = np.array([[float(x) for x in y] for y in df7.scale])[:,1:-1]

    # Distance between real and grid scales
    real_poss_dist(s7, ax[6], n=n)
    ax[6].set_xlim(-20, 330)


    # Distance between equidistant scales, and  real / grid scales
    equidistance_diff(s7, ax[7], n=n)
    ax[6].set_xlim(-20, 330)


    # Distributions for real scales, similar scales, and different scales
    iparams = {5:"20_60_420", 7:"20_60_320"}[n]
    far = np.load(f'../PossibleScales/possible_{n}_{iparams}_far100.npy')
    close = np.load(f'../PossibleScales/possible_{n}_{iparams}_close50.npy')

    lbls = ['Real', 'Grid-Close', 'Grid-Far']
    xlbls = [[f"{a} {b}" for b in ['All', 'Min', 'Max']] for a in ["adj", "2nd", "3rd"]]
    for i, (d, l) in enumerate(zip([s7, close, far], lbls)):

        # Distributions of step intervals,
        ints = np.diff(d, axis=1)
        xlo, xhi = np.min(ints), np.max(ints)
        xlo = dx * ((xlo // dx) - 1) - int(dx/2)
        xhi = dx * (3 + xhi // dx)
        bins = np.arange(xlo, xhi, dx)
        X = bins[:-1] + np.diff(bins[:2])
        ax[8].plot(X, np.histogram(ints, bins=bins, density=True)[0], label=lbls[i])

        # Distributions of each scale note,
        for j, s in enumerate(d.T):
            xlo, xhi = np.min(s), np.max(s)
            bins = np.arange(dx * ((xlo // dx) - 1), dx * (2 + xhi // dx), dx)
            X = bins[:-1] + np.diff(bins[:2])
            hist = np.histogram(s, bins=bins, density=True)[0]
            ax[j].plot(X, hist, label=lbls[i])
            ax[j].plot([(j+1)*1200/n]*2, [0, hist.max()], ':k', alpha=0.5)

            ax[j].set_xlabel(f"Note {j+2} / cents")
            ax[j].set_ylim(0, ax[j].get_ylim()[1])

    ax[0].set_ylabel("Density")
    ax[6].set_ylabel("Density")
    ax[7].set_ylabel("Density")
    ax[8].set_ylabel("Density")

    ax[6].set_xlabel("Distance to nearest scale / cents")
    ax[7].set_xlabel("Mean Note Distance from\nEquiheptatonic Scale / cents")
    ax[8].set_xlabel("Adjacent Interval / cents")

    ax[6].legend(loc='upper right', frameon=False)
    ax[7].legend(loc='upper right', frameon=False)
    ax[8].legend(loc='upper right', frameon=False)

    ax[7].set_xlim(0, 220)

    for a in ax[1:n-1]:
        a.spines['left'].set_visible(False)

    xmaj = [200, 200, 250, 350, 200, 200, 50, 100]
    for x, a in zip(xmaj, ax[:n-1] + ax[7:]):
        set_xticks(a, x, x/2, '%d')

    if n == 7:
        ax[2].set_xticklabels(['', 250, 500, 750, ''])

    for a in ax[:]:
        try:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.set_yticks([])
        except:
            pass

    ax[6].annotate('A', (-0.075, 1.00), xycoords='axes fraction', fontsize=16)
    ax[8].annotate('B', (-0.072, 1.00), xycoords='axes fraction', fontsize=16)
    ax[7].annotate('C', (-0.08, 1.00), xycoords='axes fraction', fontsize=16)
    ax[0].annotate('D', (-0.15, 1.00), xycoords='axes fraction', fontsize=16)

    fig.savefig(PATH_FIG.joinpath(f"nonscale_{n}.pdf"), bbox_inches='tight')


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


def world_map(df):
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
#   col = np.array(Paired_12.mpl_colors)[[5,1]]
    col = np.array(Set2_8.mpl_colors)[[1,0]]
    ft1 = 12

    world.plot(ax=ax[0], color=(0.8, 0.8, 0.8), edgecolor=(1.0,1.0,1.0), lw=0.2)
    world.loc[world.name.apply(lambda x: x in countries)].plot(ax=ax[0], color=(0.5, 0.5, 0.5), edgecolor=(1.0,1.0,1.0), lw=0.2)
    gdf.plot(color=col[0], ax=ax[0], markersize=gdf['count'].values*0.5, alpha=1)
    cont_df.plot(color=col[1], ax=ax[0], markersize=cont_df['count'].values)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim(-185, 185)
    ax[0].set_ylim(-60, 88)

    width = 0.4
    X = np.arange(len(Cont))
    ax[1].bar(X - width/2, theory, width, label='Theory', color=np.array(col[1])*0.9, alpha=0.8)
    ax[1].bar(X + width/2, inst, width, label='Measured', color=np.array([col[0]])*0.9, alpha=0.8)
    xtra = {1: 0.1, 2:0.05, 3:-0.05}
    for i in range(len(theory)):
        ax[1].annotate(str(theory[i]), (X[i] - 0.4 + xtra[len(str(theory[i]))], theory[i]+5), fontsize=ft1)
        ax[1].annotate(str(inst[i]), (X[i] + xtra[len(str(inst[i]))], inst[i]+5), fontsize=ft1)

    ax[1].set_xticks(X)
    [tick.label.set_fontsize(ft1) for tick in ax[1].xaxis.get_major_ticks()]
    [tick.label.set_fontsize(ft1) for tick in ax[1].yaxis.get_major_ticks()]
    ax[1].set_xticklabels(Cont, rotation=28, fontsize=ft1)
    ax[1].legend(loc='upper right', frameon=False, fontsize=ft1)
    ax[1].set_ylabel('Number of scales', fontsize=ft1+2)
    ax[1].set_ylim(0, 220)

    fig.savefig(PATH_FIG.joinpath(f"world_map.pdf"), bbox_inches='tight')
    

def annotate_scale(ax, xy, s, r=3, c='k', fs=10):
    theta = np.random.rand() * np.pi * 2
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    if x < 0:
        r *= 2
    xyt = xy + np.array([x, y])
#   xyt = xy + np.array(xyt)
    ax.annotate(s, xy=xy, xytext=xyt, arrowprops={'arrowstyle':'->'}, color=c, fontsize=fs)
    
 

def plot_scale_embedding(df, n=7, eps=2, min_samp=5, n_ex=6, seed=8158817):
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

    diatonic_modes = ['Ionian', 'Aeolian', 'Dorian', 'Phrygian', 'Lydian', 'Mixolydian', 'Locrian']
    dia_xy_vec = [(-6,-3), (4,0), (2,-2), (-5,2), (-5,2), (3,-4), (-5,-2)]
    thaats = ['Asavari', 'Bhairav', 'Bhairavi', 'Bilawal', 'Kalyan', 'Khafi', 'Khamaj', 'Marwa', 'Purvi', 'Todi']
    tha_xy_vec = [(0,-4), (2,-2), (-6,-4), (1,-4), (0,4), (-7,-2), (-5,-4), (-6,-3), (-3,3), (0,4)]
    other_scales = ["Nev'eser Makam", "Maqam Sultani Yakah", "Mela Chitrambari", "Tsinganikos",
                    "Mela Pavani", "Dastgah-e Homayun", "Maqam Nakriz", "Mela Kantamani", "Huzzam Makam", "Maqam Saba"]
    melakarta = df.loc[(df.n_notes==n)&(df.SocID=='Car9'), 'Name'].unique()

    for s, xy in zip(thaats, tha_xy_vec):
        i = np.where(df.loc[df.n_notes==n, 'Name']==s)[0][0]
        ax.annotate(s, (X[i], Y[i]), (X[i] + xy[0], Y[i] + xy[1]), arrowprops={'arrowstyle':'->'}, color='r')

    for s, xy in zip(diatonic_modes, dia_xy_vec):
        i = np.where(df.loc[df.n_notes==n, 'Name']==s)[0][0]
        ax.annotate(s, (X[i], Y[i]), (X[i] + xy[0], Y[i] + xy[1]), arrowprops={'arrowstyle':'->'}, color='k')

    for s in other_scales:
#   for s in melakarta:
        i = np.where(df.loc[df.n_notes==n, 'Name']==s)[0][0]
        annotate_scale(ax, (X[i], Y[i]), s, r=3, c='b')



    clust = DBSCAN(eps=eps, min_samples=min_samp).fit(tsne.embedding_).labels_
    N = np.max(clust) + 1
    bins = np.arange(15, 1200, 30)
    xgrid = bins[:-1] + 15
    xbar = np.arange(len(Reg))
    bar_col = list(col_key.values())

    for i, (c, nc) in enumerate(sorted(Counter(clust[clust>0]).items(), key=lambda x: x[1], reverse=True)):
        if i >= n_ex:
            continue

        alpha_shape = alphashape(tsne.embedding_[clust==c], 0.2)
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.4, color=col2[i%8]))

        sns.distplot(scale[clust==c].ravel(), kde=False, norm_hist=True, bins=bins, color=col2[i%8], ax=ax2[i,0])
#       ax2[i,0].set_title(f"C = {c}; N = {nc}")
        ax2[i,0].set_yticks([])
        ylo, yhi = ax2[i,0].get_ylim()
        scale_mean = scale[clust==c].mean(axis=0)
        for j in range(1, n):
#           ax2[i,0].plot([j*1200/n]*2, [0, yhi], ':', c='grey')
#           ax2[i,0].plot([j*1200/n-30]*2, [0, yhi], ':', c='grey')
#           ax2[i,0].plot([j*1200/n+30]*2, [0, yhi], ':', c='grey')
            ax2[i,0].fill_between([j*1200/n-30, j*1200/n+30], [0,0], [yhi,yhi], color='grey', alpha=0.5)
            ax2[i,0].plot([scale_mean[j-1]]*2, [0, yhi], '-k')
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

    is_theory = df.loc[df.n_notes==n, 'Theory'] == 'Y'
    ax3[0].scatter(X[is_theory], Y[is_theory], s=20, c=col2[0], alpha=0.5, label='Theory')
    ax3[0].scatter(X[is_theory==False], Y[is_theory==False], s=20, c=col2[1], alpha=0.5, label='Measured')
    ax3[0].legend(loc='best', frameon=False)

    equi = np.mean(np.abs(scale - np.arange(1,n)*1200/n), axis=1)
    is_equi = equi <= 30
    ax3[1].scatter(X[is_equi], Y[is_equi], s=20, c=col2[2], alpha=0.5, label='Equidistant')
    ax3[1].scatter(X[is_equi==False], Y[is_equi==False], s=20, c=col2[4], alpha=0.5, label='Non-equidistant')
    ax3[1].legend(loc='best', frameon=False)
#   im = ax3[1].scatter(X, Y, s=20, c=equi, alpha=0.5, label='Equidistant')
#   fig.colorbar(im, ax=ax3[1])
    
    soc_lbls = ['Carnatic', 'Arabic']
    for i, (soc, j) in enumerate(zip(['Car9', 'Ara11'], [3,6])):
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
        a.set_xlabel("Dimension 1")
        a.set_ylabel("Dimension 2")
    for a in ax2[:,0]:
        a.spines['left'].set_visible(False)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    handles = [Line2D([], [], ls='', marker='o', color=c) for c in col_key.values()]
    ax.legend(handles, Reg, loc='best', frameon=False)

    fig.savefig(PATH_FIG.joinpath(f"scale_embedding_{n}.pdf"), bbox_inches='tight')
    return clust, tsne.embedding_


def plot_histogram(mean_dist, xlbls, ylbls, fig='', ax=''):
    if isinstance(ax, str):
        fig, ax = plt.subplots()
    im = ax.imshow(mean_dist)
    ax.set_xticks(range(len(mean_dist)))
    ax.set_yticks(range(len(mean_dist)))
    ax.invert_yaxis()
    ax.set_xticklabels(xlbls, rotation=60)
    ax.set_yticklabels(ylbls, rotation=60)
    fig.colorbar(im, ax=ax)


def cluster_societies(df, n=7, min_samp=2, norm=False):
    soc = df.loc[df.n_notes==n, 'SocID'].values
    societies = np.array([k for k, v in df.loc[df.n_notes==n, 'SocID'].value_counts().items() if v >= min_samp])
    df_soc = pd.read_csv('../../ScalesDatabase/Metadata/societies.csv')
    soc_name = np.array([df_soc.loc[df_soc.ID==s, 'Name.x'].values[0] for s in societies])

    scale = np.array([x for x in df.loc[df.n_notes==n, 'scale']])[:,1:-1]
    dist = cdist(scale, scale, lambda x, y: np.mean(np.abs(x - y)))
    mean_dist = np.array([[np.mean(dist[soc==s1][:,soc==s2]) for s1 in societies] for s2 in societies])
    min_dist = np.array([[np.mean(np.min(dist[soc==s1][:,soc==s2], axis=0)) for s1 in societies] for s2 in societies])
    equidist = np.mean(np.abs(scale - np.arange(1,n)*1200/n), axis=1)

    if norm:
        mean_diag = np.diag(mean_dist)
        norm_factor = np.mean(np.meshgrid(mean_diag, mean_diag), axis=0)
        mean_dist = mean_dist / norm_factor

    fig = plt.figure(figsize=(18,14))
    gs = GridSpec(2,2, height_ratios=[1,0.5])
    ax = [fig.add_subplot(gs[j,i]) for j in [0,1] for i in [0,1]]
#   ax = [fig.add_subplot(gs[1,i]) for i in [0,1]] + [fig.add_subplot(gs[0,:])]
    fig.subplots_adjust(hspace=0.4)

    li = linkage(mean_dist[np.triu_indices(len(mean_dist), 1)], method='ward')
    idx = np.argsort(fcluster(li, li[0,2], criterion='distance'))
    plot_histogram(mean_dist[idx][:,idx], soc_name[idx], societies[idx], fig, ax[0])

    X = np.arange(len(societies))
    ax[2].plot(X, np.array([np.mean(equidist[soc==s1]) for s1 in societies[idx]]))
    ax[2].set_xticks(X)
    ax[2].set_xticklabels(soc_name[idx], rotation=60)
    i, j = np.triu_indices(len(min_dist), 1)
    md_sum = min_dist[i,j] + min_dist[j,i]
    li = linkage(md_sum, method='ward')
    idx = np.argsort(fcluster(li, li[0,2], criterion='distance'))
    plot_histogram(min_dist[idx][:,idx], soc_name[idx], societies[idx], fig, ax[1])

#   ax[3].plot(X, np.array([np.sum(soc==s1) for s1 in societies[idx]]))
    ax[3].plot(X, np.array([np.mean(equidist[soc==s1]) for s1 in societies[idx]]))
    ax[3].set_xticks(X)
    ax[3].set_xticklabels(soc_name[idx], rotation=60)

    ttls = ["Cluster by mean distance between all scales", "Cluster by mean distance between closest scales"]
    ylbl = ["SocID"] * 2 + ["Distance from equidistant scale"] * 2
    for i, a in enumerate(ax):
        a.set_xlabel("Society")
        if i < 2:
            a.set_title(ttls[i])

    fig.savefig(PATH_FIG.joinpath("cluster_societies.pdf"), bbox_inches='tight')



    




