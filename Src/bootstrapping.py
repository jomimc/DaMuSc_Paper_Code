import os
from pathlib import Path
import pickle

from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, lognorm, norm, binom, entropy, ks_1samp
from sklearn.manifold  import TSNE
from sklearn.preprocessing import StandardScaler
#mport umap


import octave
import process_csv
import utils

N_PROC = 24

PATH_DATA = Path("../Figures/Data")


def boot_hist(df, xsamp, ysamp, bins, s=0, n_rep=1000):
    Y = []
    for i in range(n_rep):
        if len(xsamp):
            y = []
            for c in df[xsamp].unique():
                all_y = df.loc[df[xsamp]==c, ysamp].values
                y.extend(list(np.random.choice(all_y, replace=True, size=min(s, len(all_y)))))
        else:
            y = np.random.choice(df[ysamp].values, replace=True, size=len(df))
        Y.append(np.histogram(y, bins=bins, density=True)[0])
    Y = np.array(Y)
    return Y.mean(axis=0), np.quantile(Y, 0.025, axis=0), np.quantile(Y, 0.975, axis=0)


def scale_degree(df, n_rep=1000):
    bins = np.arange(3.5, 10, 1)
    X = np.arange(4, 10, 1)
    out = {'X': X,
           'All': boot_hist(df, '', 'n_notes', bins),
           'Theory': boot_hist(df.loc[df.Theory=='Y'], '', 'n_notes', bins),
           'Measured': boot_hist(df.loc[df.Theory=='N'], '', 'n_notes', bins)}

    xsamp_list = ['Region', 'SocID']
    for xsamp, s in zip(xsamp_list, [10, 5]):
        out.update({xsamp: boot_hist(df, xsamp, 'n_notes', bins, s=s)})

    pickle.dump(out, open(PATH_DATA.joinpath("scale_degree.pickle"), 'wb'))
    return out


def unfold_list(l):
    return [x for y in l for x in y]


def boot_hist_list(df, xsamp, ysamp, bins, s=0, n_rep=1000):
    Y = []
    for i in range(n_rep):
        if len(xsamp):
            y = []
            for c in df[xsamp].unique():
                all_y = df.loc[df[xsamp]==c, ysamp].values
                y.extend(unfold_list(np.random.choice(all_y, replace=True, size=min(s, len(all_y)))))
        else:
            y = unfold_list(np.random.choice(df[ysamp].values, replace=True, size=len(df)))
        Y.append(np.histogram(y, bins=bins, density=True)[0])
    Y = np.array(Y)
    return Y.mean(axis=0), np.quantile(Y, 0.025, axis=0), np.quantile(Y, 0.975, axis=0)


def boot_list(df, ysamp='step_intervals'):
    if isinstance(df.loc[0, ysamp], str):
        df[ysamp] = df[ysamp].apply(utils.str_to_ints)

    bins = {'step_intervals':np.arange(-10, 520, 20),
            'scale': np.arange(15, 1170, 30),
            'tonic_intervals': np.arange(15, 1170, 30),
            'all_intervals': np.arange(15, 1170, 30)}[ysamp]

    X = bins[1:] - np.diff(bins[:2]) * 0.5
    out = {'X': X,
           'All': boot_hist_list(df, '', ysamp, bins),
           'Theory': boot_hist_list(df.loc[df.Theory=='Y'], '', ysamp, bins),
           'Measured': boot_hist_list(df.loc[df.Theory=='N'], '', ysamp, bins)}

    xsamp_list = ['Region', 'SocID']
    for xsamp, s in zip(xsamp_list, [10, 5]):
        out.update({xsamp: boot_hist_list(df, xsamp, ysamp, bins, s=s)})
        for n in range(4, 10):
            out.update({f"{xsamp}_{n}": boot_hist_list(df.loc[df.n_notes==n], xsamp, ysamp, bins, s=s)})

    inst_list = ['Idiophone', 'Aerophone', 'Chordophone']
    for inst in inst_list:
        out.update({inst: boot_hist_list(df.loc[df.Inst_type==inst], '', ysamp, bins)})
    
    pickle.dump(out, open(PATH_DATA.joinpath(f"{ysamp}.pickle"), 'wb'))
    return out


def get_int_prob_lognorm(df, ysamp='scale', xsamp='', s=5, mode=''):
    if mode == 'shuffle':
        Y = utils.sample_shuffled_scales(df, xsamp, s)
    elif mode == 'resample':
        idx = utils.sample_df_index(df, xsamp, s)
        alt_df = utils.create_new_scales(df.loc[idx], 1)[0]
        Y = np.array([x for y in alt_df[ysamp].values for x in y])
    else:
        Y = utils.sample_df_value(df, ysamp, xsamp, s)
    bins = np.arange(15, 5000, 30)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.

    try:
        # Get maximum-likelihood lognormal distribution
        shape, loc, scale = [0.93, -45.9, 605.4]
        params = lognorm.fit(Y, loc=loc, scale=scale)

        # Get probability of finding interval in each bin
        bin_prob = np.diff(lognorm.cdf(bins, *params))

        # Count intervals in each bin
        count = np.histogram(Y, bins=bins)[0]
        N = count.sum()

        # Get binomial probability that observed counts (or fewer) would be
        # sampled from the maximum-likelihood lognormal distribution
        prob_less_than = binom.cdf(count, N, bin_prob)
        return [count, np.ones(count.size)*N, prob_less_than]
    except Exception as e:
        print(e)
        print(ysamp, xsamp, s, mode)
        return [np.ones(X.size)*np.nan] * 3


def generate_int_prob_lognorm(df, xsamp, ysamp, nsamp, mode='', nrep=1000):
    for y in ysamp:
        for x in xsamp:
            for n in nsamp:
                for i in range(nrep):
                    yield df, y, x, n, mode


def boot_int_prob_lognorm(df, path, nrep=1000, refresh=False, mode=''):
    if path.exists() and not refresh:
        return np.load(path)
    else:
        df = df.loc[:, ['scale', 'Intervals', 'AllInts', 'Region', 'SocID']]
        bins = np.arange(15, 5000, 30)
        ysamp = ['scale', 'AllInts']
        xsamp = ['Region', 'SocID']
        nsamp = [1, 5, 10, 20, 40]
        shape = (len(ysamp), len(xsamp), len(nsamp), nrep, 3, bins.size-1)
        with Pool(N_PROC) as pool:
            res = np.array(pool.starmap(get_int_prob_lognorm, generate_int_prob_lognorm(df, xsamp, ysamp, nsamp, mode))).reshape(shape)
            np.save(path, res)
        return res


def boot_int_prob_lognorm_all(refresh=True, nrep=1000):
    df = process_csv.instrument_tunings()
    idx_list = [(df.Reduced_scale=='N'),
                (df.Reduced_scale=='N') & (df.Inst_type=='Aerophone'),
                (df.Reduced_scale=='N') & (df.Inst_type=='Chordophone'),
                (df.Reduced_scale=='N') & (df.Inst_type=='Idiophone')]
    path_list = [PATH_DATA.joinpath(f"int_prob_lognorm{x}.npy") for x in ['', '_aero', '_chord', '_idio']]
    for idx, path in zip(idx_list, path_list):
        _ = boot_int_prob_lognorm(df.loc[idx], path, refresh=refresh, nrep=nrep)

    path = PATH_DATA.joinpath(f"int_prob_lognorm_shuffle.npy")
    _ = boot_int_prob_lognorm(df.loc[idx_list[0]], path, refresh=refresh, mode="shuffle", nrep=nrep)
#   _ = boot_int_prob_lognorm(df.loc[idx_list[0]], path, refresh=True, mode="shuffle", nrep=nrep)

    path = PATH_DATA.joinpath(f"int_prob_lognorm_shuffle_nonequidistant.npy")
    idx = (df.Reduced_scale=='N') & (df.irange>100)
    print(np.sum(idx))
#   _ = boot_int_prob_lognorm(df.loc[idx], path, refresh=refresh, mode="shuffle", nrep=nrep)
    _ = boot_int_prob_lognorm(df.loc[idx], path, refresh=True, mode="shuffle", nrep=nrep)

    for i in range(10):
        path = PATH_DATA.joinpath(f"int_prob_lognorm_alt{i}.npy")
        _ = boot_int_prob_lognorm(df.loc[idx_list[0]], path, refresh=refresh, mode="resample", nrep=nrep)


def get_step_int_dist(df, ysamp='step_int', xsamp='SocID', s=5, dx=30):
    Y = utils.sample_df_value(df, ysamp, xsamp, s)
    bins = np.arange(0, 5000, 10)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.
    return np.histogram(Y, bins=bins, density=True)[0]


def generate_step_int_dist(df, xsamp, ysamp, nsamp, nrep=1000):
    for y in ysamp:
        for x in xsamp:
            for n in nsamp:
                for i in range(nrep):
                    yield df, y, x, n


def boot_step_int(refresh=True, nrep=1000, nsamp=[5], ysamp=['step_int'], xsamp=['SocID']):
    path = PATH_DATA.joinpath(f"step_int_density.npy")
    if path.exists() and not refresh:
        return np.load(path)
    else:
        df = process_csv.instrument_tunings()
        df = df.loc[(df.Reduced_scale=='N')]
        bins = np.arange(0, 5000, 10)
        if (len(nsamp) == 1) & (len(xsamp) == 1) & (len(ysamp) == 1):
            shape = (nrep, bins.size-1)
        else:
            shape = (len(ysamp), len(xsamp), len(nsamp), nrep, bins.size-1)
        with Pool(N_PROC) as pool:
            res = np.array(pool.starmap(get_step_int_dist, generate_step_int_dist(df, xsamp, ysamp, nsamp))).reshape(shape)
            np.save(path, res)
        return res
    

def run_tsne():
    poss = np.cumsum(np.load('../PossibleScales/possible_7_20_60_400.npy')[:,:-1], axis=1)
    tsne = TSNE(perplexity=20).fit(poss)
    np.save('../tsne_poss_7_20_60_400.npy', tsne.embedding_)


#ef get_umap_embedding(X):
#   reducer = umap.UMAP()
#   return reducer.fit_transform(StandardScaler().fit_transform(X))


#ef run_umap():
#   poss = np.cumsum(np.load('../PossibleScales/possible_7_20_60_400.npy')[:,:-1], axis=1)
#   embedding = get_umap_embedding(poss)
#   np.save('../umap_poss_7_20_60_400.npy', embedding)


def tmp_fn(df, ysamp='scale', xsamp='SocID', s=1, nmax=0):
    idx_samp = utils.sample_df_index(df, xsamp, s)
    if nmax == 0:
        nmax = len(idx_samp)
    Y = utils.sample_df_value(df, ysamp, xsamp, s)
    bins = np.arange(15, 5000, 30)
    dx = np.diff(bins[:2])
    X = bins[:-1] + dx / 2.

    P = []

    for i in range(1, nmax):
        # Get maximum-likelihood lognormal distribution
        idx = np.random.choice(idx_samp, size=i, replace=False)
        Y = np.array([x for j in idx for x in df.loc[j, 'scale']])
        shape, loc, scale = [0.93, -45.9, 605.4]
        params = lognorm.fit(Y, loc=loc, scale=scale)

        # Get probability of finding interval in each bin
        bin_prob = np.diff(lognorm.cdf(bins, *params))

        # Count intervals in each bin
        count = np.histogram(Y, bins=bins)[0]
        mean_euc = np.sum(bin_prob * count) / np.sum(count)
        mean_geo = np.product((bin_prob**(1/np.sum(count)))**count)
        ks_stat = ks_1samp(Y, lognorm.cdf, args=params)[1]
#       P.append([mean_geo, mean_euc])
        P.append([ks_stat])
    return np.array(P)



def boot_all():
    df = process_csv.process_data()
    _ = scale_degree(df)
    for ysamp in ['step_intervals', 'scale', 'tonic_intervals', 'all_intervals']:
        _ = boot_list(df, ysamp)
    boot_int_prob_lognorm_all(refresh=False)
    boot_step_int()


if __name__ == "__main__":

#   boot_all()
    boot_int_prob_lognorm_all(refresh=False)

#   run_umap()



