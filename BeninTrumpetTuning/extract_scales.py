"""
Code used to extract scales from
    'L. Copeland,  Pitch and tuning in Beninese brass bands, Ethnomusicology Forum 27, 213, (2018)'

Methodology:
    We take the means of Gaussian mixture models as representative values for scale notes.

    The Guassian mixture models are fitted to distributions of notes taken from performances;
    notes were computationally extracted by the original author, and are presented (in Hz) along with a
    musical score in the original paper.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import shgo, brute, basinhopping, dual_annealing
from scipy.stats import norm
import seaborn as sns


### In the end, optimization didn't work so well...
### so I just optimized by myself (I found better results than the fancy algorithms...)
### For dia: bounds = [100, 300, 420, 600, 780]
### For penta the algorithm worked well:
###       bounds = [60, 330, 470, 760]


### From this, I was able to extract the two scales.
### I used the tonic indicated in the text as the first note.
### Ther
###     dia: [0, 188, 515, 701, 893, 1001, 1200]
###     penta: [0, 227, 387, 694, 886, 1200]



def load_dia():
    path = '/home/johnmcbride/projects/Scales/TrumpetBenin/diatonic.dat'
    return np.array([l.strip('\n').split(',') for l in open(path, 'r')], float)


def load_penta():
    path = '/home/johnmcbride/projects/Scales/TrumpetBenin/pentatonic.dat'
    return [np.array(l.strip('\n').split(','), float) for l in open(path, 'r')]


def get_cents_from_ratio(ratio):
    return 1200.*np.log10(ratio)/np.log10(2)


def get_scales(data):
    return np.array([np.array([get_cents_from_ratio(f/d[0]) for f in d]) % 1200 for d in data])


def transpose_scales(data, dx=80):
    return (data + dx) % 1200 - dx


def get_ints(data):
    return np.array([[get_cents_from_ratio(f2/f1) for f1, f2 in zip(d[:-1], d[1:])] for d in data])


def evaluate_borders(borders, X):
    notes, params = fit_gauss(X, borders)
    return np.mean([p[1] for p in params])
#   return - np.sum([x for n, p in zip(notes, params) for x in norm.pdf(n, *p)])
    

def fit_gauss(X, borders):
    borders = sorted(borders) + [max(X)+1]
    idx = np.digitize(X, borders)
    notes = []
    params = []
    for i in range(len(borders)):
        n = X[idx==i]
        notes.append(n)
        mu, sigma = norm.fit(n)
        params.append((mu, max(sigma, 10)))
    return notes, params


def best_fit_gauss(X, N):
    mu0 = np.arange(0, 1200, 1200 / N)
    borders = np.convolve(mu0, np.ones(2)/2, mode='valid')
    border_range = [(max(min(X), b-300), min(b+300, max(X))) for b in borders]
    return dual_annealing(evaluate_borders, border_range, args=[X], initial_temp=9000, restart_temp_ratio=0.001, maxiter=5000)
#   return shgo(evaluate_borders, border_range, args=[X])
    
#   return brute(evaluate_borders, border_range, args=[X], Ns=5)

#   return basinhopping(evaluate_borders, borders, minimizer_kwargs={'args':(X)})


def view_borders(X, borders, params):
    fig, ax = plt.subplots()
    bins = np.arange(-70, 1250, 20)
    xgrid = np.arange(-70, 1250, 5)
    sns.distplot(X, kde=False, norm_hist=True, bins=bins)
    y = ax.get_ylim()[1]

    for b in borders:
        ax.plot([b]*2, [0,y*0.8], ':k')

    for p in params:
        Y = norm.pdf(xgrid, *p)
        ax.plot(xgrid, Y)


if __name__ == "__main__":

    pass




