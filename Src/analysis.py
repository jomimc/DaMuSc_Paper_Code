

import numpy as np
import pandas as pd

import process_csv
import utils


def effect_of_octave_cutoff():
    df_list = {o: process_csv.process_data(oct_cut=o) for o in [1, 5, 10, 20, 30, 40, 50, 75, 100]}
    tmp = {k: len(v.loc[v.Theory=='N']) for k, v in df_list.items()}
    dat = np.array([(k,v) for k, v in tmp.items()])
    np.savetxt('../effect_of_oct_cut.txt', dat, header="oct_cut, number of instrument scales")



def harmonic_intervals(df):
    pass



def octave_chance(df, n_rep=10):
    all_ints = [utils.str_to_int(y) for y in df.all_ints2]
    all_ints = [utils.str_to_int(y) for y in df.all_ints2]



