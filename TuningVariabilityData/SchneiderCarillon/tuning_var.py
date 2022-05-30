from collections import defaultdict
import re

import pandas as pd
import numpy as np


### Compare all intervals of approximately equal size,
### i.e., all fifths, all seconds, etc.
def compare_all_intervals():
    data = pd.read_csv('carillon_tuning.csv')
    scale = data.loc[1:, data.columns[1]].values.astype(float)

    all_cat = defaultdict(list)
    for i in range(len(scale)-1):
        for x in scale[i:] - scale[i]:
            all_cat[100*int(round(x/100))].append(x)

    tot = 0.0
    cnt = 0
    all_vals = []
    for k, v in all_cat.items():
        if len(v) == 1 or k == 0:
            continue
        v = np.array(v)
        tot += np.sum((v-v.mean())**2)
        cnt += len(v)
        all_vals.extend(v - v.mean())
    np.savetxt("interval_deviation_from_mean.txt", all_vals)
    print(f"Total std: {(tot/cnt)**0.5}")
    mean_len = np.mean([len(v) for k, v in all_cat.items() if len(v)>1 and k != 0])
    print(f"Mean number of intervals per category: {mean_len}")


# Only compare intervals that have the same label,
# i.e. only compare A-B with A-B in the octave above
def compare_only_equivalent_intervals():
    data = pd.read_csv('carillon_tuning.csv')
    scale = data.loc[1:, data.columns[1]].values.astype(float)
    note_name = data.loc[1:, data.columns[0]].apply(lambda s: s[s.find("(")+1:s.find(")")][:-1]).values

    all_cat = defaultdict(list)
    for i in range(len(scale)-1):
        for j in range(i+1, len(scale)):
            s_int = f"{note_name[i]}_{note_name[j]}"
            x  = scale[j] - scale[i]
            if x < 1250:
                all_cat[s_int].append(x)

    tot = 0.0
    cnt = 0
    all_vals = []
    for k, v in all_cat.items():
        if len(v) == 1:
            continue
        v = np.array(v)
        tot += np.sum((v-v.mean())**2)
        cnt += len(v)
        all_vals.extend(v - v.mean())
    np.savetxt("equiv_interval_deviation_from_mean.txt", all_vals)
    print(f"Total std: {(tot/cnt)**0.5}")


if __name__ == '__main__':

    compare_all_intervals()
    compare_only_equivalent_intervals()



