from collections import defaultdict

import pandas as pd
import numpy as np

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
#   print(k, (np.sum((v-v.mean())**2) / len(v))**0.5)
    tot += np.sum((v-v.mean())**2)
    cnt += len(v)
    all_vals.extend(v - v.mean())
np.savetxt("interval_deviation_from_mean.txt", all_vals)
print(f"Total std: {(tot/cnt)**0.5}")




