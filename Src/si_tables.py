from collections import Counter, defaultdict

import numpy as np
import pandas as pd


from process_csv import PATH_RAW, PATH_BASE


def si1(df, inst):
    df_theory = pd.read_csv(PATH_RAW.joinpath("Data/theory_scales.csv"))
    src = pd.read_csv(PATH_RAW.joinpath("Metadata/sources.csv"))
    ref_bib_key = {i:j for i, j in pd.read_csv(PATH_BASE.joinpath("Data/refID_bibtex_key.csv")).values}
    count = Counter(df.RefID)
    for i in range(1, 61):
        if i not in count.keys():
            count[i] = 0
    intro = r"Ref & RefID & Year & T & M\textsubscript{inst} &  M\textsubscript{rec} & " + \
            r"OT & OM\textsubscript{inst} &  OM\textsubscript{rec} & O\textsubscript{total} \\"
    running_total = np.zeros(7, int)
    print(intro)
    for i, (refID, c) in enumerate(sorted(count.items(), key=lambda x:x[1], reverse=True)):
        bibkey = ref_bib_key[refID]
        year = src.loc[src.RefID==refID, ['Year']].values[0][0]

        N_theory = np.sum(df_theory.RefID==refID)

        meas_type = inst.loc[inst.RefID==refID, 'Measured_type']
        N_inst = np.sum(meas_type == 'Instrument')
        N_rec = np.sum(meas_type == 'Recording')

        theory, meas_type = df.loc[df.RefID==refID, ['Theory', 'Measured_type']].values.T
        ON_theory = np.sum(theory == 'Y')
        ON_inst = np.sum(meas_type == 'Instrument')
        ON_rec = np.sum(meas_type == 'Recording')

        ON_total = ON_theory + ON_inst + ON_rec

        running_total += np.array([N_theory, N_inst, N_rec, ON_theory, ON_inst, ON_rec, ON_total])

        print(f"\\cite{{{bibkey}}} & {refID:d} & {year:d} & {N_theory} & {N_inst} & {N_rec} & {ON_theory} & {ON_inst} & {ON_rec} & {ON_total} \\\\")

    print(" &  &  & " + " & ".join(running_total.astype(str)) + r" \\")


