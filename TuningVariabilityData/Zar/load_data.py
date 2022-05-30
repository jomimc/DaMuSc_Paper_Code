
import numpy as np

def load_melodic_intervals():
    out = {}
    for line in open('melodic_intervals.txt'):
        splt = line.strip('\n').split(',')
        key = splt[0]
        data = np.array(splt[1:]).astype(float)
        out[key] = data.reshape(data.size//2,2)
    return out


def load_harmonic_intervals():
    out = {}
    for line in open('harmonic_intervals.txt'):
        splt = line.strip('\n').split(',')
        key = splt[0]
        data = np.array(splt[1:]).astype(float)
        out[key] = data.reshape(data.size//2,2)
    return out


def align_intervals(data):
    keys = np.array(list(data.keys()))
    d_len = np.array([len(v) for v in data.values()])
    i = np.argmax(d_len)
    nmax = d_len.max()

    aligned = np.zeros((len(data), nmax)) * np.nan
    aligned[i] = data[keys[i]][:,0]
    for j in range(len(data)):
        if i == j:
            continue
        for k in range(len(data[keys[j]])):
            i1 = data[keys[j]][k,0]
            l = np.argmin(np.abs(aligned[i] - i1))
            aligned[j,l] = i1
    return aligned


def get_deviation_from_mean(mat):
    return np.concatenate([mat[:,i] - np.nanmean(mat[:,i]) for i in range(mat.shape[1])])


def get_other_diff(mel, harm):
    diff = []
    for k in mel.keys():
        m = mel[k].copy() - mel[k].min()
        for k2 in harm.keys():
            if k != k2:
                for h in harm[k][:,0]:
                    diff.append(np.min(np.abs(m - h)))
    return diff


def get_mel_harm_diff(mel, harm):
    diff = []
    mel = mel.copy() - mel.min()
    for h in harm:
        diff.append(np.min(np.abs(mel - h)))
    return diff


if __name__ == "__main__":
    mel = load_melodic_intervals()
    harm = load_harmonic_intervals()

#   keys = list(mel.keys())
#   diff = [d for k in keys for d in get_mel_harm_diff(mel[k][:,0], harm[k][:,0])]

    mel_gmm_std = np.concatenate([v[:,1] for v in mel.values()])
    harm_gmm_std = np.concatenate([v[:,1] for v in harm.values()])

    print("Standard deviations of Guassian mixture models")
    print("Melodic average: ", mel_gmm_std.mean())
    print("Harmonic average ", harm_gmm_std.mean())


    mel_aligned = align_intervals(mel)
    har_aligned = align_intervals(harm)

    mel_err_mat = get_deviation_from_mean(mel_aligned)
    har_err_mat = get_deviation_from_mean(har_aligned)

    np.savetxt('melodic_scale_note_deviation_from_mean.txt', mel_err_mat)
    np.savetxt('harmonic_scale_note_deviation_from_mean.txt', har_err_mat)

    mel_err = np.nanstd(mel_err_mat)
    har_err = np.nanstd(har_err_mat)
    
    print("Standard deviation of aligned notes across different recordings")
    print("Melodic: ", mel_err)
    print("Harmonic: ", har_err)


