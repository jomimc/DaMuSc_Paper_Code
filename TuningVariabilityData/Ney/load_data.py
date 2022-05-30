
import numpy as np

def load_melodic_intervals():
    out = {}
    for line in open('tunings.txt'):
        splt = line.strip('\n').split(',')
        key = splt[0]
        data = np.array(splt[1:]).astype(float)
        out[key] = data * 1200 / 53
    return out


def get_err():
    data = load_melodic_intervals()
    mat = np.array([v for v in data.values()])
    err = np.concatenate([mat[:,i] - mat[:,i].mean() for i in range(8) if i != 1])
    np.savetxt('scale_note_deviation_from_mean.txt', err)
    print(np.std(err))

    ints = np.diff(mat, axis=1)
    err_int = np.concatenate([ints[:,i] - ints[:,i].mean() for i in range(7)])
    np.savetxt('step_interval_deviation_from_mean.txt', err_int)
    print(np.std(err_int))



if __name__ == "__main__":

    get_err()


