import numpy as np


def sampling(n, values):
    """Return sampled value from a list of values"""
    f = sampling_func(values)
    return f(n)


def sampling_func(values):
    """Return a function to sample a list of values"""

    def f(n):
        if not isinstance(n, int):
            raise ValueError(f'n ({n}) must be an integer')
        spp = len(values)
        idx = int(n % spp)
        return values[idx]

    return f


def exponenial_func(x, A, tau, C):
    return A * np.exp(x / tau) + C


def to_db(value, amp=20, masked=True):
    norm = normalize(value)
    if masked:
        db = amp * np.ma.log10(norm)  # use masked array to avoid 0s
    else:
        db = amp * np.log10(norm)  # use masked array to avoid 0s
    return db


def normalize(signal):
    signal = np.array(signal)
    maximum = np.max(np.abs(signal))
    norm = signal / maximum
    return norm


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx