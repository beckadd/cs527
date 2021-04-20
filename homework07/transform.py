import numpy as np


def transform(p, r=None, t=None):
    if r is None:
        r = np.eye(p.shape[0])
    if t is None:
        t = np.zeros(p.shape[0])
    shifts = np.outer(t, np.ones(p.shape[1]))
    return r @ (p - shifts)
