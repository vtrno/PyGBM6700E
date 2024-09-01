from numpy.linalg import lstsq
import numpy as np

def dlt(a):
    """
    Performs the DLT
    """
    m = -a[:, [-1]]
    a = a[:, :-1]
    return lstsq(a,m, rcond=None)[0].astype(np.float32).squeeze()