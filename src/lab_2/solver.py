from numpy.linalg import lstsq
import numpy as np

def dlt(a:np.ndarray):
    """
    Performs the DLT.

    Args:
        a (np.ndarray): Linear equation system matrix so that AX = 0.

    Returns:
        np.ndarray: X vector that solves AX = 0.
    """
    m = -a[:, [-1]]
    a = a[:, :-1]
    return lstsq(a,m, rcond=None)[0].astype(np.float32).squeeze()