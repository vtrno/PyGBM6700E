import numpy as np
from scipy.interpolate import CubicSpline

def interpolate(X:np.ndarray, n:int) -> np.ndarray:
    """
    Uses cubic spline interpolation to increase the number of points.

    Args:
        X (np.ndarray): Original points.
        n (int): Target number of points.

    Returns:
        np.ndarray: New interpolated points.
    """
    # Create the spline
    x = np.arange(1, len(X) + 1)
    cs = CubicSpline(x, X, axis=0)
    
    # Generate n points on the spline
    x_new = np.linspace(1, len(X), n)
    Y = cs(x_new)
    
    return Y

def sample_n_points(X:np.ndarray, n:int) -> np.ndarray:
    """
    Samples evenly spaced points from an array.

    Args:
        X (np.ndarray): Reference points, shape (N, D).
        n (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled array.
    """
    idx = np.linspace(0, X.shape[0], endpoint=False, num=n, dtype = np.int64)
    return X[idx]