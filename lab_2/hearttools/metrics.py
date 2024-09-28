import numpy as np
from scipy.stats.stats import pearsonr


def compute_stats(x:np.ndarray, y:np.ndarray) -> dict:
    """
    Computes statistics on 1D series.

    Args:
        x (np.ndarray): Data series 1.
        y (np.ndarray): Data series 2.

    Returns:
        dict: Dictionary containing the measurements.
    """
    out = {}
    for s_name, series in zip(['x', 'y'], [x, y]):
        out['mean_'+s_name] = series.mean()
        out['std_'+s_name] = series.std()
        out['min_'+s_name] = series.min()
        out['max_'+s_name] = series.max()
    out['correlation'] = pearsonr(x, y).statistic
    return out

def rms(y_true:np.ndarray, y_pred:np.ndarray):
    """
    Calculates RMS.

    Args:
        y_true (np.ndarray): Reference value.
        y_pred (np.ndarray): Calculated/predicted value.

    Returns:
        float: The calculated RMS value.
    """
    if len(y_true.shape) == 2:

        return np.sqrt(
            np.power(
                np.linalg.norm(y_pred - y_true, axis=1),
                2
                ).mean()
        )
    elif len(y_true.shape) == 1:
        return np.sqrt(
            np.power(
                y_pred - y_true,
                2
                ).mean()
        )