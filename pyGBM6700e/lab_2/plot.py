import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

COLORS = list(plt.colormaps.get("Set1").colors)
COLORS.pop(5)

def imshow(points:np.ndarray,image_size:tuple[int], labels:np.ndarray = None, figure_title:str = None) -> None:
    """
    Displays a scatter plot of points.

    Args:
        points (np.ndarray): Array of points to plot.
        image_size (tuple[int]): Size of the image (width, height).
        labels (np.ndarray, optional): Labels for each point. Defaults to None.
        figure_title (str, optional): Title of the figure. Defaults to None.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    if figure_title is not None:
        ax.set_title(figure_title)
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(0, image_size[1])
    ax.scatter(points[:, 0], points[:, 1], s=1, c=[COLORS[x] for x in labels], marker = 's')
    fig.show()