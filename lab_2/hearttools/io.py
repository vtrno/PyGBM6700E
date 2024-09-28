"""
Process files, load .mat and save figures
"""
import os
from typing import Dict

import imageio.v3 as imio
import numpy as np
from mat4py import loadmat

from .structures import Mesh


def load_data(mat_file:str) -> Dict:
    """
    Reads a .mat file, and returns its content as nested dictionaries.

    Args:
        mat_file (str): Path to the .mat file.

    Returns:
        dict: Content of the .mat file.
    """

    assert mat_file.endswith('.mat')
    data = loadmat(mat_file)
    return data

def load_mesh(filepath:str) -> Mesh:
    """
    Loads a mesh from a file and returns it as a Mesh object.

    Args:
        filepath (str): Path to the file containing the mesh data.

    Returns:
        Mesh: A Mesh object containing vertices, faces, and skeleton data.
    """
    data = load_data(filepath)
    return Mesh(
        vertices=np.asarray(data['Coronary']['isosurface']['vertices']),
        faces=np.asarray(data['Coronary']['isosurface']['faces']) - 1,
        skeleton=data['Coronary']['skeleton']
    )

def read_xray(filepath:str) -> np.ndarray:
    """
    Reads an X-ray image from a file and returns it as a NumPy array.

    Args:
        filepath (str): Path to the file containing the X-ray image.

    Returns:
        np.ndarray: The X-ray image as a NumPy array.
    """
    return imio.imread(filepath)