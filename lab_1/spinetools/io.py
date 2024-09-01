"""
Process files, load .mat and save figures
"""
import os
from typing import Dict

import numpy as np
import pandas as pd
from mat4py import loadmat


def load_data(mat_file:str) -> Dict:
    """
    Reads a .mat file, and returns its content as nested dictionnaries

    Parameters
    ----------
    mat_file : str
        Path to mat file  

    Returns
    -------
    Dict
        Content
    """

    assert mat_file.endswith('.mat')
    data = loadmat(mat_file)
    return data

def process_files(directory:str) -> pd.DataFrame:
    """
    Process mat files in a directory to create a dataframe aggregating data from different views

    Parameters
    ----------
    directory : str
        Directory containing .mat files  

    Returns
    -------
    pd.DataFrame
        Dict with 2 keys : 2d and 3d, each associated to a dataframe with columns 'view', 'bead', 'x', 'y' ('z' for the 3D view)
    """
    files = [os.path.realpath(os.path.join(directory, x)) for x in os.listdir(directory) if (x.endswith('.mat') and 'beads' in x.lower())]
    out = {}
    for file in files:
        if '2d' in os.path.basename(file).lower():
            tmp = load_data(file)
            tmp_dict = []
            for view, beads_data in tmp.items():
                for bead_name, coordinates in zip(beads_data['name'], beads_data['coord']):
                    tmp_dict.append({'view':view, 'bead':bead_name, 'x_2d':coordinates[0], 'y_2d':coordinates[1]})
            out['2d'] = pd.DataFrame.from_records(tmp_dict)
        elif '3d' in os.path.basename(file).lower():
            tmp = load_data(file)
            tmp_dict = []
            for view, beads_data in tmp.items():
                for bead_name, coordinates in zip(beads_data['name'], beads_data['coord']):
                    tmp_dict.append({'view':view, 'bead':bead_name, 'x_3d':coordinates[0], 'y_3d':coordinates[1], 'z_3d':coordinates[2]})
            out['3d'] = pd.DataFrame.from_records(tmp_dict)
        else:
            print('Ignored %s'%os.path.basename(file))
    
    return out