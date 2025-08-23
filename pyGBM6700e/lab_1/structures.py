from typing import Dict

import numpy as np
import pandas as pd

from .io import load_data
from .data import join_dataframes

class Spine():
    """
    Spine structure
    """
    def __init__(self, filepath:str) -> None:
        verts_data = load_data(filepath)
        self.__vertebrae = {}
        self.__views = list(verts_data.keys())

        for v_name in verts_data[self.__views[0]]['name']:
            self.__vertebrae[v_name] = Vertebra(v_name)
        
        for view in self.__views:
            for v_name, points_2d in zip(verts_data[view]['name'], verts_data[view]['points2D']): # points2d is a dict w/ keys name, coord
                self.__vertebrae[v_name].add_view(view, points_2d['name'], np.array(points_2d['coord']))
        
    def __repr__(self) -> str:
        return '{n_v} vertebrae with views {views}'.format(n_v = len(self.__vertebrae.keys()), views = str(self.__views))

    @property
    def views(self) -> list:
        """
        Accesses the views

        Returns
        -------
        list
            List of views names
        """
        return self.__views
    
    @property
    def vertebrae(self) -> list:
        """
        Returns a list of all vertebrae

        Returns
        -------
        list
            vertebrae that make the spine
        """
        return list(self.__vertebrae.keys())
    
    def get(self, vertebra:str) -> Dict:
        """
        Returns the vertebra

        Parameters
        ----------
        vertebra : str
            Vertebra name

        Returns
        -------
        Dict
            Points in both views making a specific vertebra
        """
        if vertebra not in self.__vertebrae:
            raise ValueError('Vertebra must be in '+ str(self.__vertebrae.keys()))
        return self.__vertebrae[vertebra]
        
class Vertebra:
    """
    Wrapper for making manipulation of views for the same Vertebra easy
    """
    def __init__(self, name:str) -> None:
        self.__name = name
        self.__data = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the dataframe containing the points information about the vertebra

        Returns
        -------
        pd.DataFrame
            Dataframe with attributes 'landmark_name', x and y for each view
        """
        return self.__data
    
    def __repr__(self) -> str:
        return self.__name
    
    def add_view(self, view_name:str, names:list, points:np.ndarray=None) -> None:
        """
        Adds points from a view to the vertebra data

        Parameters
        ----------
        view_name : str
            Name of the view to add  
        names : list
            Names of the landmarks names  
        points : np.ndarray, optional
            Coordinates of the points, by default None
        """
        tmp_df = []
        for vert_name, (x, y) in zip(names, points):
            tmp_df.append({
                'landmark_name' : vert_name,
                'x_'+view_name : x,
                'y_'+view_name : y,
            })
        tmp_df = pd.DataFrame.from_records(tmp_df)

        if self.__data is None:
            self.__data = tmp_df
        else:
            self.__data = join_dataframes(self.__data, tmp_df, 'landmark_name')