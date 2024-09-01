import os
import time

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


COLORS = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
LANDMARKS = ['Ped_Inf_R',
 'Ped_Inf_L',
 'Ped_Sup_R',
 'Ped_Sup_L',
 'Plat_Inf_Cent',
 'Plat_Sup_Cent']

class Plot3D:
    def __init__(self) -> None:
        self._figure = go.Figure(layout = go.Layout(
             scene=dict(
                 aspectmode='data'
         )))
        self._figure.update_layout(
            autosize=False,
            width=500,
            height=700,
        )

        self.__points = {'x' : [], 'y' : [], 'z' : []}
            
    def scatter_vertebrae(self, x:list, y:list, z:list, vertebra_name:str, color:str=COLORS[0], size:int=1):
        """
        Adds a scatter for a vertebra

        Parameters
        ----------
        x : list
            X coordinates for the vertebrae  
        y : list
            Y coordinates for the vertebrae  
        z : list
            Z coordinates for the vertebrae  
        vertebra_name : str
            Vertebra name  
        color : str, optional
            Color, by default COLORS[0]  
        """
        self.__points['x'].extend(x)
        self.__points['y'].extend(y)
        self.__points['z'].extend(z)
        self._figure.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker={
                    'color':color,
                    'size':size
                },
                text=LANDMARKS,
                hovertemplate='<br><b>x</b>:%{x:.2f}<br><b>y</b>:%{y:.2f}<br><b>z</b>:%{z:.2f}<br><b>Landmark</b>:%{text}',
                legendgroup=vertebra_name,
                legendgrouptitle={'text': vertebra_name},
                name = ' '
                )
            )
        line_idx = [0, 2, 3, 1, 0, 4, 1, 4, 5, 3, 5, 2]
        self._figure.add_trace(
            go.Scatter3d(
                x=np.array(x)[line_idx],
                y=np.array(y)[line_idx],
                z=np.array(z)[line_idx],
                mode='lines',
                line={
                    'color':color,
                    'width':5.
                },
                legendgroup=vertebra_name,
                legendgrouptitle={'text': vertebra_name},
                name = vertebra_name,
                showlegend = False
                )
            )

    @property
    def points(self):
        return np.array(
            [self.__points['x'],
             self.__points['y'],
             self.__points['z']]
        ).T
    
    def show(self):
        """
        Displays the figure
        """
        self._figure.show()
    
    def save(self, filename:str = None):
        """
        Saves the figure

        Parameters
        ----------
        filename : str, optional
            Filename for the output, by default None
        """
        if filename is None:
            filename = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(os.getcwd(), filename)
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            filename = filename + '.png'
        self._figure.write_image(filename)

def plot_selected_beads(beads_data:dict, selected_beads:list) -> None:
    """
    Plots a 3D view of selected beads (in green over the two calibration plates A & B). This viz does NOT include plates C and D.

    Parameters
    ----------
    beads_data : dict
        Original beads data.  
    selected_beads : list
        List of selected beads names  
    """
    fig = go.Figure(layout = go.Layout(
             scene=dict(
                 aspectmode='data'
         )))

    for _, (_, bead, x, y, z) in beads_data['3d'].iterrows():
        if 'A' in bead or 'B' in bead:
            fig.add_trace(
                go.Scatter3d(
                    x = [x],
                    y = [y],
                    z = [z],
                    name = bead,
                    mode='markers',
                    marker={'size':3 if bead in selected_beads else 2,
                            'color':'green' if bead in selected_beads else 'black'},
                    showlegend=False
                )
            )
    fig.update_layout(title = 'Selected beads for calibration')
    fig.show()