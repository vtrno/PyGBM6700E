import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import KDTree


class Pointcloud:
    def __init__(self, vertices: np.ndarray) -> None:
        """
        Initializes the Pointcloud with given vertices.

        Args:
            vertices (np.ndarray): A numpy array of vertices.
        """
        self._verts = vertices

    @property
    def vertices(self) -> np.ndarray:
        """
        Returns the vertices of the point cloud.

        Returns:
            np.ndarray: The vertices of the point cloud.
        """
        return self._verts
    
    @property
    def skeleton(self) -> np.ndarray:
        """
        Returns the skeleton of the point cloud if it exists.

        Returns:
            np.ndarray: The skeleton of the point cloud.
            None: If the skeleton does not exist.
        """
        try:
            return self._skeleton
        except AttributeError:
            return None
    
    @property
    def skeleton_labels(self) -> np.ndarray:
        """
        Returns the skeleton labels of the point cloud if they exist.

        Returns:
            np.ndarray: The skeleton labels of the point cloud.
            None: If the skeleton labels do not exist.
        """
        try:
            return self._labels
        except AttributeError:
            return None
        
    @property
    def labels(self) -> np.ndarray:
        """
        Returns the labels of the vertices if they exist.

        Returns:
            np.ndarray: The labels of the vertices.
            None: If the labels do not exist.
        """
        try:
            return self._vertices_labels
        except AttributeError:
            return None
        
    def plot(self, **kwargs) -> None:
        """
        Plots the mesh, with color labels if available. Takes any argument you'd pass to go.Figure.

        Args:
            **kwargs: Arbitrary keyword arguments passed to go.Figure.update_layout.
        """
        colors = px.colors.qualitative.Plotly
        fig = go.Figure(layout=go.Layout(
            scene=dict(
                aspectmode='data'
            )
        ))
        fig.update_layout(**kwargs)
        
        size = kwargs.pop('size', 1)
        
        if hasattr(self, 'faces'):
            fig.add_trace(
                go.Mesh3d(
                    x=self.vertices[:, 0],
                    y=self.vertices[:, 1],
                    z=self.vertices[:, 2],
                    i=self.faces[:, 0],
                    j=self.faces[:, 1],
                    k=self.faces[:, 2],
                    vertexcolor=[colors[i] for i in self.labels] if self.labels is not None else [colors[0] for _ in range(len(self.vertices))],
                )
            )
        else:
            marker_dict = {'size': size, 'color': self.labels}
            if self.labels is not None:
                del marker_dict['color']
            fig.add_trace(
                go.Scatter3d(
                    x=self.vertices[:, 0],
                    y=self.vertices[:, 1],
                    z=self.vertices[:, 2],
                    mode='markers',
                    marker=marker_dict
                )
            )
        fig.show()

class Mesh(Pointcloud):
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, skeleton: np.ndarray = None) -> None:
        """
        Mesh object.

        Args:
            vertices (np.ndarray): Array of shape (N, 3) representing the vertices.
            faces (np.ndarray): Array of shape (M, 3) representing the faces.
            skeleton (np.ndarray, optional): Array of shape (P, 3) representing the skeleton. Defaults to None.
        """
        super().__init__(vertices=vertices)
        self._faces = faces
        
        if skeleton is not None:
            self._skeleton = []
            self._labels = []
            for i, branch in enumerate(skeleton):
                self._skeleton.extend(branch[0])
                self._labels.extend([i] * len(branch[0]))
            self._skeleton = np.asarray(self._skeleton)
            self._labels = np.asarray(self._labels)
            self._kdtree = KDTree(self._skeleton)
            _, self._neigh = self._kdtree.query(self._verts)
            self._vertices_labels = self._labels[self._neigh]
    
    @property
    def faces(self) -> np.ndarray:
        """
        Returns the faces of the mesh.

        Returns:
            np.ndarray: The faces of the mesh.
        """
        return self._faces
    
    @property
    def isosurface(self) -> dict:
        """
        Returns the isosurface of the mesh.

        Returns:
            dict: A dictionary containing the vertices and triangles of the mesh.
        """
        return {'vertices': self.vertices, 'triangles': self.faces}

    def plot_compare(self, x: Pointcloud) -> None:
        """
        Plots a comparison between the skeleton of the mesh and the vertices of another point cloud.

        Args:
            x (Pointcloud): The point cloud to compare against the mesh's skeleton.

        Raises:
            ValueError: If the number of points in the skeleton and the reference point cloud do not match.
        """
        reference_points = x.vertices
        if reference_points.shape[0] != self.skeleton.shape[0]:
            raise ValueError('Please check that both structures have the same number of points. This mesh has {n0} points whereas the input point cloud has {n1}'.format(n1=reference_points.shape[0], n0=self.skeleton.shape[0]))
        
        norm_difference = np.linalg.norm(reference_points - self.skeleton, axis=1)
        
        fig = go.Figure(layout=go.Layout(
            scene=dict(
                aspectmode='data'
            )
        ))
        
        fig.add_trace(
            go.Scatter3d(
                x=self.skeleton[:, 0],
                y=self.skeleton[:, 1],
                z=self.skeleton[:, 2],
                mode='markers',
                marker={'size': 2, 'color': norm_difference, 'colorscale': 'Viridis', "colorbar": dict(thickness=20)}
            )
        )
        
        fig.show()
