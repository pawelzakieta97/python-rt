from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


class Ray:
    def __init__(self, start, direction, color):
        self.start = start
        self.direction = direction
        self.color = color


class Rays:
    def __init__(self, rays_matrix=None, rays_array=None):
        if rays_matrix is None and rays_array is None:
            rays_matrix = np.zeros((0, 9))
        self._rays_matrix = rays_matrix
        self._rays_array = rays_array
        self.count = rays_matrix.shape[0] if rays_matrix is not None else len(rays_array)
        self.additional_data = {}

    @property
    def rays_array(self):
        if self._rays_array is not None:
            return self._rays_array
        rays = []
        for r in self._rays_matrix:
            rays.append(Ray(r[:3], r[3:6], r[6:9]))
        self._rays_array = rays
        return rays

    @property
    def rays_matrix(self):
        if self._rays_matrix is not None:
            return self._rays_matrix
        rays = np.zeros((len(self._rays_array), 9))
        for i, r in enumerate(self._rays_array):
            rays[i, :3] = r.start
            rays[i, 3:6] = r.direction
            rays[i, 6:9] = r.color
        self._rays_matrix = rays
        return rays

    @property
    def starts(self):
        return self.rays_matrix[:, :3]

    @property
    def directions(self):
        return self.rays_matrix[:, 3:6]

    @property
    def colors(self):
        return self.rays_matrix[:, 6:9]

    def set_colors(self, colors: np.array):
        self.rays_matrix[:, 6:9] = colors

    def add_rays(self, rays: Rays):
        if self._rays_matrix.shape[0] == 0:
            self._rays_matrix = rays.rays_matrix
            return
        self._rays_matrix = np.vstack((self._rays_matrix, rays.rays_matrix))

    def add_data(self, additional_data: np.array, data_name=None):
        idx_start = self._rays_matrix.shape[1]
        self._rays_matrix = np.hstack(self._rays_array, additional_data)
        if data_name is not None:
            self.additional_data[data_name] = (idx_start, self._rays_matrix.shape[1])

    def get_additional_data(self, data_name):
        return self._rays_matrix[:, self.additional_data[data_name][0]: self.additional_data[data_name][1]]


class RaysPD:
    def __init__(self, data: Union[np.array, pd.DataFrame]=None, starts=None, directions=None, colors=None):
        if isinstance(data, pd.DataFrame):
            self.data = data
            return
        if starts is not None and directions is not None and colors is not None:
            self.data = pd.DataFrame(
                {
                    'x': starts[:, 0], 'y': starts[:, 1], 'z': starts[:, 2],
                    'dx': directions[:, 0], 'dy': directions[:, 1], 'dz': directions[:, 2],
                    'r': colors[:, 0], 'g': colors[:, 1], 'b': colors[:, 2]
                }
            )
            return
        self.data = pd.DataFrame(data)
        if data is not None:
            self.data.columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'r', 'g', 'b']

    @property
    def starts(self):
        return self.data[['x', 'y', 'z']].to_numpy()

    @property
    def count(self):
        return len(self.data)

    @property
    def directions(self):
        return self.data[['dx', 'dy', 'dz']].to_numpy()

    @property
    def colors(self):
        return self.data[['r', 'g', 'b']].to_numpy()

    def set_starts(self, starts: np.array):
        self.data['x'] = starts[:, 0]
        self.data['y'] = starts[:, 1]
        self.data['z'] = starts[:, 2]

    def set_directions(self, starts: np.array):
        self.data['dx'] = starts[:, 0]
        self.data['dy'] = starts[:, 1]
        self.data['dz'] = starts[:, 2]

    def set_colors(self, colors: np.array):
        self.data['r'] = colors[:, 0]
        self.data['g'] = colors[:, 1]
        self.data['b'] = colors[:, 2]

    def add_rays(self, rays: RaysPD):
        self.data = pd.concat((self.data, rays.data), axis=0)

    def add_data(self, additional_data: pd.DataFrame):
        self.data = pd.concat((self.data, additional_data), axis=1)

    def get_data(self, columns: list[str]):
        return self.data[columns].to_numpy()

    def copy(self) -> RaysPD:
        return RaysPD(self.data.copy())

    def __getitem__(self, item):
        return RaysPD(self.data.iloc[item])
