from __future__ import annotations
import numpy as np

from ray import Rays


class RaysNP(Rays):
    def __init__(self, data: np.array = None, starts=None, directions=None, colors=None):
        self.columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'r', 'g', 'b']
        self.data = None
        if data is not None:
            self.data = np.array(data)#, order='F')
            return
        if starts is not None and directions is not None and colors is not None:
            self.data = np.hstack((starts, directions, colors))#, order='F')
            return

    def get_column_indices(self, column_names: list):
        return [self.columns.index(c) for c in column_names]

    @property
    def starts(self):
        return self.data[:, :3]

    @property
    def count(self):
        return self.data.shape[0]

    @property
    def directions(self):
        return self.data[:, 3:6]

    @property
    def colors(self):
        return self.data[:, 6:9]

    def set_starts(self, starts: np.array):
        self.data[:, :3] = starts

    def set_directions(self, starts: np.array):
        self.data[:, 3:6] = starts

    def set_colors(self, colors: np.array):
        self.data[:, 6:9] = colors

    def add_rays(self, rays: RaysNP):
        if self.data is None:
            self.data = rays.data.copy()
            self.columns = [c for c in rays.columns]
            return
        self.data = np.vstack([self.data, rays.data])

    def add_data(self, additional_data: np.array, columns: list[str]):
        self.data = np.hstack((self.data, additional_data))
        self.columns = self.columns + columns

    def get_data(self, columns: list[str]):
        column_indices = self.get_column_indices(columns)
        if column_indices[-1] - column_indices[0] == len(columns):
            return self.data[:, column_indices[0]:column_indices[-1]]
        return self.data[:, column_indices]

    def copy(self) -> RaysNP:
        rays = RaysNP(self.data.copy())
        rays.columns = [c for c in self.columns]
        return rays

    def __getitem__(self, item):
        rays = RaysNP(self.data[item, :])
        rays.columns = [c for c in self.columns]
        return rays
