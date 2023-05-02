from __future__ import annotations
import numpy as np

from ray import Rays


class RaysNP(Rays):
    def __init__(self, data: np.array = None, starts=None, directions=None, colors=None):
        self.columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'r', 'g', 'b']
        self.data = None
        if data is not None:
            self.data = data
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


class RaysNP2(Rays):
    def __init__(self, data: np.array = None, starts=None, directions=None, colors=None):
        self.data = None
        if data is not None:
            self.data = data
        elif starts is not None and directions is not None and colors is not None:
            self.data = {
                'starts': np.asfortranarray(starts),
                'directions': np.asfortranarray(directions),
                'colors': np.asfortranarray(colors)
            }

    @property
    def starts(self):
        return self.data['starts']

    @property
    def count(self):
        return list(self.data.values())[0].shape[0]

    @property
    def directions(self):
        return self.data['directions']

    @property
    def colors(self):
        return self.data['colors']

    def set_starts(self, starts: np.array):
        self.data['starts'] = np.asfortranarray(starts)

    def set_directions(self, starts: np.array):
        self.data['directions'] = np.asfortranarray(starts)

    def set_colors(self, colors: np.array):
        self.data['colors'] = np.asfortranarray(colors)

    def add_rays(self, rays: RaysNP2):
        if self.data is None:
            self.data = {k: v.copy() for k, v in rays.data.items()}
            return
        for k, v in rays.data.items():
            self.data[k] = np.vstack((self.data[k], v))

    def add_data(self, additional_data: np.array, columns: list[str]):
        self.data[tuple(columns)] = additional_data

    def get_data(self, columns: list[str]):
        return self.data[tuple(columns)]

    def copy(self) -> RaysNP2:
        rays = RaysNP2({k: v.copy() for k, v in self.data.items()})
        return rays

    def __getitem__(self, item):
        d = {}
        for k, v in self.data.items():
            d[k] = v[item, :]
        # rays = RaysNP2({k: v[item, :] for k, v in self.data.items()})
        return RaysNP2(d)

class RaysNP3(Rays):
    """
    implements padding to avoid concats
    """
    def __init__(self, data: np.array = None, starts=None, directions=None, colors=None):
        self.columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'r', 'g', 'b']
        self.data = None
        if data is not None:
            self.data = data
            return
        if starts is not None and directions is not None and colors is not None:
            padding = np.zeros((starts.shape[0], 3))
            self.data = np.hstack((starts, directions, colors, padding))
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

        # self.data = np.hstack((self.data, additional_data))
        self.data[:, len(self.columns): len(self.columns)+len(columns)] = additional_data
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
