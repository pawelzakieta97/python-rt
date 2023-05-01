from __future__ import annotations

from abc import ABC
from typing import Union

import numpy as np
import pandas as pd


class Rays(ABC):
    @property
    def starts(self) -> np.array:
        raise NotImplemented

    @property
    def directions(self) -> np.array:
        raise NotImplemented

    @property
    def colors(self) -> np.array:
        raise NotImplemented

    @property
    def count(self) -> int:
        raise NotImplemented


    def set_starts(self, starts: np.array):
        raise NotImplemented

    def set_directions(self, starts: np.array):
        raise NotImplemented

    def set_colors(self, colors: np.array):
        raise NotImplemented

    def add_rays(self, rays: Rays):
        raise NotImplemented

    def add_data(self, additional_data: np.array, columns: list[str]):
        raise NotImplemented

    def get_additional_data(self, data_name) -> np.array:
        raise NotImplemented


class RaysPD(Rays):
    def __init__(self, data: Union[np.array, pd.DataFrame] = None,
                 starts=None, directions=None, colors=None):
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

    def add_data(self, additional_data, columns):
        self.data = self.data.reset_index(drop=True)
        self.data = pd.concat((self.data, pd.DataFrame(additional_data, columns=columns)), axis=1)

    def get_data(self, columns: list[str]):
        return self.data[columns].to_numpy()

    def copy(self) -> RaysPD:
        return RaysPD(self.data.copy())

    def __getitem__(self, item):
        return RaysPD(self.data.iloc[item])
