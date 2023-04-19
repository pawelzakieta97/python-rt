import numpy as np

from constants import MAX_DISTANCE
from hitable import Hittable
from ray import Rays, RaysPD


class Environment():
    def __init__(self, horizon_color=None, zenith_color=None):
        if horizon_color is None:
            horizon_color = np.array([1, 1, 1])
        if zenith_color is None:
            zenith_color = np.array([1, 0.5, 0])
        self.horizon_color = horizon_color
        self.zenith_color = zenith_color

    def hits(self, rays: RaysPD, calculate_colors=True) -> np.array:
        distances = np.ones(rays.count) * (MAX_DISTANCE - 1)
        if not calculate_colors:
            return distances
        zs = rays.directions[:, 2]
        colors = zs[:, None] * self.zenith_color[None, :] + (1-zs)[:, None] * self.horizon_color[None, :]
        colors[zs < 0, :] = 0
        return distances

    def get_colors(self, rays: RaysPD):
        zs = rays.directions[:, 2]
        colors = zs[:, None] * self.zenith_color[None, :] + (1-zs)[:, None] * self.horizon_color[None, :]
        colors[zs < 0, :] = 0
        return colors