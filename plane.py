import numpy as np

from constants import MAX_DISTANCE, MIN_DISTANCE
from hitable import Hittable
from material import Material
from ray import RaysPD


class Plane(Hittable):
    def __init__(self, normal: np.array, offset: np.array, material: Material):
        super().__init__(material)
        self.normal = normal
        self.offset = offset

    def hits(self, rays: RaysPD) -> np.array:
        ds = (self.normal[None, :] * rays.starts).sum(axis=1)
        dirs = (self.normal[None, :] * rays.directions).sum(axis=1)
        distances = (self.offset - ds) / dirs
        distances[distances<MIN_DISTANCE] = MAX_DISTANCE
        return distances

    def get_normals(self, hit_pos) -> np.array:
        return self.normal[None, :] * np.ones((hit_pos.shape[0], 1))