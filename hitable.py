from typing import Optional

import numpy as np

from material import Material
from ray import Rays, RaysPD
from ray_np import RaysNP as RaysPD


class Hittable:
    def __init__(self, material: Material):
        self.material = material

    def hits(self, rays: RaysPD) -> np.array:
        ...

    def get_normals(self, hit_pos) -> np.array:
        ...

    def scatters(self, rays: RaysPD, normals: np.array, hit_points: np.array) -> RaysPD:
        return self.material.scatters(rays, normals, hit_points)


class HittableCollection:
    def hits(self, rays: Rays, calculate_colors=True) -> np.array:
        ...