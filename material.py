import numpy as np

from ray import Rays, RaysPD
from utils import random_unit_vector


class Material:
    def __init__(self, diffuse: np.array, specular: np.array, glossiness: float):
        self.diffuse = diffuse
        self.glossiness = glossiness
        self.specular = specular

    def scatters(self, rays: RaysPD, normals: np.array, hit_points: np.array) -> RaysPD:
        diffuse_directions = normals + random_unit_vector(rays.count)
        diffuse_directions = diffuse_directions/np.linalg.norm(diffuse_directions, axis=1)[:,None]
        # eventually sample tint color based on hit location / UV coordinates
        reflected_directions = rays.directions - 2 * normals * (normals * rays.directions).sum(axis=1)[:, None]
        scattered_directions = self.glossiness * reflected_directions + (1 - self.glossiness) * diffuse_directions
        tint = self.diffuse * (1 - self.glossiness) + self.specular * self.glossiness
        scattered_rays = rays.copy()
        scattered_rays.set_starts(hit_points)
        scattered_rays.set_directions(scattered_directions)
        scattered_rays.set_colors(scattered_rays.colors * tint)
        return scattered_rays
