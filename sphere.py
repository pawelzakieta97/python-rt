import numpy as np

from constants import MAX_DISTANCE, MIN_DISTANCE
from hitable import HittableCollection, Hittable
from material import Material
from ray import Rays, RaysPD


class Sphere(Hittable):
    def __init__(self, pos: np.array, radius: float, material: Material):
        super().__init__(material)
        self.pos = pos
        self.radius = radius

    def hits(self, rays: RaysPD):
        # v = ray.start - sphere.position
        vs = rays.starts - self.pos
        # b = 2 * dot(ray.direction, v)
        bs = 2 * (rays.directions * vs).sum(axis=1)
        # delta = b * b - 4 * (dot(v, v) - sphere.radius * sphere.radius)
        deltas = bs ** 2 - 4 * ((vs * vs).sum(axis=1) - self.radius * self.radius)
        hit_indices = np.where(deltas > 0)[0]
        distances = np.ones(rays.count) * MAX_DISTANCE
        deltas = deltas[hit_indices]
        sqrt_deltas = np.sqrt(deltas)
        bs = bs[hit_indices]
        distances1 = (-bs - sqrt_deltas) / 2
        distances2 = (-bs + sqrt_deltas) / 2
        valid_hits = distances2 > MIN_DISTANCE
        distances[hit_indices[valid_hits]] = distances1[valid_hits]
        return distances

    def get_normals(self, hit_pos):
        diff = hit_pos - self.pos
        return diff/self.radius



class Spheres(HittableCollection):
    def __init__(self, spheres_array: list[Sphere], spheres_matrix: np.array = None):
        self._spheres_array = spheres_array
        self._spheres_matrix = spheres_matrix

    @property
    def spheres_array(self):
        if self._spheres_array is not None:
            return self._spheres_array
        spheres = []
        for s in self._spheres_matrix:
            spheres.append(Sphere(s[:3], s[3], Material(s[4:7], s[7:10], s[10])))
        return spheres

    @property
    def spheres_matrix(self):
        if self._spheres_matrix is not None:
            return self._spheres_matrix
        spheres = np.zeros((len(self._spheres_array)))
        for i, s in enumerate(self._spheres_array):
            spheres[:3] = self._spheres_array[i].pos
            spheres[3] = self._spheres_array[i].radius
            spheres[4:7] = self._spheres_array[i].material.diffuse
            spheres[7:10] = self._spheres_array[i].material.specular
            spheres[10] = self._spheres_array[i].material.roughness
        return spheres



Sphere.collection_class = Spheres