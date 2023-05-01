import numpy as np

from constants import MAX_DISTANCE, MIN_DISTANCE
from hitable import HittableCollection, Hittable
from material import Material
from ray import Rays, RaysPD
from ray_np import RaysNP as RaysPD


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


if __name__ == '__main__':
    count = 10_000_000
    starts = np.zeros((count, 3))
    directions = np.zeros((count, 3))
    directions[:,0] = 1
    colors = np.zeros((count, 3))
    rays = RaysPD(starts=starts, directions=directions, colors=colors)
    s = Sphere(np.array([10,0,0]), 2, None)
    import datetime
    start = datetime.datetime.now()
    res = s.hits(rays)
    print(datetime.datetime.now() - start)