from typing import Union

import numpy as np

from ray import Rays, RaysPD
from ray_np import RaysNP


class Camera:
    def __init__(self, pose=None, f=100, width=200, height=200):
        if pose is None:
            pose = np.eye(4)
        self.pose = pose
        self.f = f
        self.width = width
        self.height = height

    def emit_rays(self, c=RaysPD) -> Rays:
        x = np.arange(-self.width/2, self.width/2)[None, :] * np.ones(self.height)[:, None]
        y = np.arange(-self.width/2, self.width/2)[:, None] * np.ones(self.width)[None, :]
        z = np.ones((self.height, self.width))*self.f
        distances = np.sqrt(x**2 + y**2 + z**2)
        x = x/distances
        y = y/distances
        z = z/distances

        ray_count = self.height * self.width
        rays = np.stack((x, y, z), axis=2)
        rays = rays.reshape((ray_count, 3))
        rays = np.dot(self.pose[:3, :3], rays.T).T
        color = np.array([1, 1, 1])
        # rays = np.hstack((self.pose[:3, 3] * np.ones((ray_count, 1)), rays, color[None, :] * np.ones((ray_count, 1))))
        return c(starts=self.pose[:3, 3] * np.ones((ray_count, 1)),
                 directions=rays,
                 colors=color[None, :] * np.ones((ray_count, 1)))

    def get_indices(self, rays: Rays) -> np.array:
        rel_directions = np.dot(np.linalg.inv(self.pose)[:3, :3], rays.directions.T).T
        xs = rel_directions[:, 0] / rel_directions[:, 2] * self.f + self.width/2
        ys = rel_directions[:, 1] / rel_directions[:, 2] * self.f + self.height/2
        x_indices = (xs+0.5).astype(int)
        y_indices = (ys+0.5).astype(int)
        return x_indices, y_indices
