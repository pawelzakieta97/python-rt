import numpy as np

from constants import MAX_DISTANCE
from material import Material
from ray import RaysPD


class Light:
    def phong(self, rays: RaysPD, material: Material, hit_points: np.array, normals: np.array):
        ...

    def get_start_point(self) -> np.array:
        ...


class PointLight(Light):
    def __init__(self, position: np.array, color: np.array):
        self.position = position
        self.color = color

    def phong(self, rays: RaysPD, material: Material, hit_points: np.array, normals: np.array):
        phong = np.zeros((rays.count, 3))
        l2p = hit_points - self.position[None, :]
        l2p_norms = np.linalg.norm(l2p, axis=1)[:, None]
        directions = l2p / l2p_norms
        colors = self.color[None, :] / l2p_norms / l2p_norms
        diffuse_cos_angles = (-directions * normals).sum(axis=1)
        valid_rays = diffuse_cos_angles > 0
        phong[valid_rays] = material.diffuse * colors[valid_rays, :] * diffuse_cos_angles[valid_rays][:,None]

        # mirrorReflection = ray.direction + 2 * dot(normal, ray.direction) * normal
        # specularCosAngle = dot(mirrorReflection, -lightRay.direction)
        return phong

    def get_start_point(self) -> np.array:
        return self.position

class SunLight(Light):
    def __init__(self, direction: np.array, color: np.array):
        self.direction = direction / np.linalg.norm(direction)
        self.color = color

    def phong(self, rays: RaysPD, material: Material, hit_points: np.array, normals: np.array):
        phong = np.zeros((rays.count, 3))
        directions = self.direction[None, :] * np.ones((rays.count, 1))
        colors = self.color[None, :] * np.ones((rays.count, 1))
        diffuse_cos_angles = (-directions * normals).sum(axis=1)
        valid_rays = diffuse_cos_angles > 0
        phong[valid_rays] = material.diffuse * colors[valid_rays, :] * diffuse_cos_angles[valid_rays][:,None]

        # mirrorReflection = ray.direction + 2 * dot(normal, ray.direction) * normal
        # specularCosAngle = dot(mirrorReflection, -lightRay.direction)
        return phong

    def get_start_point(self) -> np.array:
        return -self.direction * MAX_DISTANCE * 0.9
