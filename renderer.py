import numpy as np
import pandas as pd

from camera import Camera
from constants import MIN_DISTANCE
from environment import Environment
from hitable import Hittable
from light import Light
from ray import Rays, RaysPD


def render(camera: Camera, objects: list, lights: list[Light], samples=1, bounces=2):
    img = np.zeros((camera.height, camera.width, 3))
    for i in range(samples):
        rays = camera.emit_rays()
        rays = RaysPD(rays.rays_matrix)
        idx_x, idx_y = camera.get_indices(rays)
        rays.add_data(pd.DataFrame({'idx_x': idx_x, 'idx_y': idx_y}))
        for bounce in range(bounces):
            # finding the closest instersecting object
            distances = np.ones((rays.count, len(objects)))
            for i, o in enumerate(objects):
                distances[:, i] = o.hits(rays)
            closest_objects = np.argmin(distances, axis=1)

            # aggregating rays into groups that hit the same objects
            hit_points = rays.starts + rays.directions * distances[np.arange(rays.count), closest_objects][:,None]
            obj_hits = {}
            for i, o in enumerate(objects):
                # rays that hit this object
                obj_rays_indices = closest_objects == i
                # obj_rays = RaysPD(rays.data.iloc[obj_rays_indices])
                obj_rays = rays[obj_rays_indices]
                obj_hits[o] = (obj_rays, hit_points[obj_rays_indices])

            # Updating image pixels' colors and generating a new set of rays
            rays = RaysPD()
            for obj, (obj_rays, obj_hit_points) in obj_hits.items():
                indices = obj_rays.get_data(['idx_y', 'idx_x'])
                if isinstance(obj, Environment):
                    img[indices[:, 0], indices[:, 1]] += obj.get_colors(obj_rays) * obj_rays.colors
                elif isinstance(obj, Hittable):
                    # update pixels with colors
                    normals = obj.get_normals(obj_hit_points)
                    for light in lights:
                        los = is_in_line_of_sight(light.get_start_point()[None, :] * np.ones((obj_rays.count, 1)),
                                                  obj_hit_points, objects)
                        indices_in_sight = indices[los, :]
                        img[indices_in_sight[:, 0], indices_in_sight[:, 1]] += \
                            light.phong(obj_rays[los], obj.material, obj_hit_points[los, :], normals[los, :])
                    rays.add_rays(obj.scatters(obj_rays, normals, obj_hit_points))

        pass
    return img/samples


def is_in_line_of_sight(p1: np.array, p2: np.array, objects: list[Hittable]) -> np.array:
    diffs = p2 - p1
    distances = np.linalg.norm(diffs, axis=1)
    dirs = diffs/distances[:, None]
    # ray = Ray(p1, dir, Vec3(0,0,0))
    rays = RaysPD(starts=p1, directions=dirs, colors=np.zeros(p1.shape))
    rays.add_data(pd.DataFrame({'index': np.arange(rays.count)}))
    lines_of_sights = np.ones(p1.shape[0]).astype(bool)
    for obj in objects:
        ds = obj.hits(rays)
        in_between = ds < distances - MIN_DISTANCE
        lines_of_sights[rays.get_data(['index'])[in_between]] = False

        rays = rays[np.logical_not(in_between)]
        distances = distances[np.logical_not(in_between)]
    return lines_of_sights

