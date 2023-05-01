import datetime

import numpy as np
import pandas as pd

from camera import Camera
from constants import MIN_DISTANCE
from environment import Environment
from hitable import Hittable
from light import Light
from ray_np import RaysNP
from ray import RaysPD, Rays
import logging

logger = logging.getLogger(__name__)


def render(camera: Camera, objects: list,
           lights: list[Light], samples=1, bounces=2, rays_class=RaysPD):
    img = np.zeros((camera.height, camera.width, 3))
    for sample in range(samples):
        logger.info(f'processing sample {sample+1}')
        rays = camera.emit_rays(c=rays_class)
        idx_x, idx_y = camera.get_indices(rays)
        rays.add_data(np.hstack([idx_x[:, None], idx_y[:, None]]), columns=['idx_x', 'idx_y'])
        for bounce in range(bounces):
            start = datetime.datetime.now()
            logger.info(f'processing bounce {bounce+1} with {rays.count} rays')
            # finding the closest instersecting object
            distances = np.ones((rays.count, len(objects)))
            for i, o in enumerate(objects):
                distances[:, i] = o.hits(rays)
            closest_objects = np.argmin(distances, axis=1)

            # aggregating rays into groups that hit the same objects
            hit_points = rays.starts + rays.directions * distances[np.arange(rays.count), closest_objects][:, None]
            obj_hits = {}
            for i, o in enumerate(objects):
                # rays that hit this object
                obj_rays_indices = closest_objects == i
                # obj_rays = RaysPD(rays.data.iloc[obj_rays_indices])
                obj_rays = rays[obj_rays_indices]
                obj_hits[o] = (obj_rays, hit_points[obj_rays_indices])

            # Updating image pixels' colors and generating a new set of rays
            rays = rays_class()

            light_rays = rays_class()
            for obj, (obj_rays, obj_hit_points) in obj_hits.items():
                indices = obj_rays.get_data(['idx_x', 'idx_y']).astype(int)
                if isinstance(obj, Environment):
                    img[indices[:, 1], indices[:, 0]] += obj.get_colors(obj_rays) * obj_rays.colors
                elif isinstance(obj, Hittable):
                    # update pixels with colors
                    normals = obj.get_normals(obj_hit_points)
                    for light in lights:
                        pass
                        obj_light_rays = obj_rays.copy()
                        # obj_light_rays.data = obj_light_rays.data.reset_index(drop=True)
                        l2hp = obj_hit_points - light.get_start_point(obj_rays.count)
                        obj_light_rays.set_starts(light.get_start_point(obj_rays.count))
                        obj_light_rays.set_directions(l2hp/np.linalg.norm(l2hp, axis=1)[:, None])
                        obj_light_rays.set_colors(light.phong(obj_rays, obj.material, obj_hit_points, normals) *
                                                  obj_rays.colors)
                        obj_light_rays.add_data(obj_hit_points, columns=['hit_point_x', 'hit_point_y', 'hit_point_z'])
                        light_rays.add_rays(obj_light_rays)

                    rays.add_rays(obj.scatters(obj_rays, normals, obj_hit_points))

            los = is_in_line_of_sight_rays(light_rays, light_rays.get_data(['hit_point_x', 'hit_point_y', 'hit_point_z']), objects)
            indices = light_rays.get_data(['idx_x', 'idx_y']).astype(int)
            indices_in_sight = indices[los, :]
            img[indices_in_sight[:, 1], indices_in_sight[:, 0]] += light_rays[los].colors
            logger.info(f'Bounce calculation time: {datetime.datetime.now() - start}')
        pass
    return img/samples


def is_in_line_of_sight_rays(rays: Rays, end_points, objects: list[Hittable]) -> np.array:
    # return np.ones(rays.count).astype(bool)
    diffs = end_points - rays.starts
    distances = np.linalg.norm(diffs, axis=1)
    # ray = Ray(p1, dir, Vec3(0,0,0))
    rays.add_data(np.arange(rays.count)[:, None], columns=['index'])
    lines_of_sights = np.ones(rays.count).astype(bool)
    for obj in objects:
        ds = obj.hits(rays)
        in_between = ds < distances - MIN_DISTANCE
        lines_of_sights[rays.get_data(['index']).astype(int)[in_between]] = False

        rays = rays[np.logical_not(in_between)]
        distances = distances[np.logical_not(in_between)]
    return lines_of_sights # == lines_of_sights

