import numpy as np
import cv2

from camera import Camera
from environment import Environment
from light import PointLight, SunLight
from material import Material
from plane import Plane
from populator import generate_uniformly_distributed_points
from renderer import render
from sphere import Sphere
from utils import get_translation_matrix, get_rotation_matrix_x
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)-8s')


def generate_random_spheres(density=0.1, z_range=(3, 5), x_range=(-200, 200), y_range=(0, 300)):
    spheres = []

    def viewing_cone(xs, ys):
        return (np.abs(xs) < ys) & (ys > 0)

    xs, ys = generate_uniformly_distributed_points(viewing_cone, density=density, x_range=x_range, y_range=y_range)
    print(f'{len(xs)} spheres')
    for x, y in zip(xs, ys):
        z = np.random.random() * (z_range[1] - z_range[0]) + z_range[0]
        color = np.random.random(3)
        metalicness = np.random.random()
        glossiness = np.random.random()
        reflectiveness = np.random.random()
        material = Material(diffuse=color,
                            specular=(color * metalicness + np.ones(3) - metalicness) * reflectiveness,
                            glossiness=glossiness)
        position = np.array([x, y, z])
        sphere = Sphere(position, z, material)
        spheres.append(sphere)
    return spheres


if __name__ == '__main__':
    np.random.seed(1)
    render_multiplier = 0.5
    f = int(1000 * render_multiplier)
    w = int(2000 * render_multiplier)
    h = int(2000 * render_multiplier)
    c = Camera(f=f, width=w, height=h)
    c.pose = np.dot(get_rotation_matrix_x(-np.pi * 0.6), c.pose)
    # c.pose = np.dot(get_translation_matrix([-15, 35, 12]), c.pose)
    c.pose = np.dot(get_translation_matrix([0, 7, 15]), c.pose)
    env = Environment()  # zenith_color=np.array([1,0,0]), horizon_color=np.array([1,0,0]))
    m = Material(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]), 0)
    red_diff = Material(np.array([0.0, 0.0, 1]), np.array([0.0, 0.0, 0.0]), 0)
    mirror = Material(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), 1)
    red_sphere = Sphere(np.array([8, 0, 2]), 2, red_diff)
    # red_sphere = Sphere(np.array([-6, 0, 5]), 5, Material(diffuse=np.array([0.35726976, 0.90853515, 0.62336012]), specular=np.array([0.98983121, 0.99855291, 0.99404109]), glossiness=0.5))
    mirror_sphere = Sphere(np.array([-8, 0, 5]), 5, mirror)
    # mirror_sphere = Sphere(np.array([6, 0, 5]), 5, Material(diffuse=np.array([0.6852195, 0.20445225, 0.87811744]), specular=np.array([0.99137892, 0.97821186, 0.99666193]), glossiness=0.5))
    # mirror_sphere2 = Sphere(np.array([-0, -3, 2]), 2, mirror)
    p = Plane(np.array([0, 0, 1.0]), 0, m)
    sl = SunLight(np.array([1, 0, -1]), np.array([0.7, 0.8, 0.9]))
    # sl = SunLight(np.array([1, 0, -1]), np.array([1, 1, 1]))
    img = render(camera=c,
                 objects=[env, p] + generate_random_spheres(density=0.06),
                 # objects=[env, p, red_sphere],
                 # objects=[env, p] + generate_random_spheres(count=20),
                 lights=[sl],
                 bounces=4,
                 samples=10)
    cv2.imshow('render', img)
    cv2.waitKey(0)
    pass
