import numpy as np
import cv2

from camera import Camera
from environment import Environment
from hitable import Hittable
from light import PointLight, SunLight
from material import Material
from plane import Plane
from renderer import render
from sphere import Spheres, Sphere
from utils import get_translation_matrix, get_rotation_matrix_x


if __name__ == '__main__':
    render_multiplier = 0.5
    f = int(1000 * render_multiplier)
    w = int(2000 * render_multiplier)
    h = int(2000 * render_multiplier)
    c = Camera(f=f, width=w, height=h)
    c.pose = np.dot(get_rotation_matrix_x(-np.pi/2), c.pose)
    c.pose = np.dot(get_translation_matrix([0, -15, 5]), c.pose)
    env = Environment()
    m = Material(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]), 0)
    s = Sphere(np.array([0, 0, 5]), 5, m)
    p = Plane(np.array([0, 0, 1.0]), 0, m)
    sl = SunLight(np.array([1, 0, -1]), np.array([0.7, 0.8, 0.9]))
    pl = PointLight(np.array([-10,0, 10]), np.array([0.0100, 0.0100, 0.0100]))
    img = render(camera=c, objects=[env, s, p], lights=[sl], bounces=4, samples=10)
    cv2.imshow('render', img)
    cv2.waitKey(0)
    pass
