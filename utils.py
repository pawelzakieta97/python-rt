import numpy as np


def get_rotation_matrix_x(angle):
    return np.array([[1, 0,             0,              0],
                     [0, np.cos(angle), -np.sin(angle), 0],
                     [0, np.sin(angle), np.cos(angle),  0],
                     [0, 0,             0,              1]])


def get_rotation_matrix_y(angle):
    return np.array([[np.cos(angle),    0, np.sin(angle),   0],
                     [0,                1, 0,               0],
                     [-np.sin(angle),   0, np.cos(angle),   0],
                     [0,                0, 0,               1]])


def get_rotation_matrix_z(angle):
    return np.array([[np.cos(angle),    -np.sin(angle), 0, 0],
                     [np.sin(angle),    np.cos(angle),  0, 0],
                     [0,                0,              1, 0],
                     [0,                0,              0, 1]])


def get_translation_matrix(translation):
    return np.array([[1, 0, 0, translation[0]],
                     [0, 1, 0, translation[1]],
                     [0, 0, 1, translation[2]],
                     [0, 0, 0, 1]])


def random_unit_vector(count: int):
    z = np.random.random(count) * 2 - 1
    r = np.sqrt(1-z*z)
    bearing = np.random.random(count) * 2 * np.pi
    return np.hstack([(r*np.cos(bearing))[:, None], (r*np.sin(bearing))[:, None], z[:, None]])


if __name__ == '__main__':
    r = random_unit_vector(10)
    print(r)