import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_uniformly_distributed_points(mask, density=0.3, x_range=(-100, 100), y_range=(-100,100)):
    x_size = int((x_range[1] - x_range[0]) * density)
    y_size = int((y_range[1] - y_range[0]) * density)
    count = x_size*y_size
    xs = np.random.random(count) * (x_range[1] - x_range[0]) + x_range[0]
    ys = np.random.random(count) * (y_range[1] - y_range[0]) + y_range[0]
    valid = mask(xs, ys)
    x = xs[valid]
    y = ys[valid]
    pd.DataFrame({'x': x, 'y': y}).plot.scatter(x='x', y='y')
    for i in range(20):
        x, y = relax(x, y, lr=1/density)
    pd.DataFrame({'x': x, 'y': y}).plot.scatter(x='x', y='y')
    plt.show()
    return x, y


def get_stats(xs , ys):
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    distances = dx ** 2 + dy ** 2
    distances[np.arange(distances.shape[0]), np.arange(distances.shape[0])] = 999999999
    distances = np.sqrt(distances)
    distances = distances.min(axis=1)
    return distances


def relax(xs, ys, lr=0.01):
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    distances = dx ** 2 + dy ** 2
    distances[np.arange(distances.shape[0]), np.arange(distances.shape[0])] = 999999999
    distances = np.sqrt(distances)
    fx = dx/(distances ** 3)
    fx[fx>5] = 5
    fx[fx<-5] = -5
    fy = dy/(distances ** 3)
    fy[fy>5] = 5
    fy[fy<-5] = -5
    return xs+fx.sum(axis=1)*lr, ys+fy.sum(axis=1)*lr


if __name__ == '__main__':
    def viewing_cone(xs, ys):
        return (np.abs(xs) < ys) & (ys > 0)
    x, y = generate_uniformly_distributed_points(viewing_cone, density=0.1)
    res = get_stats(x, y)
    plt.show()