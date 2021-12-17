import numpy as np


def interpolate(x, y, z):
    shape = y.shape
    assert np.all(x.shape == shape), 'r and c must have equal shape'
    x, y = np.ravel(x), np.ravel(y)
    rows, cols = z.shape[:2]
    inside = (x >= 0) & (x < cols - 1) & (y >= 0) & (y < rows - 1)
    x, y = x[inside], y[inside]
    grid_x, grid_y = np.arange(cols), np.arange(rows)
    right = np.searchsorted(grid_x, x, side='right')
    left = right - 1
    down = np.searchsorted(grid_y, y, side='right')
    up = down - 1
    x_left, x_right = grid_x[left], grid_x[right]
    y_up, y_down = grid_y[up], grid_y[down]
    z_up_left = z[y_up, x_left]
    z_down_left = z[y_down, x_left]
    z_up_right = z[y_up, x_right]
    z_down_right = z[y_down, x_right]
    alpha = x - x_left
    alpha_comp = 1 - alpha
    beta = y - y_up
    beta_comp = 1 - beta
    zi_inside = z_up_left * alpha_comp * beta_comp + \
        z_down_left * alpha_comp * beta + \
        z_up_right * alpha * beta_comp + \
        z_down_right * alpha * beta
    zi = np.zeros(np.prod(shape))
    zi[inside] = zi_inside
    zi = np.reshape(zi, shape)
    return np.clip(np.round(zi), 0., 255.).astype(np.uint8)
