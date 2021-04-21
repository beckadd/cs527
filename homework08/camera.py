from types import SimpleNamespace
import numpy as np
from geometry import transform
from matplotlib import pyplot as plt


def print_array(a, name=None, unit=None, indent=0):
    dims = a.ndim
    assert dims == 1 or dims == 2, 'Can only print vectors or matrices'
    tabs = '\t' * indent
    prefix = tabs if name is None else '{}{}: '.format(tabs, name)
    suffix = '' if unit is None else ' {}'.format(unit)
    with np.printoptions(precision=3, suppress=True):
        if dims == 1:
            print('{}{}{}'.format(prefix, a, suffix))
        else:
            if len(suffix):
                suffix = ' ({})'.format(suffix)
            print('{}'.format(prefix, suffix))
            for row in a:
                print(tabs, '\t', row, sep='', end='\n')


def fov(r, tau):
    return 2. * np.arcsin(r / tau)


def focal_distance(h, phi):
    return h / 2. / np.tan(phi / 2.)


def new_camera(radius, t, pixels, pixel_microns):
    tau = np.linalg.norm(t)
    sensor_mm = pixels * pixel_microns / 1000.
    f_mm = focal_distance(sensor_mm[1], fov(radius, tau))
    k = - t / tau
    up = np.array((0., 0., 1.))
    i = np.cross(k, up)
    i /= np.linalg.norm(i)
    j = np.cross(k, i)
    R = np.array((i, j, k))
    pixels_per_mm = np.full(2, 1000.) / pixel_microns
    principal_point = pixels / 2.
    model = SimpleNamespace(R=R, t=t, pixels=pixels, pi=principal_point,
                            f=f_mm, s=pixels_per_mm)
    return model


def print_camera(model):
    with np.printoptions(precision=3, suppress=True):
        print('Extrinsic parameters:')
        print_array(model.t, 'Origin (t)', 'mm', indent=1)
        print_array(model.R, 'Rotation matrix (R)', indent=1)

        print('\nIntrinsic parameters:')
        print('\tFocal distance (f): {:.3f} mm'.format(model.f))
        print_array(model.s, 'Scaling (s)', 'pixels per mm', indent=1)
        print_array(model.pi, 'Principal point (pi)', 'pixels', indent=1)


def project(points, camera):
    canonical = transform(points, camera.R, camera.t)
    assert np.all(canonical[2, :] > 0.), \
        'some points are at or behind the camera'
    z = np.outer(np.ones(2), canonical[2, :])
    image_points = canonical[:2, :] / z
    image_points *= camera.f
    scaled = np.diag(camera.s) @ image_points
    image_points = transform(scaled, t=-camera.pi)
    return image_points


def setup_image_figure(camera):
    figure = plt.figure(figsize=camera.pixels / 100)
    axes = figure.add_subplot(1, 1, 1)
    axes.set_position((0, 0, 1, 1))
    return axes


def finalize_image_figure(ax, camera, title=None, font_size=50, offset=(50, 100)):
    ax.autoscale_view()
    ax.set_xlim(0, camera.pixels[0])
    ax.set_ylim(camera.pixels[1], 0)
    ax.set_aspect(1.)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        plt.text(offset[0], offset[1], title, {'fontsize': font_size})
    plt.show()


def show_image_points(p, cam, title=None):
    ax = setup_image_figure(cam)
    plt.plot(p[0, :], p[1, :], '.', ms=12)
    finalize_image_figure(ax, cam, title)
