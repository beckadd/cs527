import numpy as np
from rubik import draw_cube
from camera import new_camera, print_camera
from camera import show_image_points
from matplotlib import pyplot as plt
from longuet_higgins import reconstruct
from geometry import convert_rotation, transform
from geometry import radians_to_degrees, degrees_to_radians
from labels import labels as lab


# The standard camera fixates a sphere with the given radius and centered
# at the origin of the world from a point of view with world coordinates in t.
# All coordinates are in millimeters.
def standard_camera(t, object_radius=60.):
    sensor_pixels, sensor_pixel_microns = np.array((1920, 1080)), 5.
    return new_camera(object_radius, t, sensor_pixels, sensor_pixel_microns)


def print_features(fs):
    print('{} visible features found:'.format(len(fs)))
    with np.printoptions(precision=3, suppress=True):
        for fid, point in fs.items():
            print('\t id ', fid, ': pixel: ', point['pixel'],
                  ', world: ', point['world'], sep='')


def camera_separation(vergence, height, radius):
    h2, r2 = height ** 2, radius ** 2
    d2 = h2 + r2
    inner = (d2 * np.cos(vergence) - h2) / r2
    return np.arccos(inner)


def camera_pair(vergence_degrees, height, circle_radius):
    vergence = degrees_to_radians(vergence_degrees)
    alpha = camera_separation(vergence, height, circle_radius)
    half = alpha / 2.
    theta_0 = np.pi / 4.
    ts = [np.array((circle_radius * np.cos(azimuth),
                    circle_radius * np.sin(azimuth),
                    height)) for azimuth in (theta_0 - half, theta_0 + half)]
    return [standard_camera(t) for t in ts]


def canonical_points(p, cam):
    return np.diag(1. / cam.s) @ (p - np.outer(cam.pi, np.ones(p.shape[1]))) / cam.f


def correspondences(f_dicts):
    common_ids = f_dicts[0].keys() & f_dicts[1].keys()
    corr = {'pixel': [[], []], 'world': [[], []]}
    for i in common_ids:
        for system in ('pixel', 'world'):
            for cam in range(2):
                corr[system][cam].append(list(f_dicts[cam][i][system]))
    for system in ('pixel', 'world'):
        for cam in range(2):
            corr[system][cam] = np.array(corr[system][cam]).T
    return corr


def rotation_error_degrees(r, r_hat):
    msg = 'inputs must be rotation matrices, not vectors'
    assert r.shape == (3, 3) and r_hat.shape == (3, 3), msg
    dr = r.T @ r_hat
    return radians_to_degrees(np.linalg.norm(convert_rotation(dr)))


def translation_error_degrees(t, t_hat):
    n1, n2 = np.linalg.norm(t), np.linalg.norm(t_hat)
    assert n1 > 0. and n2 > 0., 'translation cannot be zero'
    return radians_to_degrees(np.arccos(np.dot(t / n1, t_hat / n2)))


def center(pts):
    return pts - np.outer(np.mean(pts, axis=1), np.ones(pts.shape[1]))


def procrustes(p, q):
    msg = 'Columns of inputs must be 3D'
    assert p.shape[0] == 3 and q.shape[0] == 3, msg
    p_ctr, q_ctr = center(p), center(q)
    m = q_ctr @ p_ctr.T
    u, _, vt = np.linalg.svd(m)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    p_ctr = r @ p_ctr
    return p_ctr, q_ctr, r


def structure_error(p, p_hat):
    p_nearest, p_hat_ctr, _ = procrustes(p, p_hat)
    return np.linalg.norm(p_nearest - p_hat_ctr, 'fro') / np.sqrt(p.shape[1])


def side_views_3d(p, q):
    plt.figure(figsize=(16, 5))
    for side, direction in ((2, 'front'), (0, 'side'), (1, 'top')):
        p_view, q_view = np.delete(p, side, axis=0), np.delete(q, side, axis=0)
        plt.subplot(1, 3, side + 1)
        labels = [label for s, label in enumerate(('X', 'Y', 'Z')) if s != side]
        if side == 0:
            p_view, q_view = p_view[::-1, :], q_view[::-1, :]
            labels = [labels[1], labels[0]]
        plt.plot(p_view[0, :], p_view[1, :], '.', ms=6, label='true')
        plt.plot(q_view[0, :], q_view[1, :], '.', ms=6, label='reconstructed')
        if side != 1:
            y_lim = plt.ylim()
            plt.ylim((y_lim[1], y_lim[0]))
        plt.gca().set_aspect(1.)
        plt.legend()
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title('canonical {} view'.format(direction))
    plt.show()


def experiment(cube,
               height_mm=100., circle_radius_mm=150.,
               omega_degrees=15., sigma_pixels=0.,
               repetitions=30, display=False):
    cameras = camera_pair(omega_degrees, height_mm, circle_radius_mm)
    features = []
    for camera, title in zip(cameras, ('left', 'right')):
        features.append(draw_cube(cube, camera, display=display, title=title))
        if display:
            print_camera(camera)
            print()
    matches = correspondences(features)
    pixel, world = matches['pixel'], matches['world']
    t = cameras[0].R @ (cameras[1].t - cameras[0].t)
    R = cameras[1].R @ cameras[0].R.T
    P = world[0]
    P_camera_0 = transform(P, cameras[0].R, cameras[0].t) / np.linalg.norm(t)
    errors = {lab.translation: [], lab.rotation: [], lab.structure: []}
    rng = np.random.default_rng()
    for rep in range(repetitions):
        canonical = []
        for view, title in enumerate(('left', 'right')):
            noise = sigma_pixels * rng.normal(0., sigma_pixels, pixel[view].shape)
            points = pixel[view] + noise
            if display:
                show_image_points(points, cameras[view], title=title)
            canonical.append(canonical_points(points, cameras[view]))
        t_hat, R_hat, P_hat, _ = reconstruct(canonical[0], canonical[1])
        if display:
            side_views_3d(P_camera_0, P_hat)
        errors[lab.translation].append(translation_error_degrees(t, t_hat))
        errors[lab.rotation].append(rotation_error_degrees(R, R_hat))
        errors[lab.structure].append(structure_error(P_camera_0, P_hat))
    error_stats = {}
    for error_type in (lab.translation, lab.rotation, lab.structure):
        median = np.median(errors[error_type])
        deviation = np.median(np.abs(errors[error_type] - median))
        error_stats[error_type] = {lab.median: median, lab.deviation: deviation}
    distance_mm = np.sqrt(height_mm ** 2 + circle_radius_mm ** 2)
    results = {lab.distance: distance_mm,
               lab.vergence: omega_degrees,
               lab.noise: sigma_pixels,
               lab.errors: error_stats}
    return results


def print_stats(s):
    fmt = '{}: {} {:.3f}, {} {:.3f}'
    for key, value in s.items():
        if type(value) == dict:
            for sub_key, sub_value in value.items():
                print(fmt.format(sub_key, lab.median, sub_value[lab.median],
                                 lab.deviation, sub_value[lab.deviation]))
        else:
            print('{}: {:.3f}'.format(key, value))
