import numpy as np


def radians_to_degrees(x):
    return x * 180. / np.pi


def degrees_to_radians(x):
    return x * np.pi / 180.


def transform(p, r=None, t=None):
    if r is None:
        r = np.eye(p.shape[0])
    if t is None:
        t = np.zeros(p.shape[0])
    shifts = np.outer(t, np.ones(p.shape[1]))
    return r @ (p - shifts)


def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    plane_rotation = np.array([[c, -s], [s, c]])
    rx = np.eye(3)
    rx[np.ix_((1, 2), (1, 2))] = plane_rotation
    return rx


def rotation_basis(r):
    assert r.shape == (3,), 'a must be a 3-vector'
    u, _, vt = np.linalg.svd(np.expand_dims(r, axis=1))
    if vt[0, 0] < 0:  # make sure that a is not flipped
        u = -u
    if np.linalg.det(u) < 0:  # make sure that u is right-handed
        u[:, (1, 2)] = u[:, (2, 1)]
    return u


def convert_rotation(r):
    msg = 'input must be a 3-vector or a 3x3 matrix'
    assert r.shape == (3,) or r.shape == (3, 3), msg
    if r.shape == (3,):
        theta = np.linalg.norm(r)
        if theta == 0.:
            return np.eye(3)
        xform = rotation_basis(r)
        rx = rotation_x(theta)
        return xform @ rx @ xform.T
    else:
        e_values, e_vectors = np.linalg.eig(r)
        j = np.argmin(np.abs(e_values - 1))
        a = np.real(e_vectors[:, j])
        u = rotation_basis(a)
        b = u[:, 1]
        c = u[:, 2]
        s = r @ b
        theta = np.arctan2(np.dot(c, s), np.dot(b, s))
        return theta * a
