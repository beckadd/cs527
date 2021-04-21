import numpy as np


def cross(t):
    return np.array(((0, -t[2], t[1]),
                     (t[2], 0, -t[0]),
                     (-t[1], t[0], 0)))


def triangulate(p, q, t, R):
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns'
    P = np.zeros((3, n))

    i, j, k = R[0], R[1], R[2]
    kt = np.dot(k, t)
    proj = np.vstack((i, j))
    proj_t = np.dot(proj, t)

    C = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0))).astype(float)
    c = np.zeros(4)

    for m in range(n):
        C[:2, 2] = -p[:2, m]
        C[2:, :] = np.outer(q[:2, m], k) - proj
        c[2:] = kt * q[:2, m] - proj_t

        x = np.linalg.lstsq(C, c, rcond=None)
        P[:, m] = x[0]

    Q = np.dot(R, P - np.outer(t, np.ones((1, n))))
    return P, Q


# Input features are arrays of canonical_points image coordinates.
# Both arrays have shape (2, n).
def reconstruct(p, q):
    # Number of point pairs
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns'

    # Transform images from 2D to 3D in the standard reference frame
    o = np.ones((1, n)).astype(float)
    p, q = np.concatenate((p, o)), np.concatenate((q, o))

    # Set up matrix A such that A*E.flatten() = 0, where E is the essential
    # matrix.
    # This system encodes the epipolar constraint q' * E * p = 0 for each of
    # the points p and q
    A = np.zeros((n, 9))
    for k in range(n):
        A[k, :] = np.outer(q[:, k], p[:, k]).flatten()
    assert np.linalg.matrix_rank(A) >= 8, 'Insufficient rank for A'

    # The singular vector corresponding to the smallest singular value of A
    # is the arg min_{norm(e) = 1} A * e, and is the LSE estimate of E.flatten()
    _, _, VT = np.linalg.svd(A)
    E = np.reshape(VT[-1, :], (3, 3))

    # The two possible translation vectors are t and -t, where t is a unit
    # vector in the null space of E. The vector t (or -t) is also the
    # second epipole of the camera pair
    _, _, VET = np.linalg.svd(E)
    t = VET[2, :]

    # The cross-product matrix for vector t
    tx = cross(t)

    # Two rotation matrix choices are found by solving the Procrustes problem
    # for the rows of E and tx, and allowing for the ambiguity resulting
    # from the sign of the null-space vectors (both E and tx are rank 2).
    # These two choices are independent of the sign of t, because both E
    # and -E are essential matrices
    UF, _, VFT = np.linalg.svd(np.dot(E, tx))
    R1 = np.dot(UF, VFT)
    R1 *= np.linalg.det(R1)
    UF[:, 2] = -UF[:, 2]
    R2 = np.dot(UF, VFT)
    R2 *= np.linalg.det(R2)

    # Combine the two sign options for t with the two choices for R
    tList = [t, t, -t, -t]
    RList = [R1, R2, R1, R2]

    # Pick the combination of t and R that yields the greatest number of
    # positive depth (Z) values in the structure results for the frames of
    # reference of both cameras. Ideally, all depth values should be positive
    P, Q, R, npdMax = [], [], [], -1
    for k in range(4):
        tt, RR = tList[k], RList[k]
        PP, QQ = triangulate(p, q, tt, RR)
        npd = np.sum(np.logical_and(PP[2, :] > 0, QQ[2, :] > 0))
        if npd > npdMax:
            t, R, P, Q, npdMax = tt, RR, PP, QQ, npd

    return t, R, P, Q
