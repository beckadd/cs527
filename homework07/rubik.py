from types import SimpleNamespace
import numpy as np
from transform import transform
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def new_color(name, hex_code):
    return SimpleNamespace(name=name, hex=hex_code)


def signature(center):
    center = np.array(center)
    largest = np.argmax(abs(center)).item()
    s = [0, 0, 0]
    s[largest] = 1 if center[largest] > 0 else -1
    return tuple(s)


def new_face_specs(scale=1):
    specs = {
        'back': SimpleNamespace(center=(-3 * scale, 0, 0), initial_color=new_color('blue', '#0045ad')),
        'front': SimpleNamespace(center=(3 * scale, 0, 0), initial_color=new_color('green', '#009b48')),
        'left': SimpleNamespace(center=(0, -3 * scale, 0), initial_color=new_color('orange', '#ff5900')),
        'right': SimpleNamespace(center=(0, 3 * scale, 0), initial_color=new_color('red', '#b90000')),
        'down': SimpleNamespace(center=(0, 0, -3 * scale), initial_color=new_color('yellow', '#ffd500')),
        'up': SimpleNamespace(center=(0, 0, 3 * scale), initial_color=new_color('white', '#ffffff'))
    }
    return specs


face_names = {signature(spec.center): name for name, spec in new_face_specs().items()}


def snap(point):
    return np.round(point)


def zeros(p):
    return len(p) - np.count_nonzero(p)


def new_square(color, corners):
    return SimpleNamespace(color=color, corners=corners)


def polygon_centroid(corners):
    return np.mean(corners, axis=1)


def square_face(corners):
    square_center = snap(polygon_centroid(corners))
    return face_names[signature(square_center)]


def indices_other_than(d):
    indices = list(range(3))
    indices.remove(d)
    return indices


def sort_counter_clockwise(corners, indices):
    centroid = polygon_centroid(corners)
    centered = transform(corners, t=centroid)
    points = centered[indices, :]
    order = np.argsort(np.arctan2(points[1, :], points[0, :]))
    return corners[:, order]


def new_piece(center, face_specs, scale):
    squares = []
    for d in range(3):
        center_d = center[d]
        if center_d != 0:
            common_coordinate = center_d + (1 if center_d > 0 else -1)
            non_d = indices_other_than(d)
            square_corners = []
            for a in (-1, 1):
                for b in (-1, 1):
                    corner = [0, 0, 0]
                    corner[d] = common_coordinate
                    corner[non_d[0]] = center[non_d[0]] + a
                    corner[non_d[1]] = center[non_d[1]] + b
                    square_corners.append(corner)
            square_corners = np.array(square_corners).T
            square_corners = sort_counter_clockwise(square_corners, non_d)
            color = face_specs[square_face(square_corners)].initial_color
            squares.append(new_square(color, square_corners * scale))

    return squares


def new_cube(scale=10):
    assert type(scale) == int, 'scale factor for the cube must be an integer'
    face_specs = new_face_specs()
    pieces = []
    center_coordinates = (-2, 0, 2)
    for x in center_coordinates:
        for y in center_coordinates:
            for z in center_coordinates:
                center = np.array((x, y, z))
                if zeros(center) < 3:
                    pieces.append(new_piece(center, face_specs, scale))
    face_specs = new_face_specs(scale)
    return SimpleNamespace(face_specs=face_specs, pieces=pieces)


# List of the faces in a piece
def piece_faces(piece):
    return [square_face(square.corners) for square in piece]


# List of the pieces in a face_name
def face_pieces(face_name, cube):
    return [piece for piece in cube.pieces if
            face_name in piece_faces(piece)]


def face_squares(face_name, cube):
    pieces = face_pieces(face_name, cube)
    squares = []
    for piece in pieces:
        squares.extend([square for square in piece
                        if square_face(square.corners) == face_name])
    return squares


def is_visible(center, camera):
    center = np.array(center)
    ray = center - camera.t
    return np.dot(ray, center) < 0.


def project(points, camera):
    canonical = transform(points, camera.R, camera.t)
    assert np.all(canonical[2, :] > 0.), \
        'some points are at or behind the camera'
    z = np.outer(np.ones(2), canonical[2, :])
    canonical_image = camera.f * canonical[:2, :] / z
    scaled = np.diag(camera.s) @ canonical_image
    pixel_image = transform(scaled, t=-camera.pi)
    return pixel_image


def square_patch_list(square, camera):
    patches, colors, margin = [], [], 0.9
    center = polygon_centroid(square.corners)
    sticker = transform(margin * transform(square.corners, t=center), t=-center)
    corners = project(square.corners, camera)
    sticker = project(sticker, camera)
    patches.append(Polygon(corners.T, True))
    black = new_color('black', '#000000')
    colors.append(black.hex)
    patches.append(Polygon(sticker.T, True))
    colors.append(square.color.hex)
    return patches, colors


def cube_patch_list(cube, camera):
    patches, colors = [], []
    for face_name, specs in cube.face_specs.items():
        face_center = cube.face_specs[face_name].center
        if is_visible(face_center, camera):
            squares = face_squares(face_name, cube)
            for square in squares:
                square_patches, square_colors = square_patch_list(square, camera)
                patches.extend(square_patches)
                colors.extend(square_colors)
    return patches, colors


def draw_cube(cube, camera):
    patches, colors = cube_patch_list(cube, camera)
    colormap = ListedColormap(colors)
    collection = PatchCollection(patches, cmap=colormap)
    collection.set_array(np.arange(len(patches)))
    fig = plt.figure(figsize=camera.pixels/100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_position((0, 0, 1, 1))
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_xlim(0, camera.pixels[0])
    ax.set_ylim(camera.pixels[1], 0)
    ax.set_aspect(1.)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plt.show()
