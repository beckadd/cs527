from types import SimpleNamespace
import numpy as np
from camera import project, setup_image_figure, finalize_image_figure
from geometry import transform
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap


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


# List of the pieces in a face
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


def square_patch_list(square, camera):
    patches, colors, margin = [], [], 0.9
    center = polygon_centroid(square.corners)
    sticker_3d = transform(margin * transform(square.corners, t=center), t=-center)
    corners = project(square.corners, camera)
    sticker = project(sticker_3d, camera)
    patches.append(Polygon(corners.T, True))
    black = new_color('black', '#000000')
    colors.append(black.hex)
    patches.append(Polygon(sticker.T, True))
    colors.append(square.color.hex)
    features = [{'pixel': sticker[:, i], 'world': sticker_3d[:, i]}
                for i in range(sticker.shape[1])]
    return patches, colors, features


def cube_patch_list(cube, camera):
    patches, colors, features, last_id = [], [], {}, -1
    for face_name, specs in cube.face_specs.items():
        face_center = cube.face_specs[face_name].center
        if is_visible(face_center, camera):
            squares = face_squares(face_name, cube)
            for square in squares:
                square_patches, square_colors, square_features = \
                    square_patch_list(square, camera)
                patches.extend(square_patches)
                colors.extend(square_colors)
                ids = last_id + 1 + np.arange(4)
                for k, i in enumerate(ids):
                    features[i] = square_features[k]
                last_id += 4
        else:
            last_id += 36  # Skipping 36 feature IDs for the four corners of nine squares
    return patches, colors, features


def draw_cube(cube, camera, display=True, title=None):
    patches, colors, features = cube_patch_list(cube, camera)
    if display:
        colormap = ListedColormap(colors)
        collection = PatchCollection(patches, cmap=colormap)
        collection.set_array(np.arange(len(patches)))
        ax = setup_image_figure(camera)
        ax.add_collection(collection)
        finalize_image_figure(ax, camera, title)
    return features
