from imageio import imread
import matplotlib.pyplot as plt
import pickle


def pick_points(img, row=244, n=26):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.plot([0, img.shape[1]], 2 * [row], 'y')
    plt.axis('off')
    msg = 'Pick points on the yellow line'
    msg += '\n(left click: add; right click: remove last point; press enter to stop)'
    print(msg)
    pts = plt.ginput(n=-1, timeout=-1)
    return [round(p[0]) for p in pts]


circles_file = 'circles.png'
image = imread(circles_file)
points = pick_points(image)
print(points)

points_file = 'points.pkl'
with open(points_file, 'wb') as file:
    pickle.dump(points, file)
print('Horizontal point coordinates saved to pickle file {}'.format(points_file))
