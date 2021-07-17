import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import tensorflow as tf
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.metrics import pairwise_distances
from lib.utils import eulers_to_rot_mat
from .metrics import misorientation


def visu_preds_and_targets(preds, minieulers, rx, ry):
    """Shows a 2 x 3 grid plot of IPFs in preds AND targets"""

    x_map_pred, y_map_pred, z_map_pred = visualize(preds, rx, ry, reshape=False)
    x_map_pred = x_map_pred.numpy().reshape((rx, ry, 3))
    y_map_pred = y_map_pred.numpy().reshape((rx, ry, 3))
    z_map_pred = z_map_pred.numpy().reshape((rx, ry, 3))

    x_map, y_map, z_map = visualize(minieulers, rx, ry, reshape=False)
    x_map = x_map.numpy().reshape((rx, ry, 3))
    y_map = y_map.numpy().reshape((rx, ry, 3))
    z_map = z_map.numpy().reshape((rx, ry, 3))

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(231)
    ax.axis('off')
    ax.imshow(x_map_pred)
    ax.set_title('X (prediction)')
    ax = fig.add_subplot(232)
    ax.axis('off')
    ax.imshow(y_map_pred)
    ax.set_title('Y (prediction)')
    ax = fig.add_subplot(233)
    ax.axis('off')
    ax.imshow(z_map_pred)
    ax.set_title('Z (prediction)')
    ax = fig.add_subplot(234)
    ax.axis('off')
    ax.imshow(x_map)
    ax.set_title('X (ground truth)')
    ax = fig.add_subplot(235)
    ax.axis('off')
    ax.imshow(y_map)
    ax.set_title('Y (ground truth)')
    ax = fig.add_subplot(236)
    ax.axis('off')
    ax.imshow(z_map)
    ax.set_title('Z (ground truth)')
    plt.tight_layout()
    plt.show()


def evaluate_and_disptlot(model, testing_dataset, yte, title='test'):
    """Evaluates the testing dataset, calculates preds and moas, makes a distplot"""
    model.evaluate(testing_dataset)  # isn't that the average moa though?
    preds = model.predict(testing_dataset)
    moas = np.degrees(misorientation(yte.astype(np.float32), preds).numpy())
    median_moa = np.median(moas)
    ma_x, ma_y, ma_z = in_and_out_of_plane(preds, yte)

    return moas, median_moa, ma_x, ma_y, ma_z


def voronoi_IPF_plot(eulers, z_values, direction='z'):
    """
    A plotting function to visualize distributions of data in the Inverse Pole Figure (IPF) space; we are looking at a sparse
    cloud of points in orientation space (for example 1 point per grain) each with an associated value of interest (for example,
    the disorientaiton angle). The plot shows if there is a trend in the distribution of values of interest in orientation space.

    eulers: an array of shape (:,3) representing a set of ZYZ intrinsic Euler angles.
    z_values: an array of shape (:,1) representing the value of interest associated with each set of Euler angle.
    n_fib_tiles: number of Voronoi tiles to compute (the resolution of the "bins" for averaging). A good default is to set the
                 number of tiles so that there are at least 10 data points in the least populated tile.
    line_width: width of the Voronoi tile lines.
    direction: IPF plotting direction - 'x', 'y' or 'z'.
    """

    def cart2spher(x, y, z):
        radius = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / radius)
        phi = np.arctan2(y, x) + np.radians(180)
        return radius, theta, phi

    def get_region_values(vor, xyc, zees):
        vor_points = np.array(vor.points)
        regvals = np.zeros((len(vor_points)))
        n_vals_cell = np.zeros((len(vor_points)))
        knn_idx = np.argmin(pairwise_distances(
            X=xyc, Y=vor_points, metric='euclidean'), axis=1)

        neighbors_dico = {}
        for ind in knn_idx:
            neighbors_dico[ind] = []

        for ind, zval in zip(knn_idx, zees):
            n_vals_cell[ind] += 1
            regvals[ind] += (zval - regvals[ind]) / n_vals_cell[ind]  # Iterated mean

            neighbors_dico[ind].append(zval)

        regmedian = np.zeros((len(vor_points)))
        for ind, pts in neighbors_dico.items():
            regmedian[ind] = np.median(pts)

        return regvals, regmedian

    eulers_ref = np.radians(np.array([
        [0, 0, 0],  # 100
        [0, 45, 0],  # 110
        [0, 54.74, 45],  # 111

    ]))

    xvr, yvr, zvr = _compute_vectors(eulers_ref)
    yes_ref, xes_ref = _vector_project(zvr)

    xv, yv, zv = _compute_vectors(eulers)
    if direction == 'z':
        yes, xes = _vector_project(zv)
    elif direction == 'y':
        yes, xes = _vector_project(yv)
    else:
        yes, xes = _vector_project(xv)

    xy_coords = np.vstack((xes, yes)).T

    points = []
    spherical_points = []
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(400 * 16):
        theta = phi * i
        y = 1 - (i / float(400 * 16 - 1)) * 2
        radius = math.sqrt(1 - y * y)
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        spherical_points.append(cart2spher(x, y, z))
        x = x / (1 - z)
        y = y / (1 - z)
        points.append((x, y))
    spherical_points = np.array(spherical_points)
    points = np.array(points)
    filt = (spherical_points[:, 1] >= np.radians(90))
    points = points[filt]
    points = -points

    R = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    filt = (R <= 0.8)
    points = points[filt]

    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

    voronoi_plot_2d(vor, show_points=False, show_vertices=False,
                    line_width=0.0, line_colors='black', ax=ax)

    _, region_values = get_region_values(vor, xy_coords, z_values)
    mapper = plt.cm.ScalarMappable(
        norm=clrs.Normalize(vmin=0, vmax=20, clip=True),
        cmap=plt.cm.Blues)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            plt.fill(*zip(*polygon), color=mapper.to_rgba(region_values[r]))
    plt.colorbar(mapper)

    ax.set_ylim(0, 0.40)
    ax.set_xlim(0, 0.45)
    ax.axis('off')

    theta_array = np.radians(np.linspace(0, 45, 50))
    x_circle_ref = np.sqrt(2) * np.cos(theta_array) - 1.0
    y_circle_ref = np.sqrt(2) * np.sin(theta_array)

    ax.plot(x_circle_ref, y_circle_ref, color='black', linewidth=2, zorder=10)

    ax.plot([xes_ref[2], xes_ref[0], xes_ref[1]],
            [yes_ref[2], yes_ref[0], yes_ref[1]],
            color='black', linewidth=2, zorder=10)

    plt.fill([0.0, 0.0, 0.4], [0.0, 0.4, 0.4], 'black')

    xcr = x_circle_ref.copy()
    xcr = np.append(xcr, [0.45, 0.45])
    ycr = y_circle_ref.copy()
    ycr = np.append(ycr, [0.4, 0.0])
    plt.fill(xcr, ycr, 'black')

    plt.show()


@tf.function
def visualize(eulers, rx, ry, reshape=True):
    rot_mat = eulers_to_rot_mat(eulers)
    rot_mat_inv = tf.linalg.inv(rot_mat)
    output = tf.concat((_symmetrize(rot_mat_inv[:, :, 0]),
                        _symmetrize(rot_mat_inv[:, :, 1]),
                        _symmetrize(rot_mat_inv[:, :, 2])), axis=1)
    xmap = _colorize(output[:, 0:3])
    ymap = _colorize(output[:, 3:6])
    zmap = _colorize(output[:, 6:9])
    if reshape:
        xmap = xmap.reshape((rx, ry, 3))
        ymap = ymap.reshape((rx, ry, 3))
        zmap = zmap.reshape((rx, ry, 3))
    return xmap, ymap, zmap


def _compute_vectors(eulers):
    rot_mat = eulers_to_rot_mat(eulers)
    rot_mat_inv = tf.linalg.inv(rot_mat)
    xv = _symmetrize(rot_mat_inv[:, :, 0])
    yv = _symmetrize(rot_mat_inv[:, :, 1])
    zv = _symmetrize(rot_mat_inv[:, :, 2])
    return xv, yv, zv


def _vector_project(zvector):
    r_z = np.empty(zvector.shape[0])
    theta_z = np.empty(zvector.shape[0])
    for k, vect in enumerate(zvector):
        theta_z[k] = np.arctan2(vect[1], vect[0])
        r_z[k] = np.sin(np.arccos(vect[2])) / (1 + np.arccos(np.cos(vect[2])))
    xes_preds = r_z * np.cos(theta_z)
    yes_preds = r_z * np.sin(theta_z)
    return xes_preds, yes_preds


def _symmetrize(indeces):
    indeces = tf.abs(indeces)
    norms = tf.square(indeces[:, 0]) + tf.square(indeces[:, 1]) + tf.square(indeces[:, 2])
    norms = tf.expand_dims(norms, 1)
    indeces = indeces / norms
    indeces = tf.sort(indeces, axis=1, direction='ASCENDING')
    return indeces


def _colorize(indeces):
    a = tf.abs(tf.subtract(indeces[:, 2], indeces[:, 1]))
    b = tf.abs(tf.subtract(indeces[:, 1], indeces[:, 0]))
    c = indeces[:, 0]
    rgb = tf.concat((tf.expand_dims(a, -1),
                     tf.expand_dims(b, -1),
                     tf.expand_dims(c, -1)), axis=1)
    # Normalization
    maxes = tf.reduce_max(rgb, axis=1)
    a = a / maxes
    b = b / maxes
    c = c / maxes
    rgb = tf.concat((tf.expand_dims(a, -1),
                     tf.expand_dims(b, -1),
                     tf.expand_dims(c, -1)), axis=1)
    return rgb


def in_and_out_of_plane(eulers_set_1, eulers_set_2):
    """
    Computes the error in the X, Y and Z cartesian directions.
    """
    xv1, yv1, zv1 = _compute_vectors(eulers_set_1)
    xv2, yv2, zv2 = _compute_vectors(eulers_set_2)

    xv1 = tf.cast(xv1, tf.dtypes.float64)
    yv1 = tf.cast(yv1, tf.dtypes.float64)
    zv1 = tf.cast(zv1, tf.dtypes.float64)

    dot_prods_x = tf.reduce_sum(tf.multiply(xv1, xv2), axis=1)
    mean_angle_x = tf.reduce_mean(tf.acos(dot_prods_x))

    dot_prods_y = tf.reduce_sum(tf.multiply(yv1, yv2), axis=1)
    mean_angle_y = tf.reduce_mean(tf.acos(dot_prods_y))

    dot_prods_z = tf.reduce_sum(tf.multiply(zv1, zv2), axis=1)
    mean_angle_z = tf.reduce_mean(tf.acos(dot_prods_z))

    mean_angle_x = np.degrees(mean_angle_x.numpy())
    mean_angle_y = np.degrees(mean_angle_y.numpy())
    mean_angle_z = np.degrees(mean_angle_z.numpy())

    return mean_angle_x, mean_angle_y, mean_angle_z
