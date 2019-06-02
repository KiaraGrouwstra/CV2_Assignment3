from utils import load_data, load_landmarks, geo_to_im
from pinhole_camera import construct_obj_to_cam, apply_transform, \
        construct_R, construct_T
import matplotlib.pyplot as plt
import numpy as np


def create_test_data(pca_id, pca_exp, color, tri, do_plot=False):
    # parameters
    alpha = np.random.uniform(-1.0, 1.0)
    delta = np.random.uniform(-1.0, 1.0)
    omega = np.asarray([0, 0, 30])
    t = np.asarray([0, 0, 0])
    omega_c = np.asarray([0, 0, 90])
    # create test data
    geo = pca_id.sample(alpha) + pca_exp.sample(delta)
    T = construct_T(*t)
    R = construct_R(*omega)
    geo_ = apply_transform(geo, T.dot(R))
    R_c = construct_R(*omega_c)
    color_ = apply_transform(color, R_c)
    im = geo_to_im(geo_, color_, tri)
    return alpha, delta, omega, t, im

def construct_model(pca_id, pca_exp, alpha, delta, omega, t,
        resolution=(1.0, 1.0)):
    geo = pca_id.sample(alpha) + pca_exp.sample(delta)
    M = construct_obj_to_cam(omega, t, resolution)
    geo_ = apply_transform(geo, M)
    geo_gl = apply_transform(geo, construct_T(*t).dot(construct_R(*omega)))
    return geo_, geo_gl

def remove_transparency(im):
    alpha = im[:, :, -1].reshape(*im.shape[:-1], 1)
    return 1.0 - alpha * (1.0 - im[:, :, :-1])

def bilinear_interpolation(x, y, im):
    if im.shape[-1] == 4:
        im = remove_transparency(im)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    f = lambda x, y: im[y, x].reshape(-1, 3)
    x1 = np.floor(x).astype(int)
    x2 = np.ceil(x).astype(int)
    y1 = np.floor(y).astype(int)
    y2 = np.ceil(y).astype(int)
    xdiff = x2 - x1
    ydiff = y2 - y1
    xx1 = (x2 - x) / xdiff
    xx2 = (x - x1) / xdiff
    color = (y2 - y) / ydiff * (xx1 * f(x1, y1) + xx2 * f(x2, y1)) \
            + (y - y1) / ydiff * (xx1 * f(x1, y2) + xx2 * f(x2, y2))
    return color

def determine_texture(geo, im):
    mask = (geo[:, 0] >= 0) * (geo[:, 0] <= im.shape[1] - 1) \
            * (geo[:, 1] >= 0) * (geo[:, 1] <= im.shape[0] - 1)
    color = bilinear_interpolation(geo[mask, 0], geo[mask, 1], im)
    color_ = np.zeros([geo.shape[0], 3])
    color_[mask] = color
    return color_

def main():

    # load data
    color, pca_id, pca_exp, tri = load_data()
    color = color.mean
    v_idx = load_landmarks()

    # create temporary sample data
    alpha, delta, omega, t, im = create_test_data(
            pca_id, pca_exp, color, tri)

    # create model and texture it
    resolution = tuple(im.shape[:2][::-1])
    geo_, geo = construct_model(pca_id, pca_exp, alpha, delta, omega, t,
            resolution)
    color_ = determine_texture(geo_, im)
    im_ = geo_to_im(geo, color_, tri)

    # plot results
    fig, axarr = plt.subplots(1, 3)

    axarr[0].set_title('ground-truth')
    axarr[0].imshow(im)

    axarr[1].set_title('model on ground-truth')
    axarr[1].imshow(im)
    axarr[1].scatter(geo_[::20, 0], geo_[::20, 1], s=0.1, c='b')
    axarr[1].scatter(geo_[v_idx, 0], geo_[v_idx, 1], s=8, c='r')
    axarr[1].set_xlim([0, resolution[0]])
    axarr[1].set_ylim([resolution[1], 0])

    axarr[2].set_title('model')
    axarr[2].imshow(im_)
    axarr[2].set_xlim([0, resolution[0]])
    axarr[2].set_ylim([resolution[1], 0])

    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    main()
