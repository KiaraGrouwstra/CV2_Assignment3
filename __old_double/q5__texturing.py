from q2__morphable_model import load_data, plot_scene
from q3__pinhole_camera_model import to_homogenous, from_homogenous, \
        construct_R, construct_T
import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil

########################################################################
# XXX Should not be needed anymore when q4 works
#   could keep it as "if __name__ == '__main__':" part?
########################################################################
from data_def import Mesh
from q2__morphable_model import png_to_img, mesh_to_png


########################################################################
# XXX Should not be needed anymore when q4 works
#   could keep it as "if __name__ == '__main__':" part?
########################################################################
def create_test_data(pca_id, pca_exp, color, tri, do_plot=False):
    # parameters
    alpha = np.random.uniform(-1.0, 1.0)
    delta = np.random.uniform(-1.0, 1.0)
    omega = np.asarray([0, 30, 0])
    ####################################################################
    # XXX can't really properly test transformation right now...
    ####################################################################
    t = np.asarray([0, 0, 0])
    # create sample data
    geo = pca_id.sample(alpha) + pca_exp.sample(delta)
    R = construct_R(*omega)
    T = construct_T(*t)
    geo = from_homogenous(T.dot(R.dot(to_homogenous(geo).T)).T)
    R = construct_R(0, 0, 90)
    color = from_homogenous(R.dot(to_homogenous(color).T).T)
    mesh = Mesh(geo, color, tri)
    img = png_to_img(mesh_to_png(mesh))
    # plot sample data
    if do_plot:
        plt.figure()
        plt.imshow(img)
        plt.show()
    return alpha, delta, omega, t, img

########################################################################
# TODO needs testing
########################################################################
def bilinear_interpolation(x, y, f):
    x1 = floor(x)
    x2 = ceil(x)
    y1 = floor(y)
    y2 = ceil(y)
    xdiff = x2 - x1
    ydiff = y2 - y1
    xx1 = (x2 - x) / xdiff
    xx2 = (x - x1) / xdiff
    c = (y2 - y) / ydiff * (xx1 * f[x1, y1] + xx2 * f[x2, y1]) \
            + (y - y1) / ydiff * (xx1 * f[x1, y2] + xx2 * f[x2, y2])
    return c

def determine_texture(pca_id, pca_exp, alpha, delta, omega, t, img,
        # TODO remove other parameters
        tmp_color, tri):
    geo = pca_id.sample(alpha) + pca_exp.sample(delta)
    R = construct_R(*omega)
    T = construct_T(*t)
    geo = from_homogenous(T.dot(R.dot(to_homogenous(geo).T)).T)

    color = tmp_color
    print(img.shape)
    print(min(geo[:, 0]), max(geo[:, 0]))
    print(min(geo[:, 1]), max(geo[:, 1]))

#    fig, axarr = plt.subplots(1, 2)
#    plot_scene(geo, color, tri, axarr[0])
#    axarr[1].imshow(img)
#    plt.show()

    ####################################################################
    # XXX how to go from parameters to image coordinates?
    #   maybe the hint from question 4?
    ####################################################################

    return

def main():
    # load data
    pca_id, pca_exp, color, tri = load_data()
    # create temporary sample data
    alpha, delta, omega, t, img = create_test_data(
            pca_id, pca_exp, color, tri)
    determine_texture(pca_id, pca_exp, alpha, delta, omega, t, img,
            # TODO remove other parameters
            color, tri)
    return


if __name__ == '__main__':
    main()
