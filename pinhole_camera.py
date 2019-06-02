from utils import load_data, load_landmarks, geo_to_im
import numpy as np
import matplotlib.pyplot as plt
import argparse


NEAR = 300.0
FAR = 2000.0
FOVY = 0.5
CAMERA_T = np.asarray([0.0, 0.0, -400.0])

def normalize(x):
    return x / x[:, -1].reshape(-1, 1)

# to_homogenous(np.array([[3,4]]))
def to_homogenous(x):
    return np.c_[x, np.ones(x.shape[0])]

def from_homogenous(x):
    return normalize(x)[:, :-1]

def apply_transform(x, M):
    return from_homogenous(M.dot(to_homogenous(x).T).T)

def construct_V(cx, cy):
    V = np.asarray([[ cx, 0.0, 0.0,  cx],
                    [0.0, -cy, 0.0,  cy],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 1.0]])
    return V

def construct_P(near, far, fovy, aspect_ratio):
    top = np.tan(fovy / 2.0) * near
    right = top * aspect_ratio
    left = -right
    bottom = -top
    near_2 = 2 * near
    P = np.zeros([4, 4])
    P[0, 0] = near_2
    P[1, 1] = near_2
    P[2, 3] = -near_2 * far
    P[:, 2] = [right + left, top + bottom, -(far + near), -1.0]
    P /= np.asarray([right - left, top - bottom, far - near, 1.0]
            ).reshape(-1, 1)
    return P

def construct_R(theta_x, theta_y, theta_z):
    to_rad = lambda theta: theta * np.pi / 180.0
    theta_x = to_rad(theta_x)
    theta_y = to_rad(theta_y)
    theta_z = to_rad(theta_z)
    sin_x, cos_x = np.sin(theta_x), np.cos(theta_x)
    sin_y, cos_y = np.sin(theta_y), np.cos(theta_y)
    sin_z, cos_z = np.sin(theta_z), np.cos(theta_z)
    R_x = np.asarray([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]
    ])
    R_y = np.asarray([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ])
    R_z = np.asarray([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])
    R = np.eye(4)
    R[:-1, :-1] = R_z.dot(R_y.dot(R_x))
    return R

def construct_T(x, y, z):
    T = np.eye(4)
    T[:-1, -1] = [x, y, z]
    return T

def construct_obj_to_cam(omega, t, resolution=(1.0, 1.0)):
    aspect_ratio = resolution[0] / float(resolution[1])
    T = construct_T(*t)
    R = construct_R(*omega)
    model_mat = T.dot(R)
    view_mat = construct_T(*CAMERA_T)
    projection_mat = construct_P(NEAR, FAR, FOVY, aspect_ratio)
    viewport_mat = construct_V(resolution[0] / 2.0, resolution[1] / 2.0)
    M = viewport_mat.dot(projection_mat.dot(view_mat.dot(model_mat)))
    return M

def main():

    # load data
    color, pca_id, pca_exp, tri = load_data()
    color = color.mean
    v_idx = load_landmarks()

    # set geometry (use mean value)
    geo = pca_id.sample(0.0)

    if ARGS.debug:

        # parameters
        omega = [0.0, 0.0, 0.0]
        t = [0.0, 0.0, 0.0]

        # reproduce debug data
        im = geo_to_im(geo, color, tri)
        resolution = tuple(im.shape[:2][::-1])
        M = construct_obj_to_cam(omega, t, resolution)
        geo_ = apply_transform(geo, M)

        # load debug images
        test = plt.imread('debug_images/test.png')
        debug0000 = plt.imread('debug_images/debug0000.png')

        # plot results
        fig, axarr = plt.subplots(1, 3)

        axarr[0].set_title('test.png')
        axarr[0].imshow(test)

        axarr[1].set_title('debug0000.png')
        axarr[1].imshow(debug0000)

        axarr[2].set_title('reproduction')
        axarr[2].imshow(im)
        axarr[2].scatter(geo_[::20, 0], geo_[::20, 1], s=0.1, c='b')
        axarr[2].scatter(geo_[v_idx, 0], geo_[v_idx, 1], s=8, c='r')
        axarr[2].set_xlim([0, resolution[0]])
        axarr[2].set_ylim([resolution[1], 0])

        plt.tight_layout()

    else:

        # determine left and right rotated images
        R_l = construct_R(0,  10, 0)
        R_r = construct_R(0, -10, 0)
        geo_l = apply_transform(geo, R_l)
        geo_r = apply_transform(geo, R_r)

        # plot rotated images
        im_l = geo_to_im(geo_l, color, tri)
        im_r = geo_to_im(geo_r, color, tri)
        fig, axarr = plt.subplots(2)
        axarr[0].set_title('10 degree y-rotation')
        axarr[0].imshow(im_l)
        axarr[1].set_title('-10 degree y-rotation')
        axarr[1].imshow(im_r)
        plt.tight_layout()

        # parameters
        omega = [0.0, 10.0, 0.0]
        t = [0.0, 0.0, 0.0]
        resolution = tuple(im_l.shape[:2][::-1])

        # create rotated model
        M = construct_obj_to_cam(omega, t, resolution)
        geo_ = apply_transform(geo, M)

        # plot landmarks
        fig, ax = plt.subplots(1)
        ax.set_title('10 degree rotation landmarks')
        ax.scatter(geo_[v_idx, 0], geo_[v_idx, 1], c='b', s=8)
        for i in range(len(v_idx)):
            ax.text(geo_[v_idx[i], 0], geo_[v_idx[i], 1], i,
                    fontsize=8)
        ax.axis('equal')
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[::-1])
        plt.tight_layout()

    plt.show()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
            help='Run debug example')
    ARGS = parser.parse_args()
    main()
