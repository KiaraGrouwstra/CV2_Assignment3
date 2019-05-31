from q2__morphable_model import load_data, plot_scene
import numpy as np
import matplotlib.pyplot as plt


def to_homogenous(x):
    return np.c_[x, np.ones(x.shape[0])]

def from_homogenous(x):
    return x[:, :-1] / x[:, -1].reshape(-1, 1)

def construct_V(left=-1, right=1, bottom=-1, top=1):
    V = np.eye(4)
    V[0, 0] = right - left
    V[0,-1] = right + left
    V[1, 1] = top - bottom
    V[1,-1] = top + bottom
    V[-2:, -1] += 1
    V /= 2.0
    return V

def construct_P(left, right, bottom, top, near, far):
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

def read_vertex_indices(
        file_name='Landmarks68_model2017-1_face12_nomouth.anl'):
    with open(file_name, 'r') as f:
        v_idx = np.asarray([int(idx) for idx in f])
    return v_idx

def main():

    # load data and create face sample
    pca_id, pca_exp, color, tri = load_data()
    geo = pca_id.sample() + pca_exp.sample()

    # determine left and right rotated images
    geo_h = to_homogenous(geo).T
    R_left = construct_R(0, 10, 0)
    R_right = construct_R(0, -10, 0)
    geo_h_left = R_left.dot(geo_h)
    geo_h_right = R_right.dot(geo_h)
    geo_left = from_homogenous(geo_h_left.T)
    geo_right = from_homogenous(geo_h_right.T)

    # plot rotated images
    fig, axarr = plt.subplots(1, 2)
    plot_scene(geo_left, color, tri, axarr[0])
    plot_scene(geo_right, color, tri, axarr[1])

    # determine complete projected image for left rotation
    T = construct_T(0, 0, 5 * -(max(geo_h[2]) - min(geo_h[2])))
    P = construct_P(min(geo_h[0]), max(geo_h[0]),
                    min(geo_h[1]), max(geo_h[1]),
                    min(geo_h[2]), max(geo_h[2]))
    V = construct_V()
    geo_h_left_projected = V.dot(P.dot(T.dot(geo_h_left)))
    geo_left_projected = from_homogenous(geo_h_left_projected.T)

    # plot projected image
    plt.figure()
    plot_scene(geo_left_projected, color, tri)

    # load vertex indices
    v_idx = read_vertex_indices()

    # plot vertices
    plt.figure()
    plt.scatter(geo_left_projected[v_idx, 0],
            geo_left_projected[v_idx, 1])
    for i in range(len(v_idx)):
        plt.text(geo_left_projected[v_idx[i], 0],
                geo_left_projected[v_idx[i], 1], i, fontsize=7)

    plt.show()

    return

if __name__ == '__main__':
    main()
