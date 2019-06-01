from q2__morphable_model import load_data, plot_scene
import numpy as np
import matplotlib.pyplot as plt


IM_WIDTH = 1024
IM_HEIGHT = 768

NEAR = 300.0
FAR = 2000.0
FOVY = 0.5

def to_homogenous(x):
    return np.c_[x, np.ones(x.shape[0])]

def normalize(x):
    return x / x[:, -1].reshape(-1, 1)

def from_homogenous(x):
    return normalize(x)[:, :-1]

def construct_V2(cx, cy):
    V = np.asarray([[ cx, 0.0, 0.0,  cx],
                    [0.0, -cy, 0.0,  cy],
                    [0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 1.0]])
    return V

def construct_P2(near, far, fovy, aspect_ratio):
    top = np.tan(fovy / 2.0) * near
    right = top * aspect_ratio
    P = construct_P(-right, right, -top, top, near, far)
#    f_n = f - n
#    Sy = 1.0 / (np.tan(fovy / 2 * np.pi / 180))
#    Sx = Sy * width / float(height)
#    P = np.asarray([[ Sx, 0.0,           0.0,  0.0],
#                    [0.0,  Sy,           0.0,  0.0],
#                    [0.0, 0.0,     - f / f_n, -1.0],
#                    [0.0, 0.0, - f * n / f_n,  0.0]])
    return P

def construct_V(left=-1, right=1, bottom=-1, top=1):
    V = np.eye(4)
    V[0, 0] = right - left
    V[0,-1] = right + left
    V[1, 1] = top - bottom
    V[1,-1] = top + bottom
    V[-2:, -1] += 1
    V /= 2.0

    """
    l, r, b, t, n, f = left, right, bottom, top, 0, 0

    V = np.eye(4)
    w = r - l
    h = t - b
    V[0, 0] = .5 * w
    V[1, 1] = .5 * h
    V[2, 2] = .5
    V[3, 0] = .5 * (r + l)
    V[3, 1] = .5 * (t + b)
    V[3, 2] = .5
    """

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

    """
    l, r, b, t, n, f = left, right, bottom, top, near, far

    P = np.zeros((4, 4))
    w = r - l
    h = t - b
    P[0, 0] = 2 * n / w
    P[1, 1] = 2 * n / h
    P[2, 0] = (r + l) / w
    P[2, 1] = (t + b) / h
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -1
    P[3, 2] = -2 * f * n / (f - n)
    """

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

    # load vertex indices
    v_idx = read_vertex_indices()

    # load data and create face sample
    pca_id, pca_exp, color, tri = load_data()
    geo = pca_id.sample() + pca_exp.sample()


    geo_h = to_homogenous(geo).T

    T = construct_T(0, 0, -400)
    model_mat = T.dot(construct_R(0, 0, 0))
    geo_gl = model_mat.dot(geo_h)

    view_mat = construct_T(0, 0, 0)#(3 * max(geo_gl[2]) - min(geo_gl[2])))
    geo_gl_ = view_mat.dot(geo_gl)

    geo_gl__ = geo_gl[:, v_idx]

    w = max(geo_h[0]) - min(geo_h[0])
    h = max(geo_h[1]) - min(geo_h[1])

    print(IM_WIDTH/ IM_HEIGHT)
    print(w/ h)

#    projection_mat = construct_P(min(geo_gl_[0]), max(geo_gl_[0]),
#                                 min(geo_gl_[1]), max(geo_gl_[1]),
#                                               1, max(geo_gl_[2]))
#    projection_mat = construct_P(-1, 1, -1, 1, 1, 100)
#    aspect_ratio = w / h
    aspect_ratio = IM_WIDTH / IM_HEIGHT
    projection_mat = construct_P2(NEAR, FAR, FOVY, aspect_ratio)


    viewport_mat = construct_V()
#    viewport_mat = construct_V2(IM_WIDTH / 2.0, -IM_HEIGHT / 2.0)
    geo_h_ = viewport_mat.dot(projection_mat.dot(geo_gl_))

    geo_ = from_homogenous(geo_h_.T)
#    geo_ = geo_h_[:-1].T

    a = geo_.T
    print(a.shape)
    print(min(a[0]), max(a[0]))
    print(min(a[1]), max(a[1]))
    print(min(a[2]), max(a[2]))
#    print(min(a[3]), max(a[3]))

    """
    fig, axarr = plt.subplots(1, 2)
    plot_scene(from_homogenous(geo_gl.T), color, tri, axarr[0])
    plot_scene(geo_, color, tri, axarr[1])
    axarr[0].scatter(geo[v_idx, 0], geo[v_idx, 1])
    axarr[1].scatter(geo_[v_idx, 0], geo_[v_idx, 1])
    """

#    geo_ -= geo_.min(0).reshape(1, -1)

    geo__ = from_homogenous(geo_)
#    geo__[:, 0] -= IM_WIDTH / 2.0
#    geo__[:, 1] += IM_HEIGHT / 2.0



    a = geo_.T
    print(a.shape)
    print(min(a[0]), max(a[0]))
    print(min(a[1]), max(a[1]))
    print(min(a[2]), max(a[2]))


#    plot_scene(geo, color, tri, axarr[0], resolution=(IM_WIDTH, IM_HEIGHT))
#    plt.figure()
#    plot_scene(geo_, color, tri, resolution=(IM_WIDTH, IM_HEIGHT))
#    geo[:, 0] -= min(geo[:, 0])
#    geo[:, 1] -= max(geo[:, 1])
#    axarr[0].scatter(geo[v_idx, 0], -geo[v_idx, 1])
#    plt.scatter(geo_[v_idx, 0], -geo_[v_idx, 1])

    fig, axarr = plt.subplots(1, 3)
    test = plt.imread('test.png')
    debug0000 = plt.imread('debug0000.png')
    axarr[0].imshow(test)
    axarr[0].set_title('test.png')
    axarr[1].imshow(debug0000)
    axarr[1].set_title('debug0000.png')

    print(test.shape)


#    plot_scene(geo_, color, tri, axarr[1], resolution=test.shape[:2])
    axarr[2].imshow(test)
    axarr[2].scatter((geo_[::10, 0] * 0.5 + 0.5) * IM_WIDTH,
            (geo_[::10, 1] * 0.5 + 0.5) * -IM_HEIGHT + IM_HEIGHT,
            s=0.1, c='b')
    axarr[2].scatter((geo_[v_idx, 0] * 0.5 + 0.5) * IM_WIDTH,
            (geo_[v_idx, 1] * 0.5 + 0.5) * -IM_HEIGHT + IM_HEIGHT,
            s=5, c='r')
    axarr[2].set_xlim([0, IM_WIDTH])
    axarr[2].set_ylim([IM_HEIGHT, 0])
    axarr[2].set_title('result')

    plt.show()
    exit(0)


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


    geo_left_projected = geo_h_left_projected[:-1].T


    # plot projected image
    plt.figure()
    plot_scene(geo_left_projected, color, tri)

#    """
    # plot vertices
    plt.figure()
    plt.scatter(geo_left_projected[v_idx, 0],
            geo_left_projected[v_idx, 1])
    for i in range(len(v_idx)):
        plt.text(geo_left_projected[v_idx[i], 0],
                geo_left_projected[v_idx[i], 1], i, fontsize=7)
#    """

    plt.show()

    return

if __name__ == '__main__':
    main()
