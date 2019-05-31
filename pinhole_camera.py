import numpy as np
import matplotlib.pyplot as plt
from data_def import Mesh
from utils import load_data, mesh_to_png, reconstruct_face
from pdb import set_trace
import itertools
from mpl_toolkits.mplot3d import Axes3D

# - Section 3, Equation 2: [\hat{x}, \hat{y}, \hat{z}, \hat{d}] are homogeneous coordinates obtained after projection. You can remove homogeneous coordinate by dividing by \hat{d} and get u, v and depth respectively. You can check SfM lecture for more details about camera projections.
# - To convert homogeneous coordinate back to obtain u,v coordinates you just need to divide by a homogeneous coordinate, no division by depth is required.
# - Section 3: Your camera origin is at (0, 0, 0), camera view direction is (0, 0, -1). Consequently, 3D model is initially behind the camera, therefore remember to shift an object using z translation from section 4.
# - For projection matrix you can set principal point to be in the center of an image {W/2, H/2} and fovy to be 0.5.

def rotation_matrix_y(y_deg):
    """Get the Y rotation matrix (https://bit.ly/2PQ8glW) for a given rotation angle (in degrees).
       Assuming object translation to be 0.
    """
    y_rad = y_deg / 180 * np.pi 
    R = np.eye(4)
    R[0, 0] =  np.cos(y_rad)
    R[0, 2] =  np.sin(y_rad)
    R[2, 0] = -np.sin(y_rad)
    R[2, 2] =  np.cos(y_rad)
    return R


def viewport_matrix(l=-1, r=1, t=1, b=-1):
    """
    viewport matrix: http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/projection/viewport_transformation.html
    @param l: left
    @param r: right
    @param t: top
    @param b: bottom
    """
    V = np.eye(4)
    w = r - l
    h = t - b
    V[0, 0] = .5 * w
    V[1, 1] = .5 * h
    V[2, 2] = .5
    V[3, 0] = .5 * (r + l)
    V[3, 1] = .5 * (t + b)
    V[3, 2] = .5
    return V

def perspective_matrix(t, b, l, r, n, f):
    """
    perspective projection matrix: https://bit.ly/300gYmf
    @param t: top
    @param b: bottom
    @param l: left
    @param r: right
    @param n: near
    @param f: far
    """
    P = np.zeros((4, 4))
    w = r - l
    h = t - b
    P[0, 0] = 2 * n / w
    P[1, 1] = 2 * n / h
    P[2, 0] = (r + l) / w
    P[2, 1] = (t + b) / h
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -1
    P[3, 2] = 2 * f * n / (f - n)
    return P

def project_points(S, z, near):
    """inspiration: https://github.com/d4vidbiertmpl/Computer-Vision-2/blob/master/Assignment_3/solution.ipynb"""
    translation = (0, 0, z)
    R[3, 0:3] = translation
    P = perspective_matrix(1, -1, 1, -1, near, 100)
    ones = np.ones((S.shape[0], 1))
    S = np.append(S, ones, axis=1)
    V = viewport_matrix()
    # print(P, R, S.shape)
    p = V @ P @ R @ S.T

    # make it homogeneous
    # p = p / p[3, :]
    # p = V @ p
    # p = p[:2, :]
    # print(p.shape)
    return p.T


(texture, identity, expression, triangles) = load_data()
# G = reconstruct_face(identity, expression)
alpha = np.random.uniform(-1.0, 1.0)
delta = np.random.uniform(-1.0, 1.0)
G = reconstruct_face(identity, expression, alpha, delta)
(num_points, _) = G.shape
S = np.vstack((G.T, np.ones(num_points)))

fig = plt.figure()
# Rotate an object 10° and -10° around Oy and visualize result.
for i, angle in enumerate([-10, 10]):
    R = rotation_matrix_y(angle)
    G_ = (R @ S)[:3].T
    mesh = Mesh(G_, texture.mean, triangles)
    img = mesh_to_png(mesh)
    plt.subplot(2, 1, i+1)
    plt.imshow(img)
plt.savefig('pinhole_camera.png')

# vertex indexes annotation are available in the provided file
with open('Landmarks68_model2017-1_face12_nomouth.anl', 'r') as f:
    lines = f.read().splitlines()
vertex_idxs = list(map(int, lines))

# visualize facial landmark points on the 2D image plane using Eq. 2
plt.close("all")

points_ = project_points(G_, z=-200, near=1)
projections = []
originals = []
for i, pair in enumerate(points_):
    if i not in vertex_idxs:
        continue
    originals.append(G_[i][:2])
    projections.append(pair[:2])

originals = np.array(originals)
projections = np.array(projections)

numbers = [str(x) for x in range(len(originals))]

plt.scatter(projections[:, 0], projections[:, 1])
for i in range(len(originals)):
    plt.text(projections[i, 0], projections[i, 1], numbers[i], fontsize=7)

plt.scatter(originals[:, 0], originals[:, 1])
for i in range(len(originals)):
    plt.text(originals[i, 0], originals[i, 1], numbers[i], fontsize=7)
plt.show()
