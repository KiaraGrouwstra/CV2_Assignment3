import numpy as np
import matplotlib.pyplot as plt
from data_def import Mesh
from utils import load_data, mesh_to_png, reconstruct_face
from pdb import set_trace

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


def viewport_matrix(v_l=-1, v_r=1, v_t=1, v_b=-1):
    """
    viewport matrix: http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/projection/viewport_transformation.html
    @param v_l: X
    @param v_r: Y
    @param v_t: Z
    @param v_b: ?
    """
    V = np.eye(4)
    V[0, 0] = .5 * (v_r - v_l)
    V[1, 1] = .5 * (v_t - v_b)
    V[2, 2] = .5
    V[3, 0] = .5 * (v_r + v_l)
    V[3, 1] = .5 * (v_t + v_b)
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
    P[0, 0] = 2 * n / (r - l)
    P[1, 1] = 2 * n / (t - b)
    P[2, 0] = (r + l) / (r - l)
    P[2, 1] = (t + b) / (t - b)
    P[2, 2] = -(f + n) / (f - n)
    P[2, 3] = -1
    P[3, 2] = 2 * f * n / (f - n)
    return P

def project_points(S, z, near):
    """inspiration: https://github.com/d4vidbiertmpl/Computer-Vision-2/blob/master/Assignment_3/solution.ipynb"""
    translation = (0, 0, z)
    R[3, 0:3] = translation
    P = perspective_matrix(1, -1, 1, -1, near, 100)
    p = P @ R @ S
    # make it homogeneous
    V = viewport_matrix()
    p = p / p[3, :]
    p = V @ p
    p = p[:2, :]
    return p.T


(texture, identity, expression, triangles) = load_data()
G = reconstruct_face(identity, expression)
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
fig = plt.figure()
plt.imshow(img)
points_ = project_points(G_, z=-200, near=1)
for i, pair in enumerate(points_):
    plt.annotate(str(i), pair)
plt.show()
