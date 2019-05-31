import h5py
import numpy as np
import pyrender
from pyrender import Scene, Viewer, PerspectiveCamera, DirectionalLight, PointLight, OffscreenRenderer
import tempfile
import matplotlib.image as mpimg
import trimesh

from data_def import Mesh, PCAModel

NUM_IDENTITY   = 30
NUM_EXPRESSION = 20

def read_pca(bfm, path, k):
    mean = np.asarray(bfm[f'{path}/mean'],        dtype=np.float32).reshape((-1, 3))     # (N, 3)
    base = np.asarray(bfm[f'{path}/pcaBasis'],    dtype=np.float32).reshape((-1, 3, k))  # (N, 3, k)
    sig2 = np.asarray(bfm[f'{path}/pcaVariance'], dtype=np.float32)  # (k,)
    return PCAModel(mean, base, sig2)

def load_data():
    """load in the Morphace dataset; N = 28588, K = 56572"""
    bfm = h5py.File('model2017-1_face12_nomouth.h5', 'r')
    texture    = read_pca(bfm, 'color/model', 199)
    identity   = read_pca(bfm, 'shape/model', 199)
    expression = read_pca(bfm, 'expression/model', 100)
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T  # (K, 3)
    # extract 30 PC for facial identity and 20 PC for expression
    identity   = identity  .filter(NUM_IDENTITY)
    expression = expression.filter(NUM_EXPRESSION)
    return (texture, identity, expression, triangles)

def mesh_to_png(mesh, file_name=None):
    png = mesh.trimesh().scene().save_image()
    if not file_name:
        file_name = tempfile.mktemp()
    with open(file_name, 'wb') as f:
        f.write(png)
    img = mpimg.imread(file_name)
    return img

# def mesh_to_png(mesh, file_name=None, width=640, height=480, z_camera_translation=400):
#     # png = mesh.trimesh().scene().save_image()

#     mesh = trimesh.base.Trimesh(
#         vertices=mesh.vertices,
#         faces=mesh.triangles,
#         vertex_colors=mesh.colors)

#     mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, wireframe=False)

#     # compose scene
#     scene = pyrender.Scene(ambient_light=np.array([1.7, 1.7, 1.7, 1.0]), bg_color=[255, 255, 255])
#     camera = pyrender.PerspectiveCamera( yfov=np.pi / 3.0)
#     light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

#     scene.add(mesh, pose=np.eye(4))
#     scene.add(light, pose=np.eye(4))

#     # Added camera translated z_camera_translation in the 0z direction w.r.t. the origin
#     scene.add(camera, pose=[[ 1,  0,  0,  0],
#                             [ 0,  1,  0,  0],
#                             [ 0,  0,  1,  z_camera_translation],
#                             [ 0,  0,  0,  1]])

#     # render scene
#     r = pyrender.OffscreenRenderer(width, height)
#     color, _ = r.render(scene)

#     # imsave(file_name, color)
#     if not file_name:
#         file_name = tempfile.mktemp()
#     # with open(file_name, 'wb') as f:
#     #     f.write(png)
#     imsave(file_name, color)
#     img = mpimg.imread(file_name)
#     return img

def reconstruct_face(identity,
                     expression,
                     alpha=np.random.uniform(-1.0, 1.0),
                     delta=np.random.uniform(-1.0, 1.0)):
    """generate a point cloud using eq. 1.
       uniformly sample alpha and delta from -1~1.
    """
    geom = identity  .sample(alpha)
    expr = expression.sample(delta)
    G = geom + expr
    return G

def render_mesh(mesh, h=256, w=256):
    """https://pyrender.readthedocs.io/en/latest/examples/quickstart.html"""
    mesh = pyrender.Mesh.from_trimesh(mesh.trimesh())
    scene = Scene()
    scene.add(mesh)

    # z-axis away from the scene, x-axis right, y-axis up
    pose = np.eye(4)
    pose[2, 3] = 250

    # add camera
    camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    scene.add(camera, pose=pose)

    # add light
    # light = DirectionalLight(color=np.ones(3), intensity=5.0)
    light = PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=pose)

    r = OffscreenRenderer(h, w)
    color, depth = r.render(scene)
    return color
