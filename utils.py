import h5py
import numpy as np
import pyrender
from pyrender import Scene, Viewer, PerspectiveCamera, DirectionalLight, PointLight, OffscreenRenderer
from data_def import Mesh, PCAModel

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
    return (texture, identity, expression, triangles)

def mesh_to_png(file_name, mesh):
    png = mesh.trimesh().scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)

# def render_mesh(mesh, h=256, w=256):
#     """https://pyrender.readthedocs.io/en/latest/examples/quickstart.html"""
#     mesh = pyrender.Mesh.from_trimesh(mesh)
#     scene = Scene()
#     scene.add(mesh)

#     # z-axis away from the scene, x-axis right, y-axis up
#     pose = np.eye(4)
#     pose[2, 3] = 250

#     # add camera
#     camera = PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
#     scene.add(camera, pose=pose)

#     # add light
#     # light = DirectionalLight(color=np.ones(3), intensity=5.0)
#     light = PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#     scene.add(light, pose=pose)

#     r = OffscreenRenderer(h, w)
#     color, depth = r.render(scene)
#     return color
