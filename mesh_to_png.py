from data_def import Mesh
from utils import load_data, mesh_to_png
(texture, identity, expression, triangles) = load_data()
# create mesh from mean data
mean_mesh = Mesh(identity.mean, texture.mean, triangles)
mesh_to_png('debug.png', mean_mesh)
