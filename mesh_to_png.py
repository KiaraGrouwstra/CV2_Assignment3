from data_def import Mesh
from utils import load_data, mesh_to_png
(texture, identity, expression, triangles) = load_data()
# create mesh from mean data
mesh = Mesh(identity.mean, texture.mean, triangles)
png = mesh_to_png(mesh)
with open('results/debug.png', 'wb') as f:
    f.write(png)
