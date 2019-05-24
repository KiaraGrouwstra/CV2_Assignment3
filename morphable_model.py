import argparse
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from fio import load_obj, save_obj
from data_def import PCAModel, Mesh
from utils import load_data

def main(args):
    flags = ('cols', 'rows', 'num_identity', 'num_expression')
    (cols, rows, num_identity, num_expression) = itemgetter(*flags)(vars(args))

    (texture, identity, expression, triangles) = load_data()
    # extract 30 PC for facial identity and 20 PC for expression
    identity   = identity  .filter(num_identity)
    expression = expression.filter(num_expression)

    def reconstruct_face(alpha, delta):
        """generate a point cloud using eq. 1"""
        geom = identity  .sample(alpha)
        expr = expression.sample(delta)
        G = geom + expr
        return G

    plt.figure()
    # figsize=(20, 40)
    for i in range(rows * cols):
        # uniformly sample alpha and delta from -1~1
        alpha = np.random.uniform(-1.0, 1.0, num_identity)
        delta = np.random.uniform(-1.0, 1.0, num_expression)
        # generate a point cloud using eq. 1
        G = reconstruct_face(alpha, delta)
        # For vertex colour you can use mean face colour.
        mesh = Mesh(G, texture.mean, triangles)
        save_obj(f'meshes/face-{i}.mesh', mesh)
        # Provided mesh_to_png function and Mesh data structure can be used for visualization.
        img = mesh.trimesh().scene().save_image()
        # img = render_mesh(mesh)

        plt.subplot(rows, cols, i+1)
        plt.imshow(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cols', type=int, default=5, help='number of columns')
    parser.add_argument('--rows', type=int, default=5, help='number of rows')
    parser.add_argument('--num_identity',   type=int, default=30, help='number of PCs to sample for facial identity')
    parser.add_argument('--num_expression', type=int, default=20, help='number of PCs to sample for expression')
    args = parser.parse_args()
    main(args)
