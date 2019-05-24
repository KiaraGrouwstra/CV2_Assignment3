import argparse
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from fio import load_obj, save_obj
from data_def import PCAModel, Mesh
from utils import load_data, mesh_to_png, reconstruct_face

def main(args):
    flags = ('cols', 'rows')
    (cols, rows) = itemgetter(*flags)(vars(args))
    (texture, identity, expression, triangles) = load_data()

    fig = plt.figure()
    for i in range(rows * cols):
        G = reconstruct_face(identity, expression)
        # For vertex colour you can use mean face colour.
        mesh = Mesh(G, texture.mean, triangles)
        # save_obj(f'meshes/face-{i}.mesh', mesh)
        img = mesh_to_png(mesh)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cols', type=int, default=5, help='number of columns')
    parser.add_argument('--rows', type=int, default=5, help='number of rows')
    args = parser.parse_args()
    main(args)
