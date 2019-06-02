import argparse
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from fio import load_obj, save_obj
from data_def import PCAModel, Mesh
from utils import load_data, geo_to_im, render_mesh, reconstruct_face

def main(args):
    flags = ('cols', 'rows')
    (cols, rows) = itemgetter(*flags)(vars(args))
    (texture, identity, expression, triangles) = load_data()

    fig = plt.figure(figsize=(10, 10))
    for i in range(rows * cols):
        alpha = np.random.uniform(-1.0, 1.0)
        delta = np.random.uniform(-1.0, 1.0)
        G = reconstruct_face(identity, expression, alpha, delta)
        img = geo_to_im(G, texture.mean, triangles)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()
    #plt.savefig('results/morphable_model.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cols', type=int, default=5, help='number of columns')
    parser.add_argument('--rows', type=int, default=5, help='number of rows')
    args = parser.parse_args()
    main(args)
