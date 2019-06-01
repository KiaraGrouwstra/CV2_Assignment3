import h5py
import numpy as np
from data_def import PCAModel, Mesh
import matplotlib.pyplot as plt
import trimesh
import io


N_ID = 30
N_EXP = 20

def read_pca(bfm, path):
    mean = np.asarray(bfm[f'{path}/mean'], dtype=np.float32).reshape(
            (-1, 3))
    pc = np.asarray(bfm[f'{path}/pcaBasis'], dtype=np.float32).reshape(
            *mean.shape, -1)
    var = np.asarray(bfm[f'{path}/pcaVariance'], dtype=np.float32)
    return PCAModel(mean, pc, np.sqrt(var))

def load_data_all(path='model2017-1_face12_nomouth.h5'):
    bfm = h5py.File(path, 'r')
    pca_id = read_pca(bfm, 'shape/model')
    pca_exp = read_pca(bfm, 'expression/model')
    c = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape(
            (-1, 3))
    tri = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T
    return pca_id, pca_exp, c, tri

def load_data(path='model2017-1_face12_nomouth.h5'):
    pca_id, pca_exp, c, tri = load_data_all()
    pca_id.clip(N_ID)
    pca_exp.clip(N_EXP)
    return pca_id, pca_exp, c, tri

def mesh_to_png(mesh, file_name=None, resolution=(1024, 768)):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)
    png = mesh.scene().save_image(resolution)
    if file_name != None:
        with open(file_name, 'wb') as f:
            f.write(png)
    return png

def png_to_img(png):
    return plt.imread(io.BytesIO(png))

def plot_scene(geo, color, tri, ax=None, resolution=(1024, 768)):
    mesh = Mesh(geo, color, tri)
    img = png_to_img(mesh_to_png(mesh, resolution=resolution))
    if ax != None:
        ax.imshow(img)
        # ax.axis('off')
    else:
        plt.imshow(img)
    return

def main():
    pca_id, pca_exp, color, tri = load_data()
    fig, axarr = plt.subplots(2, 2)
    for r in range(axarr.shape[0]):
        for c in range(axarr.shape[1]):
            geo = pca_id.sample() + pca_exp.sample()
            plot_scene(geo, color, tri, axarr[r, c])
    plt.show()
    return


if __name__ == '__main__':
    main()
