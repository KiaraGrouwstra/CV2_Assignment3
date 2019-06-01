"""
optimize the energy function for sequences of images
Your energy optimization problem is non-convex. If your initial estimation is too far away from local optima, your solution may not be optimal. Code from (Section 3, Q2) can be used to see how far you are from the original face.
"""
import sys
import os
import dlib
import glob
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from landmarks import file_landmarks, plot_landmarks
from utils import load_data, load_landmarks, reconstruct_face
from latent_param_estimation import reconstruct_face, rotation_matrix_y, viewport_matrix, perspective_matrix, project_points, project_face, estimate_points, Model

if __name__ == "__main__":
    # load data, filter to 68 landmarks
    vertex_idxs = load_landmarks()
    (texture, identity, expression, triangles) = load_data()
    for pca in (identity, expression, texture):
        pca.mean = pca.mean[vertex_idxs]
        pca.pc   = pca.pc  [vertex_idxs]

    # get pic landmarks
    faces_folder_path = 'pics'
    files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    landmarks_pics = estimate_points(files, identity, expression)

    # Visualize predicted landmarks overlayed on ground truth.
    for landmarks, fpath in zip(landmarks_pics, files):
        ground_truth = file_landmarks(fpath)
        plot_landmarks([ground_truth, landmarks])
        plt.savefig('results/multi_estimation_' + os.path.basename(fpath))
