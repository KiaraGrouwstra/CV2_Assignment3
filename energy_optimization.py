"""
Optimize the energy function for sequences of images.
Your energy optimization problem is non-convex.
If your initial estimation is too far away from local optima, your solution may not be optimal.
Code from (Section 3, Q2) can be used to see how far you are from the original face.
"""
import os
import glob
import matplotlib.pyplot as plt
import torch

from landmarks import file_landmarks, plot_landmarks
from latent_param_estimation import estimate_points, load_morphace

if __name__ == "__main__":
    (texture, identity, expression, triangles) = load_morphace()

    # get pic landmarks
    faces_folder_path = 'pics'
    files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    landmarks_pics = estimate_points(files, identity, expression)

    # Visualize predicted landmarks overlayed on ground truth.
    for landmarks, fpath in zip(landmarks_pics, files):
        ground_truth = file_landmarks(fpath)
        plot_landmarks([ground_truth, landmarks])
        plt.savefig('results/multi_estimation_' + os.path.basename(fpath))
