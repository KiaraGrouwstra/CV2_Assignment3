"""estimate facial geometry latent parameters α,δ and object transformation ω, t for a specific 2D image with a human face using Energy minimization."""
import sys
import os
import dlib
import glob
import numpy as np
import load_landmarks from utils
import shape_to_np, detect_landmark, file_landmarks from landmarks
import rotation_matrix_y from pinhole_camera
from tqdm import tqdm

def project_face(G, omega, t):
    (num_points, _) = G.shape
    S = np.vstack((G.T, np.ones(num_points)))
    R = rotation_matrix_y(omega)
    G_ = (R @ S)[:3].T
    points = project_points(G_, near=1, translation=t)
    return points

def estimate_loss(vertex_idxs, ground_truth, identity, expression, alpha, delta, omega, t, lambda_alpha, lambda_delta):
    """calculate the loss for a specific 2D image with a human face"""
    # vertex_idxs
    G = reconstruct_face(identity, expression, alpha, delta)
    points = project_face(G, omega, t)
    # Given 68 ground truth facial landmarks the following energy can be optimized: Lfit=Llan+Lreg(3)Llan=68∑j=1∥∥pkibj−lj∥∥22(4)where pkj is a 2D projection of a landmark point kj from Landmarks68_model2017-1_face12_nomouth.anl and lj is its ground truth 2D coordinate.
    L_lan = (points - ground_truth).norm().pow(2).sum()
    # We regularize the model using Tikhonov regularization to enforce the model to predict faces closer to the mean: Lreg=λalpha30∑i=1α2i+λdelta20∑i=1δ2i(5)
    L_reg = (lambda_alpha * alpha).pow(2).sum() + (lambda_delta * delta).pow(2).sum()
    L_fit = L_lan + L_reg
    return L_fit

if __name__ == "__main__":
    # hyper-parameters
    lambda_alpha = 0.5
    lambda_delta = 0.5

    # load data
    vertex_idxs = load_landmarks()
    (texture, identity, expression, triangles) = load_data()

    # get pic landmarks
    faces_folder_path = 'pics'
    files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    # landmarks_pics = map(file_landmarks, tqdm(files))
    for f in files:
        landmarks = file_landmarks(f)
        print(landmarks)

        # alpha
        # delta
        # omega
        # t  # e.g. translation = (0, 0, -200)

        loss = estimate_loss(vertex_idxs, landmarks_pics, identity, expression, alpha, delta, omega, t, lambda_alpha, lambda_delta)
        print(loss)
        # - Assuming α, δ, ω, t to be latent parameters of your model optimize an Energy described above using Adam optimizer until convergence.
        # model = ?
        learning_rate = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Visualize predicted landmarks overlayed on ground truth.
        # - Hint: initializing transformation parameters ω and t closer to the solution may help with convergence. For example translation over z dimension can be set to be -400 in the case of projection matrix with principal point {W2, H2} and fovy = 0.5.
        # - Select hyper parameters such that α and δ to be obtained in a proper range. Report findings.
