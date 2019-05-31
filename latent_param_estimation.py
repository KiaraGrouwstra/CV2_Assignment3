"""estimate facial geometry latent parameters α,δ and object transformation ω, t for a specific 2D image with a human face using Energy minimization."""
import sys
import os
import dlib
import glob
import numpy as np
import load_landmarks from utils
import shape_to_np, detect_landmark, file_landmarks from landmarks
import rotation_matrix_y from pinhole_camera
from tqdm import tqdm, trange
import torch
from torch.autograd import Variable

def project_face(G, omega, t):
    (num_points, _) = G.shape
    S = np.vstack((G.T, np.ones(num_points)))
    R = rotation_matrix_y(omega)
    G_ = (R @ S)[:3].T
    points = project_points(G_, near=1, translation=t)
    return points

class Model(nn.Module):

    def __init__(self, landmarks_pics, identity, expression, lambda_alpha=0.5, lambda_delta=0.5):
        super(Model).__init__()

        # data
        self.landmarks_pics = landmarks_pics
        self.identity = identity
        self.expression = expression

        # hyper-parameters
        self.lambda_alpha = lambda_alpha
        self.lambda_delta = lambda_delta

        # weight parameters
        self.alpha
        self.delta
        self.omega
        self.t  # e.g. translation = (0, 0, -200)

    def forward(self, ground_truth, identity, expression, alpha, delta, omega, t):
        """calculate the loss for a specific 2D image with a human face"""
        G = reconstruct_face(self.identity, self.expression, self.alpha, self.delta)
        points = project_face(G, self.omega, self.t)
        # Given 68 ground truth facial landmarks the following energy can be optimized: Lfit=Llan+Lreg(3)Llan=68∑j=1∥∥pkibj−lj∥∥22(4)where pkj is a 2D projection of a landmark point kj from Landmarks68_model2017-1_face12_nomouth.anl and lj is its ground truth 2D coordinate.
        L_lan = (points - self.ground_truth).norm().pow(2).sum()
        # We regularize the model using Tikhonov regularization to enforce the model to predict faces closer to the mean: Lreg=λalpha30∑i=1α2i+λdelta20∑i=1δ2i(5)
        L_reg = (self.lambda_alpha * self.alpha).pow(2).sum() + (self.lambda_delta * self.delta).pow(2).sum()
        L_fit = L_lan + L_reg
        return L_fit


if __name__ == "__main__":

    # - Landmarks are a subset of vertices from the morphable model (indexes are defined by the annotation file provided), that's why you are inferring landmarks.
    # load data, filter to 68 landmarks
    # TODO: does this clash with the 30/20 filter?
    vertex_idxs = load_landmarks()
    (texture, identity, expression, triangles) = load_data()
    for pca in (identity, expression, texture):
        pca.mean = pca.mean[vertex_idxs]
        pca.pc   = pca.pc  [vertex_idxs]

    # get pic landmarks
    faces_folder_path = 'pics'
    files = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    for f in tqdm(files):
        landmarks = file_landmarks(f)
        print(landmarks)

        lr = 0.1
        model = Model(landmarks_pics, identity, expression)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # - Assuming α, δ, ω, t to be latent parameters of your model optimize an Energy described above using Adam optimizer until convergence.
        for i in trange(100):
            opt.zero_grad()
            loss = model.forward()
            print(i, loss)
            loss.backward()
            opt.step()

        print(model)
        # A = Variable(torch.ones(1, 10), requires_grad=True)

        # Visualize predicted landmarks overlayed on ground truth.
        # - Hint: initializing transformation parameters ω and t closer to the solution may help with convergence. For example translation over z dimension can be set to be -400 in the case of projection matrix with principal point {W2, H2} and fovy = 0.5.
        # - Select hyper parameters such that α and δ to be obtained in a proper range. Report findings.
