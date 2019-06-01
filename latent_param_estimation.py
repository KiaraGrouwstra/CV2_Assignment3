"""estimate facial geometry latent parameters α,δ and object transformation ω, t for a specific 2D image with a human face using Energy minimization."""
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
# from pinhole_camera import rotation_matrix_y, viewport_matrix, perspective_matrix, project_points

# from utils import reconstruct_face
# from pinhole_camera import rotation_matrix_y, viewport_matrix, perspective_matrix, project_points

def rotation_matrix_y(y_deg):
    """Get the Y rotation matrix (https://bit.ly/2PQ8glW) for a given rotation angle (in degrees).
       Assuming object translation to be 0.
    """
    y_rad = y_deg / 180 * np.pi 
    R = torch.tensor([
        [torch.cos(y_rad), 0., torch.sin(y_rad), 0.],
        [0., 1., 0., 0.],
        [-torch.sin(y_rad), 0., torch.cos(y_rad), 0.],
        [0., 0., 0., 1.],
    ])
    return R


def viewport_matrix(l=-1, r=1, t=1, b=-1):
    """
    viewport matrix: http://glasnost.itcarlow.ie/~powerk/GeneralGraphicsNotes/projection/viewport_transformation.html
    @param l: left
    @param r: right
    @param t: top
    @param b: bottom
    """
    w = r - l
    h = t - b
    V = .5 * torch.tensor([
        [w, 0., 0., 0.],
        [0., h, 0., 0.],
        [0., 0., 1., 0.],
        [r + l, t + b, 1., 2.],
    ])
    return V

def perspective_matrix(t, b, l, r, n, f):
    """
    perspective projection matrix: https://bit.ly/300gYmf
    @param t: top
    @param b: bottom
    @param l: left
    @param r: right
    @param n: near
    @param f: far
    """
    w = r - l
    h = t - b
    P = torch.tensor([
        [2. * n / w, 0., 0., 0.],
        [0., 2. * n / h, 0., 0.],
        [(r + l) / w, (t + b) / h, -(f + n) / (f - n), -1.],
        [0., 0., -2. * f * n / (f - n), 0.],
    ])
    return P


def project_points(S, near, R):
    """project points following equation 2"""
    P = perspective_matrix(1, -1, 1, -1, near, 100)
    ones = torch.ones((S.shape[0], 1))
    S = torch.cat((S, ones), dim=1)
    V = viewport_matrix()
    p = V.t() @ P.t() @ R @ S.t()
    return p.t()


# project_face(torch.tensor([[1, 2, 3]]).float(), torch.tensor(90).float(), torch.tensor((0, 0, -200)))
def project_face(G, omega, t):
    (num_points, _) = G.shape
    S = torch.cat((G.t(), torch.ones((1, num_points))))
    R = rotation_matrix_y(omega)
    G_ = (R @ S)[:3].t()
    # R[3, 0:3] = t
    R = torch.cat((R[:3], torch.cat((t, torch.tensor([1.]))).float().unsqueeze(dim=-1).t()))
    points = project_points(G_, near=1, R=R)
    return points

class Model(nn.Module):

    def __init__(self, ground_truth, identity, expression, lambda_alpha=0.5, lambda_delta=0.5):
        super(Model, self).__init__()

        # data
        self.ground_truth = torch.tensor(ground_truth).float()
        self.identity = identity
        self.expression = expression

        # hyper-parameters
        # TODO: Select hyper parameters such that α and δ to be obtained in a proper range. Report findings.
        self.lambda_alpha = lambda_alpha
        self.lambda_delta = lambda_delta

        # weight parameters
        # initializing transformation parameters ω and t closer to the solution may help with convergence. For example translation over z dimension can be set to be -400 in the case of projection matrix with principal point {W2, H2} and fovy = 0.5.
        # TODO: give np.random.uniform dimensions everywhere else as well?
        self.alpha =    nn.Parameter(torch.tensor(np.random.uniform(-1.0, 1.0, 30)).float())
        self.delta =    nn.Parameter(torch.tensor(np.random.uniform(-1.0, 1.0, 20)).float())
        self.omega =    nn.Parameter(torch.tensor(np.random.uniform(0.0, 10.0)))
        translation = (*np.random.uniform(-1.0, 1.0, 2), np.random.uniform(-400.0, 100.0))
        self.t =        nn.Parameter(torch.tensor(translation))

    # , ground_truth, identity, expression, alpha, delta, omega, t
    def forward(self):
        """calculate the loss for a specific 2D image with a human face"""

        # self.G = reconstruct_face(self.identity, self.expression, self.alpha, self.delta)

        # geom = self.identity  .sample(self.alpha)
        geom = torch.tensor(self.identity.mean) + torch.tensor(self.identity.pc) @ (self.alpha * torch.sqrt(torch.tensor(self.identity.std)))
        # expr = self.expression.sample(self.delta)
        expr = torch.tensor(self.expression.mean) + torch.tensor(self.expression.pc) @ (self.delta * torch.sqrt(torch.tensor(self.expression.std)))
        self.G = geom + expr

        self.points = project_face(self.G, self.omega, self.t)
        # Given 68 ground truth facial landmarks the following energy can be optimized: Lfit=Llan+Lreg(3)Llan=68∑j=1∥∥pkibj−lj∥∥22(4)where pkj is a 2D projection of a landmark point kj from Landmarks68_model2017-1_face12_nomouth.anl and lj is its ground truth 2D coordinate.
        L_lan = (self.points[:, 0:2] - self.ground_truth).norm().pow(2).sum()
        # We regularize the model using Tikhonov regularization to enforce the model to predict faces closer to the mean: Lreg=λalpha30∑i=1α2i+λdelta20∑i=1δ2i(5)
        L_reg = (self.lambda_alpha * self.alpha).pow(2).sum() + (self.lambda_delta * self.delta).pow(2).sum()
        L_fit = L_lan + L_reg
        return L_fit

def estimate_points(f, identity, expression):
    landmarks = file_landmarks(f)
    # print(landmarks)

    lr = 0.1
    model = Model(landmarks, identity, expression)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # - Assuming α, δ, ω, t to be latent parameters of your model optimize an Energy described above using Adam optimizer until convergence.
    for i in trange(100):
        optimizer.zero_grad()
        loss = model.forward()
        print(i, loss)
        loss.backward()
        optimizer.step()

    print(model)
    return model.points.detach().numpy()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

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
    # landmarks = file_landmarks(f)
    # ground_truths = list(map(file_landmarks, tqdm(files)))
    landmarks_pics = list(map(lambda f: estimate_points(f, identity, expression), tqdm(files)))

    # Visualize predicted landmarks overlayed on ground truth.
    for landmarks, fpath in zip(landmarks_pics, files):
        ground_truth = file_landmarks(fpath)
        print('plotting')
        plot_landmarks([ground_truth, landmarks])
        print('plotted')
        plt.savefig('estimation_' + os.path.basename(fpath))
