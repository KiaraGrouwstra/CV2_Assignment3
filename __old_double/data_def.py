import numpy as np


class Mesh:
    def __init__(self, vertices, colors, triangles):
        assert triangles.shape[1] == 3
        assert vertices.shape[1] == 3
        assert colors.shape[1] == 3
        assert vertices.shape[0] == colors.shape[0]
        
        self.vertices = vertices
        self.colors = colors
        self.triangles = triangles


class PCAModel:
    def __init__(self, mean, pc, std):
        self.mean = mean
        self.pc = pc
        self.std = std

    def clip(self, n):
        self.pc = self.pc[:, :, :min(self.pc.shape[2], n)]
        self.std = self.std[:min(self.pc.shape[0], n)]

    def sample(self, epsilon=None):
        if epsilon == None:
            epsilon = np.random.uniform(-1.0, 1.0)
        return self.mean + self.pc.dot(epsilon * self.std)
