import trimesh
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

    def trimesh(self):
        """convert to Trimesh format"""
        return trimesh.base.Trimesh(
            vertices=self.vertices,
            faces=self.triangles,
            vertex_colors=self.colors)


class PCAModel:
    def __init__(self, mean, pc, std):
        self.mean = mean
        self.pc = pc
        self.std = std

    def sample(self, off=np.random.uniform(-1.0, 1.0)):
        return self.mean + self.pc @ (off * np.sqrt(self.std))

    def filter(self, n):
        return PCAModel(self.mean, self.pc[:, :, :n], self.std[:n])
