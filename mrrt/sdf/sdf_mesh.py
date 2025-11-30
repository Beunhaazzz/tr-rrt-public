import os
import tqdm
import numpy as np
from spatialmath import SE3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model import *
from .train import *
from .dataset import *
from .to_mesh import *


class SDFMesh(object):
    """
    Class for an SDF representation of a mesh
    """
    def __init__(self, mesh_path, device, vis=None):
        self.mesh_path = mesh_path
        self.device = device
        self.vis = vis
        self.model = SDFModel().to(self.device)
        self.sampling = None
        self.mesh = None
    
    def fit(self, num_samples=1000000, num_epochs=100, lr=1e-4, batch_size=128):
        self.train_dataset = SDFDataset(
            mesh_path=self.mesh_path, num_samples=num_samples)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, 
            num_workers=1, pin_memory=True, shuffle=True)
        self._update_model_scale()

        self.criterion = nn.L1Loss(reduction='sum')
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_losses = []
        for epoch in range(num_epochs):
            if self.vis is not None:
                self.vis.text('Epoch: {}/{} ({:.2f}%)'.format(epoch + 1, num_epochs, (epoch + 1) / num_epochs * 100.0), opts={'title': "Current epoch"}, win='curr_epoch')
            train(self.train_loader, self.model, self.device,
                self.criterion, self.optimizer, epoch, train_losses, self.vis)
            if epoch % 10 == 0 or True:
                torch.save(self.model.state_dict(), self.mesh_path + '.pth')
        torch.save(self.model.state_dict(), self.mesh_path + '.pth')
        return train_losses

    def load(self):
        self._update_model_scale()
        pth_path = self.mesh_path + '.pth'
        if not os.path.isfile(pth_path):
            puzzle = os.path.basename(os.path.dirname(self.mesh_path))
            part = os.path.basename(self.mesh_path)
            raise FileNotFoundError(
                f"Missing SDF weights: '{pth_path}'.\n"
                f"Generate them first, e.g.:\n"
                f"  py .\\scripts\\fit.py --name {puzzle} --device cpu\n"
                f"This will create weights for both parts (e.g., 0.obj.pth and 1.obj.pth)."
            )
        self.model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
        self.model.to(self.device)
        self.mesh = trimesh.load(self.mesh_path)
        self.pq = trimesh.proximity.ProximityQuery(self.mesh)

    def to_mesh(self, output_name, n=100):
        return sdf_to_mesh(self.model, n, self.device, self.centroid, self.max_norm, output_name)

    def generate_sampling(self, num_samples):
        if self.mesh is None:
            self.mesh = trimesh.load(self.mesh_path)
        self.sampling, _ = trimesh.sample.sample_surface(self.mesh, num_samples, face_weight=None)
       

    def _update_model_scale(self):
        mesh = trimesh.load(self.mesh_path)
        self.centroid = mesh.centroid
        self.max_norm = np.linalg.norm(np.max(mesh.vertices - mesh.centroid))
