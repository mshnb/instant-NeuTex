import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from gridencoder import GridEncoder
from shencoder import SHEncoder
from ..networks import init_seq, positional_encoding
from utils.grid import generate_grid
from utils.cube_map import (
    convert_cube_uv_to_xyz,
    load_cube_from_single_texture,
    load_cubemap,
    sample_cubemap,
)
from PIL import Image

def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def torch_logit(x):
    return torch.log(x / (1 - x + 1e-5))

class TextureViewMlp(nn.Module):
    def __init__(
            self, uv_dim, out_channels, view_freqs, layers, width, clamp, pred_normal=False, use_bias=False
    ):
        super().__init__()
        self.uv_dim = uv_dim
        # self.view_freqs = max(view_freqs, 0)
        self.channels = out_channels

        self.uv_encoder = GridEncoder(
            input_dim=uv_dim,
            num_levels=16, level_dim=2,
            log2_hashmap_size=19,
            base_resolution=16, desired_resolution=2048
        )

        block1 = []
        block1.append(nn.Linear(32, width, bias=use_bias))
        block1.append(nn.ReLU())
        for i in range(layers[0]):
            block1.append(nn.Linear(width, width, bias=use_bias))
            block1.append(nn.ReLU())
        block1.append(nn.Linear(width, self.channels, bias=use_bias))
        self.block1 = nn.Sequential(*block1)
        init_seq(self.block1)

        self.dir_encoder = SHEncoder(input_dim=3, degree=4)

        block2 = []
        block2.append(nn.Linear(32 + 16, width, bias=use_bias))
        block2.append(nn.ReLU())
        for i in range(layers[1]):
            block2.append(nn.Linear(width, width, bias=use_bias))
            block2.append(nn.ReLU())
        block2.append(nn.Linear(width, self.channels, bias=use_bias))
        self.block2 = nn.Sequential(*block2)
        init_seq(self.block2)

        self.cubemap_ = None
        self.clamp_texture = clamp

        self.block_normal = None
        # if pred_normal:
        #     block3 = []
        #     block3.append(nn.Linear(32, width, bias=use_bias))
        #     block3.append(nn.ReLU())
        #     for i in range(2):
        #         block3.append(nn.Linear(width, width, bias=use_bias))
        #         block3.append(nn.ReLU())
        #     block3.append(nn.Linear(width, 3, bias=use_bias))
        #     self.block_normal = nn.Sequential(*block3)
        #     init_seq(self.block_normal)


    def forward(self, uv, view_dir):
        """
        Args:
            uvs: :math:`(N,Rays,Samples,U)`
            view_dir: :math:`(N,Rays,Samples,3)`
        """
        h = self.uv_encoder(uv)

        # normal = None
        # if self.block_normal is not None:
        #     normal = self.block_normal(h)
        #     normal = F.normalize(normal, dim=-1, eps=1e-6)

        diffuse = self.block1(h)
        if self.clamp_texture:
            diffuse = torch.sigmoid(diffuse)
        else:
            diffuse = F.softplus(diffuse)

        view_dir = view_dir.expand(diffuse.shape[:-1] + (view_dir.shape[-1],))
        # vp = positional_encoding(view_dir, self.view_freqs)
        vp = self.dir_encoder(view_dir)
        h = torch.cat([h, vp], dim=-1)
        specular = self.block2(h)
        if self.clamp_texture:
            specular = torch.sigmoid(specular)
        else:
            specular = F.softplus(specular)

        return {
            'color': diffuse + specular, 
            # 'normal': normal
        }

    def _export_cube(self, resolution, viewdir):
        device = next(self.block1.parameters()).device

        grid = torch.tensor(generate_grid(2, resolution)).float().to(device)
        textures = []
        for index in range(6):
            xyz = convert_cube_uv_to_xyz(index, grid)

            if viewdir is not None:
                view = torch.tensor(viewdir).float().to(device).expand_as(xyz)
                textures.append(self.forward(xyz, view))
            else:
                out = self.block1(
                    torch.cat([xyz, positional_encoding(xyz, self.num_freqs)], dim=-1)
                )
                textures.append(torch.sigmoid(self.color1(out)))

        return torch.stack(textures, dim=0)

    def _export_sphere(self, resolution, viewdir):
        with torch.no_grad():
            device = next(self.block1.parameters()).device

            grid = np.stack(
                np.meshgrid(
                    np.arange(2 * resolution), np.arange(resolution), indexing="xy"
                ),
                axis=-1,
            )
            grid = grid / np.array([2 * resolution, resolution]) * np.array(
                [2 * np.pi, np.pi]
            ) + np.array([np.pi, 0])
            x = grid[..., 0]
            y = grid[..., 1]
            xyz = np.stack(
                [-np.sin(x) * np.sin(y), -np.cos(y), -np.cos(x) * np.sin(y)], -1
            )
            xyz = torch.tensor(xyz).float().to(device)

            if viewdir is not None:
                view = torch.tensor(viewdir).float().to(device).expand_as(xyz)
                texture = self.forward(xyz, view)
            else:
                out = self.block1(
                    torch.cat([xyz, positional_encoding(xyz, self.num_freqs)], dim=-1)
                )
                texture = torch.sigmoid(self.color1(out))
            return texture['color'].flip(0)

    def _export_square(self, resolution, viewdir):
        with torch.no_grad():
            device = next(self.block1.parameters()).device

            grid = torch.tensor(generate_grid(2, resolution)).float().to(device)

            if viewdir is not None:
                view = (
                    torch.tensor(viewdir).float().to(device).expand(grid.shape[:-1] + (3,))
                )
                texture = self.forward(grid, view)
            else:
                out = self.block1(
                    torch.cat([grid, positional_encoding(grid, self.num_freqs)], dim=-1)
                )
                texture = torch.sigmoid(self.color1(out))

            return texture['color']

    def export_textures(self, resolution=512, viewdir=[0, 0, 1]):
        if self.uv_dim == 3:
            return self._export_sphere(resolution, viewdir)
        else:
            return self._export_square(resolution, viewdir)

    def import_cubemap(self, filename, mode=0):
        assert self.uv_dim == 3
        device = next(self.block1.parameters()).device
        if isinstance(filename, str):
            w, h = np.array(Image.open(filename)).shape[:2]
            if w == h:
                cube = load_cubemap([filename] * 6)
            else:
                cube = load_cube_from_single_texture(filename)
        else:
            cube = load_cubemap(filename)
        self.cubemap_ = torch.tensor(cube).float().to(device)
        self.cubemap_mode_ = mode
