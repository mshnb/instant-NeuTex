import os
import os.path as osp
import sys
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), ".."))

import torch
import numpy as np
from models import create_model
from configparser import ConfigParser
from options import TrainOptions
import trimesh

import open3d
from matplotlib.colors import cnames, hex2color

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2

from argparse import ArgumentParser

def parse_params():
    if not osp.exists('./config.ini'):
        print('can\'t find config.ini')
        exit()

    config = ConfigParser()
    config.read('./config.ini')

    params = [f'--{k}={v}' for k,v in dict(config['config']).items()]
    opt = TrainOptions().parse(params=params)
    opt.is_train = False
    return opt

def generate_uv(img_size):
    coords = (torch.arange(0, img_size) + 0.5) / img_size
    x = coords.unsqueeze(dim=-1).repeat(1, img_size).unsqueeze(dim=-1)
    y = coords.unsqueeze(dim=0).repeat(img_size, 1).unsqueeze(dim=-1)
    return torch.cat([x,y], dim=-1)

#img: [512, 512, 3]
def write_exr(img, path):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = img[:,:,::-1]
    
    img = np.concatenate([img, np.ones_like(img[...,[0]])], axis=-1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

class warpper():
    def __init__(self, opt):
        self.net = create_model(opt)
        self.net.setup(opt)
        self.net.eval()

        #mesh
        meshes, textures = self.net.visualize_mesh_3d(icosphere_division=6)
        self.mesh = meshes[0]
        self.texture = torch.pow(textures[0], 1/2.2)
        color = (255 * self.texture.data.cpu().numpy().clip(0, 1)).astype(np.uint8)
        c = np.ones((len(color), 4)) * 255
        c[:, :3] = color

        self.mesh_color = c
        trimesh.repair.fix_inversion(self.mesh)
        trimesh.repair.fix_normals(self.mesh)

        zeros = torch.zeros(1, dtype=torch.long, device=self.net.device)
        self.geometry_embedding = self.net.net_nerf_atlas.module.net_geometry_embedding(zeros)

        # point cloud
        points, normals = self.net.visualize_atlas()
        points = points.data.cpu().numpy()
        normals = normals.data.cpu().numpy()
        colors = np.array([hex2color(v) for v in cnames.values()])
        pcd_colors = []
        for p, c in zip(points, colors):
            pcd_colors.append(c + np.zeros_like(p))
        pcd_points = np.concatenate(points, axis=0)
        pcd_colors = np.concatenate(pcd_colors, axis=0)
        pcd_normals = np.concatenate(normals, axis=0)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pcd_points)
        pcd.normals = open3d.utility.Vector3dVector(pcd_normals)
        pcd.colors = open3d.utility.Vector3dVector(pcd_colors)
        self.pcd = pcd

    def visualization_mesh(self, textured=False):
        if textured:
            self.mesh.visual.vertex_colors = self.mesh_color
        else:
            self.mesh.visual.vertex_colors = np.ones_like(self.mesh_color)
        self.mesh.show(viewer="gl", smooth=True, resolution=[1280,720])

    def dump_mesh(self, path='./mesh.ply'):
        self.mesh.export(path)

    def visualization_pcd(self):
        open3d.visualization.draw_geometries([self.pcd], width=1280, height=720)

    def dump_pcd(self, path='./mesh.pcd'):
        open3d.io.write_point_cloud(path, self.pcd)

    def uv2pos(self, uv):
        img_size = uv.shape[0:2]
        uv = uv.reshape(-1, 2)
        theta = uv[..., [0]] * 2 * np.pi
        phi = uv[..., [1]] * np.pi

        r = torch.sin(phi)
        x = torch.cos(theta) * r
        y = torch.cos(phi)
        z = torch.sin(theta) * r

        uv_sqhere = torch.cat([x, y, z], dim=-1).unsqueeze(dim = 0).permute(1, 0, 2)
        uv_sqhere = uv_sqhere.to(self.net.device)

        points, normals = self.net.net_nerf_atlas.module.net_atlasnet.map_and_normal(
            self.geometry_embedding, uv_sqhere
        )

        points = points.squeeze().reshape(*img_size, 3).cpu().numpy()
        normals = normals.squeeze().reshape(*img_size, 3).cpu().numpy()
        normals = normals / (np.linalg.norm(normals, axis=-1) + 1e-15)[..., None]

        return points, normals

def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=512, help='size of output texture')
    parser.add_argument('-v', '--visualization', action='store_true', help='show mesh in a window')
    parser.add_argument('-c', '--color', action='store_true', help='mapping color when showing mesh')
    args = parser.parse_args()

    with torch.no_grad():
        opt = parse_params()
        warpped = warpper(opt)

        if args.visualization:
            warpped.visualization_mesh(args.color)
        else:
            img_size = args.size
            uv = generate_uv(img_size)
            p, n = warpped.uv2pos(uv)

            write_exr(p, './position.exr')
            write_exr(n, './normal.exr')

            print('done.')

if __name__ == "__main__":
    main()