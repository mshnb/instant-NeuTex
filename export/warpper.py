import os
import os.path as osp
import pathlib
import sys

import mcubes
import numpy as np
import open3d
import trimesh
from packaging import version as pver

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), ".."))

import math
import torch
import torch.nn.functional as F
# import numpy as np
import random
from tqdm import tqdm
from models import create_model
from configparser import ConfigParser
from options import TrainOptions

def clamp(x, l, r):
    return min(max(l, x), r)

def update_pixel(pixels, x, y, value, alpha=1.0):
    if pixels[y][x][-1] > 0:
        prev = pixels[y][x][0:3]

        cosine = (prev * value).sum()
        if cosine.item() < 1e-1:
            return

        pixel = (1 - alpha) * prev + alpha * value
    else:
        pixel = value

    pixels[y][x][0:3] = F.normalize(pixel, dim=0, eps=1e-6)
    pixels[y][x][-1] = 1

def parse_params():
    config_path = './config.ini'
    if not osp.exists(config_path):
        print('can\'t find config.ini')
        exit()

    config = ConfigParser()
    config.read(config_path)

    params = [f'--{k}={v}' for k,v in dict(config['config']).items()]
    opt = TrainOptions().parse(params=params)
    return opt


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class Neutex():
    def __init__(self, dataset, name, device=0):
        prev = os.getcwd()
        os.chdir(pathlib.Path(__file__).parent.absolute())

        opt = parse_params()
        opt.is_train = False
        opt.gpu_ids = [device]
        opt.resume_dir = osp.join(opt.checkpoints_dir, name)
        opt.data_root = dataset

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        
        self.nets = self.model.get_subnetworks()
        self.device = device
        self.uv_dim = 2 if opt.primitive_type == 'square' else 3

        os.chdir(prev)

    def trace(self, ray_resolution, ray_steps, mutation):
        # generate rays
        phi = torch.rand(ray_resolution, device=self.device) * math.pi
        theta = torch.rand(ray_resolution, device=self.device) * math.pi * 2

        o_r = torch.sin(phi)
        o_y = torch.cos(phi)
        o_x = o_r * torch.cos(theta)
        o_z = o_r * torch.sin(theta)

        origin = torch.stack([o_x, o_y, o_z], dim=1)
        dirs = -origin

        if mutation > 0:
            mutated = mutation * (torch.rand([ray_resolution, 3], device=self.device) * 2 - 1)
            dirs = mutated - origin
            dirs = F.normalize(dirs, dim=-1, eps=1e-6)

        dt = 2 / ray_steps

        samples_density = torch.empty([ray_steps, ray_resolution], device=self.device)

        # collect samples
        for step in range(ray_steps):
            # ray_resolution, 3
            pos = origin + dirs * dt * step

            # ray_resolution, 1
            sigma = self.nets['nerf'](pos)['density']
            samples_density[step] = sigma.squeeze()
        
        # ignore rays which do not hit object surface
        mask = samples_density.sum(dim=0) / ray_steps > 1.0
        samples_density = samples_density[:, mask]
        origin = origin[mask, :]
        dirs = dirs[mask, :]

        ray_resolution = samples_density.shape[-1]
        samples_uv = torch.empty([ray_steps, ray_resolution, self.uv_dim], device=self.device)

        T = torch.ones(ray_resolution, device=self.device)
        # surface_w = torch.zeros(ray_resolution, device=self.device)
        front_uv = torch.zeros([ray_resolution, self.uv_dim], device=self.device)

        # fornt
        for step in range(ray_steps):
            pos = origin + dirs * dt * step
            density = samples_density[step]
            alpha = 1.0 - torch.exp(-density * dt)

            weight = alpha * T
            T *= 1.0 - alpha

            # ray_resolution, 2
            # samples_uv[step] = self.nets['inverse_atlas'](pos)
            # surface_i = weight > surface_w
            # surface_w[surface_i] = weight[surface_i]
            # front_uv[surface_i] = samples_uv[step][surface_i]

            samples_uv[step] = self.nets['inverse_atlas'](pos)
            front_uv += weight[..., None] * samples_uv[step]

        T = torch.ones(ray_resolution, device=self.device)
        # surface_w = torch.zeros(ray_resolution, device=self.device)
        back_uv = torch.zeros([ray_resolution, self.uv_dim], device=self.device)

        # back
        for step in range(ray_steps-1, -1, -1):
            pos = origin + dirs * dt * step
            density = samples_density[step]
            alpha = 1.0 - torch.exp(-density * dt)

            weight = alpha * T
            T *= 1.0 - alpha

            # surface_i = weight > surface_w
            # surface_w[surface_i] = weight[surface_i]
            # back_uv[surface_i] = samples_uv[step][surface_i]

            back_uv += weight[..., None] * samples_uv[step]
        
        return F.normalize(front_uv, dim=-1, eps=1e-6), F.normalize(back_uv, dim=-1, eps=1e-6), dirs

    def generate_data(self, batch_size, ray_steps):
        with torch.no_grad():

            input_uv = []
            input_dir = []
            target_uv = []

            ray_resolution = batch_size
            current_size = 0
            while current_size < batch_size:
                mutation = random.random()

                front_uv, back_uv, dirs = self.trace(
                    ray_resolution=ray_resolution,
                    ray_steps=ray_steps, 
                    mutation=mutation
                )

                input_uv.append(front_uv)
                input_dir.append(dirs)
                target_uv.append(back_uv)

                current_size += front_uv.shape[0]

                if ray_resolution == batch_size:
                    ray_resolution = batch_size // 2

            input_uv = torch.cat(input_uv, dim=0)
            input_dir = torch.cat(input_dir, dim=0)
            target_uv = torch.cat(target_uv, dim=0)

            if current_size > batch_size:
                input_uv = input_uv[:batch_size, ...]
                input_dir = input_dir[:batch_size, ...]
                target_uv = target_uv[:batch_size, ...]
            
        return input_uv, input_dir, target_uv

    def extract_normal(self, ray_resolution, ray_steps, mutation):
        phi = torch.rand(ray_resolution, device=self.device) * math.pi
        theta = torch.rand(ray_resolution, device=self.device) * math.pi * 2

        o_r = torch.sin(phi)
        o_y = torch.cos(phi)
        o_x = o_r * torch.cos(theta)
        o_z = o_r * torch.sin(theta)

        origin = torch.stack([o_x, o_y, o_z], dim=1)
        dirs = -origin

        if mutation > 0:
            mutated = mutation * (torch.rand([ray_resolution, 3], device=self.device) * 2 - 1)
            dirs = mutated - origin
            dirs = F.normalize(dirs, dim=-1, eps=1e-6)

        dt = 2 / ray_steps

        T = torch.ones(ray_resolution, device=self.device)
        # surface_w = torch.zeros(ray_resolution, device=self.device)
        front_uv = torch.zeros([ray_resolution, self.uv_dim], device=self.device)
        front_normal = torch.zeros([ray_resolution, 3], device=self.device)
        acc_density = torch.zeros(ray_resolution, device=self.device) 

        # collect samples
        for step in range(ray_steps):
            pos = origin + dirs * dt * step

            outputs = self.nets['nerf'](pos)
            sigma = outputs['density'].squeeze()

            acc_density += sigma
            alpha = 1.0 - torch.exp(-sigma * dt)
            weight = alpha * T
            T *= 1.0 - alpha

            # surface_i = weight > surface_w
            # surface_w[surface_i] = weight[surface_i]
            # front_uv[surface_i] = self.nets['inverse_atlas'](pos[surface_i])
            # front_normal[surface_i] = outputs['normal'][surface_i]

            front_uv += weight[..., None] * self.nets['inverse_atlas'](pos)
            front_normal += weight[..., None] * outputs['normal']
        
        # ignore rays which do not hit object surface
        mask = acc_density / ray_steps > 1.0
        return front_uv[mask], F.normalize(front_normal[mask], dim=-1, eps=1e-6) 

    # [-1, 1] to [0, 1]
    def normalize_normal(self, uv):
        if uv.shape[-1] == 3:
            scale = 1 / math.pi
            x, y, z = torch.split(uv, [1, 1, 1], dim=-1)
            phi = torch.acos(y) * scale
            theta = (torch.atan2(z, x) * scale + 1) * 0.5
            theta = torch.fmod(theta + 0.5, 1)
            uv = torch.cat([theta, phi], dim=-1)
        else:
            uv = (uv + 1.0) * 0.5
        
        return uv

    def generate_normal_tex(self, tex_size, mutate_steps):
        normal_tex = torch.zeros(tex_size, tex_size, 4, device=self.device)

        for m in tqdm(torch.linspace(0, 2, steps=mutate_steps)):
            uv, normal = self.extract_normal(
                ray_resolution=2**13,
                ray_steps=1024,
                mutation=m
            )

            uv = self.normalize_normal(uv)

            for i in range(uv.shape[0]):
                pos_screen = uv[i] * tex_size

                left = clamp(math.floor(pos_screen[0]), 0, tex_size-1)
                right = clamp(math.ceil(pos_screen[0]), 0, tex_size-1)
                top = clamp(math.floor(pos_screen[1]), 0, tex_size-1)
                bottom = clamp(math.ceil(pos_screen[1]), 0, tex_size-1)

                n = normal[i]

                # TODO: use distance related weights instead of constant 
                update_pixel(normal_tex, left, top, n, 0.5)
                update_pixel(normal_tex, left, bottom, n, 0.5)
                update_pixel(normal_tex, right, bottom, n, 0.5)
                update_pixel(normal_tex, right, top, n, 0.5)
        
        mask = normal_tex[..., -1] > 0
        normal_tex[mask] = (normal_tex[mask] + 1) * 0.5
        return normal_tex.cpu()

    def trace_surface(self, ray_o, ray_d, t_min, t_max, steps=1024):
        batch_size = ray_o.shape[0]
        ray_o = ray_o + ray_d * t_min
        dt = (t_max - t_min) / steps

        T = torch.ones(batch_size, device=self.device)
        # surface_w = torch.zeros(batch_size, device=self.device)
        # front_uv = torch.zeros([batch_size, self.uv_dim], device=self.device)
        # front_normal = torch.zeros([batch_size, 3], device=self.device)
        integrated_front_uv = torch.zeros([batch_size, self.uv_dim], device=self.device)
        integrated_normal = torch.zeros([batch_size, 3], device=self.device)
        integrated_pos = torch.zeros([batch_size, 3], device=self.device)
        acc_density = torch.zeros(batch_size, device=self.device)

        # collect samples
        for step in range(steps):
            pos = ray_o + ray_d * dt * step

            outputs = self.nets['nerf'](pos)
            sigma = outputs['density'].squeeze()
            normal = outputs['normal']
            uv = self.nets['inverse_atlas'](pos)

            acc_density += sigma
            alpha = 1.0 - torch.exp(-sigma * dt)
            weight = alpha * T
            T *= 1.0 - alpha

            # surface_i = weight > surface_w
            # surface_w[surface_i] = weight[surface_i]
            # front_uv[surface_i] = self.nets['inverse_atlas'](pos[surface_i])
            # front_normal[surface_i] = normal[surface_i]

            integrated_front_uv += weight[..., None] * uv
            integrated_normal += weight[..., None] * normal
            integrated_pos += weight[..., None] * pos

        integrated_normal = F.normalize(integrated_normal, dim=-1, eps=1e-6)
        # ignore rays which do not hit object surface
        sample_valid = acc_density / steps > 1.0
        return integrated_front_uv, integrated_normal, sample_valid, integrated_pos

    def generate_normal_tex_by_sampling(self, tex_size, jitter=16):
        flat_size = tex_size * tex_size
        alpha = torch.zeros(flat_size, 1, device=self.device)
        normal_tex = torch.zeros(flat_size, 3, device=self.device)
        position_tex = torch.zeros(flat_size, 3, device=self.device)

        scale = 1 / tex_size
        base = torch.arange(0, tex_size, device=self.device).float()
        for j in tqdm(range(jitter)):
            samples_x = (base + torch.rand_like(base)) * scale
            samples_y = (base + torch.rand_like(base)) * scale
            samples = torch.stack(torch.meshgrid(samples_x, samples_y, indexing='xy'), dim=-1)
            samples = samples.view(-1, 2)

            if self.uv_dim == 3:
                theta, phi = torch.split(math.pi * samples, [1, 1], dim=-1)
                theta = 2 * theta

                o_r = torch.sin(phi)
                o_y = torch.cos(phi)
                o_x = o_r * torch.cos(theta)
                o_z = o_r * torch.sin(theta)

                samples = torch.cat([o_x, o_y, o_z], dim=-1)    

            #uv2pos
            pos = self.nets['atlas'].map(samples)
            #pos2normal
            outputs = self.nets['nerf'](pos)
            sigma = outputs['density'].squeeze()
            normal = outputs['normal']

            mask = sigma > 1e-4
            normal_tex[mask] += normal[mask]
            position_tex[mask] += pos[mask]
            alpha[mask] += 1
        
        mask = alpha.squeeze() > 0
        alpha[mask] = 1.0 / alpha[mask]
        normal_tex *= alpha
        position_tex *= alpha
        alpha[mask] = 1

        normal_tex[mask] = (F.normalize(normal_tex[mask], dim=-1, eps=1e-6) + 1) * 0.5
        return normal_tex, position_tex

    def get_pcd(self):
        normals, points = self.generate_normal_tex_by_sampling(512)
        points = points.cpu().numpy().astype(np.float64).reshape(-1, 3)
        normals = normals.cpu().numpy().astype(np.float64).reshape(-1, 3)

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points)
        pcd.normals = open3d.utility.Vector3dVector(normals)
        # pcd.colors = open3d.utility.Vector3dVector(colors)
        self.pcd = pcd

    def visualization_pcd(self):
        open3d.visualization.draw_geometries([self.pcd], width=1280, height=720)

    def dump_pcd(self, path='./mesh.pcd'):
        open3d.io.write_point_cloud(path, self.pcd)

    def get_mesh(self):
        mesh = trimesh.creation.icosphere(8)
        grid = torch.tensor(mesh.vertices).to(self.device).float()
        grid = grid.view(-1, 3)
        pts = self.nets['atlas'].map(grid)
        outputs = self.nets['nerf'](pts)
        normal = outputs['normal'].cpu().numpy()
        mesh.vertices = pts.cpu().numpy()

        self.mesh = mesh
        self.mesh.visual.vertex_colors = normal / 2 + 0.5

    def get_nerf_mesh(self, resolution=256, threshold=10):
        def query_func(pts):
            sigmas = []
            sigmas.append(self.nets['nerf'](pts.to(self.device))['density'])
            return torch.stack(sigmas).mean(axis=0)

        vertices, triangles = extract_geometry(torch.tensor([-1, -1, -1]), torch.tensor([1, 1, 1]),
                                               resolution=resolution, threshold=threshold, query_func=query_func)

        self.mesh = trimesh.Trimesh(vertices, triangles, process=False)

    def visualization_mesh(self, textured=False):
        # if textured:
        #     self.mesh.visual.vertex_colors = self.mesh_color
        # else:
        #     self.mesh.visual.vertex_colors = np.ones_like(self.mesh_color)
        self.mesh.show(viewer="gl", smooth=True, resolution=[1280, 720])

    def dump_mesh(self, path='./mesh.ply'):
        self.mesh.export(path)
