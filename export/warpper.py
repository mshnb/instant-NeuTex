import os
import os.path as osp
import sys
import pathlib
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

class Neutex():
    def __init__(self, name, device=0):
        prev = os.getcwd()
        os.chdir(pathlib.Path(__file__).parent.absolute())

        opt = parse_params()
        opt.is_train = False
        opt.gpu_ids = [device]
        opt.resume_dir = osp.join(opt.checkpoints_dir, name)

        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.eval()
        
        self.nets = self.model.get_subnetworks()
        self.device = device

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
        samples_uv = torch.empty([ray_steps, ray_resolution, 2], device=self.device)

        T = torch.ones(ray_resolution, device=self.device)
        surface_w = torch.zeros(ray_resolution, device=self.device)
        front_uv = torch.zeros([ray_resolution, 2], device=self.device)

        # fornt
        for step in range(ray_steps):
            pos = origin + dirs * dt * step
            density = samples_density[step]
            alpha = 1.0 - torch.exp(-density * dt)

            weight = alpha * T
            T *= 1.0 - alpha

            # ray_resolution, 2
            samples_uv[step] = self.nets['inverse_atlas'](pos)

            surface_i = weight > surface_w
            surface_w[surface_i] = weight[surface_i]
            front_uv[surface_i] = samples_uv[step][surface_i]

        T = torch.ones(ray_resolution, device=self.device)
        surface_w = torch.zeros(ray_resolution, device=self.device)
        back_uv = torch.zeros([ray_resolution, 2], device=self.device)

        # back
        for step in range(ray_steps-1, -1, -1):
            pos = origin + dirs * dt * step
            density = samples_density[step]
            alpha = 1.0 - torch.exp(-density * dt)

            weight = alpha * T
            T *= 1.0 - alpha

            surface_i = weight > surface_w
            surface_w[surface_i] = weight[surface_i]
            back_uv[surface_i] = samples_uv[step][surface_i]
        
        return front_uv, back_uv, dirs

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
        surface_w = torch.zeros(ray_resolution, device=self.device)
        front_uv = torch.zeros([ray_resolution, 2], device=self.device)
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

            surface_i = weight > surface_w
            surface_w[surface_i] = weight[surface_i]

            front_uv[surface_i] = (self.nets['inverse_atlas'](pos[surface_i]) + 1.0) * 0.5
            front_normal[surface_i] = outputs['normal'][surface_i]
        
        # ignore rays which do not hit object surface
        mask = acc_density / ray_steps > 1.0
        return front_uv[mask], front_normal[mask] 

    def generate_normal_tex(self, tex_size, mutate_steps):
        normal_tex = torch.zeros(tex_size, tex_size, 4, device=self.device)

        for m in tqdm(torch.linspace(0, 2, steps=mutate_steps)):
            uv, normal = self.extract_normal(
                ray_resolution=2**13,
                ray_steps=512,
                mutation=m
            )

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