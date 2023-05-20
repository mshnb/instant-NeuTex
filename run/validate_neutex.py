from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--spp', type=int, default=1, help='')
parser.add_argument('-s', '--size', type=int, default=512, help='resolution of image')
parser.add_argument('-d', '--device', nargs='+', type=int, default=[], help='which device to use')
parser.add_argument('--neutex', type=str, default='maitreya', help='name of the neutex model')
parser.add_argument('--dataset', type=str, default='maitreya', help='the name of dataset')
parser.add_argument('--envmap', type=str, default='twilight', help='the path of envmap')

args = parser.parse_args()

devices = args.device
if len(devices) == 0:
    device = 0
else:
    # only use single gpu
    device = devices[0]

local_rank = device
import os
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
device = 0

import sys
import pathlib

sys.path.append(osp.join(pathlib.Path(__file__).parent.absolute(), '..'))
from export.warpper import Neutex

import math
import imageio
import torch
import torch.nn.functional as F
import numpy as np


def normalize(v):
    scale = 1 / (np.linalg.norm(v, axis=-1) + 1e-15)
    return v * scale[..., None]


def refract(i, n, eta):
    cos_theta_i = -torch.sum(i * n, dim=-1)
    cos_theta_t_sqr = 1 - eta * eta * (1 - torch.pow(cos_theta_i, 2))
    cos_theta_t_sqr = torch.clamp(cos_theta_t_sqr, 0, 1)
    cos_theta_t = torch.sqrt(cos_theta_t_sqr)
    return eta * i + (eta * cos_theta_i - cos_theta_t)[..., None] * n


class Camera():
    def __init__(self, origin, target, up, fov, near=0.1, far=100):
        self.origin = np.asarray(origin).astype(np.float32)
        self.target = np.asarray(target).astype(np.float32)
        self.up = np.asarray(up).astype(np.float32)

        self.fov = fov * np.pi / 180
        self.near = near
        self.far = far

        self.dist_z = abs(self.origin[2] - self.target[2])

    def rotate(self, r):
        self.origin[0] = self.target[0] + self.dist_z * math.sin(r)
        self.origin[2] = self.target[2] + self.dist_z * math.cos(r)

    def lookat(self):
        dir = normalize(self.target - self.origin)
        left = normalize(np.cross(self.up, dir))
        new_up = np.cross(dir, left)

        mat = np.asarray([
            np.append(left, 0),
            np.append(new_up, 0),
            np.append(dir, 0),
            np.append(self.origin, 1)
        ]).transpose()

        return mat

    def lookat_3x3(self):
        dir = normalize(self.target - self.origin)
        left = normalize(np.cross(self.up, dir))
        new_up = np.cross(dir, left)

        mat = np.mat([
            left,
            new_up,
            dir
        ]).transpose()

        return mat

    def perspective(self):
        recip = 1 / (self.far - self.near)
        tan = np.tan(self.fov * 0.5)
        cot = 1 / tan

        trafo = np.diag([cot, cot, self.far * recip, 0])
        trafo[2][3] = -self.near * self.far * recip
        trafo[3][2] = 1

        return np.mat(trafo)


class Cubemap():
    def __init__(self, path, device):
        img = torch.tensor(imageio.v2.imread(path), device=device)
        self.img = img.permute(2, 0, 1)[None, ...]

    def __call__(self, uv):
        scale = 1 / math.pi
        x, y, z = torch.split(uv, [1, 1, 1], dim=-1)
        phi = torch.acos(y) * scale * 2 - 1
        theta = torch.atan2(z, x) * scale
        uv = torch.cat([theta, phi], dim=-1)

        result = F.grid_sample(self.img, uv[None, None, ...], align_corners=False, padding_mode='reflection')
        return result.squeeze().permute(1, 0)


def generate_ray(camera, samples, device):
    resolution = samples.shape[0]
    flat_size = resolution * resolution

    samples_pos = samples + np.random.rand(resolution, resolution, 2)
    samples_pos = samples_pos.reshape(flat_size, -1) * (1.0 / resolution)
    samples_pos = np.concatenate([1 - 2 * samples_pos, np.zeros([flat_size, 1]), np.ones([flat_size, 1])], axis=-1)

    mat_sample_to_camera = camera.perspective().I

    near_p = (samples_pos * mat_sample_to_camera.T).A.astype(np.float32)
    near_p = near_p[..., :3] / near_p[..., [-1]]
    ray_d = normalize(near_p)

    mat_camera_to_world = camera.lookat_3x3()
    ray_d = (ray_d * mat_camera_to_world.T).A.astype(np.float32)
    ray_o = np.repeat(camera.origin[None, ...], flat_size, axis=0)

    ray_o = torch.from_numpy(ray_o).to(device)
    ray_d = torch.from_numpy(ray_d).to(device)

    return ray_o, ray_d


def draw_gt():
    envmap_path = osp.join('../scenes/envmap', args.envmap)
    if osp.exists(envmap_path + '.hdr'):
        envmap_path += '.hdr'
    elif osp.exists(envmap_path + '.exr'):
        envmap_path += '.exr'
    else:
        print(f'{envmap_path} is not existed')
        exit()

    neutex = Neutex(args.dataset, args.neutex, device)

    cam_origin = [0, 0.25, 2.75]
    cam_target = [0, 0, 0]

    camera = Camera(
        origin=cam_origin,
        target=cam_target,
        up=[0, 1, 0],
        fov=40,
        near=0.1,
        far=4
    )

    resolution = args.size
    flat_size = resolution * resolution

    samples = np.linspace(0, resolution - 1, resolution)
    samples = np.stack(np.meshgrid(samples, samples, indexing='xy'), axis=-1)
    skybox = Cubemap(envmap_path, device)

    ior = 1.33

    pixels = torch.zeros(flat_size, 3, device=device)
    for s in range(args.spp):
        def get_normal(ray_o, ray_d):
            uv, normal, valid, pos = neutex.trace_surface(
                ray_o,
                ray_d,
                camera.near,
                camera.far,
                steps=256
            )
            return valid, normal, pos, uv

        ray_o, ray_d = generate_ray(camera, samples, device)
        # surface 0
        valid_0, normal_0, pos, _ = get_normal(ray_o, ray_d)
        ray_out_0 = refract(ray_d[valid_0], normal_0[valid_0], 1 / ior)

        # surface 1
        ray_oo = pos[valid_0] + ray_out_0 * camera.far * 0.75
        valid_1, normal_1, _, uv = get_normal(ray_oo, -ray_out_0)
        ray_out_1 = refract(ray_out_0[valid_1], -normal_1[valid_1], ior)

        ray_out_0[valid_1] = ray_out_1
        ray_d[valid_0] = ray_out_0

        ray_d = F.normalize(ray_d, dim=-1, eps=1e-6)
        pixels += skybox(ray_d)

        # pixels[valid_0] += uv
        # pixels[valid_0] += torch.cat([neutex.normalize_normal(uv), torch.zeros_like(uv[..., 0:1])], dim=-1)

    pixels = torch.cat([pixels * (1.0 / args.spp), torch.ones(flat_size, 1, device=device)], dim=-1)
    imageio.imwrite('output.exr', pixels.reshape(resolution, resolution, -1).cpu())


def show():
    neutex = Neutex(args.dataset, args.neutex, device)

    # neutex.get_pcd()
    # neutex.visualization_pcd()

    neutex.get_mesh()
    neutex.visualization_mesh()
    neutex.dump_mesh('uv.ply')

    neutex.get_nerf_mesh()
    neutex.visualization_mesh()
    neutex.dump_mesh('nerf.ply')


if __name__ == "__main__":
    with torch.no_grad():
        draw_gt()
        show()
