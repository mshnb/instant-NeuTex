import os
import os.path as osp
import numpy as np
import shutil
import math
import torch
import random
import glob
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-15)

def lookat(origin, target, up):
    dir = normalize(target - origin)
    left = normalize(np.cross(up, dir))
    new_up = np.cross(dir, left)

    mat_lookat = np.asarray([
        np.append(left, 0),
        np.append(new_up, 0),
        np.append(dir, 0),
        np.append(origin, 1)
    ]).transpose()

    return mat_lookat

parser = ArgumentParser()
parser.add_argument('-s', '--size', type=int, default=256, help='number of views')
parser.add_argument('-a', '--addition', type=int, default=64, help='number of additional views in the top sphere')
parser.add_argument('-i', '--input', type=str, default=r'./bunny/bunny.xml', help='input scene xml path')
parser.add_argument('-o', '--output', type=str, default=r'../run/bunny', help='output path of the dataset')
parser.add_argument('--gpu', action='store_true', help='use mitsuba3\'s gpu backend')
parser.add_argument('--clear', action='store_true', help='delete all prev data')
parser.add_argument('--mi', type=str, default='../../mitsuba3/build/mitsuba', help='locate mitsuba')
args = parser.parse_args()

mi = args.mi

if not os.path.exists(mi):
    print(f'can\'t find mitsuba in {mi}')
    exit()

input_path = args.input
output_path = args.output
if args.clear and os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_path + '/data', exist_ok=True)

backend_str = 'cuda_rgb' if args.gpu else 'scalar_rgb'

radius = 3.5
center = torch.asarray([0, 0.5, 0])

mutation = 0.1

total_size = args.size + args.addition
campos_list = torch.empty(total_size, 3)
camat_list = torch.empty(total_size, 3)
mat_list = torch.empty(total_size, 3, 3)

for i in tqdm(range(args.size)):
    theta = 2 * math.pi * i / args.size
    cosPhi = random.random() * 2 - 1
    sinPhi = math.sqrt(1 - cosPhi * cosPhi)

    dist = radius + (random.random() * 2 - 1) * mutation
    campos = center + torch.asarray([math.cos(theta) * sinPhi, cosPhi, math.sin(theta) * sinPhi]) * dist
    camat = center + torch.rand(3) * mutation

    campos_list[i] = campos
    camat_list[i] = camat
    mat_list[i] = torch.from_numpy(lookat(campos, camat, np.asarray([0, 1, 0]))[:3,:3])

    campos_str = ','.join([str(n) for n in campos.tolist()])
    camat_str = ','.join([str(n) for n in camat.tolist()])
    rendering_cmd = f'{mi} -m {backend_str} -o {output_path}/data/{i:04d}.exr -Dcampos={campos_str} -Dcamat={camat_str} {input_path}'
    os.popen(rendering_cmd).read()

if args.addition > 0:
    for i in tqdm(range(args.addition)):
        theta = 2 * math.pi * (random.random() * mutation + i / args.addition)
        cosPhi = random.random()
        sinPhi = math.sqrt(1 - cosPhi * cosPhi)

        dist = radius + (random.random() * 2 - 1) * mutation
        campos = center + torch.asarray([math.cos(theta) * sinPhi, cosPhi, math.sin(theta) * sinPhi]) * dist
        camat = center + torch.rand(3) * mutation

        i += args.size
        campos_list[i] = campos
        camat_list[i] = camat
        mat_list[i] = torch.from_numpy(lookat(campos, camat, np.asarray([0, 1, 0]))[:3,:3])

        campos_str = ','.join([str(n) for n in campos.tolist()])
        camat_str = ','.join([str(n) for n in camat.tolist()])
        rendering_cmd = f'{mi} -m {backend_str} -o {output_path}/data/{i:04d}.exr -Dcampos={campos_str} -Dcamat={camat_str} {input_path}'
        os.popen(rendering_cmd).read()

img_list = glob.glob(osp.join(output_path, 'data', '[0-9]*.exr'))
for img_path in tqdm(img_list):
    img = torch.tensor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED)).float()
    alpha = img[..., [-1]] * 255.0
    bgr = torch.pow(img[..., :-1], 1.0/2.2) * 255.0
    cv2.imwrite(img_path.replace('.exr', '.png'), torch.cat([bgr, alpha], dim=-1).numpy())
    os.remove(img_path)

np.save(f'{output_path}/in_camOrgs.npy', campos_list.numpy())
np.save(f'{output_path}/in_camAts.npy', camat_list.numpy())
np.save(f'{output_path}/in_camExtrinsics.npy', mat_list.numpy())

print('done.')