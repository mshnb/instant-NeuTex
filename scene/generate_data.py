import os
import numpy as np
import shutil
import math
import torch
import random
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
parser.add_argument('-s', '--size', type=int, default=128, help='number of views')
parser.add_argument('-i', '--input', type=str, default=r'./bunny/bunny.xml', help='input scene xml path')
parser.add_argument('-o', '--output', type=str, default=r'../run/bunny', help='output path of the dataset')
parser.add_argument('--gpu', action='store_true', help='use mitsuba3\'s gpu backend')
parser.add_argument('--clear', action='store_true', help='delete all prev data')
args = parser.parse_args()

if not os.path.exists('../mitsuba'):
    print('can\'t find mitsuba in ../')
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

campos_list = torch.empty(args.size, 3)
camat_list = torch.empty(args.size, 3)
mat_list = torch.empty(args.size, 3, 3)

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
    rendering_cmd = f'../mitsuba -m {backend_str} -o {output_path}/data/{i:04d}.exr -Dcampos={campos_str} -Dcamat={camat_str} {input_path}'
    os.popen(rendering_cmd)

np.save(f'{output_path}/in_camOrgs.npy', campos_list.numpy())
np.save(f'{output_path}/in_camAts.npy', camat_list.numpy())
np.save(f'{output_path}/in_camExtrinsics.npy', mat_list.numpy())

print('done.')