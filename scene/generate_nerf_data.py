import os
import os.path as osp
import numpy as np
import shutil
import math
import torch
import random
import json
import glob
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
from tqdm import tqdm
from argparse import ArgumentParser

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-15)

def convert(v):
    return np.asarray([v[2], v[0], v[1]])

def camera2world(origin, target, up):
    dir = normalize(target - origin)
    left = normalize(np.cross(up, dir))
    new_up = np.cross(dir, left)

    mat_c2w = np.mat([
        np.append(convert(-left), 0),
        np.append(convert(new_up), 0),
        np.append(convert(-dir), 0),
        np.append(convert(origin), 1)
    ]).T

    return mat_c2w.A

parser = ArgumentParser()
parser.add_argument('-s', '--size', type=int, default=64, help='number of views')
parser.add_argument('-i', '--input', type=str, default=r'./scenes/refract-bunny/bunny_gt.xml', help='input scene xml path')
parser.add_argument('-o', '--output', type=str, default=r'./dataset/bunny', help='output dir of the dataset')
parser.add_argument('-a', '--addition', type=int, default=32, help='number of additional views in the top sphere')
parser.add_argument('--gpu', action='store_true', help='use mitsuba3\'s gpu backend')
parser.add_argument('--exr', action='store_true', help='output exr image')
parser.add_argument('--clear', action='store_true', help='delete all prev data')
parser.add_argument('--mi', type=str, default=r'../mitsuba3/build/mitsuba', help='locate mitsuba')
args = parser.parse_args()

mi = args.mi
if not os.path.exists(mi):
    print(f'can\'t find mitsuba in {mi}')
    exit()

random.seed(404)

input_path = args.input
output_path = args.output
if args.clear and os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

backend_str = 'cuda_rgb' if args.gpu else 'scalar_rgb'

fov = np.pi * 40 / 180.0
radius = 2
up = np.asarray([0, 1, 0])

def render_views(size, suffix, output_img_path, mutation, idx_offset=0, top_hemisphere=False):
    file_list = []
    mat_list = torch.empty(size, 4, 4)
    for i in tqdm(range(size)):

        if top_hemisphere:
            theta = 2 * math.pi * (random.random() * mutation + i / size)
            cosPhi = random.random()
        else:
            theta = 2 * math.pi * i / size
            cosPhi = random.random() * 2 - 1

        sinPhi = math.sqrt(1 - cosPhi * cosPhi)
        dist = radius + (random.random() * 2 - 1) * mutation * 5
        campos = torch.asarray([math.cos(theta) * sinPhi, cosPhi, math.sin(theta) * sinPhi]) * dist
        camat = torch.rand(3) * mutation * 0.5

        name_str = f'{i+idx_offset:04d}'
        file_list.append(f'./{suffix}/{name_str}')
        mat_list[i] = torch.from_numpy(camera2world(campos, camat, up))

        campos_str = ','.join([str(n) for n in campos.tolist()])
        camat_str = ','.join([str(n) for n in camat.tolist()])
        rendering_cmd = f'{mi} -m {backend_str} -o {output_img_path}/{name_str}.exr -Dcampos={campos_str} -Dcamat={camat_str} {input_path}'
        os.popen(rendering_cmd).read()
    
    return file_list, mat_list

def generate(suffix, size, size_addition, mutation):
    print(f'generating {suffix}...')

    output_img_path = osp.join(output_path, suffix)
    os.makedirs(output_img_path, exist_ok=True)

    file_list, mat_list = render_views(size, suffix, output_img_path, mutation=mutation)
    if size_addition > 0:
        file_list_add, mat_list_add = render_views(size_addition, suffix, output_img_path, mutation=mutation, idx_offset=size, top_hemisphere=True)
        file_list = file_list + file_list_add
        mat_list = torch.cat([mat_list, mat_list_add], dim=0)

    if not args.exr:
        img_list = glob.glob(osp.join(output_img_path, '[0-9]*.exr'))
        for img_path in tqdm(img_list):
            img = torch.tensor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED)).float()
            alpha = img[..., [-1]] * 255.0
            bgr = torch.pow(img[..., :-1], 1.0/2.2) * 255.0
            cv2.imwrite(img_path.replace('.exr', '.png'), torch.cat([bgr, alpha], dim=-1).numpy())
            os.remove(img_path)

    frame_list = []
    for i in range(size + size_addition):
        frame_list.append({
            "file_path" : file_list[i],
            "transform_matrix" : mat_list[i].tolist()
        })

    json_dict = {
        "camera_angle_x" : fov,
        "frames": frame_list
    }

    with open(osp.join(output_path, f'transforms_{suffix}.json'), 'w') as f:
        json.dump(json_dict, f, indent = 4)

# generate train and test datasaet
generate('train', args.size, args.addition, 0.05)
generate('test', args.size // 2, args.addition // 2, 0.02)

print('done.')