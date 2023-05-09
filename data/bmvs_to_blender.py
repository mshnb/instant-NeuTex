import copy
import json
import os
import cv2
import numpy as np
from tqdm import tqdm

from scene.generate_nerf_data import convert

root = r"D:\shares\NeuS"
instance_dir = 'bmvs_sculpture'


def to16b(img):
    img = img.clip(0, 1) * 65535
    return img.astype(np.uint16)


def opencv_to_gl(pose):
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pose[:3, :3] = pose[:3, :3] @ mat
    return pose


def get_offset(poses):
    eyes = np.stack([pose[:3, 3] for pose in poses])

    scale = eyes.max(axis=0) - eyes.min(axis=0)
    print(f'scale : {scale}')

    offset = -(eyes.max(axis=0) + eyes.min(axis=0)) / 2
    print(f'offset : {offset}')

    return scale / 2, offset


def s(pose, scale, offset):
    pose[:3, 3] = (pose[:3, 3] + offset) / scale
    # print(pose[:3, 3])
    return pose.tolist()


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    right = R[0, :]
    up = R[1, :]
    m_dir = R[2, :]
    pos = (t[:3] / t[3])[:, 0]

    pose = np.mat([
        np.append(convert(right), 0),
        np.append(convert(up), 0),
        np.append(convert(m_dir), 0),
        np.append(convert(pos), 1)
    ]).T.A

    return intrinsics, pose


def main():
    os.chdir(os.path.join(root, instance_dir))

    image_dir = 'image'
    n_images = len(os.listdir(image_dir))

    cam_file = 'cameras_sphere.npz'
    camera_dict = np.load(cam_file)
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for mat in world_mats:
        P = mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(opencv_to_gl(pose))

    train_json = dict()

    train_json['fl_y'] = intrinsics[1][1]
    train_json['h'] = int(intrinsics[1, 2] * 2)
    train_json['fl_x'] = intrinsics[0][0]
    train_json['w'] = int(intrinsics[0, 2] * 2)

    scale, offset = get_offset(pose_all)

    train_json['enable_depth_loading'] = True
    train_json['integer_depth_scale'] = 1 / 65535

    train_json['frames'] = []

    test_json = copy.deepcopy(train_json)

    need_handle_mask = False
    if not os.path.exists('data'):
        os.makedirs('data')
        need_handle_mask = True

    for i in tqdm(range(n_images)):
        frames = train_json['frames']

        if need_handle_mask:
            img = cv2.imread(os.path.join('image', '{:03d}.png'.format(i)), -1)
            mask = cv2.imread(os.path.join('mask', '{:03d}.png'.format(i)), -1)
            cv2.imwrite(os.path.join('data', '{:04d}.png'.format(i)), np.concatenate([img, mask[..., 0:1]], axis=-1))

        frame = {
            'file_path': f'./data/{i:04d}',
            'transform_matrix': s(pose_all[i], scale.max(), offset)
        }
        frames.append(frame)

    with open('transforms_train.json', 'w') as f:
        json.dump(train_json, f, indent=4)
    with open('transforms_test.json', 'w') as f:
        json.dump(test_json, f, indent=4)
    with open('transforms_val.json', 'w') as f:
        json.dump(test_json, f, indent=4)


if __name__ == '__main__':
    main()
