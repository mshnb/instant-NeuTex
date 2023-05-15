import copy
import json
import os
import cv2
import numpy as np
from tqdm import tqdm

root = r"D:\shares"
instance_dir = 'buddha'


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

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def main():
    os.chdir(os.path.join(root, instance_dir))

    cam_file = 'cameras.npz'
    camera_dict = np.load(cam_file)

    intrinsics_all = {}
    pose_all = {}
    for id, mat in camera_dict.items():
        P = mat.astype(np.float32)
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all[id] = intrinsics
        pose_all[id] = opencv_to_gl(pose)

    train_json = dict()

    train_json['fl_y'] = intrinsics[1][1]
    train_json['h'] = int(intrinsics[1, 2] * 2)
    train_json['fl_x'] = intrinsics[0][0]
    train_json['w'] = int(intrinsics[0, 2] * 2)

    scale, offset = get_offset(pose_all.values())

    train_json['frames'] = []

    test_json = copy.deepcopy(train_json)

    for pose_id in tqdm(pose_all):
        id = int(pose_id.split('_')[-1])
        frames = train_json['frames']
        frame = {
            'file_path': f'./frames/{id:06d}',
            'transform_matrix': s(pose_all[pose_id], scale.max(), offset)
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
