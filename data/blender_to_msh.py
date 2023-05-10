import json
import math
import os
import shutil

import cv2
import numpy as np

from scene.generate_data import lookat

root = r"D:\shares\NeuS"
instance_dir = 'bmvs_sculpture'


def convert_inv(v):
    return np.asarray([v[1], v[2], v[0]])


def main():
    os.chdir(os.path.join(root, instance_dir))

    with open('transforms_train.json', 'r') as f:
        json_dic = json.load(f)

    path = json_dic['frames'][0]['file_path']
    img = cv2.imread(path + '.png', -1)
    print(img.shape)
    h, w = img.shape[0], img.shape[1]

    dir = os.path.dirname(path)
    if not os.path.exists('data'):
        shutil.copy(dir, 'data')

    f_x = json_dic['fl_x']
    fov_x = math.degrees(2 * math.atan(w / 2 / f_x))
    f_y = json_dic['fl_y']
    fov_y = math.degrees(2 * math.atan(h / 2 / f_y))
    print(fov_x, fov_y)

    pos_list = []
    tar_list = []
    rotation_list = []

    for i, frame in enumerate(json_dic['frames']):
        pose = np.array(frame['transform_matrix'])
        # pose[:3, :3] = pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        pos = convert_inv(pose[:3, 3])
        dir = -convert_inv(pose[:3, 2])
        target = pos + dir
        print(i, pos, target)

        pos_list.append(pos)
        tar_list.append(target)
        rotation_list.append(lookat(pos, target)[:3, :3])

    np.save('in_camOrgs.npy', np.stack(pos_list))
    np.save('in_camAts.npy', np.stack(tar_list))
    np.save('in_camExtrinsics.npy', np.stack(rotation_list))
    np.save('fov.npy', [fov_x, fov_y])


if __name__ == '__main__':
    main()
