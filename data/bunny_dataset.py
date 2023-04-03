import math
import numpy as np
import torch
import os
import os.path as osp
import glob
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
from .base_dataset import BaseDataset

def perspective(fov, near, far):
    recip = 1 / (far - near)

    tan = math.tan(0.5 * fov * math.pi / 180)
    cot = 1 / tan

    trafo = np.diag([cot, cot, far * recip, 0])
    trafo[2][3] = -near * far * recip
    trafo[3][2] = 1

    return np.asmatrix(trafo)

def perspective_projection(film_size, fov_x, near_clip, far_clip):
    aspect = film_size[0] / film_size[1]

    mat_scale = np.asmatrix(np.diag([-0.5, -0.5*aspect, 1, 1]))

    translate = np.identity(4)
    translate[...,3] = np.asarray([-1, -1/aspect, 0, 1])
    mat_translate = np.asmatrix(translate)

    mat_perspective = perspective(fov_x, near_clip, far_clip)

    return mat_scale * mat_translate * mat_perspective

def get_rays_dir(pixelcoords, height, width, rot, sample_to_camera):
    scale_x = 1 / width
    scale_y = 1 / height

    # pixelcoords: H x W x 2
    x = pixelcoords[..., 0] * scale_x
    y = pixelcoords[..., 1] * scale_y
    z = np.zeros_like(x)
    w = np.ones_like(x)
    dirs = np.stack([x, y, z, w], axis=-1)

    dirs = np.sum(sample_to_camera.T[None, None, :, :].A * dirs[..., None], axis=-2)
    dirs = dirs[...,:3] / dirs[..., [3]]
    dirs = dirs / (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-15)
    dirs = np.sum(rot[:3,:3].T[None, None, :, :] * dirs[..., None], axis=-2)

    return dirs

class BunnyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--random_sample",
            type=str,
            default="no_crop",
            choices=["no_crop", "random", "balanced", "patch"],
            help="method for sampling from the image",
        )
        parser.add_argument(
            "--random_sample_size",
            type=int,
            default=64,
            help="square root of the number of random samples",
        )
        parser.add_argument(
            "--use_test_data", type=int, default=-1, help="train or test dataset",
        )
        parser.add_argument(
            "--test_views", type=str, default="6,34,73,112,155,190,221,248", help="held out views",
        )

        return parser

    def initialize(self, opt):
        self.opt = opt
        self.data_dir = opt.data_root

        self.campos = np.load(self.data_dir + "/in_camOrgs.npy")
        self.camat = np.load(self.data_dir + "/in_camAts.npy")
        self.extrinsics = np.load(self.data_dir + "/in_camExtrinsics.npy")

        self.total = self.campos.shape[0]

        if os.path.isfile(self.data_dir + '/exclude.txt'):
            with open(self.data_dir + '/exclude.txt', 'r') as f:
                exclude_views = [int(x) for x in f.readline().strip().split(',')]
        else:
            exclude_views = []

        if os.path.isfile(self.data_dir + '/test_views.txt'):
            with open(self.data_dir + '/test_views.txt', 'r') as f:
                test_views = [int(x) for x in f.readline().strip().split(',')]
        else:
            test_views = [int(x) for x in opt.test_views.split(',')]

        if self.opt.use_test_data > 0:
            self.indexes = test_views
            assert len(self.indexes) > 0
        else:
            self.indexes = [i for i in range(self.total) if i not in test_views and i not in exclude_views]
            assert len(self.indexes) == self.campos.shape[0] - len(test_views) - len(exclude_views)

        print("Total views:", self.total)

        print("Loading data in memory")
        self.gt_image = []
        self.gt_mask = []
        files = sorted(glob.glob(osp.join(self.data_dir, 'data', '[0-9]*.exr')), key=lambda path: int(path[-8:-4]))
        for file in files:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)[:,:,::-1]
            img = np.power(img, 1.0/2.2)
            mask = torch.from_numpy(np.sum(img, axis=-1, keepdims=True) > 0).long()

            self.gt_image.append(img)
            self.gt_mask.append(mask)

        print("Finish loading")

        self.height = self.gt_image[0].shape[0]
        self.width = self.gt_image[0].shape[1]
        print("center cam pos: ", self.campos[128])
        self.center_cam_pos = self.campos[128]

        self.m_sample_to_camera = perspective_projection([self.height,self.width], 19.5, 1e-2, 1e4).I

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]

        item = {}
        gt_image = self.gt_image[idx]
        gt_mask = self.gt_mask[idx]
        
        item["gt_mask"] = gt_mask.permute(2, 0, 1).float()

        height = gt_image.shape[0]
        width = gt_image.shape[1]

        campos = self.campos[idx]
        camrot = self.extrinsics[idx]
        item["campos"] = torch.from_numpy(campos).float()

        dist = np.linalg.norm(self.camat[idx] - campos)
        item["near"] = torch.FloatTensor([dist - 1.0])
        item["far"] = torch.FloatTensor([dist + 1.0])

        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(0, width - subsamplesize + 1)
            indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32),
            )
        elif self.opt.random_sample == "random":
            px = np.random.randint(
                0, width, size=(subsamplesize, subsamplesize)
            ).astype(np.float32)
            py = np.random.randint(
                0, height, size=(subsamplesize, subsamplesize)
            ).astype(np.float32)
        elif self.opt.random_sample == "balanced":
            px, py, trans = self.proportional_select(gt_mask)
            item["transmittance"] = torch.from_numpy(trans).float().contiguous()
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32),
            )

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        raydir = get_rays_dir(
            pixelcoords, self.height, self.width, camrot, self.m_sample_to_camera
        )

        raydir = np.reshape(raydir, (-1, 3))
        item["raydir"] = torch.from_numpy(raydir).float()
        gt_image = gt_image[py.astype(np.int32), px.astype(np.int32), :]
        gt_image = np.reshape(gt_image, (-1, 3))
        item["gt_image"] = torch.from_numpy(gt_image).float().contiguous()
        item["background_color"] = torch.from_numpy(np.zeros(3)).float().contiguous()

        return item

    def proportional_select(self, mask):
        # random select 3/4 pixels from foreground
        # random select 1/4 pixels from background
        subsamplesize = self.opt.random_sample_size

        fg_index = np.where(mask > 0)
        #  print(fg_index)
        fg_yx = np.stack(fg_index, axis=1)  # n x 2
        fg_num = fg_yx.shape[0]

        bg_index = np.where(mask == 0)
        bg_yx = np.stack(bg_index, axis=1)
        bg_num = bg_yx.shape[0]

        select_fg_num = int(self.opt.random_sample_size * self.opt.random_sample_size * 3.0 / 4.0)
        select_bg_num = subsamplesize * subsamplesize - select_fg_num

        if select_fg_num > fg_num:
            select_fg_num = fg_num

        if select_bg_num > bg_num:
            select_bg_num = bg_num

        fg_index = np.random.choice(range(fg_num), select_fg_num)
        bg_index = np.random.choice(range(bg_num), select_bg_num)

        px = np.concatenate((fg_yx[fg_index, 1], bg_yx[bg_index, 1]))
        py = np.concatenate((fg_yx[fg_index, 0], bg_yx[bg_index, 0]))

        px = px.astype(np.float32)  # + 0.5 * np.random.uniform(-1, 1)
        py = py.astype(np.float32)  # + 0.5 * np.random.uniform(-1, 1)

        px = np.clip(px, 0, mask.shape[1] - 1)
        py = np.clip(py, 0, mask.shape[0] - 1)

        px = np.reshape(px, (subsamplesize, subsamplesize))
        py = np.reshape(py, (subsamplesize, subsamplesize))

        trans = np.zeros(len(fg_index) + len(bg_index))
        trans[len(fg_index) :] = 1

        return px, py, trans

    def get_item(self, idx):
        item = self.__getitem__(idx)

        for key, value in item.items():
            if isinstance(value, np.int64):
                item[key] = torch.LongTensor([value])
            else:
                item[key] = value.unsqueeze(0)

        return item
