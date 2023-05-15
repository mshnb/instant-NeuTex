import os
import cv2
import numpy as np
import collections
from tqdm.contrib import tenumerate

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def read_cameras_text(path) -> "dict[int, Camera]":
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0].replace(',', ''))
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_images_text(path) -> "dict[int, Image]":
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0].replace(',', ''))
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8].replace(',', ''))
                image_name = elems[9]
                _ = fid.readline()
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name)
    return images


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def main():
    scene = r'D:\shares\buddha'
    os.chdir(scene)
    cameras_file = "sparse/cameras.txt"
    images_file = "sparse/images.txt"
    scale = 1

    cameras = read_cameras_text(cameras_file)
    image_infos = read_images_text(images_file)

    camera_only = True
    use_depth = False
    camera_dict = {}
    for i, imId in tenumerate(image_infos):
        imInfo = image_infos[imId]
        camInfo = cameras[imInfo.camera_id]
        h = camInfo.height
        w = camInfo.width
        if not camera_only:
            os.makedirs(f"image", exist_ok=True)
            imName = "images/{}".format(imInfo.name)
            im = cv2.imread(imName)
            if scale > 1:
                im = cv2.resize(im, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
            cv2.imwrite("scan{}/image/{:04d}.png".format(scan_id, i), im)
        if use_depth:
            os.makedirs(f"depth", exist_ok=True)
            depth = "dense/stereo/depth_maps/{}.geometric.bin".format(imInfo.name)
            depth = read_array(depth)
            if scale > 1:
                depth = cv2.resize(depth, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
            cv2.imwrite("scan{}/depth/{:04d}.exr".format(scan_id, i), depth)
        assert camInfo.model == "SIMPLE_PINHOLE"
        f, cx, cy = camInfo.params
        K = np.eye(4, dtype=np.float32)
        K[0, 0] = f / scale
        K[1, 1] = f / scale
        K[0, 2] = cx / scale
        K[1, 2] = cy / scale
        R = qvec2rotmat(imInfo.qvec)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = imInfo.tvec
        P = K @ w2c
        camera_dict[f"world_mat_{i}"] = P
        # camera_dict[f"intrinsics_{i}"] = K
        # camera_dict[f"extrinsics_{i}"] = w2c

    np.savez_compressed("cameras.npz", **camera_dict)


if __name__ == '__main__':
    main()
