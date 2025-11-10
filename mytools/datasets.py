import os, glob
import torch, sys
from torch.utils.data import Dataset
import pickle
import random
import numpy as np
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalWarningDisplay(False)


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


def sitkReadImage(dir_arr):
    result = []
    for dir in dir_arr:
        X = sitk.ReadImage(dir)
        X = sitk.GetArrayFromImage(X)
        result.append(X)
    return tuple(result)


def final_result(img_arr):
    result = []
    for img in img_arr:
        X = np.ascontiguousarray(img)
        X = torch.from_numpy(X)
        result.append(X)
    return tuple(result)


class OASISTrainDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # inter-patient
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = self.img_type([x[None, ...], y[None, ...]])
        # x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        del x_seg, y_seg
        return final_result([x, y])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class OASISTestDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # inter-patient
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = self.img_type([x[None, ...], y[None, ...]])
        x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        return final_result([x, y, x_seg, y_seg])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class LPBATrainDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.atlas_dir = atlas_dir
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # patient-to-atlas
        x_path = self.paths[index]
        y_path = self.atlas_dir['img']
        x, y = sitkReadImage([x_path, y_path])
        x, y = self.img_type([x[None, ...], y[None, ...]])
        # x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        return final_result([x, y])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class LPBATestDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.atlas_dir = atlas_dir
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # patient-to-atlas
        x_path = self.paths[index]
        x_seg_path = x_path.replace('delineation_l_norm/test', 'label')
        x_seg_path = x_seg_path.replace('skullstripped', 'structure.label')
        y_path = self.atlas_dir['img']
        y_seg_path = self.atlas_dir['seg']

        x, x_seg, y, y_seg = sitkReadImage([x_path, x_seg_path, y_path, y_seg_path])
        x, y = self.img_type([x[None, ...], y[None, ...]])
        x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        return final_result([x, y, x_seg, y_seg])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class IXITrainDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.atlas_dir = atlas_dir
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # patient-to-atlas
        x_path = self.paths[index]
        x, x_seg = pkload(x_path)
        y, y_seg = pkload(self.atlas_dir)
        x, y = self.img_type([x[None, ...], y[None, ...]])
        # x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        del x_seg, y_seg
        return final_result([x, y])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class IXITestDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.atlas_dir = atlas_dir
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # patient-to-atlas
        x_path = self.paths[index]
        x, x_seg = pkload(x_path)
        y, y_seg = pkload(self.atlas_dir)

        x, y = self.img_type([x[None, ...], y[None, ...]])
        x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        return final_result([x, y, x_seg, y_seg])  # moving, fixed

    def __len__(self):
        return len(self.paths)


def center(arr):
    c = np.sort(np.nonzero(arr))[:,[0,-1]]
    return np.mean(c, axis=-1).astype('int16')


def cropByCenter(image,center,final_shape):
    c = center
    crop = np.array([s // 2 for s in final_shape])
    # 0 axis
    cropmin, cropmax = c[0] - crop[0], c[0] + crop[0]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[0]
    if cropmax > image.shape[0]:
        cropmax = image.shape[0]
        cropmin = image.shape[0] - final_shape[0]
    image = image[cropmin:cropmax, :, :]
    # 1 axis
    cropmin, cropmax = c[1] - crop[1], c[1] + crop[1]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[1]
    if cropmax > image.shape[1]:
        cropmax = image.shape[1]
        cropmin = image.shape[1] - final_shape[1]
    image = image[:, cropmin:cropmax, :]

    # 2 axis
    cropmin, cropmax = c[2] - crop[2], c[2] + crop[2]
    if cropmin < 0:
        cropmin = 0
        cropmax = final_shape[2]
    if cropmax > image.shape[2]:
        cropmax = image.shape[2]
        cropmin = image.shape[2] - final_shape[2]
    image = image[:, :, cropmin:cropmax]
    return image


class MindboggleTrainDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.atlas_dir = atlas_dir
        self.img_type = img_type
        self.seg_type = seg_type
        self.img_size = img_size
        self.img_name = 't1weighted_brain.MNI152.nii.gz'
        self.seg_name = 'labels.DKT31.manual.MNI152.nii.gz'

    def __getitem__(self, index):
        # inter-patient
        x_path = self.paths[index]['x'] + self.img_name
        y_path = self.paths[index]['y'] + self.img_name
        x, y = sitkReadImage([x_path, y_path])

        # crop by center
        x_center = center(x)
        x = cropByCenter(x, x_center)
        y_center = center(y)
        y = cropByCenter(y, y_center)

        # min-max norm
        x = min_max_normalize(x)
        y = min_max_normalize(y)

        # transform data type
        x, y = self.img_type([x, y])
        # x_seg, y_seg = self.seg_type([x_seg, y_seg])
        return final_result([x[None, ...], y[None, ...]])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class MindboggleTestDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.atlas_dir = atlas_dir
        self.img_type = img_type
        self.seg_type = seg_type
        self.img_size = img_size
        self.img_name = 't1weighted_brain.MNI152.nii.gz'
        self.seg_name = 'labels.DKT31.manual.MNI152.nii.gz'

    def __getitem__(self, index):
        # inter-patient
        x_path = self.paths[index]['x'] + self.img_name
        x_seg_path = self.paths[index]['x'] + self.seg_name
        y_path = self.paths[index]['y'] + self.img_name
        y_seg_path = self.paths[index]['y'] + self.seg_name
        x, x_seg, y, y_seg = sitkReadImage([x_path, x_seg_path, y_path, y_seg_path])

        # crop by center
        x_center = center(x)
        x = cropByCenter(x, x_center)
        x_seg = cropByCenter(x_seg, x_center)
        y_center = center(y)
        y = cropByCenter(y, y_center)
        y_seg = cropByCenter(y_seg, y_center)

        # min-max norm
        x = min_max_normalize(x)
        y = min_max_normalize(y)

        # transform data type
        x, y = self.img_type([x, y])
        x_seg, y_seg = self.seg_type([x_seg, y_seg])
        return final_result([x[None, ...], y[None, ...], x_seg[None, ...], y_seg[None, ...]])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class AbdomenTrainDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # inter-patient
        path = self.paths[index]
        x = np.load(path)

        y_data_file = os.path.dirname(path.replace("/CT/", "/MRI/"))
        y_data_path = y_data_file + '/' + random.choice(os.listdir(y_data_file))
        y = np.load(y_data_path)

        x, y = self.img_type([x[None, ...], y[None, ...]])
        return final_result([x, y])  # moving, fixed

    def __len__(self):
        return len(self.paths)


class AbdomenTestDataset(Dataset):
    def __init__(self, data_path, atlas_dir, img_type, seg_type, img_size):
        self.paths = data_path
        self.img_type = img_type
        self.seg_type = seg_type

    def __getitem__(self, index):
        # inter-patient
        path = self.paths[index]
        x = np.load(path)
        x_seg = np.load(path.replace("/data/", "/mask/"))

        y = np.load(path.replace("/CT/", "/MRI/"))
        y_seg = np.load(path.replace("/CT/", "/MRI/").replace("/data/", "/mask/"))

        x, y = self.img_type([x[None, ...], y[None, ...]])
        x_seg, y_seg = self.seg_type([x_seg[None, ...], y_seg[None, ...]])
        return final_result([x, y, x_seg, y_seg])  # moving, fixed

    def __len__(self):
        return len(self.paths)
