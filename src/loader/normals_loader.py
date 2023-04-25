import numpy as np
from torch.utils.data import Dataset
from src.loader.laserscan import SemLaserScan

def prep_long_2_right_up(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+2] - orig[:, i+2, j+2]
            v2 = orig[:, i+2, j+4] - orig[:, i+2, j+2]
            cr = np.cross(v1, v2)
            norm_mat[:, i, j] = cr
    return norm_mat

def prep_long_2_half_right_up(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i+1, j+2] - orig[:, i+2, j+2]
            v2 = orig[:, i+2, j+4] - orig[:, i+2, j+2]
            cr = np.cross(v1, v2)
            norm_mat[:, i, j] = cr
    return norm_mat


def prep_stock(orig):
    norm_mat = np.zeros_like(orig)

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            norm_mat[:, i, j] = np.cross(v1, v2)
    return norm_mat

def prep_cross_aver(orig):
    norm_mat = np.zeros_like(orig)
    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i+1, j] - orig[:, i+1, j+1]
            v4 = orig[:, i+2, j+1] - orig[:, i+1, j+1]
            norm_mat[:, i, j] = np.mean((np.cross(v1, v2), np.cross(v3, v4)), axis = 0)
    return norm_mat


def prep_cross_stacked(orig):
    norm_mat = np.zeros((6, 64, 1024))

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i+1, j] - orig[:, i+1, j+1]
            v4 = orig[:, i+2, j+1] - orig[:, i+1, j+1]
            cr1 = np.cross(v1, v2)
            cr2 = np.cross(v3, v4)
            norm_mat[:, i, j] = np.hstack((cr1, cr2))
    return norm_mat

def prep_long_short_stacked(orig):
    norm_mat = np.zeros((6, 64, 1024))

    orig = np.pad(orig, pad_width=2)[2:5]

    for i in range(norm_mat.shape[1]):
        for j in range(norm_mat.shape[2]):
            v1 = orig[:, i, j+1] - orig[:, i+1, j+1]
            v2 = orig[:, i+1, j+2] - orig[:, i+1, j+1]
            v3 = orig[:, i, j+2] - orig[:, i+2, j+2]
            v4 = orig[:, i+2, j+4] - orig[:, i+2, j+2]
            cr1 = np.cross(v1, v2)
            cr2 = np.cross(v3, v4)
            norm_mat[:, i, j] = np.hstack((cr1, cr2))
    return norm_mat

class CustomKittiWithOrig(Dataset):
    def __init__(self, dir, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.ls.proj_xyz, self.ls.proj_sem_label, self.ls.points, self.ls.sem_label


class CustomKitti(Dataset):
    def __init__(self, dir, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.ls.proj_xyz, self.ls.proj_sem_label
    
class CustomKittiProcessing(Dataset):
    def __init__(self, dir, proc_func, mode="train"):
        self.ls = SemLaserScan(project=True, nclasses=100)
        self.mode = mode
        self.len = 4541
        self.train_len = 3700
        self.val_len = 300
        self.test_len = self.len - self.train_len - self.val_len
        self.label_folder = dir + "plane_labels/"
        self.file_folder = dir + "velodyne/"
        self.proc_func = proc_func
    def __len__(self):
        if self.mode == "train":
            return self.train_len
        if self.mode == "val":
            return self.val_len
        return self.test_len
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += self.train_len + self.val_len
        if self.mode == "val":
            idx += self.train_len
        self.ls.open_scan(self.file_folder + '{0:06d}.bin'.format(idx))
        self.ls.open_label(self.label_folder + 'label-{0:06d}.npy'.format(idx))
        self.ls.proj_sem_label[self.ls.proj_sem_label != 0] = 1
        self.ls.proj_xyz = np.transpose(self.ls.proj_xyz, (2, 0, 1))
        return self.proc_func(self.ls.proj_xyz), self.ls.proj_sem_label

class CustomKittiPrep(Dataset):
    def __init__(self, path_d, path_l):
        self.data = np.load(path_d, allow_pickle=True)
        self.labels = np.load(path_l,  allow_pickle=True)

        print(self.data.shape)
        print(self.data[0].shape)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]