from torch.utils.data import Dataset
import os
from osgeo import gdal
import torch


class myDataset(Dataset):

    def __init__(self, img_fold, sentinel_fold, img_set, label_fold, build_fold, label_set, norm_value=0):
        super(myDataset, self).__init__()

        self.img_fold = img_fold
        self.sentinel_fold = sentinel_fold
        self.label_fold = label_fold
        self.build_fold = build_fold
        self.img_url = img_set
        self.label_url = label_set
        self.len = len(self.img_url)
        self.norm_value = norm_value

    def __getitem__(self, index):
        # Sentinel-2
        srcds = gdal.Open(os.path.join(self.img_fold, self.img_url[index]))
        img = srcds.ReadAsArray(0, 0, srcds.RasterXSize, srcds.RasterYSize)
        # Sentinel-1
        srcds = gdal.Open(os.path.join(self.sentinel_fold, self.img_url[index]))
        sentinel = srcds.ReadAsArray(0, 0, srcds.RasterXSize, srcds.RasterYSize)
        # Footprint-label
        srcds = gdal.Open(os.path.join(self.label_fold, self.label_url[index]))
        label = srcds.ReadAsArray(0, 0, srcds.RasterXSize, srcds.RasterYSize)
        # Building-label
        srcds = gdal.Open(os.path.join(self.build_fold, self.label_url[index]))
        build = srcds.ReadAsArray(0, 0, srcds.RasterXSize, srcds.RasterYSize)
        del srcds
        img = torch.from_numpy(img)
        sentinel = torch.from_numpy(sentinel)
        label = torch.from_numpy(label)
        build = torch.from_numpy(build)
        build = torch.unsqueeze(build, dim=0)
        if self.norm_value != 0:
            img = img / float(self.norm_value)
            sentinel = sentinel / float(self.norm_value)

        return img, sentinel, label, build

    def __len__(self):
        return self.len
