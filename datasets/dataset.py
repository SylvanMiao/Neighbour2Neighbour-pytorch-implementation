from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

"""
掩码与子采样操作放到训练脚本中处理，每个batch的mask一致
"""

class Neighbour2Neighbour(Dataset):
    def __init__(self, data_path, patch_size=(512, 512)):
        self.data_path = Path(data_path)
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif')
        self.paths = []
        for ext in exts:
            self.paths.extend(sorted(self.data_path.glob(ext)))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        if img.mode in ('I;16', 'I;16B', 'I;16L'):
            arr = np.array(img, dtype=np.uint16).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            arr = arr / 65535.0
        else:
            arr = np.array(img.convert('L'), dtype=np.float32)
            arr = arr[:, :, None]
            arr = arr / 255.0

        source, _ = self.random_crop(arr, self.patch_size)

        # (H, W, C) → (C, H, W)
        source = torch.from_numpy(source.transpose(2, 0, 1)).float()
        return source

    def random_crop(self, img, patch_size):
        if not isinstance(patch_size, tuple):
            raise TypeError('patch_size must be tuple')

        h, w, _ = img.shape
        if h == patch_size[0] and w == patch_size[1]:
            return img, (0, 0, h, w)
        if h < patch_size[0] or w < patch_size[1]:
            raise ValueError('patch_size must be <= image size')

        top = np.random.randint(0, h - patch_size[0])
        left = np.random.randint(0, w - patch_size[1])

        patch = img[top:top + patch_size[0], left:left + patch_size[1], :]
        return patch, (top, left, top + patch_size[0], left + patch_size[1])
