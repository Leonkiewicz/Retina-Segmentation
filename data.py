import os
import numpy as np
import cv2
from glob import glob

import torch
from torch.utils.data import Dataset


class RetinaDataset(Dataset):
    def __init__(self, image_path, mask_path):

        self.image_path = image_path
        self.mask_path = mask_path

    
    def __getitem__(self, index):
        """ Read in the image """
        image = cv2.imread(self.image_path[index], cv2.IMREAD_COLOR)
        image = image/255.
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Read in the mask """
        mask = cv2.imread(self.mask_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask


    def __len__(self):
        return len(self.image_path)
