import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import PIL
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ColorJitter, ToPILImage, ToTensor

from utils import Downsample

def load_mask(sampling_ratio):
    assert sampling_ratio in [10, 20, 30, 40, 50]
    from scipy.io import loadmat
    mask = loadmat('MICCAI/mask/Gaussian1D/GaussianDistribution1DMask_{}.mat'.format(sampling_ratio))
    mask = mask['maskRS1']
    print('mask:', np.mean(mask))
    return mask

class MRIDataset(data.Dataset):
    def __init__(self, image_set, transform, config):
        self.root = config.root
        self.image_set = image_set
        self.transform = transform
        self.ids = [i.strip() for i in open(self.root + self.image_set + '.txt').readlines()]

        self.mask = load_mask(config.sampling_ratio)
        #self.DAGAN = 'SegChallenge' in root and image_set == 'test'

    def __getitem__(self, index):
        x = cv2.imread(os.path.join(self.root, self.image_set, self.ids[index]), cv2.IMREAD_GRAYSCALE)
        if x.shape != (256, 256):
            x = cv2.resize(x, (256, 256))

        # data augmentation
        if self.transform:
            transformations = Compose(
                [ToPILImage(),
                 RandomRotation(degrees=10, resample=PIL.Image.BICUBIC),
                 #RandomAffine(degrees=10, translate=(-25, 25), scale=(0.90, 1.10), resample=PIL.Image.BILINEAR),
                 RandomHorizontalFlip(),
                 RandomResizedCrop(size=256, scale=(0.90, 1), ratio=(0.95, 1.05), interpolation=PIL.Image.BICUBIC),
                 #ColorJitter(brightness=0.05),
                 #CenterCrop(size=(256, 256)),
                 ToTensor(),
                ])
            x = x[..., np.newaxis]
            x = transformations(x).float().numpy() * 255
            x = x[0]

        image, _, _ = Downsample(x, self.mask)

        x = x / 255.
        image = image / 255.
        
        target = torch.from_numpy(x).float().unsqueeze(0)
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = [0] # return something to be compatible with fastMRI dataset
        return target, image, mask

    def __len__(self):
        return len(self.ids)
