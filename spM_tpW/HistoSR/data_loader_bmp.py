import os
import time
import pickle
import lmdb
import numpy as np
import random
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from natsort import natsorted
def pil2tensor(image, dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))

def get_loader(lmdb_dir, npy_dir, batch_size, stage, num_workers, bi='bilinear'):
    """
    This is to generate input_data(96*96) and output_target(192*192)
    No upsample
    For final_up network
    """

    transform_data = transforms.ToTensor()
    transform_target = transforms.ToTensor()

    if stage == 'train':
        dataset = NUMPY_Dataset(lmdb_dir,npy_dir,transform_data, transform_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory=False)
    else:
        dataset = NUMPY_Dataset(lmdb_dir,npy_dir,transform_data, transform_target)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=False)
    return dataloader

class NUMPY_Dataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_dir, image_dir, transform_data=None, transform_target=None):

        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(lmdb_dir, max_readers=10, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

        cache_file = os.path.join(lmdb_dir, 'cache')# should add cache_dir
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file,'rb'))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key,_ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file,'wb'))

        img_list = os.listdir(image_dir)
        self.image_dir = image_dir
        self.generated_data = natsorted(img_list)

        self.transform_data = transform_data
        self.transform_target = transform_target


    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            dictbuf_bytes = txn.get(self.keys[index])

        dictbuf = pickle.loads(dictbuf_bytes)# imageDict has two keys 'image' 'label'
        # data_buf = dictbuf['data']
        target_buf = dictbuf['target']

        # img_data = BytesIO()
        # img_data.write(data_buf)
        # img_data.seek(0)
        # image_data = np.asarray(Image.open(img_data))
        img_target = BytesIO()
        img_target.write(target_buf)
        img_target.seek(0)
        image_target = np.asarray(Image.open(img_target))

        image_data = np.asarray(Image.open(os.path.join(self.image_dir,self.generated_data[index])))
        image_data = np.transpose(image_data, [2,0,1])
        image_data = image_data[::-1,:,:]
        image_data = image_data.astype(np.float32) / 255.0
        # image_data = pil2tensor(image_data, np.float32)
        # upscale the degraded image via bicubic intepolation
        # factor = image_target.shape[1] // image_data.shape[1]
        # ch, h, w = image_data.shape
        # image_data = image_data.reshape([1,ch,h,w])
        # image_data = F.interpolate(image_data, scale_factor=factor, mode='bicubic')
        # _, ch, h, w = image_data.shape
        # image_data = image_data.reshape([ch,h,w])

        image_target = image_target.astype(np.float32) / 255.0
        image_target = pil2tensor(image_target, np.float32)

        return image_data, image_target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__+'('+self.lmdb_dir+')'




if __name__ == "__main__":
    train_loader = get_loader(
        './dataset/HistoSR/bicubic/train_lmdb',
        './results/bicubic/train',
        batch_size=12, 
        stage='train', 
        num_workers=4)
    import cv2
    for i, (image_data, ori_image_data) in enumerate(train_loader):
        # if you want to see the loaded data
        print("image_data:",image_data.max(),image_data.min())
        idx = 2
        lr = cv2.imwrite('lr.bmp',np.transpose(image_data[idx,:,:,:].numpy(),[1,2,0])*255)
        hr = cv2.imwrite('hr.bmp',np.transpose(ori_image_data[idx,:,:,:].numpy(),[1,2,0])*255)
        exit(0)
        print("data:",image_data.shape,ori_image_data.shape)












