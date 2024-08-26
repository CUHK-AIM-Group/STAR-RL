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

def get_loader(npy_dir, batch_size, stage, num_workers,test_list_dir=None, bi='bilinear', use_bicubic_upsample=False, getFileName=False):
    """
    This is to generate input_data(96*96) and output_target(192*192)
    No upsample
    For final_up network
    """

    transform_data = transforms.ToTensor()
    transform_target = transforms.ToTensor()

    if stage == 'train':
        dataset = NUMPY_Dataset(npy_dir,transform_data, transform_target, use_bicubic_upsample=use_bicubic_upsample, getFileName=getFileName)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = num_workers, pin_memory=False)
    else:
        dataset = NUMPY_Dataset(npy_dir,transform_data, transform_target, test_list_dir=test_list_dir, use_bicubic_upsample=use_bicubic_upsample, getFileName=getFileName)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = num_workers, pin_memory=False)
    return dataloader

class NUMPY_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform_data=None, transform_target=None, test_list_dir=None, use_bicubic_upsample=False, getFileName=False):
        if test_list_dir != None: # for testset
            img_list_file = open(test_list_dir, 'r')
            img_list_tmp = img_list_file.readlines()
            img_list = []
            for imgname in img_list_tmp:
                img_list.append(imgname.strip())
        else:
            img_list = os.listdir(os.path.join(image_dir,'hr'))
        self.image_dir = image_dir
        self.generated_data = natsorted(img_list)
        # print(self.generated_data)

        self.transform_data = transform_data
        self.transform_target = transform_target

        self.use_bicubic_upsample = use_bicubic_upsample

        self.getFileName = getFileName

    def __getitem__(self, index):
        # env = self.env
        # with env.begin(write=False) as txn:
        #     dictbuf_bytes = txn.get(self.keys[index])

        # dictbuf = pickle.loads(dictbuf_bytes)# imageDict has two keys 'image' 'label'
        # # data_buf = dictbuf['data']
        # target_buf = dictbuf['target']

        # # img_data = BytesIO()
        # # img_data.write(data_buf)
        # # img_data.seek(0)
        # # image_data = np.asarray(Image.open(img_data))
        # img_target = BytesIO()
        # img_target.write(target_buf)
        # img_target.seek(0)
        # image_target = np.asarray(Image.open(img_target))
        lr_dir = os.path.join(self.image_dir,'lr') 
        if self.use_bicubic_upsample : 
            lr_dir = os.path.join(self.image_dir,'lr_bi') 
        
        img_path = os.path.join(lr_dir,self.generated_data[index])
        if not os.path.exists(img_path):
            img_path = img_path.replace('png', 'bmp')

        image_data = np.asarray(Image.open(img_path).convert('RGB'))#.convert('RGB'))
        image_data = np.transpose(image_data, [2,0,1])
        image_data = image_data.astype(np.float32) / 255.0
        # print("image_data:",image_data.shape,"img_path:",img_path)

        hr_dir = os.path.join(self.image_dir,'hr') 
        hr_img_path = os.path.join(hr_dir,self.generated_data[index])
        image_target = np.asarray(Image.open(hr_img_path).convert('RGB'))#.convert('RGB'))
        image_target = np.transpose(image_target, [2,0,1])
        image_target = image_target.astype(np.float32) / 255.0
        # print("image_target:",image_target.shape,"hr_img_path:",hr_img_path)
        # image_data = pil2tensor(image_data, np.float32)
        # upscale the degraded image via bicubic intepolation
        # factor = image_target.shape[1] // image_data.shape[1]
        # ch, h, w = image_data.shape
        # image_data = image_data.reshape([1,ch,h,w])
        # image_data = F.interpolate(image_data, scale_factor=factor, mode='bicubic')
        # _, ch, h, w = image_data.shape
        # image_data = image_data.reshape([ch,h,w])

        # image_target = image_target.astype(np.float32) / 255.0
        # image_target = pil2tensor(image_target, np.float32)

        if self.getFileName:
            return image_data, image_target, self.generated_data[index]

        return image_data, image_target

    def __len__(self):
        return len(self.generated_data)

    def __repr__(self):
        return self.__class__.__name__+'('+self.lmdb_dir+')'




if __name__ == "__main__":
    train_loader = get_loader(
        '/media/jay/data/SuperResolution/HistoSR/test',
        batch_size=12, 
        stage='train', 
        num_workers=4)
    import cv2
    for i, (image_data, ori_image_data) in enumerate(train_loader):
        # if you want to see the loaded data
        print("image_data:",image_data.max(),image_data.min())
        lr = cv2.imwrite('lr.bmp',np.transpose(image_data[0,:,:,:].numpy(),[1,2,0])*255)
        hr = cv2.imwrite('hr.bmp',np.transpose(ori_image_data[0,:,:,:].numpy(),[1,2,0])*255)

        print("data:",image_data.shape,ori_image_data.shape)
        input()











