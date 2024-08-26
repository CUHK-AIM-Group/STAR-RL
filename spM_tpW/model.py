import numpy as np
from numpy.random import randn
import torch
from torch.nn import Conv2d, Linear, Sigmoid
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

# coordinate-classify pixel-oriented patch agent
class CoordinateClassifyPixelPatchAgent_sm(torch.nn.Module):
    def __init__(self, config):
        super(CoordinateClassifyPixelPatchAgent_sm, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor
        if self.use_pixel_oriented_patch:
            self.num_actions = config.patch_height * config.patch_width

        self.noise_scale = config.noise_scale
        self.pw = config.patch_width
        self.ph = config.patch_height
        assert self.pw == self.ph , 'The width and height of patch are not equal.'

        # shared backbone
        backbone = []
        backbone += [ Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 64 x 192x192
                      nn.InstanceNorm2d(64),
                      nn.LeakyReLU(0.2, True),
                      nn.MaxPool2d(2, stride=2), #  128 x 96 x 96
                      Conv2d(64, 64, kernel_size=1, stride=1, padding=0), # 128 x 96 x 96
                      nn.InstanceNorm2d(64),
                      nn.LeakyReLU(0.2, True)
                      ]
        self.backbone = nn.Sequential(*backbone)

        # actor 
        ## col classification
        col_layers = []
        col_layers += [ nn.MaxPool2d((2,1), stride=(2,1)), #  128 x 48 x 96
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), # 128 x 48 x 96
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((2,1), stride=(2,1)), #  128 x 24 x 96
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 24 x 96
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((2,1), stride=(2,1)), #  128 x 24 x 96
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 12 x 96
                        nn.InstanceNorm2d(64),
                        nn.MaxPool2d((12,1), stride=(12,1)), #  128 x 1 x 96
                        Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 1 x 96
        self.act_col_layers = nn.Sequential(*col_layers)
        self.act_col_last_layer = nn.Linear(config.patch_width, config.patch_width) # bs x 96
                        
        ## row classfication 
        row_layers = []
        row_layers += [ nn.MaxPool2d((1,2), stride=(1,2)), #  128 x 96 x 48
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), # 128 x 96 x 48 
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,2), stride=(1,2)), #  128 x 96 x 24
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 96 x 24
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,2), stride=(1,2)), #  128 x 96 x 12
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 96 x 12
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,12), stride=(1,12)), #  128 x 96 x 1
                        Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 96 x 1
        self.act_row_layers = nn.Sequential(*row_layers)
        self.act_row_last_layer = nn.Linear(config.patch_height, config.patch_height) # bs x 96

        # critic
        self.cri_conv0 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 128 x 48 x 48
        self.cri_bn0 = nn.InstanceNorm2d(128)
        self.cri_conv1 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)    # 256 x 24 x 24
        self.cri_bn1 = nn.InstanceNorm2d(256)
        self.cri_conv2 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    # 512 x 12 x 12
        self.cri_bn2 = nn.InstanceNorm2d(512)
        self.cri_conv3 = Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 6 x 6
        self.cri_bn3 = nn.InstanceNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(6, stride=6) # 1024 x 1 x 1
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 1)    # 1
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = self.backbone(x)
        # actor
        ## column classification
        act_h_col = self.act_col_layers(h) # bs x 1 x 1 x 96
        ### flatten
        act_h_col = torch.flatten(act_h_col, start_dim=1, end_dim=3) # bs x pw
        act_col = self.act_col_last_layer(act_h_col) 
        act_col = F.softmax(act_col, dim=1)

        ## row classification
        act_h_row = self.act_row_layers(h) # bs x 1 x 96 x 1 
        ### flatten
        act_h_row = torch.flatten(act_h_row, start_dim=1, end_dim=3) # bs x ph
        act_row = self.act_row_last_layer(act_h_row) 
        act_row = F.softmax(act_row, dim=1)

        # concat both column and row classification
        action = torch.cat((act_row.unsqueeze(1), act_col.unsqueeze(1)), dim=1)

        # if add_noise:
        #     action = action + torch.from_numpy(randn(bs, 2, self.pw).astype(np.float32)).cuda() * self.noise_scale
        
        
        # critic
        cri_h = F.relu(self.cri_bn0(self.cri_conv0(h))) # 256 x 48 x 48
        cri_h = F.relu(self.cri_bn1(self.cri_conv1(cri_h))) # 256 x 48 x 48
        cri_h = F.relu(self.cri_bn2(self.cri_conv2(cri_h))) # 512 x 24 x 24
        cri_h = F.relu(self.cri_bn3(self.cri_conv3(cri_h))) # 1024 x 12 x 12
        cri_h = self.maxpool(cri_h) # bs x 1024 x 1 x 1
        cri_h = torch.flatten(cri_h, start_dim=1, end_dim=3) # bs x 1024
        cri_h = F.relu(self.cri_fc1(cri_h))
        cri_out = self.cri_fc2(cri_h) # bs x 1
        return  action, cri_out


# coordinate-classify pixel-oriented patch agent
class CoordinateClassifyPixelPatchAgent(torch.nn.Module):
    def __init__(self, config):
        super(CoordinateClassifyPixelPatchAgent, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor
        if self.use_pixel_oriented_patch:
            self.num_actions = config.patch_height * config.patch_width

        self.noise_scale = config.noise_scale
        self.pw = config.patch_width
        self.ph = config.patch_height
        assert self.pw == self.ph , 'The width and height of patch are not equal.'

        # shared backbone
        backbone = []
        backbone += [ Conv2d(3, 64, kernel_size=7, stride=1, padding=3), # 64 x 192x192
                      nn.InstanceNorm2d(64),
                      nn.LeakyReLU(0.2, True),
                      Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 96 x 96
                      nn.InstanceNorm2d(128),
                      nn.LeakyReLU(0.2, True)]
        self.backbone = nn.Sequential(*backbone)

        # actor 
        ## col classification
        col_layers = []
        col_layers += [ Conv2d(128, 256, kernel_size=3, stride=(2,1), padding=1), # 128 x 48 x 96
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(256, 512, kernel_size=3, stride=(2,1), padding=1), # 512 x 24 x 96
                        nn.InstanceNorm2d(512),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(512, 1024, kernel_size=3, stride=(2,1), padding=1), #  1024 x 12 x 96
                        nn.InstanceNorm2d(1024),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((12,1), stride=(12,1)), #  1024 x 1 x 96
                        Conv2d(1024, 1, kernel_size=3, stride=1, padding=1),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 1 x 96
        self.act_col_layers = nn.Sequential(*col_layers)
        self.act_col_last_layer = nn.Linear(config.patch_width, config.patch_width) # bs x 96
                        
        ## row classfication 
        row_layers = []
        row_layers += [ Conv2d(128, 256, kernel_size=3, stride=(1,2), padding=1), # 128 x 96 x 48 
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(256, 512, kernel_size=3, stride=(1,2), padding=1), # 512 x 96 x 24 
                        nn.InstanceNorm2d(512),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(512, 1024, kernel_size=3, stride=(1,2), padding=1), #  1024 x 96 x 12 
                        nn.InstanceNorm2d(1024),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,12), stride=(1,12)), #  1024 x 96 x 1 
                        Conv2d(1024, 1, kernel_size=3, stride=1, padding=1),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 96 x 1
        self.act_row_layers = nn.Sequential(*row_layers)
        self.act_row_last_layer = nn.Linear(config.patch_height, config.patch_height) # bs x 96

        # critic
        self.cri_conv1 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)    # 256 x 48 x 48
        self.cri_bn1 = nn.InstanceNorm2d(256)
        self.cri_conv2 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    # 512 x 24 x 24
        self.cri_bn2 = nn.InstanceNorm2d(512)
        self.cri_conv3 = Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 12 x 12
        self.cri_bn3 = nn.InstanceNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 1024 x 1 x 1
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 1)    # 1
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = self.backbone(x)
        # actor
        ## column classification
        act_h_col = self.act_col_layers(h) # bs x 1 x 1 x 96
        ### flatten
        act_h_col = torch.flatten(act_h_col, start_dim=1, end_dim=3) # bs x pw
        act_col = self.act_col_last_layer(act_h_col) 
        act_col = F.softmax(act_col, dim=1)

        ## row classification
        act_h_row = self.act_row_layers(h) # bs x 1 x 96 x 1 
        ### flatten
        act_h_row = torch.flatten(act_h_row, start_dim=1, end_dim=3) # bs x ph
        act_row = self.act_row_last_layer(act_h_row) 
        act_row = F.softmax(act_row, dim=1)

        # concat both column and row classification
        action = torch.cat((act_row.unsqueeze(1), act_col.unsqueeze(1)), dim=1)

        # if add_noise:
        #     action = action + torch.from_numpy(randn(bs, 2, self.pw).astype(np.float32)).cuda() * self.noise_scale
        
        
        # critic
        cri_h = F.relu(self.cri_bn1(self.cri_conv1(h))) # 256 x 48 x 48
        cri_h = F.relu(self.cri_bn2(self.cri_conv2(cri_h))) # 512 x 24 x 24
        cri_h = F.relu(self.cri_bn3(self.cri_conv3(cri_h))) # 1024 x 12 x 12
        cri_h = self.maxpool(cri_h) # bs x 1024 x 1 x 1
        cri_h = torch.flatten(cri_h, start_dim=1, end_dim=3) # bs x 1024
        cri_h = F.relu(self.cri_fc1(cri_h))
        cri_out = self.cri_fc2(cri_h) # bs x 1
        return  action, cri_out

# pixel-oriented patch agent
class PixelPatchAgent(torch.nn.Module):
    def __init__(self, config):
        super(PixelPatchAgent, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor
        if self.use_pixel_oriented_patch:
            self.num_actions = config.patch_height * config.patch_width

        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 64 x 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 128 x 96 x 96

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(128)
        # actor layers
        self.act_conv1 = Conv2d(128, 64, kernel_size=3, stride=1, padding=1)    # 64 x 96 x 96
        self.act_conv2 = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)    # 1 x 96 x 96
        self.act_bn1 = nn.InstanceNorm2d(64)
        self.act_bn2 = nn.InstanceNorm2d(1)
        self.act_fc1 = Linear(self.num_actions, self.num_actions)    # 4 -> num
        # critic layers
        self.cri_conv1 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)    # 256 x 48 x 48
        self.cri_bn1 = nn.InstanceNorm2d(256)
        self.cri_conv2 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    # 512 x 24 x 24
        self.cri_bn2 = nn.InstanceNorm2d(512)
        self.cri_conv3 = Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 12 x 12
        self.cri_bn3 = nn.InstanceNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 1024 x 1 x 1
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 1)    # 2
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        
        # actor
        act_h = F.relu(self.act_bn1(self.act_conv1(h)))
        act_h = F.relu(self.act_bn2(self.act_conv2(act_h))) # bs x 1 x ph x pw
        # flatten
        act_h = torch.flatten(act_h, start_dim=1, end_dim=3) # bs x (phxpw)
        action = self.act_fc1(act_h)
        if add_noise:
            action = action + torch.from_numpy(randn(bs, self.num_actions).astype(np.float32)).cuda() * self.noise_scale
        
        if self.use_discrete_action:
            action = F.softmax(action, dim=1)
        else:
            action = F.sigmoid(action) 
        
        # critic
        cri_h = F.relu(self.cri_bn1(self.cri_conv1(h))) # 256 x 48 x 48
        cri_h = F.relu(self.cri_bn2(self.cri_conv2(cri_h))) # 512 x 24 x 24
        cri_h = F.relu(self.cri_bn3(self.cri_conv3(cri_h))) # 1024 x 12 x 12
        cri_h = self.maxpool(cri_h) # bs x 1024 x 1 x 1
        cri_h = torch.flatten(cri_h, start_dim=1, end_dim=3) # bs x 1024
        cri_h = F.relu(self.cri_fc1(cri_h))
        cri_out = self.cri_fc2(cri_h) # bs x 1

        return  action, cri_out

class PatchAgent_4(torch.nn.Module):
    def __init__(self, config):
        super(PatchAgent_4, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor

        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 1024 x 12 x 12

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(6, stride=6) # 1024 x 2 x 2
        # actor layers
        self.act_conv1 = Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)    # 1024 x 2 x 2
        self.act_bn1 = nn.BatchNorm2d(1)
        self.act_fc1 = Linear(4, self.num_actions)    # 4 -> num
        # critic layers
        self.cri_conv1 = Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 2 x 2
        self.cri_bn1 = nn.BatchNorm2d(1024)
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 1)    # 2
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        # h = F.relu(self.bn6(self.conv6(h)))

        h_pool = self.maxpool(h) # bs x 1024 x 2 x 2
        bs, ch, _, _ = h_pool.shape
        # h_pool = h_pool.reshape([bs, ch])

        # actor
        act_h = F.relu(self.act_bn1(self.act_conv1(h_pool)))
        # flatten
        act_h = act_h.reshape((bs, 4))
        action = self.act_fc1(act_h)
        if add_noise:
            action = action + torch.from_numpy(randn(bs, self.num_actions).astype(np.float32)).cuda() * self.noise_scale
        
        if self.use_discrete_action:
            action = F.softmax(action, dim=1)
        else:
            action = F.sigmoid(action) 

        # critic
        cri_h = F.relu(self.cri_bn1(self.cri_conv1(h_pool))) # 1024 x 1 x 1
        cri_h = cri_h.reshape((bs, 1024))
        cri_h = F.relu(self.cri_fc1(cri_h))
        cri_out = self.cri_fc2(cri_h) # bs x 1
        
        return  action, cri_out

class PatchAgent_3(torch.nn.Module):
    def __init__(self, config):
        super(PatchAgent_3, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor

        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 1024 x 12 x 12

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(128)
        self.bn3 = nn.InstanceNorm2d(256)
        self.bn4 = nn.InstanceNorm2d(512)
        self.bn5 = nn.InstanceNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 2048 x 1 x 1
        # actor layers
        self.act_fc1 = Linear(1024, 128)    # 1024
        self.act_fc2 = Linear(128, 128)    # 512
        self.act_fc3 = Linear(128, self.num_actions)    # 2
        # critic layers
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 128)    # 512
        self.cri_fc3 = Linear(128, 1)    # 2
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        # h = F.relu(self.bn6(self.conv6(h)))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])

        # actor
        act_h = F.relu(self.act_fc1(h_pool))
        act_h = F.relu(self.act_fc2(act_h))
        action = self.act_fc3(act_h) # bs x 2 
        if add_noise:
            action = action + torch.from_numpy(randn(bs, self.num_actions).astype(np.float32)).cuda() * self.noise_scale
        
        if self.use_discrete_action:
            action = F.softmax(action, dim=1)
        else:
            action = F.sigmoid(action) # bs x 2

        # critic
        cri_h = F.relu(self.cri_fc1(h_pool))
        cri_h = F.relu(self.cri_fc2(cri_h))
        cri_out = self.cri_fc3(cri_h) # bs x 2 
        
        return  action, cri_out

class PatchAgent_2(torch.nn.Module):
    def __init__(self, config):
        super(PatchAgent_2, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor

        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 1024 x 12 x 12

        self.bn1 = nn.InstanceNorm2d(64)
        self.bn2 = nn.InstanceNorm2d(128)
        self.bn3 = nn.InstanceNorm2d(256)
        self.bn4 = nn.InstanceNorm2d(512)
        self.bn5 = nn.InstanceNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(6, stride=6) # 1024 x 2 x 2
        # actor layers
        self.act_conv1 = Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)    # 1024 x 2 x 2
        self.act_bn1 = nn.InstanceNorm2d(1)
        self.act_fc1 = Linear(4, self.num_actions)    # 4 -> num
        # critic layers
        self.cri_conv1 = Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 2 x 2
        self.cri_bn1 = nn.InstanceNorm2d(1024)
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 1)    # 2
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        # h = F.relu(self.bn6(self.conv6(h)))

        h_pool = self.maxpool(h) # bs x 1024 x 2 x 2
        bs, ch, _, _ = h_pool.shape
        # h_pool = h_pool.reshape([bs, ch])

        # actor
        act_h = F.relu(self.act_bn1(self.act_conv1(h_pool)))
        # flatten
        act_h = act_h.reshape((bs, 4))
        action = self.act_fc1(act_h)
        if add_noise:
            action = action + torch.from_numpy(randn(bs, self.num_actions).astype(np.float32)).cuda() * self.noise_scale
        
        if self.use_discrete_action:
            action = F.softmax(action, dim=1)
        else:
            action = F.sigmoid(action) 

        # critic
        cri_h = F.relu(self.cri_bn1(self.cri_conv1(h_pool))) # 1024 x 1 x 1
        cri_h = cri_h.reshape((bs, 1024))
        cri_h = F.relu(self.cri_fc1(cri_h))
        cri_out = self.cri_fc2(cri_h) # bs x 1
        
        return  action, cri_out

class PatchAgent(torch.nn.Module):
    def __init__(self, config):
        super(PatchAgent, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor

        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 1024 x 12 x 12

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 2048 x 1 x 1
        # actor layers
        self.act_fc1 = Linear(1024, 128)    # 1024
        self.act_fc2 = Linear(128, 128)    # 512
        self.act_fc3 = Linear(128, self.num_actions)    # 2
        # critic layers
        self.cri_fc1 = Linear(1024, 128)    # 1024
        self.cri_fc2 = Linear(128, 128)    # 512
        self.cri_fc3 = Linear(128, 1)    # 2
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        # h = F.relu(self.bn6(self.conv6(h)))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])

        # actor
        act_h = F.relu(self.act_fc1(h_pool))
        act_h = F.relu(self.act_fc2(act_h))
        action = self.act_fc3(act_h) # bs x 2 
        if add_noise:
            action = action + torch.from_numpy(randn(bs, self.num_actions).astype(np.float32)).cuda() * self.noise_scale
        
        if self.use_discrete_action:
            action = F.softmax(action, dim=1)
        else:
            action = F.sigmoid(action) # bs x 2

        # critic
        cri_h = F.relu(self.cri_fc1(h_pool))
        cri_h = F.relu(self.cri_fc2(cri_h))
        cri_out = self.cri_fc3(cri_h) # bs x 2 
        
        return  action, cri_out

class PatchPicker(torch.nn.Module):
    def __init__(self, config):
        super(PatchPicker, self).__init__()
        self.no_sigmoid = config.no_sigmoid
        self.predict_shift = config.predict_shift
        self.predict_abs_pos = config.predict_abs_pos
        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 12 x 12
        self.conv6 = Conv2d(1024, 2048, kernel_size=3,  stride=2, padding=1)    # 2048 x 6 x 6
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        self.bn6 = nn.BatchNorm2d(2048)
        # maxpooling
        self.maxpool = nn.MaxPool2d(6, stride=6) # 2048 x 1 x 1
        self.fc1 = Linear(2048, 1024)    # 1024
        self.fc2 = Linear(1024, 512)    # 512
        self.fc3 = Linear(512, 2)    # 2
        

    def forward(self, x, add_noise=True):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])
        h_fc = F.relu(self.fc1(h_pool))
        h_fc = F.relu(self.fc2(h_fc))
        out = self.fc3(h_fc) # bs x 2 
        if self.no_sigmoid: 
            if self.predict_shift or self.predict_abs_pos:
                return out
            # rearrange to [0,1]
            out = out - out.min()
            out = out / out.max()
            return out
        else:
            if add_noise:
                out = out + torch.from_numpy(randn(bs, 2).astype(np.float32)).cuda() * self.noise_scale
            return  F.sigmoid(out)

class PatchCritic(torch.nn.Module):
    def __init__(self):
        super(PatchCritic, self).__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 12 x 12

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 1024 x 1 x 1
        self.fc1 = Linear(2, 1024) 
        self.fc2 = Linear(1024*2, 1024) 
        self.fc3 = Linear(1024, 512)
        self.fc4 = Linear(512, 256) 
        self.fc5 = Linear(256, 1)

    def forward(self, state, action):
        # state : bs x ch x h x w ; action: bs x 2
        h = F.relu(self.bn1(self.conv1(state)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])

        p = F.relu(self.fc1(action))
        val = torch.cat([h_pool, p], dim=1)
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = F.relu(self.fc4(val))
        val = self.fc5(val)

        return  val

class PatchCritic_gt(torch.nn.Module):
    def __init__(self, config):
        super(PatchCritic_gt, self).__init__()

        self.conv1 = Conv2d(3+3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 12 x 12

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 1024 x 1 x 1
        self.fc1 = Linear(2, 1024) 
        self.fc2 = Linear(1024*2, 1024) 
        self.fc3 = Linear(1024, 512)
        self.fc4 = Linear(512, 256) 
        self.fc5 = Linear(256, 1)

    def forward(self, state, action, gt):
        # state : bs x ch x h x w ; action: bs x 2
        h = F.relu(self.bn1(self.conv1(torch.cat([state,gt],dim=1))))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])

        p = F.relu(self.fc1(action))
        val = torch.cat([h_pool, p], dim=1)
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = F.relu(self.fc4(val))
        val = self.fc5(val)

        return  val

class PatchCritic_maskAction(torch.nn.Module): # convert the action to mask
    def __init__(self, config):
        super(PatchCritic_maskAction, self).__init__()
        self.ph = config.patch_height
        self.pw = config.patch_width
        self.H = config.image_height
        self.W = config.image_width
        assert self.ph == self.pw and self.H and self.W , "Patch height must be equal to its width; Same as the image height and width."
        
        self.conv1 = Conv2d(3+1, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 12 x 12
        self.conv6 = Conv2d(1024, 2048, kernel_size=3,  stride=2, padding=1)    # 6 x 6
        # maxpooling
        self.maxpool = nn.MaxPool2d(6, stride=6) # 2048 x 1 x 1
        self.fc1 = Linear(2048, 1024) 
        self.fc2 = Linear(1024, 512) 
        self.fc3 = Linear(512, 256)
        self.fc4 = Linear(256, 128) 
        self.fc5 = Linear(128, 1)

    def convert_action_to_mask(self, action, image):
        # map the position from [0,1] to [0,W], just multiply by width or height
        pos_x, pos_y = (self.W * action[:,0]).ceil(), (self.H * action[:,1]).ceil()
        # print("1 - x:",pos_x[0],"y:",pos_y[0])
        pos_x = torch.tensor(pos_x, dtype=torch.int)
        pos_y = torch.tensor(pos_y, dtype=torch.int)
        # image : b x ch x H x W
        # patch should in the image
        left, top = pos_x - int(self.pw / 2), pos_y - int(self.ph / 2)
        # print("1 - len(left), len(top):",len(left), len(top))
        left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
        # print("2 - len(left), len(top):",len(left), len(top))
        start_x = torch.where(left > (self.W - self.pw), (self.W - self.pw) * torch.ones_like(left), left)
        start_y = torch.where(top > (self.H - self.ph), (self.H - self.ph) * torch.ones_like(top), top)
        # print("len(start_x):",len(start_x))
        # print("len(start_y):",len(start_y))
        bs, ch, h, w = image.shape
        mask_like_action = torch.zeros((bs,1,h,w))
        for i in range(bs):
            mask_like_action[i,:, start_x[i]:start_x[i]+self.pw, start_y[i]:start_y[i]+self.ph] = 1 
        return mask_like_action.cuda()

    def forward(self, state, action):
        # state : bs x ch x h x w ; action: bs x 2
        # convert action to mask
        mask_like_action = self.convert_action_to_mask(action, state)

        h = F.relu(self.conv1(torch.cat([state, mask_like_action], dim=1)))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))

        h = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h.shape
        h = h.reshape([bs, ch])

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h) 
        h = self.fc4(h) 
        val = self.fc5(h)

        return  val

class MyFcn(torch.nn.Module):
    def __init__(self, config):
        super(MyFcn, self).__init__()

        self.noise_scale = config.noise_scale
        self.num_parameters = len(config.parameters_scale)
        ndf = config.hidden_feat_ch
        self.conv1 = Conv2d(3, ndf, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv5_pi = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_pi = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_pi = Conv2d(ndf, config.num_actions, kernel_size=3, stride=1, padding=1)

        self.conv5_V = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_V = Conv2d(ndf + self.num_parameters, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_V = Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)
        
        self.conv5_p = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_p = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_p = Conv2d(ndf, self.num_parameters, kernel_size=3, stride=1, padding=1)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def forward(self, x, flag_a2c=True, add_noise=False):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        if not flag_a2c:
            h = h.detach()

        # pi branch
        h_pi = F.relu(self.conv5_pi(h))
        h_pi = F.relu(self.conv6_pi(h_pi))
        pi_out = F.softmax(self.conv7_pi(h_pi), dim=1)

        # p branch
        p_out = F.relu(self.conv5_p(h))
        p_out = F.relu(self.conv6_p(p_out))
        p_out = self.conv7_p(p_out)
        if flag_a2c:
            if add_noise:
                p_out = p_out.data + torch.from_numpy(randn(p_out.shape[0], p_out.shape[1], 1, 1).astype(np.float32)).cuda() * self.noise_scale
                p_out = Variable(p_out)
            else:
                p_out = p_out.detach()
        p_out = F.sigmoid(p_out)

        # V branch
        h_v = F.relu(self.conv5_V(h))
        h_v = torch.cat((h_v, p_out), dim=1)
        h_v = F.relu(self.conv6_V(h_v))
        v_out = self.conv7_V(h_v)
       
        return pi_out, v_out, self.parse_p(p_out)



class MyFcn_sm(torch.nn.Module):
    def __init__(self, config):
        super(MyFcn_sm, self).__init__()

        self.noise_scale = config.noise_scale
        self.num_parameters = len(config.parameters_scale)

        ndf= 32 
        self.conv1 = Conv2d(3, ndf, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv5_pi = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_pi = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_pi = Conv2d(ndf, config.num_actions, kernel_size=3, stride=1, padding=1)

        self.conv5_V = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_V = Conv2d(ndf + self.num_parameters, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_V = Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1)
        
        self.conv5_p = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_p = Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_p = Conv2d(ndf, self.num_parameters, kernel_size=3, stride=1, padding=1)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def forward(self, x, flag_a2c=True, add_noise=False):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        if not flag_a2c:
            h = h.detach()

        # pi branch
        h_pi = F.relu(self.conv5_pi(h))
        h_pi = F.relu(self.conv6_pi(h_pi))
        pi_out = F.softmax(self.conv7_pi(h_pi), dim=1)

        # p branch
        p_out = F.relu(self.conv5_p(h))
        p_out = F.relu(self.conv6_p(p_out))
        p_out = self.conv7_p(p_out)
        if flag_a2c:
            if add_noise:
                p_out = p_out.data + torch.from_numpy(randn(p_out.shape[0], p_out.shape[1], 1, 1).astype(np.float32)).cuda() * self.noise_scale
                p_out = Variable(p_out)
            else:
                p_out = p_out.detach()
        p_out = F.sigmoid(p_out)

        # V branch
        h_v = F.relu(self.conv5_V(h))
        h_v = torch.cat((h_v, p_out), dim=1)
        h_v = F.relu(self.conv6_V(h_v))
        v_out = self.conv7_V(h_v)
       
        return pi_out, v_out, self.parse_p(p_out)


class MyFcn_noCritic(torch.nn.Module):
    def __init__(self, config):
        super(MyFcn_noCritic, self).__init__()

        self.noise_scale = config.noise_scale
        self.num_parameters = len(config.parameters_scale)

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv5_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_pi = Conv2d(64, config.num_actions, kernel_size=3, stride=1, padding=1)

        # self.conv5_V = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        # self.conv6_V = Conv2d(64 + self.num_parameters, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.conv7_V = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.conv5_p = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_p = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_p = Conv2d(64, self.num_parameters, kernel_size=3, stride=1, padding=1)

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def forward(self, x, flag_a2c=True, add_noise=False):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        if not flag_a2c:
            h = h.detach()

        # pi branch
        h_pi = F.relu(self.conv5_pi(h))
        h_pi = F.relu(self.conv6_pi(h_pi))
        pi_out = F.softmax(self.conv7_pi(h_pi), dim=1)

        # p branch
        p_out = F.relu(self.conv5_p(h))
        p_out = F.relu(self.conv6_p(p_out))
        p_out = self.conv7_p(p_out)
        if flag_a2c:
            if add_noise:
                p_out = p_out.data + torch.from_numpy(randn(p_out.shape[0], p_out.shape[1], 1, 1).astype(np.float32)).cuda() * self.noise_scale
                p_out = Variable(p_out)
            else:
                p_out = p_out.detach()
        p_out = F.sigmoid(p_out)

        # V branch
        # h_v = F.relu(self.conv5_V(h))
        # h_v = torch.cat((h_v, p_out), dim=1)
        # h_v = F.relu(self.conv6_V(h_v))
        # v_out = self.conv7_V(h_v)
       
        return pi_out, None, self.parse_p(p_out)

# coordinate-classify pixel-oriented patch agent
class CoordinateClassifyPixelPatchAgent_noCritic(torch.nn.Module):
    def __init__(self, config):
        super(CoordinateClassifyPixelPatchAgent_noCritic, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor
        if self.use_pixel_oriented_patch:
            self.num_actions = config.patch_height * config.patch_width

        self.noise_scale = config.noise_scale
        self.pw = config.patch_width
        self.ph = config.patch_height
        assert self.pw == self.ph , 'The width and height of patch are not equal.'

        # shared backbone
        backbone = []
        backbone += [ Conv2d(3, 64, kernel_size=7, stride=1, padding=3), # 64 x 192x192
                      nn.InstanceNorm2d(64),
                      nn.LeakyReLU(0.2, True),
                      Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 128 x 96 x 96
                      nn.InstanceNorm2d(128),
                      nn.LeakyReLU(0.2, True)]
        self.backbone = nn.Sequential(*backbone)

        # actor 
        ## col classification
        col_layers = []
        col_layers += [ Conv2d(128, 256, kernel_size=3, stride=(2,1), padding=1), # 128 x 48 x 96
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(256, 512, kernel_size=3, stride=(2,1), padding=1), # 512 x 24 x 96
                        nn.InstanceNorm2d(512),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(512, 1024, kernel_size=3, stride=(2,1), padding=1), #  1024 x 12 x 96
                        nn.InstanceNorm2d(1024),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((12,1), stride=(12,1)), #  1024 x 1 x 96
                        Conv2d(1024, 1, kernel_size=3, stride=1, padding=1),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 1 x 96
        self.act_col_layers = nn.Sequential(*col_layers)
        self.act_col_last_layer = nn.Linear(config.patch_width, config.patch_width) # bs x 96
                        
        ## row classfication 
        row_layers = []
        row_layers += [ Conv2d(128, 256, kernel_size=3, stride=(1,2), padding=1), # 128 x 96 x 48 
                        nn.InstanceNorm2d(256),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(256, 512, kernel_size=3, stride=(1,2), padding=1), # 512 x 96 x 24 
                        nn.InstanceNorm2d(512),
                        nn.LeakyReLU(0.2, True),
                        Conv2d(512, 1024, kernel_size=3, stride=(1,2), padding=1), #  1024 x 96 x 12 
                        nn.InstanceNorm2d(1024),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,12), stride=(1,12)), #  1024 x 96 x 1 
                        Conv2d(1024, 1, kernel_size=3, stride=1, padding=1),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 96 x 1
        self.act_row_layers = nn.Sequential(*row_layers)
        self.act_row_last_layer = nn.Linear(config.patch_height, config.patch_height) # bs x 96

        # critic
        # self.cri_conv1 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)    # 256 x 48 x 48
        # self.cri_bn1 = nn.InstanceNorm2d(256)
        # self.cri_conv2 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    # 512 x 24 x 24
        # self.cri_bn2 = nn.InstanceNorm2d(512)
        # self.cri_conv3 = Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 12 x 12
        # self.cri_bn3 = nn.InstanceNorm2d(1024)
        # # maxpooling
        # self.maxpool = nn.MaxPool2d(12, stride=12) # 1024 x 1 x 1
        # self.cri_fc1 = Linear(1024, 128)    # 1024
        # self.cri_fc2 = Linear(128, 1)    # 1
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = self.backbone(x)
        # actor
        ## column classification
        act_h_col = self.act_col_layers(h) # bs x 1 x 1 x 96
        ### flatten
        act_h_col = torch.flatten(act_h_col, start_dim=1, end_dim=3) # bs x pw
        act_col = self.act_col_last_layer(act_h_col) 
        act_col = F.softmax(act_col, dim=1)

        ## row classification
        act_h_row = self.act_row_layers(h) # bs x 1 x 96 x 1 
        ### flatten
        act_h_row = torch.flatten(act_h_row, start_dim=1, end_dim=3) # bs x ph
        act_row = self.act_row_last_layer(act_h_row) 
        act_row = F.softmax(act_row, dim=1)

        # concat both column and row classification
        action = torch.cat((act_row.unsqueeze(1), act_col.unsqueeze(1)), dim=1)

        # if add_noise:
        #     action = action + torch.from_numpy(randn(bs, 2, self.pw).astype(np.float32)).cuda() * self.noise_scale
        
        
        # critic
        # cri_h = F.relu(self.cri_bn1(self.cri_conv1(h))) # 256 x 48 x 48
        # cri_h = F.relu(self.cri_bn2(self.cri_conv2(cri_h))) # 512 x 24 x 24
        # cri_h = F.relu(self.cri_bn3(self.cri_conv3(cri_h))) # 1024 x 12 x 12
        # cri_h = self.maxpool(cri_h) # bs x 1024 x 1 x 1
        # cri_h = torch.flatten(cri_h, start_dim=1, end_dim=3) # bs x 1024
        # cri_h = F.relu(self.cri_fc1(cri_h))
        # cri_out = self.cri_fc2(cri_h) # bs x 1
        return  action, None #cri_out




# coordinate-classify pixel-oriented patch agent
class CoordinateClassifyPixelPatchAgent_sm_noCritic(torch.nn.Module):
    def __init__(self, config):
        super(CoordinateClassifyPixelPatchAgent_sm_noCritic, self).__init__()
        # self.no_sigmoid = config.no_sigmoid
        # self.predict_shift = config.predict_shift
        # self.predict_abs_pos = config.predict_abs_pos
        # for continuous action
        self.num_actions = 2
        # for discrete action
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        if self.use_discrete_action:
            self.num_actions = config.factor * config.factor
        if self.use_pixel_oriented_patch:
            self.num_actions = config.patch_height * config.patch_width

        self.noise_scale = config.noise_scale
        self.pw = config.patch_width
        self.ph = config.patch_height
        assert self.pw == self.ph , 'The width and height of patch are not equal.'

        # shared backbone
        backbone = []
        backbone += [ Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 64 x 192x192
                      nn.InstanceNorm2d(64),
                      nn.LeakyReLU(0.2, True),
                      nn.MaxPool2d(2, stride=2), #  128 x 96 x 96
                      Conv2d(64, 64, kernel_size=1, stride=1, padding=0), # 128 x 96 x 96
                      nn.InstanceNorm2d(64),
                      nn.LeakyReLU(0.2, True)
                      ]
        self.backbone = nn.Sequential(*backbone)

        # actor 
        ## col classification
        col_layers = []
        col_layers += [ nn.MaxPool2d((2,1), stride=(2,1)), #  128 x 48 x 96
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), # 128 x 48 x 96
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((2,1), stride=(2,1)), #  128 x 24 x 96
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 24 x 96
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((2,1), stride=(2,1)), #  128 x 24 x 96
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 12 x 96
                        nn.InstanceNorm2d(64),
                        nn.MaxPool2d((12,1), stride=(12,1)), #  128 x 1 x 96
                        Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 1 x 96
        self.act_col_layers = nn.Sequential(*col_layers)
        self.act_col_last_layer = nn.Linear(config.patch_width, config.patch_width) # bs x 96
                        
        ## row classfication 
        row_layers = []
        row_layers += [ nn.MaxPool2d((1,2), stride=(1,2)), #  128 x 96 x 48
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), # 128 x 96 x 48 
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,2), stride=(1,2)), #  128 x 96 x 24
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 96 x 24
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,2), stride=(1,2)), #  128 x 96 x 12
                        Conv2d(64, 64, kernel_size=1, stride=(1,1), padding=0), #  128 x 96 x 12
                        nn.InstanceNorm2d(64),
                        nn.LeakyReLU(0.2, True),
                        nn.MaxPool2d((1,12), stride=(1,12)), #  128 x 96 x 1
                        Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
                        nn.InstanceNorm2d(1),
                        nn.LeakyReLU(0.2, True)] #  1 x 96 x 1
        self.act_row_layers = nn.Sequential(*row_layers)
        self.act_row_last_layer = nn.Linear(config.patch_height, config.patch_height) # bs x 96

        # critic
        # self.cri_conv0 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 128 x 48 x 48
        # self.cri_bn0 = nn.InstanceNorm2d(128)
        # self.cri_conv1 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)    # 256 x 24 x 24
        # self.cri_bn1 = nn.InstanceNorm2d(256)
        # self.cri_conv2 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)    # 512 x 12 x 12
        # self.cri_bn2 = nn.InstanceNorm2d(512)
        # self.cri_conv3 = Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)    # 1024 x 6 x 6
        # self.cri_bn3 = nn.InstanceNorm2d(1024)
        # # maxpooling
        # self.maxpool = nn.MaxPool2d(6, stride=6) # 1024 x 1 x 1
        # self.cri_fc1 = Linear(1024, 128)    # 1024
        # self.cri_fc2 = Linear(128, 1)    # 1
        

    def forward(self, x, add_noise=False):
        # shared backbone
        h = self.backbone(x)
        # actor
        ## column classification
        act_h_col = self.act_col_layers(h) # bs x 1 x 1 x 96
        ### flatten
        act_h_col = torch.flatten(act_h_col, start_dim=1, end_dim=3) # bs x pw
        act_col = self.act_col_last_layer(act_h_col) 
        act_col = F.softmax(act_col, dim=1)

        ## row classification
        act_h_row = self.act_row_layers(h) # bs x 1 x 96 x 1 
        ### flatten
        act_h_row = torch.flatten(act_h_row, start_dim=1, end_dim=3) # bs x ph
        act_row = self.act_row_last_layer(act_h_row) 
        act_row = F.softmax(act_row, dim=1)

        # concat both column and row classification
        action = torch.cat((act_row.unsqueeze(1), act_col.unsqueeze(1)), dim=1)

        # if add_noise:
        #     action = action + torch.from_numpy(randn(bs, 2, self.pw).astype(np.float32)).cuda() * self.noise_scale
        
        
        # # critic
        # cri_h = F.relu(self.cri_bn0(self.cri_conv0(h))) # 256 x 48 x 48
        # cri_h = F.relu(self.cri_bn1(self.cri_conv1(cri_h))) # 256 x 48 x 48
        # cri_h = F.relu(self.cri_bn2(self.cri_conv2(cri_h))) # 512 x 24 x 24
        # cri_h = F.relu(self.cri_bn3(self.cri_conv3(cri_h))) # 1024 x 12 x 12
        # cri_h = self.maxpool(cri_h) # bs x 1024 x 1 x 1
        # cri_h = torch.flatten(cri_h, start_dim=1, end_dim=3) # bs x 1024
        # cri_h = F.relu(self.cri_fc1(cri_h))
        # cri_out = self.cri_fc2(cri_h) # bs x 1
        return  action, None #cri_out

