import numpy as np
from numpy.random import randn
import torch
from torch.nn import Conv2d, Linear, Sigmoid
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn


class PatchPicker(torch.nn.Module):
    def __init__(self, config):
        super(PatchPicker, self).__init__()

        self.noise_scale = config.noise_scale
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=1, padding=3)      # 192x192
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)    # 96 x 96
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2, padding=1)   # 48 x 48
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2, padding=1)   # 24 x 24
        self.conv5 = Conv2d(512, 1024, kernel_size=3,  stride=2, padding=1)    # 12 x 12
        self.conv6 = Conv2d(1024, 2048, kernel_size=3,  stride=2, padding=1)    # 2048 x 6 x 6
        # maxpooling
        self.maxpool = nn.MaxPool2d(6, stride=6) # 2048 x 1 x 1
        self.fc1 = Linear(2048, 1024)    # 1024
        self.fc2 = Linear(1024, 512)    # 512
        self.fc3 = Linear(512, 2)    # 2
        

    def forward(self, x, add_noise=True):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])
        h_fc = self.fc1(h_pool)
        h_fc = self.fc2(h_fc)
        out = self.fc3(h_fc) # bs x 2 
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
        # maxpooling
        self.maxpool = nn.MaxPool2d(12, stride=12) # 1024 x 1 x 1
        self.fc1 = Linear(2, 1024) 
        self.fc2 = Linear(1024*2, 1024) 
        self.fc3 = Linear(1024, 512)
        self.fc4 = Linear(512, 256) 
        self.fc5 = Linear(256, 1)

    def forward(self, state, action):
        # state : bs x ch x h x w ; action: bs x 2
        h = F.relu(self.conv1(state))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))

        h_pool = self.maxpool(h) # bs x 2048 x 1 x 1
        bs, ch, _, _ = h_pool.shape
        h_pool = h_pool.reshape([bs, ch])

        p = self.fc1(action)
        val = torch.cat([h_pool, p], dim=1)
        val = self.fc2(val)
        val = self.fc3(val) 
        val = self.fc4(val) 
        val = self.fc5(val)

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


class MyFcn_bkp(torch.nn.Module):
    def __init__(self, config):
        super(MyFcn_bkp, self).__init__()

        self.noise_scale = config.noise_scale
        self.num_parameters = len(config.parameters_scale)

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv5_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_pi = Conv2d(64, config.num_actions, kernel_size=3, stride=1, padding=1)

        self.conv5_V = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_V = Conv2d(64 + self.num_parameters, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_V = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
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
        h_v = F.relu(self.conv5_V(h))
        h_v = torch.cat((h_v, p_out), dim=1)
        h_v = F.relu(self.conv6_V(h_v))
        v_out = self.conv7_V(h_v)
       
        return pi_out, v_out, self.parse_p(p_out)
