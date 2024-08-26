import numpy as np
import sys
import cv2
import torch
# from skimage.measure import compare_ssim
import pytorch_ssim

class Env_patch():
    def __init__(self, config):
        self.image = None
        self.previous_image = None
        # setting for patch cropper
        self.ph = config.patch_height
        self.pw = config.patch_width
        self.H = config.image_height
        self.W = config.image_width
        assert self.ph == self.pw and self.H and self.W , "Patch height must be equal to its width; Same as the image height and width."
        self.start_x = None
        self.start_y = None
    
    def reset(self, ori_image, image):
        self.ori_image = ori_image.copy()
        self.image = image.copy()
        self.previous_image = None
        return

    def step(self, actions):
        self.previous_image = self.image.copy()
        pos_x = actions[:,0]
        pos_y = actions[:,1]
        # crop the patches and obtain the start_x and start_y
        cropped_image, cropped_ori = self.crop_patch(pos_x, pos_y)
        # replace the lr patches with hr patches
        self.recover_patches(cropped_ori)
        # reward = pytorch_ssim.ssim(torch.from_numpy(self.ori_image).cuda(),torch.from_numpy(self.previous_image).cuda()) \
        #         - pytorch_ssim.ssim(torch.from_numpy(self.ori_image).cuda(),torch.from_numpy(self.image).cuda())
        bs, ch, h, w = self.ori_image.shape
        reward = np.sum(np.abs(self.ori_image - self.previous_image) * 255  - np.abs(self.ori_image - self.image) * 255, axis=(1,2,3)) / (ch*h*w)
        return self.image, reward
    
    def crop_patch(self, pos_x, pos_y):
        # map the position from [0,1] to [0,W], just multiply by width or height
        # print("0 - x:",pos_x[0],"y:",pos_y[0])
        pos_x, pos_y = (self.W * pos_x).ceil(), (self.H * pos_y).ceil()
        # print("1 - x:",pos_x[0],"y:",pos_y[0])
        # pos_x = torch.tensor(pos_x, dtype=torch.int)
        # pos_y = torch.tensor(pos_y, dtype=torch.int)
        pos_x = pos_x.int()
        pos_y = pos_y.int()
        # print("2 - x:",pos_x[0],"y:",pos_y[0])
        # image : b x ch x H x W
        # patch should in the image
        left, top = pos_x - int(self.pw / 2), pos_y - int(self.ph / 2)
        # print("1 - len(left), len(top):",len(left), len(top))
        left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
        # print("2 - len(left), len(top):",len(left), len(top))
        start_x = torch.where(left > (self.W - self.pw), (self.W - self.pw) * torch.ones_like(left), left)
        start_y = torch.where(top > (self.H - self.ph), (self.H - self.ph) * torch.ones_like(top), top)
        # print("3 - x:",start_x[0],"y:",start_y[0])
        # print("len(start_x):",len(start_x))
        # print("len(start_y):",len(start_y))
        bs, ch, h, w = self.image.shape
        patch_list = []
        gt_patch_list = []
        for i in range(bs):
            patch_list.append(self.image[i:i+1, :, start_x[i]:start_x[i]+self.pw, start_y[i]:start_y[i]+self.ph]) # 1*C*patch_w*patch_h
            gt_patch_list.append(self.ori_image[i:i+1, :, start_x[i]:start_x[i]+self.pw, start_y[i]:start_y[i]+self.ph]) # 1*C*patch_w*patch_h
        
        output = np.concatenate(patch_list, axis=0) # B*C*patch_w*patch_h
        gt = np.concatenate(gt_patch_list, axis=0)

        self.start_x = start_x
        self.start_y = start_y

        return output, gt
        
    def recover_patches(self, recovered_patches):
        bs, ch, h, w = recovered_patches.shape
        for i in range(bs):
            self.image[i,:,self.start_x[i]:self.start_x[i]+self.pw, self.start_y[i]:self.start_y[i]+self.ph] = recovered_patches[i]
        return 


class Env():
    def __init__(self, config):
        self.image = None
        self.previous_image = None

        self.num_actions = config.num_actions
        self.actions = config.actions

        self.parameters_scale = config.parameters_scale
        self.parameters = dict()
        self.set_param([0.5] * len(self.parameters_scale))

        self.reward_method = config.reward_method 
        self.no_addition_subtraction = config.no_addition_subtraction
        self.shift_add_sub = config.shift_add_sub
        self.useless_actions_list = config.useless_actions_list
        self.use_diag_sobel = False
        self.diag_sobel_index = []
        self.use_hor_ver_sobel = True
        if hasattr(config, "use_diag_sobel"):
            self.use_diag_sobel = config.use_diag_sobel
            self.diag_sobel_index = config.diag_sobel_index
        if hasattr(config, "rm_hor_ver_sobel"):
            self.use_hor_ver_sobel = not config.rm_hor_ver_sobel
        
    def reset(self, ori_image, image):
        self.ori_image = ori_image
        self.image = image
        self.previous_image = None

        return

    def set_param(self, p):
        for i, k in enumerate(sorted(self.parameters_scale.keys())):
            self.parameters[k] = p[i] * self.parameters_scale[k]
        return

    def step(self, act):
        self.previous_image = self.image.copy()
        # print("===========================")
        # print("self.image:",self.image.shape,self.image.max(),self.image.min(),self.image.dtype)
        canvas = [np.zeros(self.image.shape, self.image.dtype) for _ in range(self.num_actions + 1)]
        b, c, h, w = self.image.shape
        for i in range(b):
            # do nothing
            canvas[0][i] = self.image[i]

            if not self.no_addition_subtraction:
                canvas[self.actions['subtraction']][i] = self.image[i] - self.shift_add_sub / 255
                canvas[self.actions['addition']][i] = self.image[i] + self.shift_add_sub / 255

            if ( not 'box' in self.useless_actions_list) and np.sum(act[i] == self.actions['box']) > 0:
                canvas[self.actions['box']][i] = np.transpose(cv2.boxFilter(self.image[i].transpose([1,2,0]), ddepth=-1, ksize=(5,5)),[2,0,1])

            if ( not 'bilateral' in self.useless_actions_list) and np.sum(act[i] == self.actions['bilateral']) > 0:
                canvas[self.actions['bilateral']][i] = np.transpose(cv2.bilateralFilter(self.image[i].transpose([1,2,0]), d=5, sigmaColor=0.1, sigmaSpace=5),[2,0,1])

            if True:
                canvas[self.actions['Gaussian']][i] = np.transpose(cv2.GaussianBlur(self.image[i].transpose([1,2,0]), ksize=(5,5), sigmaX=0.5),[2,0,1])

            if ( not 'median' in self.useless_actions_list) and np.sum(act[i] == self.actions['median']) > 0:
                canvas[self.actions['median']][i] = np.transpose(cv2.medianBlur(self.image[i].transpose([1,2,0]), ksize=5),[2,0,1])

            if np.sum(act[i] == self.actions['Laplace']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['unsharp']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp']][i] = self.image[i] * (1 + amount) - canvas[self.actions['Gaussian']][i] * amount

            if self.use_hor_ver_sobel:
                if np.sum(act[i] == self.actions['Sobel_v1']) > 0:
                    p = self.parameters['Sobel_v1'][i]
                    k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                    canvas[self.actions['Sobel_v1']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

                if np.sum(act[i] == self.actions['Sobel_v2']) > 0:
                    p = self.parameters['Sobel_v2'][i]
                    k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                    canvas[self.actions['Sobel_v2']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

                if np.sum(act[i] == self.actions['Sobel_h1']) > 0:
                    p = self.parameters['Sobel_h1'][i]
                    k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                    canvas[self.actions['Sobel_h1']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

                if np.sum(act[i] == self.actions['Sobel_h2']) > 0:
                    p = self.parameters['Sobel_h2'][i]
                    k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                    canvas[self.actions['Sobel_h2']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if self.use_diag_sobel:
                if 1 in self.diag_sobel_index and np.sum(act[i] == self.actions['Sobel_d1']) > 0:
                    p = self.parameters['Sobel_d1'][i]
                    k = np.array([[-2 * p, -p, 0],[-p, 1, p],[0, p, 2 * p]])
                    canvas[self.actions['Sobel_d1']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

                if 2 in self.diag_sobel_index and np.sum(act[i] == self.actions['Sobel_d2']) > 0:
                    p = self.parameters['Sobel_d2'][i]
                    k = np.array([[0, -p, -2 * p],[p, 1, -p],[2 * p, p, 0]])
                    canvas[self.actions['Sobel_d2']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

                if 3 in self.diag_sobel_index and np.sum(act[i] == self.actions['Sobel_d3']) > 0:
                    p = self.parameters['Sobel_d3'][i]
                    k = np.array([[2 * p, p, 0],[p, 1, -p],[0, -p, -2 * p]])
                    canvas[self.actions['Sobel_d3']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

                if 4 in self.diag_sobel_index and np.sum(act[i] == self.actions['Sobel_d4']) > 0:
                    p = self.parameters['Sobel_d4'][i]
                    k = np.array([[0, p, 2 * p],[-p, 1, p],[-2 * p, -p, 0]])
                    canvas[self.actions['Sobel_d4']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])


        for a in range(1, self.num_actions + 1):
            self.image = np.where(act[:,np.newaxis,:,:] == a, canvas[a], self.image)
        self.image = np.clip(self.image, 0, 1)

        if self.reward_method == 'square':
            reward = np.square(self.ori_image - self.previous_image) * 255 - np.square(self.ori_image - self.image) * 255
        elif self.reward_method == 'abs':
            reward = np.abs(self.ori_image - self.previous_image) * 255 - np.abs(self.ori_image - self.image) * 255

        return self.image, reward 

class Env_RGB():
    def __init__(self, config):
        self.image = None
        self.previous_image = None

        self.num_actions = config.num_actions
        self.actions = config.actions

        self.parameters_scale = config.parameters_scale
        self.parameters = dict()
        self.set_param([0.5] * len(self.parameters_scale))

        self.reward_method = config.reward_method 
        self.no_addition_subtraction = config.no_addition_subtraction
        self.shift_add_sub = config.shift_add_sub
        self.useless_actions_list = config.useless_actions_list

    def reset(self, ori_image, image):
        self.ori_image = ori_image
        self.image = image
        self.previous_image = None

        return

    def set_param(self, p):
        for i, k in enumerate(sorted(self.parameters_scale.keys())):
            self.parameters[k] = p[i] * self.parameters_scale[k]
        return

    def step(self, act):
        self.previous_image = self.image.copy()
        # print("===========================")
        # print("self.image:",self.image.shape,self.image.max(),self.image.min(),self.image.dtype)
        canvas = [np.zeros(self.image.shape, self.image.dtype) for _ in range(self.num_actions + 1)]
        b, c, h, w = self.image.shape
        
        for i in range(b):
            # do nothing
            canvas[0][i] = self.image[i]
            if not self.no_addition_subtraction : 
                # subtraction for R/G/B
                # subtraction for R
                canvas[self.actions['subtraction_R']][i, 0] = ( self.image[i, 0] * 255 - self.shift_add_sub ) / 255.
                canvas[self.actions['subtraction_R']][i, 1:] = self.image[i, 1:] 
                # subtraction for G
                canvas[self.actions['subtraction_G']][i, 1] = ( self.image[i, 1] * 255 - self.shift_add_sub ) / 255.
                canvas[self.actions['subtraction_G']][i, 0::2] = self.image[i, 0::2] 
                # subtraction for G
                canvas[self.actions['subtraction_B']][i, 2] = ( self.image[i, 2] * 255 - self.shift_add_sub ) / 255.
                canvas[self.actions['subtraction_B']][i, 0:2] = self.image[i, 0:2] 

                # addition for R/G/B
                # addition for R
                canvas[self.actions['addition_R']][i, 0] = ( self.image[i, 0] * 255 + self.shift_add_sub ) / 255.
                canvas[self.actions['addition_R']][i, 1:] = self.image[i, 1:] 
                # addition for G
                canvas[self.actions['addition_G']][i, 1] = ( self.image[i, 1] * 255 + self.shift_add_sub ) / 255.
                canvas[self.actions['addition_G']][i, 0::2] = self.image[i, 0::2] 
                # addition for G
                canvas[self.actions['addition_B']][i, 2] = ( self.image[i, 2] * 255 + self.shift_add_sub ) / 255.
                canvas[self.actions['addition_B']][i, 0:2] = self.image[i, 0:2] 


                # subtraction & addition for RGB 
                canvas[self.actions['subtraction']][i] = ( self.image[i] * 255 - self.shift_add_sub ) / 255
                canvas[self.actions['addition']][i] = ( self.image[i] * 255 + self.shift_add_sub ) / 255

            if ( not 'box' in self.useless_actions_list) and np.sum(act[i] == self.actions['box']) > 0:
                canvas[self.actions['box']][i] = np.transpose(cv2.boxFilter(self.image[i].transpose([1,2,0]), ddepth=-1, ksize=(5,5)),[2,0,1])

            if ( not 'bilateral' in self.useless_actions_list) and np.sum(act[i] == self.actions['bilateral']) > 0:
                canvas[self.actions['bilateral']][i] = np.transpose(cv2.bilateralFilter(self.image[i].transpose([1,2,0]), d=5, sigmaColor=0.1, sigmaSpace=5),[2,0,1])

            if True:
                canvas[self.actions['Gaussian']][i] = np.transpose(cv2.GaussianBlur(self.image[i].transpose([1,2,0]), ksize=(5,5), sigmaX=0.5),[2,0,1])

            if ( not 'median' in self.useless_actions_list) and  np.sum(act[i] == self.actions['median']) > 0:
                canvas[self.actions['median']][i] = np.transpose(cv2.medianBlur(self.image[i].transpose([1,2,0]), ksize=5),[2,0,1])

            if np.sum(act[i] == self.actions['Laplace']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['unsharp']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp']][i] = self.image[i] * (1 + amount) - canvas[self.actions['Gaussian']][i] * amount

            if np.sum(act[i] == self.actions['Sobel_v1']) > 0:
                p = self.parameters['Sobel_v1'][i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                canvas[self.actions['Sobel_v1']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['Sobel_v2']) > 0:
                p = self.parameters['Sobel_v2'][i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                canvas[self.actions['Sobel_v2']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['Sobel_h1']) > 0:
                p = self.parameters['Sobel_h1'][i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                canvas[self.actions['Sobel_h1']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['Sobel_h2']) > 0:
                p = self.parameters['Sobel_h2'][i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                canvas[self.actions['Sobel_h2']][i] = np.transpose(cv2.filter2D(self.image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            ## -------------------- actions for R channel -------------------- ##
            if ( not 'box' in self.useless_actions_list) and np.sum(act[i] == self.actions['box_R']) > 0:
                canvas[self.actions['box_R']][i, 0] = cv2.boxFilter(self.image[i, 0], ddepth=-1, ksize=(5,5))
                canvas[self.actions['box_R']][i, 1:] = self.image[i, 1:] 

            if ( not 'bilateral' in self.useless_actions_list) and np.sum(act[i] == self.actions['bilateral_R']) > 0:
                canvas[self.actions['bilateral_R']][i, 0] = cv2.bilateralFilter(self.image[i, 0], d=5, sigmaColor=0.1, sigmaSpace=5)
                canvas[self.actions['bilateral_R']][i, 1:] = self.image[i, 1:] 

            if True:
                canvas[self.actions['Gaussian_R']][i, 0] = cv2.GaussianBlur(self.image[i, 0], ksize=(5,5), sigmaX=0.5)
                canvas[self.actions['Gaussian_R']][i, 1:] = self.image[i, 1:] 

            if ( not 'median' in self.useless_actions_list) and np.sum(act[i] == self.actions['median_R']) > 0:
                canvas[self.actions['median_R']][i, 0] = cv2.medianBlur(self.image[i, 0], ksize=5)
                canvas[self.actions['median_R']][i, 1:] = self.image[i, 1:] 

            if np.sum(act[i] == self.actions['Laplace_R']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace_R']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
                canvas[self.actions['Laplace_R']][i, 1:] = self.image[i, 1:] 

            if np.sum(act[i] == self.actions['unsharp_R']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp_R']][i, 0] = self.image[i, 0] * (1 + amount) - canvas[self.actions['Gaussian_R']][i, 0] * amount
                canvas[self.actions['unsharp_R']][i, 1:] = self.image[i, 1:] 

            if np.sum(act[i] == self.actions['Sobel_v1_R']) > 0:
                p = self.parameters['Sobel_v1'][i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                canvas[self.actions['Sobel_v1_R']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
                canvas[self.actions['Sobel_v1_R']][i, 1:] = self.image[i, 1:] 

            if np.sum(act[i] == self.actions['Sobel_v2_R']) > 0:
                p = self.parameters['Sobel_v2'][i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                canvas[self.actions['Sobel_v2_R']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
                canvas[self.actions['Sobel_v2_R']][i, 1:] = self.image[i, 1:] 

            if np.sum(act[i] == self.actions['Sobel_h1_R']) > 0:
                p = self.parameters['Sobel_h1'][i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                canvas[self.actions['Sobel_h1_R']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
                canvas[self.actions['Sobel_h1_R']][i, 1:] = self.image[i, 1:] 

            if np.sum(act[i] == self.actions['Sobel_h2_R']) > 0:
                p = self.parameters['Sobel_h2'][i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                canvas[self.actions['Sobel_h2_R']][i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
                canvas[self.actions['Sobel_h2_R']][i, 1:] = self.image[i, 1:] 

            ## -------------------- actions for G channel -------------------- ##
            if  ( not 'box' in self.useless_actions_list) and np.sum(act[i] == self.actions['box_G']) > 0:
                canvas[self.actions['box_G']][i, 1] = cv2.boxFilter(self.image[i, 1], ddepth=-1, ksize=(5,5))
                canvas[self.actions['box_G']][i, 0::2] = self.image[i, 0::2] 

            if ( not 'bilateral' in self.useless_actions_list) and np.sum(act[i] == self.actions['bilateral_G']) > 0:
                canvas[self.actions['bilateral_G']][i, 1] = cv2.bilateralFilter(self.image[i, 1], d=5, sigmaColor=0.1, sigmaSpace=5)
                canvas[self.actions['bilateral_G']][i, 0::2] = self.image[i, 0::2] 

            if True:
                canvas[self.actions['Gaussian_G']][i, 1] = cv2.GaussianBlur(self.image[i, 1], ksize=(5,5), sigmaX=0.5)
                canvas[self.actions['Gaussian_G']][i, 0::2] = self.image[i, 0::2] 

            if ( not 'median' in self.useless_actions_list) and np.sum(act[i] == self.actions['median_G']) > 0:
                canvas[self.actions['median_G']][i, 1] = cv2.medianBlur(self.image[i, 1], ksize=5)
                canvas[self.actions['median_G']][i, 0::2] = self.image[i, 0::2] 

            if np.sum(act[i] == self.actions['Laplace_G']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace_G']][i, 1] = cv2.filter2D(self.image[i, 1], -1, kernel=k)
                canvas[self.actions['Laplace_G']][i, 0::2] = self.image[i, 0::2] 

            if np.sum(act[i] == self.actions['unsharp_G']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp_G']][i, 1] = self.image[i, 1] * (1 + amount) - canvas[self.actions['Gaussian_G']][i,1] * amount
                canvas[self.actions['unsharp_G']][i, 0::2] = self.image[i, 0::2] 

            if np.sum(act[i] == self.actions['Sobel_v1_G']) > 0:
                p = self.parameters['Sobel_v1'][i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                canvas[self.actions['Sobel_v1_G']][i, 1] = cv2.filter2D(self.image[i, 1], -1, kernel=k)
                canvas[self.actions['Sobel_v1_G']][i, 0::2] = self.image[i, 0::2] 

            if np.sum(act[i] == self.actions['Sobel_v2_G']) > 0:
                p = self.parameters['Sobel_v2'][i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                canvas[self.actions['Sobel_v2_G']][i, 1] = cv2.filter2D(self.image[i, 1], -1, kernel=k)
                canvas[self.actions['Sobel_v2_G']][i, 0::2] = self.image[i, 0::2] 

            if np.sum(act[i] == self.actions['Sobel_h1_G']) > 0:
                p = self.parameters['Sobel_h1'][i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                canvas[self.actions['Sobel_h1_G']][i, 1] = cv2.filter2D(self.image[i, 1], -1, kernel=k)
                canvas[self.actions['Sobel_h1_G']][i, 0::2] = self.image[i, 0::2] 

            if np.sum(act[i] == self.actions['Sobel_h2_G']) > 0:
                p = self.parameters['Sobel_h2'][i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                canvas[self.actions['Sobel_h2_G']][i, 1] = cv2.filter2D(self.image[i, 1], -1, kernel=k)
                canvas[self.actions['Sobel_h2_G']][i, 0::2] = self.image[i, 0::2] 

            ## -------------------- actions for B channel -------------------- ##
            if ( not 'box' in self.useless_actions_list) and np.sum(act[i] == self.actions['box_B']) > 0:
                canvas[self.actions['box_B']][i, 2] = cv2.boxFilter(self.image[i, 2], ddepth=-1, ksize=(5,5))
                canvas[self.actions['box_B']][i, 0:2] = self.image[i, 0:2] 

            if ( not 'bilateral' in self.useless_actions_list) and np.sum(act[i] == self.actions['bilateral_B']) > 0:
                canvas[self.actions['bilateral_B']][i, 2] = cv2.bilateralFilter(self.image[i, 2], d=5, sigmaColor=0.1, sigmaSpace=5)
                canvas[self.actions['bilateral_B']][i, 0:2] = self.image[i, 0:2] 

            if True:
                canvas[self.actions['Gaussian_B']][i, 2] = cv2.GaussianBlur(self.image[i, 2], ksize=(5,5), sigmaX=0.5)
                canvas[self.actions['Gaussian_B']][i, 0:2] = self.image[i, 0:2] 

            if ( not 'median' in self.useless_actions_list) and np.sum(act[i] == self.actions['median_B']) > 0:
                canvas[self.actions['median_B']][i, 2] = cv2.medianBlur(self.image[i, 2], ksize=5)
                canvas[self.actions['median_B']][i, 0:2] = self.image[i, 0:2] 

            if np.sum(act[i] == self.actions['Laplace_B']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace_B']][i, 2] = cv2.filter2D(self.image[i, 2], -1, kernel=k)
                canvas[self.actions['Laplace_B']][i, 0:2] = self.image[i, 0:2] 

            if np.sum(act[i] == self.actions['unsharp_B']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp_B']][i, 2] = self.image[i, 2] * (1 + amount) - canvas[self.actions['Gaussian_B']][i,2] * amount
                canvas[self.actions['unsharp_B']][i, 0:2] = self.image[i, 0:2] 

            if np.sum(act[i] == self.actions['Sobel_v1_B']) > 0:
                p = self.parameters['Sobel_v1'][i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                canvas[self.actions['Sobel_v1_B']][i, 2] = cv2.filter2D(self.image[i, 2], -1, kernel=k)
                canvas[self.actions['Sobel_v1_B']][i, 0:2] = self.image[i, 0:2] 

            if np.sum(act[i] == self.actions['Sobel_v2_B']) > 0:
                p = self.parameters['Sobel_v2'][i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                canvas[self.actions['Sobel_v2_B']][i, 2] = cv2.filter2D(self.image[i, 2], -1, kernel=k)
                canvas[self.actions['Sobel_v2_B']][i, 0:2] = self.image[i, 0:2] 

            if np.sum(act[i] == self.actions['Sobel_h1_B']) > 0:
                p = self.parameters['Sobel_h1'][i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                canvas[self.actions['Sobel_h1_B']][i, 2] = cv2.filter2D(self.image[i, 2], -1, kernel=k)
                canvas[self.actions['Sobel_h1_B']][i, 0:2] = self.image[i, 0:2] 

            if np.sum(act[i] == self.actions['Sobel_h2_B']) > 0:
                p = self.parameters['Sobel_h2'][i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                canvas[self.actions['Sobel_h2_B']][i, 2] = cv2.filter2D(self.image[i, 2], -1, kernel=k)
                canvas[self.actions['Sobel_h2_B']][i, 0:2] = self.image[i, 0:2] 

        for a in range(1, self.num_actions + 1):
            self.image = np.where(act[:,np.newaxis,:,:] == a, canvas[a], self.image)
        self.image = np.clip(self.image, 0, 1)

        if self.reward_method == 'square':
            reward = np.square(self.ori_image - self.previous_image) * 255 - np.square(self.ori_image - self.image) * 255
        elif self.reward_method == 'abs':
            reward = np.abs(self.ori_image - self.previous_image) * 255 - np.abs(self.ori_image - self.image) * 255

        return self.image, reward 