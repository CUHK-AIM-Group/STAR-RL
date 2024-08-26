import numpy as np
import sys
import cv2
import torch
from skimage.measure import compare_ssim
import pytorch_ssim

class Env_patch_agent2Cond():
    def __init__(self, config):
        self.image = None
        self.previous_image = None
        # setting for patch cropper
        self.predict_shift = config.predict_shift
        self.predict_abs_pos = config.predict_abs_pos
        self.weight_reward_mae = config.weight_reward_mae
        self.weight_reward_iou = config.weight_reward_iou
        self.bias_reward_mae = config.bias_reward_mae
        self.bias_reward_iou = config.bias_reward_iou
        self.ph = config.patch_height
        self.pw = config.patch_width
        self.H = config.image_height
        self.W = config.image_width
        assert self.ph == self.pw and self.H and self.W , "Patch height must be equal to its width; Same as the image height and width."
        self.start_x = None
        self.start_y = None
        self.use_iou_reward = config.use_iou_reward
        if self.use_iou_reward:
            self.mask_previous = None
            self.mask_current = None
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        self.factor = config.factor
        # for coordinate classification
        self.use_coordinate_classify_agent = config.use_coordinate_classify_agent
        # use agent2 in agent1 
        self.use_agent2_in_agent1 = config.use_agent2_in_agent1
        self.num_actions_agent2 = config.num_actions
        self.actions = config.actions # for agent2
        self.parameters_scale = config.parameters_scale
        self.parameters = dict()
        self.set_param_agent2([0.5] * len(self.parameters_scale))
        self.T2_in_agent1 = config.T2_in_agent1
        self.useless_actions_list = config.useless_actions_list
        self.add_subtract_shift = config.add_subtract_shift
    def set_param_agent2(self, p): # set params for agent2
        for i, k in enumerate(sorted(self.parameters_scale.keys())):
            self.parameters[k] = p[i] * self.parameters_scale[k]
        return

    def reset(self, ori_image, image):
        self.ori_image = ori_image.copy()
        self.image = image.copy()
        self.previous_image = None
        if self.use_iou_reward:
            self.mask_previous = None
            self.mask_current = None
        return

    def step_agent2(self, act, image, ori_image=None, count_reward=False):
        if count_reward:
            previous_image = image.copy()
        canvas = [np.zeros(image.shape, image.dtype) for _ in range(self.num_actions_agent2 + 1)]
        b, c, h, w = image.shape
        for i in range(b):
            # do nothing
            canvas[0][i] = image[i]

            canvas[self.actions['subtraction']][i] = image[i] - self.add_subtract_shift / 255
            canvas[self.actions['addition']][i] = image[i] + self.add_subtract_shift / 255

            if ( not 'box' in self.useless_actions_list) and np.sum(act[i] == self.actions['box']) > 0:
                canvas[self.actions['box']][i] = np.transpose(cv2.boxFilter(image[i].transpose([1,2,0]), ddepth=-1, ksize=(5,5)),[2,0,1])

            if ( not 'bilateral' in self.useless_actions_list) and np.sum(act[i] == self.actions['bilateral']) > 0:
                canvas[self.actions['bilateral']][i] = np.transpose(cv2.bilateralFilter(image[i].transpose([1,2,0]), d=5, sigmaColor=0.1, sigmaSpace=5),[2,0,1])

            if True:
                canvas[self.actions['Gaussian']][i] = np.transpose(cv2.GaussianBlur(image[i].transpose([1,2,0]), ksize=(5,5), sigmaX=0.5),[2,0,1])

            if ( not 'median' in self.useless_actions_list) and np.sum(act[i] == self.actions['median']) > 0:
                canvas[self.actions['median']][i] = np.transpose(cv2.medianBlur(image[i].transpose([1,2,0]), ksize=5),[2,0,1])

            if np.sum(act[i] == self.actions['Laplace']) > 0:
                p = self.parameters['Laplace'][i]
                k = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                canvas[self.actions['Laplace']][i] = np.transpose(cv2.filter2D(image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['unsharp']) > 0:
                amount = self.parameters['unsharp'][i]
                canvas[self.actions['unsharp']][i] = image[i] * (1 + amount) - canvas[self.actions['Gaussian']][i] * amount

            if np.sum(act[i] == self.actions['Sobel_v1']) > 0:
                p = self.parameters['Sobel_v1'][i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                canvas[self.actions['Sobel_v1']][i] = np.transpose(cv2.filter2D(image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['Sobel_v2']) > 0:
                p = self.parameters['Sobel_v2'][i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                canvas[self.actions['Sobel_v2']][i] = np.transpose(cv2.filter2D(image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['Sobel_h1']) > 0:
                p = self.parameters['Sobel_h1'][i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                canvas[self.actions['Sobel_h1']][i] = np.transpose(cv2.filter2D(image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

            if np.sum(act[i] == self.actions['Sobel_h2']) > 0:
                p = self.parameters['Sobel_h2'][i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                canvas[self.actions['Sobel_h2']][i] = np.transpose(cv2.filter2D(image[i].transpose([1,2,0]), -1, kernel=k),[2,0,1])

        for a in range(1, self.num_actions_agent2 + 1):
            image = np.where(act[:,np.newaxis,:,:] == a, canvas[a], image)
        image = np.clip(image, 0, 1)
        if count_reward:
            reward = np.abs(ori_image - previous_image) * 255 - np.abs(ori_image - image) * 255
            return image, reward
        return image 

    def step(self, actions, use_iou_reward=False, sr_model=None, use_discriminator=False, reward_idx=0):
        self.previous_image = self.image.copy()
        # crop the patches and obtain the start_x and start_y
        cropped_image, cropped_ori = self.crop_patch(actions)


        for i in range(self.T2_in_agent1):
            # super-resolve the patches with agent2
            pi_out, _, p = sr_model(torch.from_numpy(cropped_image).cuda())
            _, actions = torch.max(pi_out.detach().data, dim=1)
            ## set params 
            p = p.cpu().data.numpy().transpose(1, 0)
            self.set_param_agent2(p)
            if use_discriminator and reward_idx == i: 
                cropped_patches = cropped_image.copy()
                cropped_image_sr, reward_each = self.step_agent2(actions.cpu().numpy(), cropped_image, ori_image=cropped_ori, count_reward=True)
            else:
                cropped_image_sr = self.step_agent2(actions.cpu().numpy(), cropped_image)
            cropped_image = cropped_image_sr


        # replace the lr patches with hr patches
        if self.use_agent2_in_agent1:
            self.recover_patches(cropped_image_sr, getOverlapMask=use_iou_reward)
        else:
            self.recover_patches(cropped_ori, getOverlapMask=use_iou_reward)

        # reward = pytorch_ssim.ssim(torch.from_numpy(self.ori_image).cuda(),torch.from_numpy(self.previous_image).cuda()) \
        #         - pytorch_ssim.ssim(torch.from_numpy(self.ori_image).cuda(),torch.from_numpy(self.image).cuda())
        bs, ch, h, w = self.ori_image.shape
        reward = np.sum(np.abs(self.ori_image - self.previous_image) * 255  - np.abs(self.ori_image - self.image) * 255, axis=(1,2,3)) / (ch*self.ph*self.pw)
        reward = reward.reshape((bs, 1)) 
        # get difference mask between previous and current image
        reward = reward * self.weight_reward_mae + self.bias_reward_mae
        # calculate the overlap between the current and previous patch
        if use_iou_reward:
            reward_iou = self.cal_iou_reward() * self.weight_reward_iou + self.bias_reward_iou
            reward += reward_iou
            return self.image, reward, reward_iou
        if use_discriminator :
            return self.image, reward, 0, cropped_patches, reward_each

        return self.image, reward, 0
    
    def cal_iou_reward(self):
        if not type(self.mask_previous) is np.ndarray:
            bs, ch, h, w = self.ori_image.shape
            return np.zeros((bs))
        overlap_mask = self.mask_current + self.mask_previous
        overlap_mask[overlap_mask==1] = 0
        overlap_mask[overlap_mask==2] = 1
        overlap_mask = np.sum(np.sum(overlap_mask, axis=1), axis=1) # bs x 1
        # print("overlap_mask:",overlap_mask.shape)
        iou = - overlap_mask / (self.pw * self.ph * 2 - overlap_mask)
        return iou

    def crop_patch(self, actions):
        if self.use_discrete_action:
            if self.use_coordinate_classify_agent:
                left = torch.from_numpy(actions[:,0])
                top = torch.from_numpy(actions[:,1])
            elif self.use_pixel_oriented_patch : # convert the vector to position
                left = torch.from_numpy(actions // self.ph) # actually, this code may be wrong;it should be actions // (H-self.ph)
                top = torch.from_numpy(actions % self.pw)
            else:
                # actions : (bs,) - indicates the patch index
                # left for rows, top for cols ; sorry for the misnamed
                left = torch.from_numpy(actions // self.factor * self.ph)
                top = torch.from_numpy(actions % self.factor * self.pw)
        else: # continous action
            pos_x = actions[:,0]
            pos_y = actions[:,1]
            # map the position from [0,1] to [0,W], just multiply by width or height
            if self.predict_shift:
                # bs x 1 -> pos_x , pos_y
                bs = len(pos_x)
                pos_mean = np.random.normal(0,50, (bs,2)) + np.array([[0.5*self.H,0.5*self.W]])
                pos_mean = torch.from_numpy(pos_mean).cuda()
                pos_x += pos_mean[:,0]
                pos_y += pos_mean[:,1]
                pos_x = pos_x.ceil()
                pos_y = pos_y.ceil()
            elif self.predict_abs_pos:
                pos_x = pos_x.ceil()
                pos_y = pos_y.ceil()
            else:
                pos_x, pos_y = (self.W * pos_x).ceil(), (self.H * pos_y).ceil()
            pos_x = pos_x.int()
            pos_y = pos_y.int()
            # image : b x ch x H x W
            # patch should in the image
            left, top = pos_x - int(self.pw / 2), pos_y - int(self.ph / 2)
        # force the position within the image size
        left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))

        start_x = torch.where(left > (self.W - self.pw), (self.W - self.pw) * torch.ones_like(left), left)
        start_y = torch.where(top > (self.H - self.ph), (self.H - self.ph) * torch.ones_like(top), top)

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
        
    def recover_patches(self, recovered_patches, getOverlapMask=False):
        if getOverlapMask:
            bs, ch, h, w = self.image.shape
            if type(self.mask_current) is np.ndarray:
                self.mask_previous = self.mask_current.copy() 
            self.mask_current = np.zeros((bs, h, w)) 

        bs, ch, h, w = recovered_patches.shape
        for i in range(bs):
            self.image[i,:,self.start_x[i]:self.start_x[i]+self.pw, self.start_y[i]:self.start_y[i]+self.ph] = recovered_patches[i]
            if getOverlapMask:
                self.mask_current[i,self.start_x[i]:self.start_x[i]+self.pw, self.start_y[i]:self.start_y[i]+self.ph] = 1
        return 

class Env_patch():
    def __init__(self, config):
        self.image = None
        self.previous_image = None
        # setting for patch cropper
        self.predict_shift = config.predict_shift
        self.predict_abs_pos = config.predict_abs_pos
        self.weight_reward_mae = config.weight_reward_mae
        self.weight_reward_iou = config.weight_reward_iou
        self.bias_reward_mae = config.bias_reward_mae
        self.bias_reward_iou = config.bias_reward_iou
        self.ph = config.patch_height
        self.pw = config.patch_width
        self.H = config.image_height
        self.W = config.image_width
        assert self.ph == self.pw and self.H and self.W , "Patch height must be equal to its width; Same as the image height and width."
        self.start_x = None
        self.start_y = None
        self.use_iou_reward = config.use_iou_reward
        if self.use_iou_reward:
            self.mask_previous = None
            self.mask_current = None
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        self.factor = config.factor
        # for coordinate classification
        self.use_coordinate_classify_agent = config.use_coordinate_classify_agent
        # self.previous_iou = 0
        # self.current_iou = 0

    def reset(self, ori_image, image):
        self.ori_image = ori_image.copy()
        self.image = image.copy()
        self.previous_image = None
        if self.use_iou_reward:
            self.mask_previous = None
            self.mask_current = None
        return

    def step(self, actions, use_iou_reward=False):
        self.previous_image = self.image.copy()
        # crop the patches and obtain the start_x and start_y
        cropped_image, cropped_ori = self.crop_patch(actions)
        # replace the lr patches with hr patches
        self.recover_patches(cropped_ori, getOverlapMask=use_iou_reward)
        # reward = pytorch_ssim.ssim(torch.from_numpy(self.ori_image).cuda(),torch.from_numpy(self.previous_image).cuda()) \
        #         - pytorch_ssim.ssim(torch.from_numpy(self.ori_image).cuda(),torch.from_numpy(self.image).cuda())
        bs, ch, h, w = self.ori_image.shape
        reward = np.sum(np.abs(self.ori_image - self.previous_image) * 255  - np.abs(self.ori_image - self.image) * 255, axis=(1,2,3)) / (ch*self.ph*self.pw)
        reward = reward.reshape((bs, 1)) 
        # get difference mask between previous and current image
        reward = reward * self.weight_reward_mae + self.bias_reward_mae
        # calculate the overlap between the current and previous patch
        if use_iou_reward:
            reward_iou = self.cal_iou_reward() * self.weight_reward_iou + self.bias_reward_iou
            reward += reward_iou
            return self.image, reward, reward_iou
        return self.image, reward
    
    def cal_iou_reward(self):
        if not type(self.mask_previous) is np.ndarray:
            bs, ch, h, w = self.ori_image.shape
            return np.zeros((bs))
        overlap_mask = self.mask_current + self.mask_previous
        overlap_mask[overlap_mask==1] = 0
        overlap_mask[overlap_mask==2] = 1
        overlap_mask = np.sum(np.sum(overlap_mask, axis=1), axis=1) # bs x 1
        # print("overlap_mask:",overlap_mask.shape)
        iou = - overlap_mask / (self.pw * self.ph * 2 - overlap_mask)
        return iou

    def crop_patch(self, actions):
        if self.use_discrete_action:
            if self.use_coordinate_classify_agent:
                left = torch.from_numpy(actions[:,0])
                top = torch.from_numpy(actions[:,1])
            elif self.use_pixel_oriented_patch : # convert the vector to position
                left = torch.from_numpy(actions // self.ph)
                top = torch.from_numpy(actions % self.pw)
            else:
                # actions : (bs,) - indicates the patch index
                # left for rows, top for cols ; sorry for the misnamed
                left = torch.from_numpy(actions // self.factor * self.ph)
                top = torch.from_numpy(actions % self.factor * self.pw)
        else: # continous action
            pos_x = actions[:,0]
            pos_y = actions[:,1]
            # map the position from [0,1] to [0,W], just multiply by width or height
            if self.predict_shift:
                # bs x 1 -> pos_x , pos_y
                bs = len(pos_x)
                pos_mean = np.random.normal(0,50, (bs,2)) + np.array([[0.5*self.H,0.5*self.W]])
                pos_mean = torch.from_numpy(pos_mean).cuda()
                pos_x += pos_mean[:,0]
                pos_y += pos_mean[:,1]
                pos_x = pos_x.ceil()
                pos_y = pos_y.ceil()
            elif self.predict_abs_pos:
                pos_x = pos_x.ceil()
                pos_y = pos_y.ceil()
            else:
                pos_x, pos_y = (self.W * pos_x).ceil(), (self.H * pos_y).ceil()
            pos_x = pos_x.int()
            pos_y = pos_y.int()
            # image : b x ch x H x W
            # patch should in the image
            left, top = pos_x - int(self.pw / 2), pos_y - int(self.ph / 2)
        # force the position within the image size
        left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))

        start_x = torch.where(left > (self.W - self.pw), (self.W - self.pw) * torch.ones_like(left), left)
        start_y = torch.where(top > (self.H - self.ph), (self.H - self.ph) * torch.ones_like(top), top)

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
        
    def recover_patches(self, recovered_patches, getOverlapMask=False):
        if getOverlapMask:
            bs, ch, h, w = self.image.shape
            if type(self.mask_current) is np.ndarray:
                self.mask_previous = self.mask_current.copy() 
            self.mask_current = np.zeros((bs, h, w)) 

        bs, ch, h, w = recovered_patches.shape
        for i in range(bs):
            self.image[i,:,self.start_x[i]:self.start_x[i]+self.pw, self.start_y[i]:self.start_y[i]+self.ph] = recovered_patches[i]
            if getOverlapMask:
                self.mask_current[i,self.start_x[i]:self.start_x[i]+self.pw, self.start_y[i]:self.start_y[i]+self.ph] = 1
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
    
        self.useless_actions_list = config.useless_actions_list

        self.add_subtract_shift = config.add_subtract_shift
    
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

            canvas[self.actions['subtraction']][i] = self.image[i] - self.add_subtract_shift / 255
            canvas[self.actions['addition']][i] = self.image[i] + self.add_subtract_shift / 255

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

        for a in range(1, self.num_actions + 1):
            self.image = np.where(act[:,np.newaxis,:,:] == a, canvas[a], self.image)
        self.image = np.clip(self.image, 0, 1)

        if self.reward_method == 'square':
            reward = np.square(self.ori_image - self.previous_image) * 255 - np.square(self.ori_image - self.image) * 255
        elif self.reward_method == 'abs':
            reward = np.abs(self.ori_image - self.previous_image) * 255 - np.abs(self.ori_image - self.image) * 255

        return self.image, reward 
