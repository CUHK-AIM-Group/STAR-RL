import numpy as np
import skimage.measure
# import scipy
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
import pytorch_ssim
from torch.optim import lr_scheduler
import torch.optim as optim

def fft_shift(x):
    fft = fftpack.fft2(x)
    fft = fftpack.fftshift(fft)
    return fft


def shift_ifft(fft):
    fft = fftpack.ifftshift(fft)
    x = fftpack.ifft2(fft)
    return x


def Downsample(x, mask):
    fft = fftpack.fft2(x)
    fft_good = fftpack.fftshift(fft)
    fft_bad = fft_good * mask
    fft = fftpack.ifftshift(fft_bad)
    x = fftpack.ifft2(fft)
#    x = np.abs(x)
    x = np.real(x)
    return x, fft_good, fft_bad


def SSIM(x_good, x_bad):
    # assert len(x_good.shape) == 2
    # ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
    # ssim_res = ssim(np.transpose(x_good,[1,2,0]), np.transpose(x_bad,[1,2,0]), multichannel=True, data_range=255)
    ssim_res = pytorch_ssim.ssim(torch.from_numpy(x_good).unsqueeze(0), torch.from_numpy(x_bad).unsqueeze(0))
    return ssim_res


def PSNR(x_good, x_bad):
    # assert len(x_good.shape) == 2
    # psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
    psnr_res = psnr(x_good, x_bad)
    return psnr_res


def NMSE(x_good, x_bad):
    # assert len(x_good.shape) == 2
    nmse_a_0_1 = np.sum((x_good - x_bad) ** 2)
    nmse_b_0_1 = np.sum(x_good ** 2)
    # this is DAGAN implementation, which is wrong
    nmse_a_0_1, nmse_b_0_1 = np.sqrt(nmse_a_0_1), np.sqrt(nmse_b_0_1)
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1
    return nmse_0_1


def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)


def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)


def computeNMSE(o_, p_, i_):
    return NMSE(o_, p_), NMSE(o_, i_)


def DC(x_good, x_rec, mask):
    fft_good = fft_shift(x_good)
    fft_rec = fft_shift(x_rec)
    fft = fft_good * mask + fft_rec * (1 - mask)
    x = shift_ifft(fft)
    x = np.real(x)
    #x = np.abs(x)
    return x


def adjust_learning_rate(optimizer, iters, base_lr, policy_parameter, policy='step', multiple=[1]):
    '''
    source: https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/utils.py
    '''
    if policy == 'fixed':
        lr = base_lr
    elif policy == 'step':
        lr = base_lr * (policy_parameter['gamma'] ** (iters // policy_parameter['step_size']))
    elif policy == 'exp':
        lr = base_lr * (policy_parameter['gamma'] ** iters)
    elif policy == 'inv':
        lr = base_lr * ((1 + policy_parameter['gamma'] * iters) ** (-policy_parameter['power']))
    elif policy == 'multistep':
        lr = base_lr
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
            else:
                break
    elif policy == 'poly':
        lr = base_lr * ((1 - iters * 1.0 / policy_parameter['max_iter']) ** policy_parameter['power'])
    elif policy == 'sigmoid':
        lr = base_lr * (1.0 / (1 + math.exp(-policy_parameter['gamma'] * (iters - policy_parameter['stepsize']))))
    elif policy == 'multistep-poly':
        lr = base_lr
        stepstart = 0
        stepend = policy_parameter['max_iter']
        for stepvalue in policy_parameter['stepvalue']:
            if iters >= stepvalue:
                lr *= policy_parameter['gamma']
                stepstart = stepvalue
            else:
                stepend = stepvalue
                break
        lr = max(lr * policy_parameter['gamma'], lr * (1 - (iters - stepstart) * 1.0 / (stepend - stepstart)) ** policy_parameter['power'])

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    return lr


def set_requires_grad_via_name(nets, requires_grad=False, name=None):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for key, value in dict(net.named_parameters()).items():
            # for param in net.parameters():
                if not name in key :
                    value.requires_grad = requires_grad

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def hard_update(target, source):
    """
    Copies the parameters from source network to target network
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.module.parameters(), source.module.parameters()):
            target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    :param target: Target network (PyTorch)
    :param source: Source network (PyTorch)
    :return:
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def update_learning_rate(scheduler, optimizer):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        return lr, 'learning rate %.7f -> %.7f' % (old_lr, lr)

def get_scheduler(optimizer, n_epochs=None, n_epochs_decay=None, lr_decay_iters=None, lr_policy='linear'):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

class crop_and_paste:
    def __init__(self, config):
        # setting for patch cropper
        self.predict_shift = config.predict_shift
        self.predict_abs_pos = config.predict_abs_pos
        self.ph = config.patch_height
        self.pw = config.patch_width
        self.H = config.image_height
        self.W = config.image_width
        self.use_discrete_action = config.use_discrete_action
        self.use_pixel_oriented_patch = config.use_pixel_oriented_patch
        self.factor = config.factor
        self.use_coordinate_classify_agent = config.use_coordinate_classify_agent
        
        assert self.ph == self.pw and self.H and self.W , "Patch height must be equal to its width; Same as the image height and width."

    def crop_patches(self, actions, image, origin_image, rtn_pos=False):
        if self.use_discrete_action: # actions : (bs,) - indicates the patch index
            if self.use_coordinate_classify_agent:
                left = torch.from_numpy(actions[:,0])
                top = torch.from_numpy(actions[:,1])
            elif self.use_pixel_oriented_patch : # convert the vector to position
                left = torch.from_numpy(actions // self.ph)
                top = torch.from_numpy(actions % self.pw)
            else:
                # left for rows, top for cols ; sorry for the misnamed
                left = torch.from_numpy(actions // self.factor * self.ph)
                top = torch.from_numpy(actions % self.factor * self.pw)
        else: # use the continous actions
            # map the position from [0,1] to [0,W], just multiply by width or height
            pos_x = actions[:,0]
            pos_y = actions[:,1]
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
            # print("1 - x:",pos_x[0],"y:",pos_y[0])
            pos_x = torch.tensor(pos_x, dtype=torch.int)
            pos_y = torch.tensor(pos_y, dtype=torch.int)
            # image : b x ch x H x W
            # patch should in the image
            left, top = pos_x - int(self.pw / 2), pos_y - int(self.ph / 2)
        # force the position within the image size
        left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))

        start_x = torch.where(left > (self.W - self.pw), (self.W - self.pw) * torch.ones_like(left), left)
        start_y = torch.where(top > (self.H - self.ph), (self.H - self.ph) * torch.ones_like(top), top)

        bs, ch, h, w = image.shape
        patch_list = []
        gt_patch_list = []
        for i in range(bs):
            patch_list.append(image[i:i+1, :, start_x[i]:start_x[i]+self.pw, start_y[i]:start_y[i]+self.ph]) # 1*C*patch_w*patch_h
            gt_patch_list.append(origin_image[i:i+1, :, start_x[i]:start_x[i]+self.pw, start_y[i]:start_y[i]+self.ph]) # 1*C*patch_w*patch_h
        
        output = np.concatenate(patch_list, axis=0) # B*C*patch_w*patch_h
        gt = np.concatenate(gt_patch_list, axis=0)
        if rtn_pos:
            return output, gt, start_x, start_y
        return output, gt #, start_x, start_y
        
    def paste(self, recovered_patches, input_image, start_x, start_y):
        bs, ch, h, w = recovered_patches.shape
        for i in range(bs):
            input_image[i,:,start_x[i]:start_x[i]+self.pw, start_y[i]:start_y[i]+self.ph] = recovered_patches[i]
        return input_image


if __name__ == "__main__":
    pass
