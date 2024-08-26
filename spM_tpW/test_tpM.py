import os
import sys
import time
import argparse
import numpy as np
import cv2
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from env import Env, Env_patch
from model import MyFcn, PatchPicker, PatchAgent, CoordinateClassifyPixelPatchAgent, PixelPatchAgent, PatchAgent_2,CoordinateClassifyPixelPatchAgent_sm
from pixel_wise_a2c import PixelWiseA2C, PatchWiseAC, PatchWiseAC_discrete, PatchWiseAC_discrete_coordinate_classify
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE, crop_and_paste
from networks import NLayerDiscriminator, GANLoss, MLP, MLP_512

from tqdm import tqdm
import copy
def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MICCAI', type=str,
                        dest='dataset', help='to use dataset.py and config.py in which directory')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--model_name', type=str, help='the folder of the pretrained model')
    parser.add_argument('--episodes', type=str, help='which model to load')
    parser.add_argument('--episode_len_test', type=int, default=None, help='episode length for agent2')
    parser.add_argument('--episode_len_patch_test', type=int, default=None, help='episode length for agent1')
    parser.add_argument('--entropy_th', type=float, default=None, help='threshold for entropy')
    parser.add_argument('--reward_th', type=int, default=None, help='threshold for reward')
    parser.add_argument('--cut_edge',action="store_true",default=False, help='whether cut the edge during evaluation')    
    parser.add_argument('--use_discriminator',action="store_true",default=False, help='whether use discriminator')    
    parser.add_argument('--disc_th', type=float, default=0.5, help='threshold for discriminator value')
    parser.add_argument('--actionMap_reward_save_dir', type=str,default=None, help='the folder of saving action maps and reward')
    parser.add_argument('--discriminator_model_path', type=str, default=None, help='discriminator_model_path')
    parser.add_argument('--save_images',action="store_true",default=False, help='whether save images')    
    

    return parser.parse_args()


def validation_agent1(model, picker, a2c, pAC, config, early_break=True, batch_size=None, verbose=False, valid_mode=False, current_episode=0, save_dir=None, print_log=None):
    if batch_size is None:
        batch_size = config.batch_size
    env = Env(config)
    env_p = Env_patch(config)

    # tool for cropping and pasting
    tool = crop_and_paste(config)

    results_dir = save_dir if valid_mode else 'results/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    from HistoSR import data_loader_lmdb
    test_loader = data_loader_lmdb.get_loader(
        os.path.join(config.root, config.data_degradation, 'test_lmdb'), 
        batch_size=batch_size,
        stage='test', num_workers=1)

    reward_sum = 0
    p_list = defaultdict(list)
    # PSNR_dict = defaultdict(list)
    # SSIM_dict = defaultdict(list)
    # NMSE_dict = defaultdict(list)
    PSNR_list = []
    SSIM_list = []
    NMSE_list = []
    count = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len_test))
    image_history = dict()

    actions_list = []

    model.eval()
    picker.eval()

    for i, (image, ori_image) in enumerate(tqdm(test_loader)):
        count += 1
        # if early_break and count == 101: # test only part of the dataset
        #     break
        # if count % 100 == 0:
        #     print('tested: ', count)

        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()

        for t in range(config.episode_len_patch):
            if verbose:
                image_history[t] = image
            # pick a patch via Agent 1
            actions, v_out_p = picker(torch.from_numpy(image).cuda())
            if config.use_discrete_action:
                if config.use_coordinate_classify_agent:
                    _, actions_row = torch.max(actions[:,0].data, dim=1)
                    _, actions_col = torch.max(actions[:,1].data, dim=1)
                    actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
                else:
                    _, actions = torch.max(actions.data, dim=1)
                actions = actions.cpu().numpy()
            
            image_patches, ori_image_patches, start_x, start_y = tool.crop_patches(actions, image, ori_image, rtn_pos=True)
            if i == 0 and valid_mode :
                actions_list.append(actions)
            # env.reset(ori_image=ori_image_patches, image=image_patches) 

            # for t in range(config.episode_len):
            #     image_input = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
            #     pi_out, v_out, p = model(image_input, flag_a2c=True)

            #     p = p.permute(1, 0).cpu().data.numpy()
            #     env.set_param(p)
            #     p_list[t].append(p)

            #     actions = a2c.act(pi_out, deterministic=True)
            #     last_image = image_patches.copy()
            #     image_patches, reward = env.step(actions)
            #     image_patches = np.clip(image_patches, 0, 1)

            #     reward_sum += np.mean(reward)

            #     actions = actions.astype(np.uint8)
            #     prob = pi_out.cpu().data.numpy()
            #     total = actions.size
            #     for n in range(config.num_actions):
            #         actions_prob[n, t] += np.sum(actions==n) / total

            # paste the reovered image patches on the original image
            image = tool.paste(ori_image_patches, image, start_x, start_y)
            image = np.clip(image, 0, 1)
            '''    # draw action distribution on pixels
                for j in range(ori_image.shape[0]):
                    if i > 0:
                        break
                    for dd in ['results/actions/']:#, 'results/time_steps/']:
                        if not os.path.exists(dd + str(j)):
                            os.makedirs(dd + str(j))
                            # os.mkdir(dd + str(j))
                    a = actions[j].astype(np.uint8)
                    total = a.size
                    canvas = last_image[j, 0].copy()
                    unchanged_mask = np.abs(last_image[j, 0] - image[j, 0]) < (1 / 255) # some pixel values are not changed
                    for n in range(config.num_actions):
                        A = np.tile(canvas[..., np.newaxis], (1, 1, 3)) * 255
                        a_mask = (a==n) & (1 - unchanged_mask).astype(np.bool)
                        A[a_mask, 2] += 250
                        cv2.imwrite('results/actions/' + str(j) + '/' + str(n) + '_' + str(t) +'.bmp', A)
                    cv2.imwrite('results/actions/' + str(j) + '/' + str(t) + '_unchanged.jpg', np.abs(last_image[j, 0] - image[j, 0]) * 255 * 255)
                    # cv2.imwrite('results/actions/' + str(t) + '_unchanged.jpg', np.abs(last_image[j, 0] - image[j, 0]) * 255 * 255)'''
        # print("ori_image[0]",ori_image[0].shape,ori_image[0].dtype,ori_image[0].max(),ori_image[0].min())
        # print("previous_image[0]:", previous_image[0].shape,previous_image[0].dtype,previous_image[0].max(),previous_image[0].min())
        # print("image[0]:",image[0].shape,image[0].dtype,image[0].max(),image[0].min())
        
        if valid_mode and i == 0 :
            tmp_tensor = []
        for j in range(ori_image.shape[0]):
            PSNR_list.append(computePSNR(ori_image[j], previous_image[j], image[j])) 
            SSIM_list.append(computeSSIM(ori_image[j], previous_image[j], image[j]))
            NMSE_list.append(computeNMSE(ori_image[j], previous_image[j], image[j]))
            
            if valid_mode:
                if i == 0 : 
                    tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j]), np.abs(previous_image[j] - image[j])), axis=2), [1,2,0])
                    tmp_tensor.append(tensor_cat)
            else:
                tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j])), axis=2), [1,2,0])
                cv2.imwrite(os.path.join(results_dir, str(i)+'_'+str(j)+'.bmp'), tensor_cat * 255)
            # draw output of different timesteps
            if verbose:
                cv2.imwrite('results/time_steps/'+str(i)+'_'+str(j)+'.bmp', np.concatenate([image_history[jj][0] for jj in range(config.episode_len_test)] + [image[0], ori_image[0]], axis=1) * 255)

        if valid_mode and i == 0 : 
            cv2.imwrite(os.path.join(results_dir, str(current_episode) + '_'+ str(i)+'.bmp'), np.concatenate(tmp_tensor, axis=0) * 255)
            # print the actions list for the first batch image
            if not print_log == None:
                # actions_list : [(bs,), (bs,) ....]
                for idx in range(len(actions_list)): 
                    print("t:",idx, "actions:", actions_list[idx], file=print_log)
    # print('actions_prob', actions_prob / count)

    # for key in PSNR_dict.keys():
    #     PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
    #     print('number of test images: ', len(PSNR_list))
    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    nmse_res = np.mean(np.array(NMSE_list), axis=0)
    
    print('PSNR', psnr_res, file=print_log)
    print('SSIM', ssim_res, file=print_log)
    print('NMSE', nmse_res, file=print_log)

    # for t in range(config.episode_len):
    #     print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward, file=print_log)

    model.train()
    picker.train()

    return avg_reward, psnr_res, ssim_res, nmse_res


def validation(model, picker, a2c, pAC, config, early_break=True, batch_size=None, verbose=False, valid_mode=False, current_episode=0, save_dir=None, print_log=None):
    if batch_size is None:
        batch_size = config.batch_size
    env = Env(config)
    env_p = Env_patch(config)

    # tool for cropping and pasting
    tool = crop_and_paste(config)

    results_dir = save_dir if valid_mode else 'results/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if config.load_spectral_sr: 
        from HistoSR import data_loader_bmp
        test_loader = data_loader_lmdb.get_loader(
            os.path.join(config.root, config.data_degradation, 'test_lmdb'), 
            config.spectral_sr_test_root,
            batch_size=batch_size,
            stage='test', num_workers=1)
    else:
        from HistoSR import data_loader_lmdb
        test_loader = data_loader_lmdb.get_loader(
            os.path.join(config.root, config.data_degradation, 'test_lmdb'), 
            batch_size=batch_size,
            stage='test', num_workers=1)

    reward_sum = 0
    p_list = defaultdict(list)
    # PSNR_dict = defaultdict(list)
    # SSIM_dict = defaultdict(list)
    # NMSE_dict = defaultdict(list)
    PSNR_list = []
    SSIM_list = []
    NMSE_list = []
    count = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len_test))
    image_history = dict()
    # for display actions
    actions_list = []
    for i, (image, ori_image) in enumerate(tqdm(test_loader)):
        count += 1
        # if early_break and count == 101: # test only part of the dataset
        #     break
        # if count % 100 == 0:
        #     print('tested: ', count)

        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()

        for t in range(config.episode_len_patch):
            if verbose:
                image_history[t] = image
            # pick a patch via Agent 1
            actions, v_out_p = picker(torch.from_numpy(image).cuda())
            if config.use_discrete_action:
                if config.use_coordinate_classify_agent:
                    _, actions_row = torch.max(actions[:,0].data, dim=1)
                    _, actions_col = torch.max(actions[:,1].data, dim=1)
                    actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
                else:
                    _, actions = torch.max(actions.data, dim=1)
                actions = actions.cpu().numpy()
            image_patches, ori_image_patches, start_x, start_y = tool.crop_patches(actions, image, ori_image, rtn_pos=True)
            if i == 0 and valid_mode :
                actions_list.append(actions)

            env.reset(ori_image=ori_image_patches, image=image_patches) 

            for t in range(config.episode_len_test):
                image_input = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                pi_out, v_out, p = model(image_input, flag_a2c=True)

                p = p.permute(1, 0).cpu().data.numpy()
                env.set_param(p)
                p_list[t].append(p)

                actions = a2c.act(pi_out, deterministic=True)
                last_image = image_patches.copy()
                image_patches, reward = env.step(actions)
                image_patches = np.clip(image_patches, 0, 1)

                reward_sum += np.mean(reward)

                actions = actions.astype(np.uint8)
                prob = pi_out.cpu().data.numpy()
                total = actions.size
                for n in range(config.num_actions):
                    actions_prob[n, t] += np.sum(actions==n) / total

            # paste the reovered image patches on the original image
            image = tool.paste(image_patches, image, start_x, start_y)
            image = np.clip(image, 0, 1)
            '''    # draw action distribution on pixels
                for j in range(ori_image.shape[0]):
                    if i > 0:
                        break
                    for dd in ['results/actions/']:#, 'results/time_steps/']:
                        if not os.path.exists(dd + str(j)):
                            os.makedirs(dd + str(j))
                            # os.mkdir(dd + str(j))
                    a = actions[j].astype(np.uint8)
                    total = a.size
                    canvas = last_image[j, 0].copy()
                    unchanged_mask = np.abs(last_image[j, 0] - image[j, 0]) < (1 / 255) # some pixel values are not changed
                    for n in range(config.num_actions):
                        A = np.tile(canvas[..., np.newaxis], (1, 1, 3)) * 255
                        a_mask = (a==n) & (1 - unchanged_mask).astype(np.bool)
                        A[a_mask, 2] += 250
                        cv2.imwrite('results/actions/' + str(j) + '/' + str(n) + '_' + str(t) +'.bmp', A)
                    cv2.imwrite('results/actions/' + str(j) + '/' + str(t) + '_unchanged.jpg', np.abs(last_image[j, 0] - image[j, 0]) * 255 * 255)
                    # cv2.imwrite('results/actions/' + str(t) + '_unchanged.jpg', np.abs(last_image[j, 0] - image[j, 0]) * 255 * 255)'''
        # print("ori_image[0]",ori_image[0].shape,ori_image[0].dtype,ori_image[0].max(),ori_image[0].min())
        # print("previous_image[0]:", previous_image[0].shape,previous_image[0].dtype,previous_image[0].max(),previous_image[0].min())
        # print("image[0]:",image[0].shape,image[0].dtype,image[0].max(),image[0].min())

        if valid_mode and i == 0 :
            tmp_tensor = []
        for j in range(ori_image.shape[0]):
            PSNR_list.append(computePSNR(ori_image[j], previous_image[j], image[j])) 
            SSIM_list.append(computeSSIM(ori_image[j], previous_image[j], image[j]))
            NMSE_list.append(computeNMSE(ori_image[j], previous_image[j], image[j]))
            # if 'fastMRI' in config.dataset:
            #     mask_j = mask.numpy()[j]
            #     mask_j = np.tile(mask_j, (image.shape[2] ,1))
            # else:
            #     mask_j = test_loader.dataset.mask
            # image_with_DC = DC(ori_image[j, 0], image[j, 0], mask_j)
            # image_with_DC = np.clip(image_with_DC, 0, 1)
            # for k in range(2):
            #     key = ['wo', 'DC'][k]
            #     tmp_image = [image[j, 0], image_with_DC][k]
            #     PSNR_dict[key].append(computePSNR(ori_image[j, 0], previous_image[j, 0], tmp_image)) 
            #     SSIM_dict[key].append(computeSSIM(ori_image[j, 0], previous_image[j, 0], tmp_image))
            #     NMSE_dict[key].append(computeNMSE(ori_image[j, 0], previous_image[j, 0], tmp_image))
            #     if verbose:
            #         print(j, key, PSNR_dict[key][-1], SSIM_dict[key][-1], NMSE_dict[key][-1])

            # draw input, output and error maps
            # print("ori_image[j]:",ori_image[j].shape)
            # print("tensor:",tensor_cat.shape)
            if valid_mode:
                if i == 0 : 
                    tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j]), np.abs(previous_image[j] - image[j])), axis=2), [1,2,0])
                    tmp_tensor.append(tensor_cat)
            else:
                tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j])), axis=2), [1,2,0])
                cv2.imwrite(os.path.join(results_dir, str(i)+'_'+str(j)+'.bmp'), tensor_cat * 255)
            # draw output of different timesteps
            if verbose:
                cv2.imwrite('results/time_steps/'+str(i)+'_'+str(j)+'.bmp', np.concatenate([image_history[jj][0] for jj in range(config.episode_len_test)] + [image[0], ori_image[0]], axis=1) * 255)

        if valid_mode and i == 0 : 
            cv2.imwrite(os.path.join(results_dir, str(current_episode) + '_'+ str(i)+'.jpg'), np.concatenate(tmp_tensor, axis=0) * 255)
            # print the actions list for the first batch image
            if not print_log == None:
                print("Validation:", file=print_log)
                # actions_list : [(bs,), (bs,) ....]
                for idx in range(len(actions_list)): 
                    print("t:",idx, "actions:", actions_list[idx], file=print_log)
    actions_prob = actions_prob / count
    print('actions_prob:', file=print_log)
    print(actions_prob, file=print_log)

    # for key in PSNR_dict.keys():
    #     PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
    #     print('number of test images: ', len(PSNR_list))
    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    nmse_res = np.mean(np.array(NMSE_list), axis=0)
    
    print('PSNR', psnr_res)
    print('SSIM', ssim_res)
    print('NMSE', nmse_res)

    for t in range(config.episode_len_test):
        print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward)

    return avg_reward, psnr_res, ssim_res, nmse_res, actions_prob

def test(model, picker, model_discri, a2c, pAC, config,  batch_size=None, verbose=False, results_dir=None, use_entropy=False, entropy_th=4.55, reward_th=None, cut_edge=False, use_discriminator=False, print_log=None, test_mode=False, disc_th=0.5, actmp_rwd_save_dir=None, save_images=False):
    if batch_size is None:
        batch_size = config.batch_size
    env = Env(config)
    env_p = Env_patch(config)

    # tool for cropping and pasting
    tool = crop_and_paste(config)
    if config.use_HistoSR_dataset:
        from HistoSR import data_loader_shuffled_data
        test_loader = data_loader_shuffled_data.get_loader(
            config.histosr_data_test,
            batch_size=batch_size, 
            stage='test', 
            num_workers=config.workers,
            test_list_dir=config.test_list_dir,
            getFileName=True)
    else:
        from HistoSR import data_loader_lmdb
        test_loader = data_loader_lmdb.get_loader(
            os.path.join(config.root, config.data_degradation, 'test_lmdb'), 
            batch_size=batch_size,
            stage='test', num_workers=1)

    reward_sum = 0
    p_list = defaultdict(list)
    # PSNR_dict = defaultdict(list)
    # SSIM_dict = defaultdict(list)
    # NMSE_dict = defaultdict(list)
    PSNR_list = []
    SSIM_list = []
    NMSE_list = []
    count = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len_test))
    image_history = dict()
    if save_images:
        perform_file = open(os.path.join(results_dir, 'eval.txt'), 'w')
    # position_file = open(os.path.join(results_dir, 'position.txt'), 'w')
    entropy_row_dict = {}
    entropy_col_dict = {}
    reward_agent1_dict = {}
    discriminator_dict = {}
    discriminator_loss_dict = {}
    action_map_dict = {}
    threshold = entropy_th #4.554
    GANCriterion = GANLoss().cuda() 
    for tt in range(config.episode_len_patch):
        entropy_row_dict['t'+str(tt)]=[]
        entropy_col_dict['t'+str(tt)]=[]
        reward_agent1_dict['t'+str(tt)] = []
        discriminator_dict['t'+str(tt)] = []
        discriminator_loss_dict['t'+str(tt)] = []
        action_map_dict['t'+str(tt)] = []

    for i, (image, ori_image, filenames) in enumerate(tqdm(test_loader)):
        # if i > 5:
        #     break
        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()
        last_image = image.copy()
        position_x_list = []
        position_y_list = []
        for tt in range(config.episode_len_patch):
            if verbose:
                image_history[t] = image
            # pick a patch via Agent 1
            actions, _ = picker(torch.from_numpy(image).cuda())
            tmp_actions = copy.deepcopy(actions.data)
            agent1_actions = actions.detach().clone()
            if config.use_discrete_action:
                if config.use_coordinate_classify_agent:
                    _, actions_row = torch.max(actions[:,0].data, dim=1)
                    _, actions_col = torch.max(actions[:,1].data, dim=1)
                    actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
                    if use_entropy:
                        tmp_actions = torch.clamp(tmp_actions, min=0)
                        n, num_actions, _ = tmp_actions.shape
                        # for row
                        log_pi_reshape = torch.log(torch.clamp(tmp_actions[:,0,:], min=1e-9, max=1-1e-9))
                        entropy_row = -torch.sum(tmp_actions[:,0,:] * log_pi_reshape, dim=-1).view(n,1)
                        entropy_row_dict['t'+str(tt)].append(entropy_row.cpu().numpy())
                        # for col
                        log_pi_reshape = torch.log(torch.clamp(tmp_actions[:,1,:], min=1e-9, max=1-1e-9))
                        entropy_col = -torch.sum(tmp_actions[:,1,:] * log_pi_reshape, dim=-1).view(n,1)
                        entropy_col_dict['t'+str(tt)].append(entropy_col.cpu().numpy())
                else:
                    _, actions = torch.max(actions.data, dim=1)
                actions = actions.cpu().numpy()
            image_patches, ori_image_patches, start_x, start_y = tool.crop_patches(actions, image, ori_image, rtn_pos=True)
            env.reset(ori_image=ori_image_patches, image=image_patches) 

            if use_discriminator : # use discriminator to check reward
                '''print("image_patches:",image_patches.shape,image_patches.max(),image_patches.min())'''
                if config.use_action_input:
                    bs, ch, h = agent1_actions.shape
                    if config.action_input_type == 'map' :
                        row = agent1_actions[:,0,:].reshape(bs,h,1).repeat(1,1,h)
                        col = agent1_actions[:,1,:].reshape(bs,1,h).repeat(1,h,1)
                        actions_map = row*col
                    else:
                        actions_map = agent1_actions.reshape(bs, ch*h)
                if config.use_img_patches:
                    out_discr = model_discri(torch.from_numpy(image_patches).cuda())
                if config.use_action_input:
                    if config.action_input_type == 'map':
                        actions_map = actions_map.unsqueeze(1)
                    action_map_dict['t'+str(tt)].append(actions_map)
                    out_discr = model_discri(actions_map)
                out_sigmoid = torch.sigmoid(out_discr)
                # print("out_disc:",out_discr)
                # out_discr = model_discri(torch.from_numpy(image_patches).cuda())

            for t in range(config.episode_len_test):
                image_input = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                pi_out, v_out, p = model(image_input, flag_a2c=True)

                p = p.permute(1, 0).cpu().data.numpy()
                env.set_param(p)
                p_list[t].append(p)

                actions = a2c.act(pi_out, deterministic=True)
                last_image = image_patches.copy()
                image_patches, reward = env.step(actions)
                image_patches = np.clip(image_patches, 0, 1)

                reward_sum += np.mean(reward)

                actions = actions.astype(np.uint8)
                prob = pi_out.cpu().data.numpy()
                total = actions.size
                for n in range(config.num_actions):
                    actions_prob[n, t] += np.sum(actions==n) / total
            # image/image_patches : bs x ch x h x w; 
            # entropy_col/row : bs x 1 
            if use_entropy:
                old_image = copy.deepcopy(image)
            if reward_th != None:
                last_image = copy.deepcopy(image)
            if use_discriminator :
                last_image = copy.deepcopy(image)

            # paste the reovered image patches on the original image
            image = tool.paste(image_patches, image, start_x, start_y)
            image = np.clip(image, 0, 1)
            # count the reward for agent 1
            reward_agent1 = np.sum(np.abs(ori_image - last_image) * 255  - np.abs(ori_image - image) * 255, axis=(1,2,3)) / (3*config.patch_height*config.patch_width)
            reward_agent1_dict['t'+str(tt)].append(reward_agent1)

            if use_discriminator : # use discriminator to check reward
                num = out_discr.shape[0]
                loss_discr = GANCriterion(out_discr, torch.from_numpy(reward_agent1).cuda().reshape(num, 1))
                discriminator_dict['t'+str(tt)].append(out_sigmoid.detach().cpu().numpy().reshape(num, 1))
                tmp = loss_discr.detach().cpu().numpy()
                tmp = np.repeat(tmp, num, axis=0)
                # print("tmp:",tmp.shape)
                discriminator_loss_dict['t'+str(tt)].append(tmp)#reward_agent1.reshape(num,1))#
                # selection
                # print("image:",type(image),"previous_image:",type(previous_image))
                image = torch.where(out_sigmoid.reshape(num, 1, 1, 1).cuda() > disc_th, torch.from_numpy(image).cuda(), torch.from_numpy(last_image).cuda()).cpu().numpy()

            if use_entropy and threshold !=None:
                # check images
                num, _ = entropy_row.shape
                # print("image:",type(image),"previous_image:",type(previous_image))
                image = torch.where(entropy_row.reshape(num, 1, 1, 1) <= threshold, torch.from_numpy(image).cuda(), torch.from_numpy(old_image).cuda()).cpu().numpy()

            if reward_th != None:
                # check images
                num = reward_agent1.shape[0]
                # print("image:",type(image),"previous_image:",type(previous_image))
                image = torch.where(torch.from_numpy(reward_agent1).reshape(num, 1, 1, 1).cuda() > reward_th, torch.from_numpy(image).cuda(), torch.from_numpy(last_image).cuda()).cpu().numpy()

            position_x_list.append(start_x.reshape([-1,1]))
            position_y_list.append(start_y.reshape([-1,1]))

        position_x_list = np.concatenate(position_x_list, axis=1)
        position_y_list = np.concatenate(position_y_list, axis=1)

        for j in range(ori_image.shape[0]):
            if cut_edge: 
                PSNR_list.append(computePSNR(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8])) 
                SSIM_list.append(computeSSIM(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8]))
                NMSE_list.append(computeNMSE(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8]))
            else:
                PSNR_list.append(computePSNR(ori_image[j], previous_image[j], image[j])) 
                SSIM_list.append(computeSSIM(ori_image[j], previous_image[j], image[j]))
                NMSE_list.append(computeNMSE(ori_image[j], previous_image[j], image[j]))
            # tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j]), np.abs(previous_image[j] - image[j])), axis=2), [1,2,0])
            if save_images:
                filename =os.path.join(results_dir, filenames[j])
                img = np.transpose(image[j], [1,2,0]) * 255
                cv2.imwrite(filename, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                perform_file.write('{}\t'.format(filenames[j]))
                perform_file.write('{}\t{}\t{}\t'.format(PSNR_list[count][0],float(SSIM_list[count][0]),NMSE_list[count][0]))
                perform_file.write('{}\t{}\t{}\n'.format(PSNR_list[count][1],float(SSIM_list[count][1]),NMSE_list[count][1]))
            # perform_file.write('{}\t{}\t{}\t{}\{}\t{}\t{}\t{}\t{}\n'.format(filenames[j], \
            #                                             PSNR_list[count][0],float(SSIM_list[count][0]),NMSE_list[count][0],\
            #                                             PSNR_list[count][1],float(SSIM_list[count][1]),NMSE_list[count][1]))
            # draw output of different timesteps
            if verbose:
                cv2.imwrite(results_dir+'/'+str(i)+'_'+str(j)+'.bmp', np.concatenate([image_history[jj][0] for jj in range(config.episode_len)] + [image[0], ori_image[0]], axis=1) * 255)
            # write down the position
            '''position_file.write('%d\t'%(count))
            for k in range(config.episode_len_patch):
                position_file.write('(%d,%d)\t'%(position_x_list[j,k],position_y_list[j,k]))
            position_file.write('\n')'''
            count += 1

    if use_entropy:
        for tt in range(config.episode_len_patch):
            reward_agent1_dict['t'+str(tt)]=np.concatenate(reward_agent1_dict['t'+str(tt)], axis=0)
            entropy_row_dict['t'+str(tt)]=np.concatenate(entropy_row_dict['t'+str(tt)], axis=0)
            entropy_col_dict['t'+str(tt)]=np.concatenate(entropy_col_dict['t'+str(tt)], axis=0)
            discriminator_dict['t'+str(tt)]=np.concatenate(discriminator_dict['t'+str(tt)], axis=0)
            discriminator_loss_dict['t'+str(tt)]=np.concatenate(discriminator_loss_dict['t'+str(tt)], axis=0)
            action_map_dict['t'+str(tt)]=torch.cat(action_map_dict['t'+str(tt)], dim=0)
            '''print("reward - t"+str(tt)+" - mean: ",np.mean(reward_agent1_dict['t'+str(tt)]),"max:",np.max(reward_agent1_dict['t'+str(tt)]),"min:",np.min(reward_agent1_dict['t'+str(tt)]),"shape:",reward_agent1_dict['t'+str(tt)].shape)
            print("entropy_row - t"+str(tt)+" - mean: ",np.mean(entropy_row_dict['t'+str(tt)]),"max:",np.max(entropy_row_dict['t'+str(tt)]),"min:",np.min(entropy_row_dict['t'+str(tt)]),"shape:",entropy_row_dict['t'+str(tt)].shape)
            print("entropy_col - t"+str(tt)+" - mean: ",np.mean(entropy_col_dict['t'+str(tt)]),"max:",np.max(entropy_col_dict['t'+str(tt)]),"min:",np.min(entropy_col_dict['t'+str(tt)]),"shape:",entropy_col_dict['t'+str(tt)].shape)
            print("discri_value - t"+str(tt)+" - mean: ",np.mean(discriminator_dict['t'+str(tt)]),"max:",np.max(discriminator_dict['t'+str(tt)]),"min:",np.min(discriminator_dict['t'+str(tt)]),"shape:",discriminator_dict['t'+str(tt)].shape)
            print("discri_loss - t"+str(tt)+" - mean: ",np.mean(discriminator_loss_dict['t'+str(tt)]),"max:",np.max(discriminator_loss_dict['t'+str(tt)]),"min:",np.min(discriminator_loss_dict['t'+str(tt)]),"shape:",discriminator_loss_dict['t'+str(tt)].shape)'''
        
        n, _ = entropy_row_dict['t0'].shape
        # n = len(PSNR_list)
        for idx in range(n):
            str_tmp = "%d - PSNR : %.3f - Row : "%(idx, PSNR_list[idx][1])
            for tt in range(config.episode_len_patch):
                dis_val =  discriminator_dict['t'+str(tt)][idx]
                dis_loss =  discriminator_loss_dict['t'+str(tt)][idx]
                reward_tmp = reward_agent1_dict['t'+str(tt)][idx]
                if reward_th != None and reward_agent1_dict['t'+str(tt)][idx] < reward_th:
                    reward_tmp = 0
                str_tmp += "t%d : r=%.3f, dis=%.3f, rwd=%.3f " % (tt, reward_tmp, dis_val, dis_loss) # loss
            if idx == 0:
                print(str_tmp, file=print_log)
            if test_mode :
                print(str_tmp)
        
        if actmp_rwd_save_dir != None:
            action_map_list = []
            reward_list = []
            for tt in range(config.episode_len_patch):
                reward_list.append(reward_agent1_dict['t'+str(tt)])
                action_map_list.append(action_map_dict['t'+str(tt)])
            action_map_list = torch.cat(action_map_list, dim=0).cpu().numpy()
            reward_list = np.concatenate(reward_list, axis=0)
            print("action_map_list:",action_map_list.shape)
            print("reward_list:",reward_list.shape)
        if actmp_rwd_save_dir != None:
            path_actmp = os.path.join(actmp_rwd_save_dir, "T1_%d_T2_%d_action_maps"%(config.episode_len_patch,config.episode_len_test))
            np.save(path_actmp, action_map_list)
            path_reward = os.path.join(actmp_rwd_save_dir, "T1_%d_T2_%d_reward"%(config.episode_len_patch,config.episode_len_test))
            np.save(path_reward, reward_list)

        # for idx in range(n):
        #     str_tmp = "%d - PSNR : %.3f - Row : "%(idx, PSNR_list[idx][1])
        #     str_col = "| Col : "
        #     for tt in range(config.episode_len_patch):
        #         dis_val =  discriminator_dict['t'+str(tt)][idx]
        #         dis_loss =  discriminator_loss_dict['t'+str(tt)][idx]
        #         reward_tmp = reward_agent1_dict['t'+str(tt)][idx]
        #         if reward_th != None and reward_agent1_dict['t'+str(tt)][idx] < reward_th:
        #             reward_tmp = 0
        #         if threshold != None and entropy_row_dict['t'+str(tt)][idx] > threshold:
        #             str_tmp += "t%d:%.3f, r=%.3f, dis=%.3f, loss=%.3f " % (tt, 0, reward_tmp, dis_val, dis_loss)
        #             str_col += "t%d:%.3f " % (tt, 0)
        #         else:
        #             str_tmp += "t%d:%.3f, r=%.3f, dis=%.3f, loss=%.3f " % (tt, entropy_row_dict['t'+str(tt)][idx], reward_tmp, dis_val, dis_loss)
        #             str_col += "t%d:%.3f " % (tt, entropy_col_dict['t'+str(tt)][idx])#, dis_val, dis_loss)
        #     print(str_tmp, str_col)

    '''position_file.close()'''
    # print('actions_prob', actions_prob / count)

    # for key in PSNR_dict.keys():
    #     PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
    #     print('number of test images: ', len(PSNR_list))
    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    nmse_res = np.mean(np.array(NMSE_list), axis=0)
    
    print('PSNR', psnr_res, file=print_log)
    print('SSIM', ssim_res, file=print_log)
    print('NMSE', nmse_res, file=print_log)
    if test_mode:
        print('PSNR', psnr_res)
        print('SSIM', ssim_res)
        print('NMSE', nmse_res)

    for t in range(config.episode_len_test):
        print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1), file=print_log)
        if test_mode:   
            print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward, file=print_log)
    if test_mode:   
        print('test finished: reward ', avg_reward)

    if save_images:
        perform_file.write("Bicubic: {}\t{}\t{}\n".format(psnr_res[0],ssim_res[0],nmse_res[0]))
        perform_file.write("Ours: {}\t{}\t{}\n".format(psnr_res[1],ssim_res[1],nmse_res[1]))
        perform_file.close()

    return avg_reward, psnr_res, ssim_res, nmse_res



if __name__ == "__main__":
    args = parse()
    sys.path.append(args.dataset)
    from config_tpM import config

    if args.episode_len_test != None:
        config.episode_len_test = args.episode_len_test
    if args.episode_len_patch_test != None:
        config.episode_len_patch = args.episode_len_patch_test
    torch.backends.cudnn.benchmark = True
    if args.actionMap_reward_save_dir != None and not os.path.exists(args.actionMap_reward_save_dir):
        os.makedirs(args.actionMap_reward_save_dir)
    results_dir = os.path.join('./results', args.model_name, str(args.episodes))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if args.episodes == 'best':
        model_path = os.path.join('./logs', args.model_name, 'models', str(args.episodes))
    else:
        model_path = os.path.join('./logs', args.model_name, 'models', args.model_name[:-2] + '_' + str(args.episodes))
    
    model_path_agent = model_path
    

    if config.use_discrete_action:
        agent1_model_path = model_path + '_agent1.pth'
    else: # use continous actions
        agent1_model_path = model_path + '_agent1_actor.pth'
    if config.valid_old_agent2 : 
        agent2_model_path = model_path + '_old_agent2.pth'
    else:
        agent2_model_path = model_path + '_agent2.pth'
    
    if config.resume_model != '' and config.resume_agent1 != '':
        agent1_model_path = config.resume_agent1
        agent2_model_path = config.resume_model

    print("agent1_model_path:",agent1_model_path)
    print("agent2_model_path:",agent2_model_path)
    # env = Env(config)
    model = MyFcn(config)
    if config.use_coordinate_classify_agent:
        if config.use_sm_agent1:
            picker = CoordinateClassifyPixelPatchAgent_sm(config)
        else:
            picker = CoordinateClassifyPixelPatchAgent(config)  
    elif config.use_pixel_oriented_patch:
        picker = PixelPatchAgent(config)  
    elif config.use_discrete_action: 
        picker = PatchAgent_2(config) 
    else:
        picker = PatchPicker(config) 
    # discriminator:
    # model_discri = NLayerDiscriminator(n_layers=config.n_layers_discr, \
    #                                     image_size=config.patch_height \
    #                                         if config.discriminator_in_agent2 \
    #                                         else config.image_height )
    if config.action_input_type == 'map' :
        model_discri = NLayerDiscriminator(input_nc = config.discr_input_nc,
                                            n_layers=config.n_layers_discr, \
                                            image_size=config.patch_height \
                                                if config.discriminator_in_agent2 \
                                                else config.image_height ) 
    else:
        if config.last_layer_ch == 512:
            model_discri = MLP_512(input_nc=config.patch_height*2)
        else:
            model_discri = MLP(input_nc=config.patch_height*2)
    
    discri_model_path = model_path + '_discrim.pth' if args.discriminator_model_path == None else args.discriminator_model_path
    model_discri.load_state_dict(torch.load(discri_model_path))
    model.load_state_dict(torch.load(agent2_model_path))
    picker.load_state_dict(torch.load(agent1_model_path))
    model_discri = torch.nn.DataParallel(model_discri, device_ids=args.gpu).cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    picker = torch.nn.DataParallel(picker, device_ids=args.gpu).cuda()
    a2c = PixelWiseA2C(config)
    if config.use_coordinate_classify_agent:
        pAC = PatchWiseAC_discrete_coordinate_classify(config) 
    elif config.use_discrete_action: 
        pAC = PatchWiseAC_discrete(config) 
    else:
        pAC = PatchWiseAC(config) 
    model_discri.eval()
    model.eval()
    picker.eval()
    # avg_reward, psnr_res, ssim_res = test(model, a2c, config, early_break=False, batch_size=50, verbose=True)
    avg_reward, psnr_res, ssim_res, nmse_res = test(model, picker, model_discri, a2c, pAC, config, batch_size=12, results_dir=results_dir, use_entropy=True, entropy_th=args.entropy_th, reward_th=args.reward_th, cut_edge=args.cut_edge, use_discriminator=args.use_discriminator,test_mode=True, disc_th=args.disc_th, actmp_rwd_save_dir=args.actionMap_reward_save_dir,save_images=args.save_images)
