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

from env import Env, Env_patch, Env_RGB
from model import MyFcn, PatchPicker
from pixel_wise_a2c import PixelWiseA2C, PatchWiseAC
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE, crop_and_paste

from tqdm import tqdm
from piq import fsim, gmsd

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MICCAI', type=str,
                        dest='dataset', help='to use dataset.py and config.py in which directory')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--model_name', type=str, help='the folder of the pretrained model')
    parser.add_argument('--episodes', type=str, help='which model to load')
    parser.add_argument('--episode_len', type=int, default=None, help='how many episode to run during inference')
    parser.add_argument('--cut_edge',action="store_true",default=False, help='whether cut the edge during evaluation')    
    parser.add_argument('--save_images',action="store_true",default=False, help='whether save images')    

    return parser.parse_args()



def validation(model,a2c, config, early_break=False, batch_size=None, verbose=False, valid_mode=False, current_episode=0, save_dir=None, cut_edge=False, save_images=False):
    if batch_size is None:
        batch_size = config.batch_size
    env =  Env(config) if not config.use_RGB_actions else Env_RGB(config)
    # env_p = Env_patch(config)

    # tool for cropping and pasting
    # tool = crop_and_paste(config)

    results_dir = save_dir #if valid_mode else 'results/'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    if config.use_HistoSR_dataset:
        from HistoSR import data_loader_shuffled_data
        test_loader = data_loader_shuffled_data.get_loader(
            config.histosr_data_test,
            batch_size=batch_size, 
            stage='test', 
            num_workers=config.workers,
            test_list_dir=config.test_list_dir,
            use_bicubic_upsample=config.use_bicubic_upsample,
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
    fsim_res = 0
    gmsd_res = 0
    GMSD_list = []
    FSIM_list = []

    count = 0
    img_idx = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len))
    image_history = dict()
    if save_images:
        perform_file = open(os.path.join(results_dir, 'eval.txt'), 'w')

    for i, (image, ori_image, filenames) in enumerate(tqdm(test_loader)):
        count += 1
        if early_break and count == 100 : #401: # test only part of the dataset
            break
        # if count % 100 == 0:
        #     print('tested: ', count)

        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()

        # for t in range(config.episode_len_patch):
        if verbose:
            image_history[t] = image
        # pick a patch via Agent 1
        # actions = picker(torch.from_numpy(image).cuda())
        # image_patches, ori_image_patches, start_x, start_y = tool.crop_patches(actions[:,0].detach(), actions[:,1].detach(), image, ori_image, rtn_pos=True)
        
        env.reset(ori_image=ori_image, image=image) 

        for t in range(config.episode_len):
            image_input = Variable(torch.from_numpy(image).cuda(), volatile=True)
            pi_out, v_out, p = model(image_input, flag_a2c=True)

            p = p.permute(1, 0).cpu().data.numpy()
            env.set_param(p)
            p_list[t].append(p)

            actions = a2c.act(pi_out, deterministic=True)
            last_image = image.copy()
            image, reward = env.step(actions)
            image = np.clip(image, 0, 1)

            reward_sum += np.mean(reward)

            actions = actions.astype(np.uint8)
            prob = pi_out.cpu().data.numpy()
            total = actions.size
            for n in range(config.num_actions):
                actions_prob[n, t] += np.sum(actions==n) / total

            # paste the reovered image patches on the original image
            # image = tool.paste(image_patches, image, start_x, start_y)
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

        # compute fsim/gmsd
        if cut_edge:
            fsim_val = fsim(torch.tensor(image[:,:, 8:-8, 8:-8]).cuda(), torch.tensor(ori_image[:, :, 8:-8, 8:-8]).cuda(), data_range=255, reduction='none')
            gmsd_val = gmsd(torch.tensor(image[:,:, 8:-8, 8:-8]).cuda(), torch.tensor(ori_image[:, :, 8:-8, 8:-8]).cuda(), data_range=255, reduction='none')
        else:
            fsim_val = fsim(torch.tensor(image).cuda(), torch.tensor(ori_image).cuda(), data_range=255, reduction='none')
            gmsd_val = gmsd(torch.tensor(image).cuda(), torch.tensor(ori_image).cuda(), data_range=255, reduction='none')
        
        GMSD_list.extend(gmsd_val.cpu().numpy())
        FSIM_list.extend(fsim_val.cpu().numpy())

        if valid_mode and i == 0 :
            tmp_tensor = []
        for j in range(ori_image.shape[0]):
            if cut_edge : 
                PSNR_list.append(computePSNR(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8])) 
                SSIM_list.append(computeSSIM(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8]))
                NMSE_list.append(computeNMSE(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8]))
            else:
                PSNR_list.append(computePSNR(ori_image[j], previous_image[j], image[j])) 
                SSIM_list.append(computeSSIM(ori_image[j], previous_image[j], image[j]))
                NMSE_list.append(computeNMSE(ori_image[j], previous_image[j], image[j]))
            
            if save_images:
                filename =os.path.join(results_dir, filenames[j])
                img = np.transpose(image[j], [1,2,0]) * 255
                cv2.imwrite(filename, cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
                perform_file.write('{}\t'.format(filenames[j]))
                perform_file.write('{}\t{}\t{}\t'.format(PSNR_list[img_idx][0],float(SSIM_list[img_idx][0]),NMSE_list[img_idx][0]))
                perform_file.write('{}\t{}\t{}\n'.format(PSNR_list[img_idx][1],float(SSIM_list[img_idx][1]),NMSE_list[img_idx][1]))
            img_idx += 1
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
                    tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j])), axis=2), [1,2,0])
                    tmp_tensor.append(tensor_cat)
            '''else:
                tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j])), axis=2), [1,2,0])
                cv2.imwrite(os.path.join(results_dir, str(i)+'_'+str(j)+'.bmp'), tensor_cat * 255)'''
            # draw output of different timesteps
            if verbose:
                cv2.imwrite('results/time_steps/'+str(i)+'_'+str(j)+'.bmp', np.concatenate([image_history[jj][0] for jj in range(config.episode_len)] + [image[0], ori_image[0]], axis=1) * 255)

        if valid_mode and i == 0 : 
            cv2.imwrite(os.path.join(results_dir, str(current_episode) + '_'+ str(i)+'.bmp'), np.concatenate(tmp_tensor, axis=0) * 255)
    actions_prob = actions_prob / count
    print('actions_prob:')
    print(actions_prob)# / count)

    # for key in PSNR_dict.keys():
    #     PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
    #     print('number of test images: ', len(PSNR_list))
    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    nmse_res = np.mean(np.array(NMSE_list), axis=0)
    
    gmsd_res = np.mean(np.array(GMSD_list), axis=0)
    fsim_res = np.mean(np.array(FSIM_list), axis=0)
    
    print('PSNR', psnr_res)
    print('SSIM', ssim_res)
    print('NMSE', nmse_res)
    print('GMSD', gmsd_res)
    print('FSIM', fsim_res)

    for t in range(config.episode_len):
        print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward)

    if save_images:
        perform_file.write("Bicubic: {}\t{}\t{}\n".format(psnr_res[0],ssim_res[0],nmse_res[0]))
        perform_file.write("Ours: {}\t{}\t{}\n".format(psnr_res[1],ssim_res[1],nmse_res[1]))
        perform_file.close()

    return avg_reward, psnr_res, ssim_res, nmse_res, actions_prob

def test(model, a2c, config,  batch_size=None, verbose=False, results_dir=None, cut_edge=False):
    if batch_size is None:
        batch_size = config.batch_size
    env = Env(config)
    # env_p = Env_patch(config)

    # tool for cropping and pasting
    # tool = crop_and_paste(config)

    if config.use_HistoSR_dataset:
        from HistoSR import data_loader_shuffled_data
        test_loader = data_loader_shuffled_data.get_loader(
            config.histosr_data_test,
            batch_size=batch_size, 
            stage='test', 
            num_workers=config.workers,
            test_list_dir=config.test_list_dir,
            use_bicubic_upsample=config.use_bicubic_upsample)
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
    fsim_res = 0
    gmsd_res = 0
    GMSD_list = []
    FSIM_list = []

    count = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len))
    image_history = dict()

    position_file = open(os.path.join(results_dir, 'position.txt'), 'w')

    for i, (image, ori_image) in enumerate(tqdm(test_loader)):

        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()

        position_x_list = []
        position_y_list = []
        for t in range(config.episode_len_patch):
            if verbose:
                image_history[t] = image
            # pick a patch via Agent 1
            actions = picker(torch.from_numpy(image).cuda())
            image_patches, ori_image_patches, start_x, start_y = tool.crop_patches(actions[:,0].detach(), actions[:,1].detach(), image, ori_image, rtn_pos=True)
            
            env.reset(ori_image=ori_image_patches, image=image_patches) 

            for t in range(config.episode_len):
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
            position_x_list.append(start_x.reshape([-1,1]))
            position_y_list.append(start_y.reshape([-1,1]))

        # compute fsim/gmsd
        if cut_edge:
            fsim_val = fsim(torch.tensor(image[:,:, 8:-8, 8:-8]).cuda(), torch.tensor(ori_image[:, :, 8:-8, 8:-8]).cuda(), data_range=255, reduction='none')
            gmsd_val = gmsd(torch.tensor(image[:,:, 8:-8, 8:-8]).cuda(), torch.tensor(ori_image[:, :, 8:-8, 8:-8]).cuda(), data_range=255, reduction='none')
        else:
            fsim_val = fsim(torch.tensor(image).cuda(), torch.tensor(ori_image).cuda(), data_range=255, reduction='none')
            gmsd_val = gmsd(torch.tensor(image).cuda(), torch.tensor(ori_image).cuda(), data_range=255, reduction='none')
        
        GMSD_list.extend(gmsd_val)
        FSIM_list.extend(fsim_val)
        # fsim_res += sum(fsim_val)/ori_image.shape[0]
        # gmsd_res += sum(gmsd_val)/ori_image.shape[0]

        position_x_list = np.concatenate(position_x_list, axis=1)
        position_y_list = np.concatenate(position_y_list, axis=1)
        for j in range(ori_image.shape[0]):
            if cut_edge : 
                PSNR_list.append(computePSNR(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8])) 
                SSIM_list.append(computeSSIM(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8]))
                NMSE_list.append(computeNMSE(ori_image[j, :, 8:-8, 8:-8], previous_image[j, :, 8:-8, 8:-8], image[j, :, 8:-8, 8:-8]))
            else:
                PSNR_list.append(computePSNR(ori_image[j], previous_image[j], image[j])) 
                SSIM_list.append(computeSSIM(ori_image[j], previous_image[j], image[j]))
                NMSE_list.append(computeNMSE(ori_image[j], previous_image[j], image[j]))
        
            tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j])), axis=2), [1,2,0])
            # cv2.imwrite(os.path.join(results_dir, str(count)+'.bmp'), tensor_cat * 255)
            # draw output of different timesteps
            if verbose:
                cv2.imwrite(results_dir+'/'+str(i)+'_'+str(j)+'.bmp', np.concatenate([image_history[jj][0] for jj in range(config.episode_len)] + [image[0], ori_image[0]], axis=1) * 255)
            # write down the position
            position_file.write('%d\t'%(count))
            for k in range(config.episode_len_patch):
                position_file.write('(%d,%d)\t'%(position_x_list[j,k],position_y_list[j,k]))
            position_file.write('\n')
            count += 1


    position_file.close()
    # print('actions_prob', actions_prob / count)

    # for key in PSNR_dict.keys():
    #     PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
    #     print('number of test images: ', len(PSNR_list))
    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    nmse_res = np.mean(np.array(NMSE_list), axis=0)

    gmsd_res = np.mean(np.array(GMSD_list), axis=0)
    fsim_res = np.mean(np.array(FSIM_list), axis=0)
    
    print('PSNR', psnr_res)
    print('SSIM', ssim_res)
    print('NMSE', nmse_res)
    print('GMSD', gmsd_res)
    print('FSIM', fsim_res)

    for t in range(config.episode_len):
        print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward)

    return avg_reward, psnr_res, ssim_res, nmse_res



if __name__ == "__main__":
    args = parse()
    sys.path.append(args.dataset)
    from config import config

    if args.episode_len != None:
        config.episode_len = args.episode_len

    torch.backends.cudnn.benchmark = True
    
    results_dir = os.path.join('./results', args.model_name, args.episodes + '_' + config.data_degradation + '_T_' + str(config.episode_len))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_path = os.path.join('./logs', args.model_name, 'models', args.model_name + '_' + args.episodes)
    if args.episodes == 'best':
        model_path = os.path.join('./logs', args.model_name, 'models', args.episodes)

    # agent1_model_path = model_path + '_agent1_actor.pth'
    agent2_model_path = model_path + '_agent2.pth'
    # env = Env(config)
    model = MyFcn(config)
    picker = PatchPicker(config)
    model.load_state_dict(torch.load(agent2_model_path))
    # picker.load_state_dict(torch.load(agent1_model_path))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    # picker = torch.nn.DataParallel(picker, device_ids=args.gpu).cuda()
    a2c = PixelWiseA2C(config)
    # pAC = PatchWiseAC(config)
    picker.eval()
    model.eval()
    # avg_reward, psnr_res, ssim_res = test(model, a2c, config, early_break=False, batch_size=50, verbose=True)
    # avg_reward, psnr_res, ssim_res, nmse_res = test(model, a2c, config, batch_size=50, results_dir=results_dir)
    validation(model,a2c, config, batch_size=50, valid_mode=False, current_episode=None, save_dir=results_dir,  cut_edge=args.cut_edge,save_images=args.save_images)
