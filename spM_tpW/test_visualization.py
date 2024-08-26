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
from model import MyFcn, PatchPicker, PatchAgent, CoordinateClassifyPixelPatchAgent,CoordinateClassifyPixelPatchAgent_sm, PixelPatchAgent, PatchAgent_2
from pixel_wise_a2c import PixelWiseA2C, PatchWiseAC, PatchWiseAC_discrete, PatchWiseAC_discrete_coordinate_classify
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE, crop_and_paste

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
    

    return parser.parse_args()


def test(model, picker, a2c, pAC, config,  batch_size=None, verbose=False, results_dir=None, phase='test',alpha=0.5):

    # rgb
    color_list  =  {'nothing': (0,0,0),
                'Gaussian': (0, 0, 255), # blue
                'Laplace': (255, 0, 255), # purple like
                'Sobel_v1': (255,0,0),  # red
                'Sobel_v2': (255, 165, 0), # orange
                'Sobel_h1': (255, 255, 0), # yellow
                'Sobel_h2': (0, 255, 0),   # green
                'unsharp': (0, 139, 139),  # DarkCyan
                'subtraction': (139, 129, 76),  # LightGoldenrod4	
                'addition': (139, 0, 0)}# DarkRed	

    if batch_size is None:
        batch_size = config.batch_size
    env = Env(config)
    env_p = Env_patch(config)

    # tool for cropping and pasting
    tool = crop_and_paste(config)

    if config.use_shuffled_dataset:
        from HistoSR import data_loader_shuffled_data
        test_loader = data_loader_shuffled_data.get_loader(
            config.shuffled_data_test,
            batch_size=batch_size, 
            stage='test', 
            num_workers=config.workers,
            test_list_dir=config.test_list_dir)
    else:
        from HistoSR import data_loader_lmdb
        test_loader = data_loader_lmdb.get_loader(
            os.path.join(config.root, config.data_degradation, phase+'_lmdb'), 
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
    actions_prob = np.zeros((config.num_actions, config.episode_len_test),np.uint8)
    image_history = dict()

    image_dir = os.path.join(results_dir, 'testset')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    # position_file = open(os.path.join(results_dir, 'position.txt'), 'w')
    # all_data_npy = np.zeros([len(test_loader)*batch_size, 3, config.image_height, config.image_width])
    # print("all_data_npy:",all_data_npy.shape)
    save_idx = 0
    for i, (image, ori_image) in enumerate(tqdm(test_loader)):
        if i > 20: 
            break
        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()

        position_x_list = []
        position_y_list = []
        # input image 
        cv2.imwrite(os.path.join(image_dir, str(save_idx)+'.bmp'), cv2.cvtColor(np.transpose(image[0], [1,2,0]) * 255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(image_dir, str(save_idx)+'_gt.bmp'), cv2.cvtColor(np.transpose(ori_image[0], [1,2,0]) * 255, cv2.COLOR_BGR2RGB))
        for tt in range(config.episode_len_patch):
            if verbose:
                image_history[t] = image
            # pick a patch via Agent 1
            actions, _ = picker(torch.from_numpy(image).cuda())
            actions_raw = actions.detach().cpu().numpy()
            if config.use_discrete_action:
                if config.use_coordinate_classify_agent:
                    _, actions_row = torch.max(actions[:,0].data, dim=1)
                    _, actions_col = torch.max(actions[:,1].data, dim=1)
                    actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
                else:
                    _, actions = torch.max(actions.data, dim=1)
                actions = actions.cpu().numpy()
            image_patches, ori_image_patches, start_x, start_y = tool.crop_patches(actions, image, ori_image, rtn_pos=True)
            env.reset(ori_image=ori_image_patches, image=image_patches) 
            # picked patches
            cv2.imwrite(os.path.join(image_dir, str(save_idx)+'_patch_before_T_'+str(tt)+'.bmp'), cv2.cvtColor(np.transpose(np.clip(image_patches[0], 0, 1), [1,2,0]) * 255, cv2.COLOR_BGR2RGB))

            save_img_path_list = []
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

                # visualization the actions
                for j in range(ori_image.shape[0]):
                    # if i > 1 : break
                    # if not os.path.exists:
                    #     os.makedirs(os.path.join(image_dir, str(j)))
                    a = actions[j].astype(np.uint8)
                    canvas = last_image[j].copy()
                    ch, h, w = canvas.shape
                    unchanged_mask = np.abs(last_image[j] - image_patches[j]) < (1 / 255) # some pixel values are not changed
                    unchanged_mask = unchanged_mask[0,:,:]

                    A = np.zeros([h,w,ch],np.float32)
                    for action_name in config.actions:
                        act_idx = config.actions[action_name]
                        color = color_list[action_name]
                        a_mask = (a==act_idx) #& (1-unchanged_mask).astype(np.bool)
                        A[a_mask,:] = color
                    
                    canvas_each_act = np.transpose(canvas*255, [1,2,0]).astype(np.uint8)
                    A = cv2.cvtColor(A.astype(np.uint8),cv2.COLOR_RGB2BGR)
                    canvas_each_act = cv2.cvtColor(canvas_each_act,cv2.COLOR_RGB2BGR)
                    cv2.addWeighted(A, alpha, canvas_each_act, 1 - alpha,0, canvas_each_act)
                    img_path = os.path.join(image_dir, str(i)+'_T1_'+str(tt)+'_T2_'+str(t)+'.png')
                    save_img_path_list.append((j,img_path))
                    cv2.imwrite(img_path, canvas_each_act)

                    img_after_path = os.path.join(image_dir, str(i)+'_patch_after_T1_'+str(tt)+'_T2_'+str(t)+'.bmp')
                    cv2.imwrite(img_after_path,  cv2.cvtColor(np.transpose(np.clip(image_patches[j], 0, 1), [1,2,0]) * 255, cv2.COLOR_BGR2RGB))
                    """ for n in range(config.num_actions):
                        config.actions[]
                        canvas_each_act = np.transpose(canvas.copy()*255, [1,2,0]).astype(np.uint8)
                        A = np.zeros([h,w,ch],np.float32)
                        a_mask = (a==n) & (1-unchanged_mask).astype(np.bool)
                        A[a_mask,0] = 255
                        A = cv2.cvtColor(A.astype(np.uint8),cv2.COLOR_RGB2BGR)
                        canvas_each_act = cv2.cvtColor(canvas_each_act,cv2.COLOR_RGB2BGR)
                        print("canvas_each_act:",canvas_each_act.shape,canvas_each_act.max(),canvas_each_act.min())
                        print("A:",A.shape,A.max(),A.min())
                        cv2.addWeighted(A, alpha, canvas_each_act, 1 - alpha,0, canvas_each_act)
                        img_path = os.path.join(image_dir, str(i)+'_act_'+str(n)+'_t_'+str(t)+'.bmp')
                        cv2.imwrite(img_path, canvas_each_act) """

                
            # save the specific patches:
            cv2.imwrite(os.path.join(image_dir, str(save_idx)+'_patch_after_T_'+str(tt)+'.bmp'), cv2.cvtColor(np.transpose(np.clip(image_patches[0], 0, 1), [1,2,0]) * 255, cv2.COLOR_BGR2RGB))
            np.save(os.path.join(image_dir, str(save_idx)+ '_actions_T_'+str(tt)), actions_raw[0])


            old_image = copy.deepcopy(image)

            # paste the reovered image patches on the original image
            image = tool.paste(image_patches, image, start_x, start_y)
            image = np.clip(image, 0, 1)

            reward_agent1 = np.sum(np.abs(ori_image - old_image) * 255  - np.abs(ori_image - image) * 255, axis=(1,2,3)) / (3*config.patch_height*config.patch_width)
            # print("i:",i,"tt:",tt, "reward:",np.mean(reward_agent1))
            # rename image list 
            for index in range(len(save_img_path_list)):
                idx = save_img_path_list[index][0]
                img_path = save_img_path_list[index][1]
                # print(idx, img_path)
                oldname = img_path
                newname = img_path.split('.png')[0] + '_' + str(round(reward_agent1[idx], 2)) + '.png'
                # print("newname:",newname)
                os.rename(oldname,newname) 

            cv2.imwrite(os.path.join(image_dir, str(save_idx)+'_recovered_T_'+str(tt)+'.bmp'), cv2.cvtColor(np.transpose(image[0], [1,2,0]) * 255, cv2.COLOR_BGR2RGB))

            position_x_list.append(start_x.reshape([-1,1]))
            position_y_list.append(start_y.reshape([-1,1]))
        save_idx += 1

        position_x_list = np.concatenate(position_x_list, axis=1)
        position_y_list = np.concatenate(position_y_list, axis=1)
        for j in range(ori_image.shape[0]):
            PSNR_list.append(computePSNR(ori_image[j], previous_image[j], image[j])) 
            SSIM_list.append(computeSSIM(ori_image[j], previous_image[j], image[j]))
            NMSE_list.append(computeNMSE(ori_image[j], previous_image[j], image[j]))

            # all_data_npy[count, :, :, :] = np.uint8(image[j] * 255)

            # tensor_cat = np.transpose(np.concatenate((ori_image[j], previous_image[j], image[j], np.abs(ori_image[j] - image[j]), np.abs(previous_image[j] - image[j])), axis=2), [1,2,0])
            # cv2.imwrite(os.path.join(image_dir, str(count)+'.bmp'), np.transpose(image[j], [1,2,0]) * 255)
            # cv2.imwrite(os.path.join(image_dir, str(count)+'_'+str(round(PSNR_list[count][1], 3))+'.bmp'), np.transpose(image[j], [1,2,0]) * 255)
            # draw output of different timesteps
            if verbose:
                cv2.imwrite(results_dir+'/'+str(i)+'_'+str()+'.bmp', np.concatenate([image_history[jj][0] for jj in range(config.episode_len)] + [image[0], ori_image[0]], axis=1) * 255)
            # write down the position
            # position_file.write('%d\t'%(count))
            # for k in range(config.episode_len_patch):
            #     position_file.write('(%d,%d)\t'%(position_x_list[j,k],position_y_list[j,k]))
            # position_file.write('\n')
            count += 1
    print("count:",count)
    # position_file.close()
    # print('actions_prob', actions_prob / count)

    # for key in PSNR_dict.keys():
    #     PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
    #     print('number of test images: ', len(PSNR_list))
    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    nmse_res = np.mean(np.array(NMSE_list), axis=0)
    # save numpy data
    # np.save(os.path.join(results_dir, phase + '_PSNR_' + str(round(psnr_res[1], 3)) + '_SSIM_' + str(round(ssim_res[1], 3)) + '_uint8.npy'), all_data_npy)

    print('PSNR', psnr_res)
    print('SSIM', ssim_res)
    print('NMSE', nmse_res)

    for t in range(config.episode_len_test):
        print('parameters at {}: '.format(t), np.mean(np.concatenate(p_list[t], axis=1), axis=1))

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward)

    return avg_reward, psnr_res, ssim_res, nmse_res



if __name__ == "__main__":
    args = parse()
    sys.path.append(args.dataset)
    from config import config

    if args.episode_len_test != None:
        config.episode_len_test = args.episode_len_test
    if args.episode_len_patch_test != None:
        config.episode_len_patch = args.episode_len_patch_test
    torch.backends.cudnn.benchmark = True
    
    results_dir = os.path.join('./visualization', args.model_name, str(args.episodes))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if args.episodes == 'best':
        model_path = os.path.join('./logs', args.model_name, 'models', str(args.episodes))
    else:
        model_path = os.path.join('./logs', args.model_name, 'models', args.model_name[:-2] + '_' + str(args.episodes))
    if config.use_discrete_action:
        agent1_model_path = model_path + '_agent1.pth'
    else: # use continous actions
        agent1_model_path = model_path + '_agent1_actor.pth'
    if config.valid_old_agent2 : 
        agent2_model_path = model_path + '_old_agent2.pth'
    else:
        agent2_model_path = model_path + '_agent2.pth'
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
    model.load_state_dict(torch.load(agent2_model_path))
    picker.load_state_dict(torch.load(agent1_model_path))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    picker = torch.nn.DataParallel(picker, device_ids=args.gpu).cuda()
    a2c = PixelWiseA2C(config)
    if config.use_coordinate_classify_agent:
        pAC = PatchWiseAC_discrete_coordinate_classify(config) 
    elif config.use_discrete_action: 
        pAC = PatchWiseAC_discrete(config) 
    else:
        pAC = PatchWiseAC(config) 
    model.eval()
    picker.eval()
    # avg_reward, psnr_res, ssim_res = test(model, a2c, config, early_break=False, batch_size=50, verbose=True)
    avg_reward, psnr_res, ssim_res, nmse_res = test(model, picker, a2c, pAC, config, batch_size=1, results_dir=results_dir, phase='test')
