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

from env import Env, Env_RGB#, Env_patch
from model import MyFcn#, PatchPicker, PatchCritic
from pixel_wise_a2c import PixelWiseA2C#, PatchWiseAC
from test import test, validation

from utils import adjust_learning_rate
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE, set_requires_grad#, crop_and_paste
from shutil import copyfile

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MICCAI', type=str,
                        dest='dataset', help='to use dataset.py and config.py in which directory')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')

    return parser.parse_args()


def train():
    torch.backends.cudnn.benchmark = False

    # load config
    args = parse()
    sys.path.append(args.dataset)
    from config import config
    assert config.switch % config.iter_size == 0
    time_tuple = time.localtime(time.time())
    log_dir = './logs/' + '_'.join(map(lambda x: str(x), time_tuple[1:5]))
    model_dir = os.path.join(log_dir, 'models')
    results_dir = os.path.join(log_dir, 'results')
    print('log_dir: ', log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # copy the config file to log dir
    copyfile(os.path.join(args.dataset,'config.py'), os.path.join(log_dir, 'config.py'))
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    # validation log
    val_log = open(os.path.join(log_dir, 'val.txt'), 'a+')
    # loss log
    loss_log = open(os.path.join(log_dir, 'loss.txt'), 'a+')
    # loss log
    # actions_log = open(os.path.join(log_dir, 'log_actions.txt'), 'a+')
    now = time.strftime("%c")
    val_log.write('================ Validation results (%s) ================\n' % now)
    loss_log.write('================ Training loss (%s) ================\n' % now)
    # actions_log.write('================ Action log (%s) ================\n' % now)
    # if not os.path.exists('model/'):
    #     os.mkdir('model/')

    # dataset
    # from dataset import MRIDataset
    # train_loader = torch.utils.data.DataLoader(
    #     dataset = MRIDataset(image_set='train', transform=True, config=config),
    #     batch_size=config.batch_size, shuffle=True,
    #     num_workers=config.workers, pin_memory=True)

    if config.use_HistoSR_dataset:
        from HistoSR import data_loader_shuffled_data
        train_loader = data_loader_shuffled_data.get_loader(
            config.histosr_data_train,
            batch_size=config.batch_size, 
            stage='train', 
            num_workers=config.workers,
            use_bicubic_upsample=config.use_bicubic_upsample)
    else:
        from HistoSR import data_loader_lmdb
        train_loader = data_loader_lmdb.get_loader(
            os.path.join(config.root, config.data_degradation, 'train_lmdb'), 
            batch_size=config.batch_size, 
            stage='train', 
            num_workers=config.workers)
    
    
    # agent 1
    # env_p = Env_patch(config)
    # pAC = PatchWiseAC(config)
    # agent 2
    env = Env(config) if not config.use_RGB_actions else Env_RGB(config)
    a2c = PixelWiseA2C(config)
    # tool for cropping and pasting
    # tool = crop_and_paste(config)

    episodes = 0
    model = MyFcn(config)
    if len(config.resume_model) > 0: # resume training
        model.load_state_dict(torch.load(config.resume_model))
        episodes = int(config.resume_model.split('.')[0].split('_')[-1])
        print('resume from episodes {}'.format(episodes))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    # construct optimizers for a2c and ddpg - for pi parameters
    parameters_wo_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' not in key]
    optimizer = torch.optim.Adam(parameters_wo_p, config.base_lr)

    parameters_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' in key]
    #parameters_pi = [value for key, value in dict(model.module.named_parameters()).items() if '_pi.' in key]
    params = [
        {'params': parameters_p, 'lr': config.base_lr},
    ]
    optimizer_p = torch.optim.SGD(params, config.base_lr)

    # ----- model for patch picker ------
    '''picker = PatchPicker(config)
    episodes_picker = 0
    if len(config.resume_picker) > 0: # resume training
        picker.load_state_dict(torch.load(config.resume_picker))
        episodes_picker = int(config.resume_picker.split('.')[0].split('_')[-1])
        print('resume picker model from episodes {}'.format(episodes_picker))
    picker = torch.nn.DataParallel(picker, device_ids=args.gpu).cuda()    
    params = [{'params': picker.module.parameters(), 'lr': config.base_lr},]
    optimizer_picker = torch.optim.SGD(params, config.base_lr)

    # ----- model for patch critic -------
    patchCritic = PatchCritic()
    episodes_patchCritic = 0
    if len(config.resume_patchCritic) > 0 : # resume training
        patchCritic.load_state_dict(torch.load(config.resume_patchCritic))
        episodes_patchCritic = int(config.resume_patchCritic.split('.')[0].split('_')[-1])
        print('resume patch critic model from episodes {}'.format(episodes_patchCritic))
    patchCritic = torch.nn.DataParallel(patchCritic, device_ids=args.gpu).cuda()    
    params = [{'params': patchCritic.module.parameters(), 'lr': config.base_lr},]
    optimizer_patchCritic = torch.optim.SGD(params, config.base_lr)'''

    # training
    flag_a2c = True # if True, use a2c; if False, use ddpg
    flag_updateCrt = True # if True, update critic; if False, update actor
    # best psnr
    best_psnr = 0
    best_psnr_epoch = -1 
    while episodes < config.num_episodes:

        for i, (image_data, ori_image_data) in enumerate(train_loader): # ori_image for target image; image for degraded image
            # print("image_data:",image_data.max(),image_data.min())
            # print("ori_image_data:",ori_image_data.max(), ori_image_data.min())
            # log 
            loss_log.write('Episode %d - ' % (episodes))
            # adjust learning rate
            learning_rate = adjust_learning_rate(optimizer, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            _ = adjust_learning_rate(optimizer_p, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            # _ = adjust_learning_rate(optimizer_picker, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            # _ = adjust_learning_rate(optimizer_patchCritic, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)

            # ori_image = ori_image_data.numpy()
            # image = image_data.numpy()

            # env_p.reset(ori_image=ori_image, image=image)
            # reward_picker = np.array((0))#np.zeros((1))

            # ---------- Agent 1 ---------- #
            '''if not flag_updateCrt:
                v_out_dict = dict()
            # turn out to set require grad as True
            set_requires_grad(patchCritic, flag_updateCrt)
            # print("episodes:",episodes,"flag_updateCrt:",flag_updateCrt)
            for t in range(config.episode_len_patch):
                # print("episode_t:",t)
                image_input = Variable(torch.from_numpy(image).cuda())
                reward_picker_input = Variable(torch.from_numpy(reward_picker).cuda())
                # predict the position (x,y)
                actions = picker(image_input)
                # get the value function
                if flag_updateCrt:
                    v_out = patchCritic(image_input, actions.detach())
                else :
                    v_out = patchCritic(image_input, actions)
                    v_out_dict[t] = - v_out.mean()
                # store the previous reward, and value for current state and action
                pAC.act(v_out, reward_picker_input)
                # step
                previous_image = image.copy()
                image, reward_picker = env_p.step(actions)
                # for display
                if not(episodes % config.display):
                    print('\nUpdate critic: ', flag_updateCrt)
                    print('episode {}: reward@{} = {:.4f}'.format(episodes, t, np.mean(reward_picker)))
                    print("PSNR: {:.5f} -> {:.5f}".format(*computePSNR(ori_image[0], previous_image[0], image[0])))
                    print("SSIM: {:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0], previous_image[0], image[0])))

            # compute loss and backpropagate
            if flag_updateCrt :  # for critic
                loss_critic = pAC.compute_loss(reward=Variable(torch.from_numpy(reward_picker).cuda()))
                # loss_critic = losses_critic #/ config.iter_size
                loss_critic.backward()
            else:              # for actor
                loss_actor = sum(v_out_dict.values()) * config.ac1_c_loss_coeff #/ config.iter_size
                pAC.reset()
                loss_actor.backward()
            # update parameters
            # if not((episodes+1) % config.iter_size):
            # -- optimize critic -- #
            if flag_updateCrt:
                optimizer_patchCritic.step()
                optimizer_patchCritic.zero_grad()
                optimizer_picker.zero_grad()
                writer.add_scalar('agent1_mc_loss', float(loss_critic.cpu().data.numpy()), episodes)
                loss_log.write('agent1_mc_loss : %.3f ' % (float(loss_critic.cpu().data.numpy())))
            # -- optimize actor -- #
            else:
                set_requires_grad(patchCritic, False)
                optimizer_picker.step()
                optimizer_picker.zero_grad()
                optimizer_patchCritic.zero_grad()
                for l in v_out_dict.keys():
                    writer.add_scalar('agent1_v_out_{}'.format(l), float(v_out_dict[l].cpu().data.numpy()), episodes)
                    loss_log.write('agent1_v_out_%s : %.3f ' % (l,float(v_out_dict[l].cpu().data.numpy())))
            #  train actor and critic alternatively
            if not((episodes+1) % config.switch_iter_agent1):
                flag_updateCrt = not flag_updateCrt
                if episodes < config.ac1_warm_up_episodes-1 :
                    flag_updateCrt = True'''
            

            # ---------- Agent 2 ---------- #
            ori_image = ori_image_data.numpy()
            image = image_data.numpy()
            reward = np.zeros((1))

            # pick a patch via Agent 1
            # actions = picker(torch.from_numpy(image).cuda())
            # image, ori_image = tool.crop_patches(actions[:,0].detach(), actions[:,1].detach(), image, ori_image)
            env.reset(ori_image=ori_image, image=image) 

            # forward
            if not flag_a2c:
                v_out_dict = dict()
            for t in range(config.episode_len):
                image_input = Variable(torch.from_numpy(image).cuda())
                reward_input = Variable(torch.from_numpy(reward).cuda())
                pi_out, v_out, p = model(image_input, flag_a2c, add_noise=flag_a2c)
                if flag_a2c:
                    actions = a2c.act_and_train(pi_out, v_out, reward_input)
                else:
                    v_out_dict[t] = - v_out.mean()
                    actions = a2c.act(pi_out, deterministic=True)
 
                p = p.cpu().data.numpy().transpose(1, 0)
                env.set_param(p)
                previous_image = image
                image, reward = env.step(actions)

                if not(episodes % config.display):
                    print('\na2c: ', flag_a2c)
                    print('episode {}: reward@{} = {:.4f}'.format(episodes, t, np.mean(reward)))
                    for k, v in env.parameters.items(): 
                        print(k, ' parameters: ', v.mean())
                    # for image level
                    print("PSNR: {:.5f} -> {:.5f}".format(*computePSNR(ori_image[0], previous_image[0], image[0])))
                    print("SSIM: {:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0], previous_image[0], image[0])))
                    # write down the reward for each T
                    writer.add_scalar('agent2_reward_t_{}'.format(t), float(np.mean(reward)), episodes)

                image = np.clip(image, 0, 1)


            # compute loss and backpropagate
            if flag_a2c:
                losses = a2c.stop_episode_and_compute_loss(reward=Variable(torch.from_numpy(reward).cuda()), done=True)
                loss = sum(losses.values()) #/ config.iter_size
                loss.backward()
            else:
                loss = sum(v_out_dict.values()) * config.c_loss_coeff #/ config.iter_size
                loss.backward()

            if not(episodes % config.display):
                print('\na2c: ', flag_a2c)
                print('episode {}: loss = {}'.format(episodes, float(loss.data)))

            # update model and write into tensorboard
            # if not(episodes % config.iter_size):
            if flag_a2c:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_p.zero_grad()
                for l in losses.keys():
                    writer.add_scalar(l, float(losses[l].cpu().data.numpy()), episodes)
                    loss_log.write('agent2_%s : %.3f ' % (l,float(losses[l].cpu().data.numpy())))
            else:
                optimizer_p.step()
                optimizer_p.zero_grad()
                optimizer.zero_grad()
                for l in v_out_dict.keys():
                    writer.add_scalar('agent2_v_out_{}'.format(l), float(v_out_dict[l].cpu().data.numpy()), episodes)
                    loss_log.write('agent2_v_out_%s : %.3f ' % (l,float(v_out_dict[l].cpu().data.numpy())))
            writer.add_scalar('lr', float(learning_rate), episodes)
            for k, v in env.parameters.items():
                writer.add_scalar(k, float(v.mean()), episodes)
            loss_log.write('\n')
            if not(episodes % config.switch):
                flag_a2c = not flag_a2c
                if episodes < config.warm_up_episodes:
                    flag_a2c = True

            episodes += 1

            # save model
            if not(episodes % config.save_episodes):
                torch.save(model.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:5])) + '_' + str(episodes) + '_agent2.pth'))
                # torch.save(picker.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_agent1_actor.pth'))
                # torch.save(patchCritic.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_agent1_critc.pth'))
                print('model saved')

            # # test model
            if not(episodes % config.test_episodes) or episodes == 1:
                # avg_reward, psnr_res, ssim_res, nmse_res = validation(model, picker, a2c, pAC, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir)
                avg_reward, psnr_res, ssim_res, nmse_res, actions_prob = validation(model,a2c, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir, early_break=config.early_break, cut_edge=config.cut_edge_test)
                writer.add_scalar('test reward', avg_reward, episodes)
                writer.add_scalar('test psnr', psnr_res[1], episodes)
                writer.add_scalar('test ssim', ssim_res[1], episodes)
                writer.add_scalar('test nmse', nmse_res[1], episodes)

                for ii in range(actions_prob.shape[1]): # write down the probability of actions
                    # action_dict = {}
                    action_dict = {"nothing": actions_prob[0, ii]}
                    for jj, item in enumerate(config.actions):
                        action_dict[item] = actions_prob[jj+1, ii]
                        writer.add_scalars("action_%05d"%episodes, action_dict, ii)

                if episodes == 1 :
                    val_log.write('Before - PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (psnr_res[0],ssim_res[0],nmse_res[0]))
                val_log.write('Episode %d - reward: %.3f, PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (episodes, avg_reward,psnr_res[1],ssim_res[1],nmse_res[1]))
                val_log.flush()
                loss_log.flush()

                # save the model with best psnr
                if best_psnr < psnr_res[1]:
                    best_psnr = psnr_res[1]
                    best_psnr_epoch = episodes
                    torch.save(model.module.state_dict(), os.path.join(model_dir, 'best_agent2.pth'))
                    print('Saving the best model with PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (psnr_res[1],ssim_res[1],nmse_res[1]))

            if episodes == config.num_episodes:
                writer.close()
                val_log.close()
                loss_log.close()
                break

if __name__ == "__main__":
    train()
