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

from env import Env, Env_patch, Env_patch_agent2Cond
from model import MyFcn, PatchPicker, PatchCritic, PatchCritic_maskAction, PatchCritic_gt, PatchAgent, PatchAgent_2, PatchAgent_3, PatchAgent_4, PixelPatchAgent, CoordinateClassifyPixelPatchAgent, CoordinateClassifyPixelPatchAgent_sm
from networks import NLayerDiscriminator, GANLoss
from pixel_wise_a2c import PixelWiseA2C, PatchWiseAC, PatchWiseAC_discrete, PatchWiseAC_discrete_coordinate_classify
from test import test, validation, validation_agent1

from utils import adjust_learning_rate, set_requires_grad_via_name
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE, set_requires_grad, crop_and_paste, hard_update
from shutil import copyfile

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MICCAI', type=str,
                        dest='dataset', help='to use dataset.py and config.py in which directory')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--output_name', default='log.txt', type=str, help='log file name')

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
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # copy the config file to log dir
    copyfile(os.path.join(args.dataset,'config.py'), os.path.join(log_dir, 'config.py'))
    writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    # validation log
    val_log = open(os.path.join(log_dir, 'val.txt'), 'a+')
    # loss log
    loss_log = open(os.path.join(log_dir, 'loss.txt'), 'a+')
    # out log
    print_log = open(os.path.join('./outputs', args.output_name), 'w')
    now = time.strftime("%c")
    val_log.write('================ Validation results (%s) ================\n' % now)
    loss_log.write('================ Training loss (%s) ================\n' % now)
    print_log.write('================ Logging (%s) ================\n' % now)
    print('log_dir: '+ log_dir, file=print_log)
    val_log.flush()
    loss_log.flush()
    print_log.flush()
    # if not os.path.exists('model/'):
    #     os.mkdir('model/')

    # dataset
    if config.load_syn_sr: 
        from HistoSR import data_loader_npy
        train_loader = data_loader_npy.get_loader(
            os.path.join(config.root, config.data_degradation, 'train_lmdb'),
            config.syn_sr_train_root,
            batch_size=config.batch_size, 
            stage='train', 
            num_workers=config.workers)
    elif config.use_HistoSR_dataset:
        from HistoSR import data_loader_shuffled_data
        train_loader = data_loader_shuffled_data.get_loader(
            config.histosr_data_train,
            batch_size=config.batch_size, 
            stage='train', 
            num_workers=config.workers)
    else:
        from HistoSR import data_loader_lmdb
        train_loader = data_loader_lmdb.get_loader(
            os.path.join(config.root, config.data_degradation, 'train_lmdb'), 
            batch_size=config.batch_size, 
            stage='train', 
            num_workers=config.workers)
    
    # agent 1
    env_p = Env_patch_agent2Cond(config) if config.use_agent2_in_agent1 else Env_patch(config)
    if config.use_coordinate_classify_agent:
        pAC = PatchWiseAC_discrete_coordinate_classify(config) 
    elif config.use_discrete_action:
        pAC = PatchWiseAC_discrete(config) 
    else: 
        pAC = PatchWiseAC(config) 
    # agent 2
    env = Env(config)
    a2c = PixelWiseA2C(config)
    # tool for cropping and pasting
    tool = crop_and_paste(config)

    # ----- model for the agent2 --------
    episodes = 0
    model = MyFcn(config)
    if len(config.resume_model) > 0: # resume training
        model.load_state_dict(torch.load(config.resume_model))
        resume_episodes = config.resume_model.split('/')[-1]
        print('resume from episodes {}'.format(resume_episodes), file=print_log)
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
    # --- copy agent2 model ---
    if config.use_agent2_in_agent1:
        old_agent2_model = MyFcn(config)
        old_agent2_model = torch.nn.DataParallel(old_agent2_model, device_ids=args.gpu).cuda()
        hard_update(old_agent2_model, model)

    # ----- model for the agent1 --------
    if config.use_coordinate_classify_agent:
        if config.use_sm_agent1:
            model_agent1 = CoordinateClassifyPixelPatchAgent_sm(config)
        else:
            model_agent1 = CoordinateClassifyPixelPatchAgent(config)  
    elif config.use_pixel_oriented_patch:
        model_agent1 = PixelPatchAgent(config)  
    else:
        model_agent1 = PatchAgent_2(config) 
    if len(config.resume_agent1) > 0: # resume training
        agent1_episode = 0
        model_agent1.load_state_dict(torch.load(config.resume_agent1))
        agent1_episode = config.resume_agent1.split('/')[-1]#.split('_')[-1])
        print('resume from episodes {}'.format(agent1_episode), file=print_log)
    model_agent1 = torch.nn.DataParallel(model_agent1, device_ids=args.gpu).cuda()
    # optimizer for actor
    # use SGD
    # optimizer_picker = torch.optim.SGD(params_actor, config.base_lr)
    # use Adam
    optimizer_picker = torch.optim.Adam(model_agent1.module.parameters(), config.base_lr_agent1)

    #----------------- discriminator for termination --------------- #
    if config.use_discrminator:
        model_discri = NLayerDiscriminator(n_layers=config.n_layers_discr, \
                                            image_size=config.patch_height \
                                                if config.discriminator_in_agent2 \
                                                else config.image_height )
        optimizer_discri = torch.optim.Adam(model_discri.parameters(),lr=config.base_lr_discr, betas=(config.beta1_disc, config.beta1_disc))
        GANCriterion = GANLoss().cuda()
        model_discri = torch.nn.DataParallel(model_discri, device_ids=args.gpu).cuda()
        print("discriminator:", model_discri, file=print_log)
    # training
    flag_a2c = True # if True, use a2c; if False, use ddpg
    # best psnr
    best_psnr = 0
    best_psnr_epoch = -1 
    while episodes < config.num_episodes:

        for i, (image_data, ori_image_data) in enumerate(train_loader): # ori_image for target image; image for degraded image
            # log 
            loss_log.write('Episode %d - ' % (episodes))
            # adjust learning rate
            learning_rate = adjust_learning_rate(optimizer, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            _ = adjust_learning_rate(optimizer_p, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            _ = adjust_learning_rate(optimizer_picker, episodes, config.base_lr_agent1, policy=config.lr_policy, policy_parameter=config.policy_parameter)

            ori_image = ori_image_data.numpy()
            image = image_data.numpy()

            env_p.reset(ori_image=ori_image, image=image)
            reward_picker = np.array((0))#np.zeros((1))

            # ---------- Agent 1 ---------- #
            # turn out to set require grad as True
            for t in range(config.episode_len_patch):
                # print("episode_t:",t)
                image_input = Variable(torch.from_numpy(image).cuda())
                reward_picker_input = Variable(torch.from_numpy(reward_picker).cuda())
                # predict the action and value
                actions, v_out = model_agent1(image_input, add_noise=False)
                # sample the action
                # print("tmp_act:",actions, file=print_log)
                actions = pAC.act(v_out, reward_picker_input, actions, isTrain=True, deterministic=True)

                # step
                previous_image = image.copy()
                if config.discriminator_in_old_agent2:
                    image, reward_picker, reward_iou, input_patches, reward_patches = env_p.step(actions,\
                                                                use_iou_reward=config.use_iou_reward, \
                                                                sr_model=old_agent2_model if config.use_agent2_in_agent1 else None,
                                                                use_discriminator=config.use_discrminator, reward_idx=config.reward_idx)
                else:
                    image, reward_picker, reward_iou = env_p.step(actions,\
                                                                use_iou_reward=config.use_iou_reward, \
                                                                sr_model=old_agent2_model if config.use_agent2_in_agent1 else None) # use agent2 in agent1
                # discriminator update in agent1
                if config.use_discrminator and (not config.discriminator_in_agent2):
                    out_discr = model_discri(torch.from_numpy(input_patches).cuda()) \
                                            if config.discriminator_in_old_agent2 \
                                            else model_discri(image_input) 
                    optimizer_discri.zero_grad()
                    loss_discr = GANCriterion(out_discr, torch.mean(torch.from_numpy(reward_patches).cuda(), dim=(1,2,3)).reshape(config.batch_size, 1)) \
                                            if config.discriminator_in_old_agent2 \
                                            else GANCriterion(out_discr, torch.from_numpy(reward_picker).cuda())
                    loss_discr.backward()
                    optimizer_discri.step()
                    writer.add_scalar('Discriminator_{}'.format(t), float(loss_discr.cpu().data.numpy()), episodes)
                    loss_log.write('Dis_%d : %.3f ' % (t, float(loss_discr.cpu().data.numpy())))
                
                # for display
                if not(episodes % config.display):
                    print('\nAgent1: ', file=print_log)
                    print("actions:",actions, file=print_log)
                    print("y:", env_p.start_x,file=print_log)
                    print("x:", env_p.start_y,file=print_log)
                    if config.use_iou_reward:
                        print('episode {}: reward@{} = {:.4f} iou_reward@{} = {:.4f}'.format(episodes, t, np.mean(reward_picker),t, np.mean(reward_iou)), file=print_log)
                        writer.add_scalar('agent1_iou_reward_{}'.format(t), float(np.mean(reward_iou)), episodes)
                    else:
                        print('episode {}: reward@{} = {:.4f}'.format(episodes, t, np.mean(reward_picker)), file=print_log)
                    writer.add_scalar('agent1_reward_{}'.format(t), float(np.mean(reward_picker)), episodes)
                    print("PSNR: {:.5f} -> {:.5f}".format(*computePSNR(ori_image[0], previous_image[0], image[0])), file=print_log)
                    print("SSIM: {:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0], previous_image[0], image[0])), file=print_log)

            # compute loss and backpropagate
            loss_critic_out = pAC.compute_loss(reward=Variable(torch.from_numpy(reward_picker).cuda()))
            if config.use_discrete_action:
                loss_critic = sum(loss_critic_out.values())
            # loss_critic = losses_critic #/ config.iter_size
            loss_critic.backward()
            # update parameters
            # if not((episodes+1) % config.iter_size):
            # -- optimize agent1 -- #
            optimizer_picker.step()
            optimizer_picker.zero_grad()
            if config.use_discrete_action: 
                for l in loss_critic_out.keys():
                    writer.add_scalar('agent1_%s' % (l), float(loss_critic_out[l].cpu().data.numpy()), episodes)
                    loss_log.write('agent1_%s : %.3f ' % (l, float(loss_critic_out[l].cpu().data.numpy())))
            else:
                writer.add_scalar('agent1_mc_loss', float(loss_critic.cpu().data.numpy()), episodes)
                loss_log.write('agent1_mc_loss : %.3f ' % (float(loss_critic.cpu().data.numpy())))

            # ---------- Agent 2 ---------- #
            ori_image = ori_image_data.numpy()
            image = image_data.numpy()
            reward = np.zeros((1))

            # pick a patch via Agent 1
            agent1_actions, _ = model_agent1(torch.from_numpy(image).cuda())
            if config.use_discrete_action:
                if config.use_coordinate_classify_agent:
                    _, actions_row = torch.max(agent1_actions[:,0].data, dim=1)
                    _, actions_col = torch.max(agent1_actions[:,1].data, dim=1)
                    agent1_actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
                else:
                    _, agent1_actions = torch.max(agent1_actions.data, dim=1)

                agent1_actions = agent1_actions.cpu().numpy()
            image, ori_image = tool.crop_patches(agent1_actions, image, ori_image)
            env.reset(ori_image=ori_image, image=image) 

            # forward
            if not flag_a2c:
                v_out_dict = dict()
            for t in range(config.episode_len):
                image_input = Variable(torch.from_numpy(image).cuda())
                reward_input = Variable(torch.from_numpy(reward).cuda())
                pi_out, v_out, p = model(image_input, flag_a2c, add_noise=True)
                if flag_a2c:
                    actions = a2c.act_and_train(pi_out, v_out, reward_input)
                else:
                    v_out_dict[t] = - v_out.mean()
                    actions = a2c.act(pi_out, deterministic=True)
 
                p = p.cpu().data.numpy().transpose(1, 0)
                env.set_param(p)
                previous_image = image
                image, reward = env.step(actions)

                # discriminator update in agent2 when 
                if config.use_discrminator and config.discriminator_in_agent2 and t == 0:
                    out_discr = model_discri(image_input)
                    optimizer_discri.zero_grad()
                    t0_reward = torch.mean(torch.from_numpy(reward).cuda(), dim=(1,2,3)).reshape(config.batch_size, 1)
                    loss_discr = GANCriterion(out_discr, t0_reward)
                    loss_discr.backward()
                    optimizer_discri.step()
                    writer.add_scalar('Discriminator', float(loss_discr.cpu().data.numpy()), episodes)
                    loss_log.write('Dis : %.3f ' % (float(loss_discr.cpu().data.numpy())))

                if not(episodes % config.display):
                    print('\na2c: ', flag_a2c, file=print_log)
                    print("actions:", agent1_actions, file=print_log)
                    print('episode {}: reward@{} = {:.4f}'.format(episodes, t, np.mean(reward)), file=print_log)
                    for k, v in env.parameters.items(): 
                        print(k, ' parameters: ', v.mean(), file=print_log)
                    # for image level
                    print("PSNR: {:.5f} -> {:.5f}".format(*computePSNR(ori_image[0], previous_image[0], image[0])), file=print_log)
                    print("SSIM: {:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0], previous_image[0], image[0])), file=print_log)
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
                print('\na2c: ', flag_a2c, file=print_log)
                print('episode {}: loss_agent1 = {} loss_agent2 = {}'.format(episodes, float(loss_critic.data), float(loss.data)), file=print_log)
                print_log.flush()

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
                if config.valid_old_agent2:
                    torch.save(old_agent2_model.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_old_agent2.pth'))
                torch.save(model.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_agent2.pth'))
                torch.save(model_agent1.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_agent1.pth'))
                if config.use_discrminator:
                    torch.save(model_discri.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_discrim.pth'))

                # torch.save(picker.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_agent1_actor.pth'))
                # torch.save(patchCritic.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_agent1_critc.pth'))
                print('model saved', file=print_log)

            # # test model
            if not(episodes % config.test_episodes): #or episodes == 1:
                model_agent1.eval()
                if config.valid_old_agent2:
                    old_agent2_model.eval()
                    avg_reward, psnr_res, ssim_res, nmse_res, actions_prob = validation(old_agent2_model, model_agent1, a2c, pAC, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir, print_log=print_log, early_break=config.early_break, cut_edge=config.cut_edge_test)
                    old_agent2_model.train()
                else:
                    model.eval()
                    avg_reward, psnr_res, ssim_res, nmse_res, actions_prob = validation(model, model_agent1, a2c, pAC, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir, print_log=print_log, early_break=config.early_break, cut_edge=config.cut_edge_test)
                    model.train()
                model_agent1.train()
                # avg_reward, psnr_res, ssim_res, nmse_res = validation_agent1(model, model_agent1, a2c, pAC, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir, print_log=print_log)
                writer.add_scalar('test reward', avg_reward, episodes)
                writer.add_scalar('test psnr', psnr_res[1], episodes)
                writer.add_scalar('test ssim', ssim_res[1], episodes)
                writer.add_scalar('test nmse', nmse_res[1], episodes)

                for ii in range(actions_prob.shape[1]): # write down the probability of actions
                    action_dict = {"nothing": actions_prob[0, ii]}
                    for jj, item in enumerate(config.actions): # do nothing
                        action_dict[item] = actions_prob[jj + 1, ii]
                        writer.add_scalars("action_%05d"%episodes, action_dict, ii)

                if episodes == 1 :
                    val_log.write('Before - PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (psnr_res[0],ssim_res[0],nmse_res[0]))
                val_log.write('Episode %d - reward: %.3f, PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (episodes, avg_reward,psnr_res[1],ssim_res[1],nmse_res[1]))
                val_log.flush()
                
                # save the model with best psnr
                if best_psnr < psnr_res[1]:
                    best_psnr = psnr_res[1]
                    best_psnr_epoch = episodes
                    if config.valid_old_agent2:
                        torch.save(old_agent2_model.module.state_dict(), os.path.join(model_dir, 'best_old_agent2.pth'))
                    torch.save(model.module.state_dict(), os.path.join(model_dir,  'best_agent2.pth'))
                    torch.save(model_agent1.module.state_dict(), os.path.join(model_dir, 'best_agent1.pth'))
                    if config.use_discrminator:
                        torch.save(model_discri.module.state_dict(), os.path.join(model_dir, 'best_discrim.pth'))

                    print('Saving the best model with PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (psnr_res[1],ssim_res[1],nmse_res[1]), file=print_log)

            if not (episodes % config.display):
                loss_log.flush()
                print_log.flush()
            

            # copy the params from current agent2 model to old agent2 model
            if config.use_agent2_in_agent1 and not(episodes % config.update_old_agent2_model_episode):
                hard_update(old_agent2_model, model)
                print('Episode {} - update the old agent2 model'.format(episodes), file=print_log)
            
            if episodes == config.num_episodes:
                writer.close()
                val_log.close()
                loss_log.close()
                print_log.close()
                break

if __name__ == "__main__":
    train()
