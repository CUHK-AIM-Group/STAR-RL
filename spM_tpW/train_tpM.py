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
from networks import NLayerDiscriminator, GANLoss, MLP, MLP_512
from pixel_wise_a2c import PixelWiseA2C, PatchWiseAC, PatchWiseAC_discrete, PatchWiseAC_discrete_coordinate_classify
# from test import test, validation, validation_agent1
from test_tpM import test, validation, validation_agent1

from utils import adjust_learning_rate, set_requires_grad_via_name, get_scheduler, update_learning_rate
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
    from config_tpM import config
    assert config.switch % config.iter_size == 0
    time_tuple = time.localtime(time.time())
    log_dir = './logs/' + '_'.join(map(lambda x: str(x), time_tuple[1:5]))
    model_dir = os.path.join(log_dir, 'models')
    results_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # copy the config file to log dir
    copyfile(os.path.join(args.dataset,'config_tpM.py'), os.path.join(log_dir, 'config.py'))
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
    if config.use_HistoSR_dataset:
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
    '''parameters_wo_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' not in key]
    optimizer = torch.optim.Adam(parameters_wo_p, config.base_lr)

    parameters_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' in key]'''
    #parameters_pi = [value for key, value in dict(model.module.named_parameters()).items() if '_pi.' in key]
    '''params = [
        {'params': parameters_p, 'lr': config.base_lr},
    ]
    optimizer_p = torch.optim.SGD(params, config.base_lr)'''
    

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

    model_agent1.eval()
    model.eval()
    # optimizer for actor
    # use SGD
    # optimizer_picker = torch.optim.SGD(params_actor, config.base_lr)
    # use Adam
    #optimizer_picker = torch.optim.Adam(model_agent1.module.parameters(), config.base_lr_agent1)

    #----------------- discriminator for termination --------------- #
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
            

    optimizer_discri = torch.optim.Adam(model_discri.parameters(),lr=config.base_lr_discr, betas=(config.beta1_disc, config.beta1_disc))
    GANCriterion = GANLoss().cuda()
    model_discri = torch.nn.DataParallel(model_discri, device_ids=args.gpu).cuda()
    schedulers_discr = get_scheduler(optimizer_discri, n_epochs=config.num_episodes/2, n_epochs_decay=config.num_episodes/2, lr_policy=config.lr_policy_disc) 
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
            

            ori_image_whole = ori_image_data.numpy()
            image_whole = image_data.numpy()

            image_patches = []
            image_patches_reward = []
            agent1_actions_list = []
            # ---------- Agent 1 ---------- #
            # turn out to set require grad as True
            for tt in range(config.episode_len_patch):

                # ---------- Agent 2 ---------- #
                # pick a patch via Agent 1
                agent1_actions, v_out = model_agent1(torch.from_numpy(image_whole).cuda())
                agent1_actions = agent1_actions.detach()
                v_out = v_out.detach()
                if config.use_action_input:
                    bs, ch, h = agent1_actions.shape
                    if config.action_input_type == 'map' :
                        row = agent1_actions[:,0,:].reshape(bs,h,1).repeat(1,1,h)
                        col = agent1_actions[:,1,:].reshape(bs,1,h).repeat(1,h,1)
                        actions_map = row*col
                    else:
                        actions_map = agent1_actions.clone().reshape(bs, ch*h)
                    agent1_actions_list.append(actions_map)
                # if config.use_discrete_action:
                #     if config.use_coordinate_classify_agent:
                #         _, actions_row = torch.max(agent1_actions[:,0].data, dim=1)
                #         _, actions_col = torch.max(agent1_actions[:,1].data, dim=1)
                #         agent1_actions = torch.cat((actions_row.unsqueeze(1), actions_col.unsqueeze(1)), dim=1)
                #     else:
                #         _, agent1_actions = torch.max(agent1_actions.data, dim=1)
                agent1_actions = pAC.act(v_out, 0, agent1_actions, isTrain=True, deterministic=True)
                
                # agent1_actions = agent1_actions.cpu().numpy()
                # print actions
                # print("t:",tt,file=print_log)
                # print("actions:", agent1_actions, file=print_log)

                image, ori_image, start_x, start_y  = tool.crop_patches(agent1_actions, image_whole, ori_image_whole, rtn_pos=True)
                env.reset(ori_image=ori_image, image=image) 
                reward = np.zeros((1))
                if config.use_img_patches:
                    image_patches.append(image.copy())
                previous_image_whole = image_whole.copy()

                # forward
                for t in range(config.episode_len):
                    image_input = Variable(torch.from_numpy(image).cuda())
                    reward_input = Variable(torch.from_numpy(reward).cuda())
                    pi_out, v_out, p = model(image_input, flag_a2c, add_noise=True)
                    pi_out = pi_out.detach()
                    v_out = v_out.detach() 
                    p = p.detach()
                    actions = a2c.act(pi_out, deterministic=True, store_reward=True, reward=reward_input)
    
                    p = p.cpu().data.numpy().transpose(1, 0)
                    env.set_param(p)
                    previous_image = image
                    image, reward = env.step(actions)

                    '''if not(episodes % config.display):
                        # print('\na2c: ', flag_a2c, file=print_log)
                        print('episode {}: reward@{} = {:.4f}'.format(episodes, t, np.mean(reward)), file=print_log)
                        for k, v in env.parameters.items(): 
                            print(k, ' parameters: ', v.mean(), file=print_log)
                        # for image level
                        print("PSNR: {:.5f} -> {:.5f}".format(*computePSNR(ori_image[0], previous_image[0], image[0])), file=print_log)
                        print("SSIM: {:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0], previous_image[0], image[0])), file=print_log)
                        # write down the reward for each T
                        writer.add_scalar('agent2_reward_t_{}'.format(t), float(np.mean(reward)), episodes)'''

                    image = np.clip(image, 0, 1)

                image_whole = tool.paste(image, image_whole, start_x, start_y)
                image_whole = np.clip(image_whole, 0, 1)

                agent1_each_reward = np.sum(np.abs(ori_image_whole - previous_image_whole) * 255  - np.abs(ori_image_whole - image_whole) * 255, axis=(1,2,3)) / (config.input_ch*config.image_height*config.image_width)
                image_patches_reward.append(agent1_each_reward) # bs x 1 -> concat -> bsxT1 x 1 

                a2c.reset()

                # get final reward
                # final_reward = a2c.get_reward(reward=Variable(torch.from_numpy(reward).cuda()), done=True)
                # final_reward = torch.mean(final_reward, dim=(1,2,3)).reshape(config.batch_size, 1)

                # for k, v in env.parameters.items():
                #     writer.add_scalar(k, float(v.mean()), episodes)
            del agent1_actions
            del v_out
            del ori_image_whole
            del previous_image_whole
            del image_whole
            del image, ori_image
            del pi_out, p 
            # train discriminator
            if config.use_img_patches:
                image_patches = torch.from_numpy(np.concatenate(image_patches, axis=0))
            if config.use_action_input:
                actions_input = torch.cat(agent1_actions_list, dim=0)
                if config.action_input_type == 'map':
                    actions_input = actions_input.unsqueeze(1)
            image_patches_reward = torch.from_numpy(np.concatenate(image_patches_reward,axis=0)).cuda().reshape(config.batch_size*config.episode_len_patch, 1)
            if config.use_img_patches:
                out_discr = model_discri(Variable(image_patches).cuda())
            else: 
                out_discr = model_discri(actions_input)
            
            optimizer_discri.zero_grad()
            loss_discr, label = GANCriterion(out_discr, image_patches_reward,rtn_gt=True)
            loss_discr.backward()
            optimizer_discri.step()
            
            total = label.shape[0]
            nonzero_label = int(torch.sum(label))
            zero_label = int(total - nonzero_label)
            nonzero_ratio = round(nonzero_label*1.0/total, 3)
            zero_ratio = 1 - nonzero_ratio
            if not(episodes % config.display):
                print("episodes=", episodes, file=print_log)
                print("1 : 0 - {} : {} ".format(nonzero_label, zero_label), file=print_log)
                print('nonzero ratio:', nonzero_ratio, file=print_log)
                print_log.flush()


            writer.add_scalar('Discriminator', float(loss_discr.cpu().data.numpy()), episodes)
            loss_log.write('Dis : %.3f ' % (float(loss_discr.cpu().data.numpy())))

            loss_log.write('\n')
            if not(episodes % config.display):
                out_sig = torch.sigmoid(out_discr).cpu()
                print('final reward = %.3f' % (float(np.mean(image_patches_reward.cpu().numpy()))),file=print_log)
                str_tmp_out = ""
                str_tmp_rwd = ""
                for idx in range(config.episode_len_patch): # 
                    idx_bs = idx * config.batch_size
                    str_tmp_out += "outsig_%d = %.3f " % (idx, float(out_sig[idx_bs]))
                    str_tmp_rwd += "reward_%d = %.3f " % (idx, float(image_patches_reward[idx_bs].cpu().data))
                print(str_tmp_out, file=print_log)
                print(str_tmp_rwd, file=print_log)
                # print('episode {}: final reward = {}'.format(episodes, float(np.mean(image_patches_reward.cpu().numpy()))), file=print_log)
                loss_log.flush()
                print_log.flush()

            del image_patches
            del image_patches_reward

            episodes += 1

            # update lr 
            lr, lr_res = update_learning_rate(schedulers_discr, optimizer_discri)
            writer.add_scalar('lr', float(lr), episodes)


            # save model
            if not(episodes % config.save_episodes):
                torch.save(model_discri.module.state_dict(), os.path.join(model_dir, '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '_discrim.pth'))
                print('model saved', file=print_log)

            # # test model
            """ if not(episodes % config.test_episodes) or episodes == 1:
                # avg_reward, psnr_res, ssim_res, nmse_res, actions_prob = validation(model, model_agent1, a2c, pAC, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir, print_log=print_log)
                model_discri.eval()
                avg_reward, psnr_res, ssim_res, nmse_res = test(model, model_agent1, model_discri,a2c, pAC, config, batch_size=10, results_dir=results_dir, use_discriminator=True, use_entropy=True, print_log=print_log)
                model_discri.train()
                # model.train()
                # model_agent1.train()
                # avg_reward, psnr_res, ssim_res, nmse_res = validation_agent1(model, model_agent1, a2c, pAC, config, batch_size=10, valid_mode=True, current_episode=episodes, save_dir=results_dir, print_log=print_log)
                writer.add_scalar('test reward', avg_reward, episodes)
                writer.add_scalar('test psnr', psnr_res[1], episodes)
                writer.add_scalar('test ssim', ssim_res[1], episodes)
                writer.add_scalar('test nmse', nmse_res[1], episodes)

                # for ii in range(actions_prob.shape[1]): # write down the probability of actions
                #     action_dict = {"nothing": actions_prob[0, ii]}
                #     for jj, item in enumerate(config.actions): # do nothing
                #         action_dict[item] = actions_prob[jj + 1, ii]
                #         writer.add_scalars("action_%05d"%episodes, action_dict, ii)

                if episodes == 1 :
                    val_log.write('Before - PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (psnr_res[0],ssim_res[0],nmse_res[0]))
                val_log.write('Episode %d - reward: %.3f, PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (episodes, avg_reward,psnr_res[1],ssim_res[1],nmse_res[1]))
                val_log.flush()
                
                # save the model with best psnr
                if best_psnr < psnr_res[1]:
                    best_psnr = psnr_res[1]
                    best_psnr_epoch = episodes
                    if config.use_discrminator:
                        torch.save(model_discri.module.state_dict(), os.path.join(model_dir, 'best_discrim.pth'))

                    print('Saving the best model with PSNR: %.3f, SSIM:%.3f NMSE:%.3f\n' % (psnr_res[1],ssim_res[1],nmse_res[1]), file=print_log) """

            if not (episodes % config.display_learning_rate):
                print(lr_res, file=print_log)
                loss_log.flush()
                print_log.flush()
            

            if episodes == config.num_episodes:
                writer.close()
                val_log.close()
                loss_log.close()
                print_log.close()
                break

if __name__ == "__main__":
    train()
