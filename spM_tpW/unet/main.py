"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import cv2
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

import sys
sys.path.append('..')
from utils import PSNR, SSIM, NMSE, DC, computePSNR, computeSSIM, computeNMSE

from unet_model import UnetModel
from args import Args
sys.path.append('../fastMRI/')
from subsample import MaskFunc
from dataset import MRIDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(args):
    from config import config
    train_data = MRIDataset(image_set='train', transform=False, config=config)
    dev_data = MRIDataset(image_set='test', transform=False, config=config)
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = []#[dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        #batch_size=args.batch_size,
        batch_size=10,
        num_workers=1,
        pin_memory=False,
    )
    #display_loader = DataLoader(
    #    dataset=display_data,
    #    batch_size=16,
    #    num_workers=8,
    #    pin_memory=True,
    #)
    return train_loader, dev_loader, None#display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    iter = 0
    while True:
        for _, data in enumerate(data_loader):
            iter += 1
            target, input, _ = data
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            output = model(input)
            loss = F.l1_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss = 0.99 * avg_loss + 0.01 * float(loss.data) if iter > 0 else loss.item()
            writer.add_scalar('TrainLoss', float(loss.data), global_step + iter)

            if iter % args.report_interval == 0:
                logging.info(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss = {float(loss.data):.4g} Avg Loss = {avg_loss:.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
            start_iter = time.perf_counter()
            if args.iters_per_epoch and iter == args.iters_per_epoch:
                break

        return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    early_break = True


    PSNR_dict = defaultdict(list)
    SSIM_dict = defaultdict(list)
    NMSE_dict = defaultdict(list)
    count = 0

    for i, (target, input, mask) in enumerate(data_loader):
        ori_image = target.numpy()
        previous_image = input.numpy()
        mask = mask.numpy()
        count += 1
        print(count)
        if early_break and count == 101:
            break
        if count % 100 == 0:
            print('tested: ', count)

        input = Variable(input, volatile=True).cuda()
        target = Variable(target).cuda()
        output = model(input)#.squeeze(1)

        loss = F.mse_loss(output, target, size_average=False)
        losses.append(float(loss.data))

        image = output.cpu().data.numpy()
        for ii in range(image.shape[0]):
            m = min(float(np.min(ori_image[ii, 0])), 0)
            def rescale(x):
                return (x - m) / (6 - m)
            ori_image[ii, 0] = rescale(ori_image[ii, 0])
            previous_image[ii, 0] = rescale(previous_image[ii, 0])
            image[ii, 0] = rescale(image[ii, 0])
            image_with_DC = DC(ori_image[ii, 0], image[ii, 0], mask[ii])

            for k in range(2):
                key = ['wo', 'DC'][k]
                tmp_image = [image[ii, 0], image_with_DC][k]
                PSNR_dict[key].append(computePSNR(ori_image[ii, 0], previous_image[ii, 0], tmp_image))
                SSIM_dict[key].append(computeSSIM(ori_image[ii, 0], previous_image[ii, 0], tmp_image))
                NMSE_dict[key].append(computeNMSE(ori_image[ii, 0], previous_image[ii, 0], tmp_image))

            cv2.imwrite('unet_results/'+str(i)+'_'+str(ii)+'.bmp', np.concatenate((ori_image[ii, 0], previous_image[ii, 0], image[ii, 0], np.abs(ori_image[ii, 0] - image[ii, 0]) * 10), axis=1) * 255)
    writer.add_scalar('Dev_Loss', np.mean(losses), epoch)


    for key in PSNR_dict.keys():
        PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
        print('number of test images: ', len(PSNR_list))
        psnr_res = np.mean(np.array(PSNR_list), axis=0)
        ssim_res = np.mean(np.array(SSIM_list), axis=0)
        nmse_res = np.mean(np.array(NMSE_list), axis=0)

        print('PSNR', psnr_res)
        print('SSIM', ssim_res)
        print('NMSE', nmse_res)

    return np.mean(losses), time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).cuda()#to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(args.exp_dir / 'summary')

    if args.test:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        start_epoch = checkpoint['epoch']
        del checkpoint
    elif args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        if args.test:
            print('evaluating')
            dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
            exit()

        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        if (epoch + 1) % 5 == 0:
            #dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)

            is_new_best = True #dev_loss < best_dev_loss
            best_dev_loss = 0 #min(best_dev_loss, dev_loss)
            save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
            logging.info(
                    'saved',
                    #f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                    #f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--iters-per-epoch', type=int, default=0, help='Number of iterations per epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=40, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--test', action='store_true', default=False)
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
