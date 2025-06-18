import json
import os
import random
import re
import subprocess
import sys
import time
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Union
import argparse
import torch.distributed as dist

def parse_arg():
    parser = argparse.ArgumentParser(description='Wasserstein Quantizer Under VAR Experimental Settings.') 

    ###Dataset and Dataloader Configuration
    parser.add_argument('--dataset_dir', default="/projects/yuanai/data/", type=str, help='the directory of dataset') 
    parser.add_argument('--dataset_name', default='FFHQ', help='the name of dataset', choices=['ImageNet', 'FFHQ'])
    parser.add_argument('--global_batch_size', type=int, default=128, help="the size of batch samples")
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--resolution', type=int, choices=[256], default=256, help='resolution of train and test')
    parser.add_argument('--channels', default=3, type=int, metavar='N', help='the channels of images')
    
    ###Model Configuration
    parser.add_argument('--ms_patch_size', default="1_2_3_4_5_6_8_10_13_16", type=str, help='multi-scale patch size.')
    parser.add_argument('--max_patch_size', default=16, type=int, help='the maximum patch size.')
    parser.add_argument('--codebook_size', default=16384, type=int, help='the size of codebook.')
    parser.add_argument('--codebook_dim', default=8, type=int, help='the dimension of codebook vectors.')
    parser.add_argument('--z_channels', default=256, type=int, help='the resolution of latent variables.')
    parser.add_argument('--factor', default=16, type=int, help='the downscale factor of vanilla image to the latent variable', choices=[16])
   
    ###Loss Configuration
    parser.add_argument('--alpha', type=float, default=1.0, help=" the hyperparameter of commit_loss.")
    parser.add_argument('--beta', type=float, default=1.0, help=" the hyperparameter of commit_loss.")
    parser.add_argument('--gamma', type=float, default=0.5, help="the hyperparameter of wasserstein_loss.")
    parser.add_argument('--rate_d', type=float, default=0.2, help="discriminator loss weight for gan training")

    ###Training Configuration
    parser.add_argument('--model_name', default='VAR', help='the name of models.', choices=['VAR'])
    parser.add_argument('--resume', action='store_true', help='reloading model from specified checkpoint.')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs, 10 epochs for ImageNet, 30 epochs for FFHQ datasets.")
    parser.add_argument('--eval_epochs', type=int, default=1, help="epochs for each eval, 1 epochs for ImageNet, 5 epochs for FFHQ datasets.")
    parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate for encoder-decoder architecture.')
    parser.add_argument('--dropout', help='dropout for the model', type=float, default=0.0)
    parser.add_argument('--seed', help='random seed', type=int, default=3407)
    parser.add_argument('--warmup_iters', help='Number of steps for warmup of lr', type=int, default=5000)
    parser.add_argument('--decay_iters', help='Number of steps for cosine decay of lr', type=int, default=40000)
    parser.add_argument('--weight_decay', help='weight decay for optimizer', type=float, default=0.05)
    parser.add_argument("--disc_epoch", type=int, default=4, help="iteration to start discriminator training and loss")
    parser.add_argument('--accumulation', help='Number of batches to accumulate before backward step.', type=int, default=1)

    parser.add_argument('--checkpoint_dir', default="/projects/yuanai/projects/WassersteinVQ/VAR/checkpoint/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--results_dir', default="/projects/yuanai/projects/WassersteinVQ/VAR/results/", type=str, help='the directory of results.')
    parser.add_argument('--saver_dir', default="/projects/yuanai/projects/WassersteinVQ/VAR/saver/", type=str, help='the directory of saver.')
    parser.add_argument('--nnodes', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training.')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend.')
    args = parser.parse_args()

    if args.dataset_name == "FFHQ":
        args.epochs = 200
        args.disc_epoch = 60
        args.eval_epochs = 10
        args.rate_d = 0.2
        args.warmup_iters = 20000
        args.decay_iters = 100000
        
    elif args.dataset_name == "ImageNet":
        args.epochs = 18
        args.disc_epoch = 5
        args.eval_epochs = 1
        args.rate_d = 0.2
        args.warmup_iters = 10000
        args.decay_iters = 80000

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.batch_size = round(args.global_batch_size/args.world_size)
    args.workers = min(max(0, args.workers), args.batch_size)
    args.ms_token_size = tuple(map(int, args.ms_patch_size.replace('-', '_').split('_')))
    
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
    args.results_dir = os.path.join(args.results_dir, args.dataset_name)
    args.saver_dir = os.path.join(args.saver_dir, args.dataset_name)
        
    args.data_pre = '{}_{}'.format(args.dataset_name, args.resolution)
    args.model_pre = 'model_{}_{}_{}'.format(args.codebook_size, args.codebook_dim, args.factor)
    args.loss_pre = 'loss_{}_{}_{}_{}'.format(args.alpha, args.beta, args.gamma, args.rate_d)
    args.training_pre = '{}_{}_{}'.format(args.model_name, args.epochs, args.disc_epoch)
    args.saver_name_pre = args.training_pre + '_' + args.data_pre + '_' + args.model_pre + '_' + args.loss_pre
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return args