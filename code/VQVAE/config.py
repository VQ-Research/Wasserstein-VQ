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

def parse_arg():
    parser = argparse.ArgumentParser(description='Various Quantizer Under VQVAE Experimental Settings.') 

    ###Dataset and Dataloader Configuration
    ## BC: /mmfs1/data/fangxian/data;
    parser.add_argument('--dataset_dir', default="/mmfs1/data/fangxian/data/", type=str, help='the directory of dataset') 
    parser.add_argument('--dataset_name', default='ImageNet', help='the name of dataset', choices=['ImageNet', 'FFHQ'])
    parser.add_argument('--batch_size', type=int, default=32, help="the size of batch samples")
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--resolution', default=256, type=int, metavar='N', help='resolution of train and test')
    parser.add_argument('--channels', default=3, type=int, metavar='N', help='the channels of images')
    
    ###Model Configuration
    parser.add_argument('--codebook_size', default=8192, type=int, help='the size of codebook.')
    parser.add_argument('--codebook_dim', default=8, type=int, help='the dimension of codebook vectors.')
    parser.add_argument('--z_channels', default=256, type=int, help='the resolution of latent variables.')
    parser.add_argument('--factor', default=16, type=int, help='the downscale factor of vanilla image to the latent variable', choices=[16, 8, 4])
   
    ###Loss Configuration
    parser.add_argument('--beta', type=float, default=1.0, help=" the hyperparameter of vq_loss.")
    parser.add_argument('--gamma', type=float, default=0.0, help="the hyperparameter of wasserstein_loss.")
    parser.add_argument('--alpha', type=float, default=1.0, help=" the hyperparameter of commit_loss.")

    ###Training Configuration
    parser.add_argument('--quantizer_name', default='wasserstein_quantizer', help='the name of models.', choices=['wasserstein_quantizer', 'vanilla_quantizer', 'ema_quantizer', 'online_quantizer'])
    parser.add_argument('--resume', action='store_true', help='reloading model from specified checkpoint.')
    parser.add_argument('--epochs', type=int, default=20, help="training epochs, 10 epochs for ImageNet, 50 epochs for FFHQ datasets.")
    parser.add_argument('--eval_epochs', type=int, default=1, help="epochs for each eval, 1 epochs for ImageNet, 5 epochs for FFHQ datasets.")
    parser.add_argument('--lr', default=5e-5, type=float, metavar='LR', help='initial learning rate for encoder-decoder architecture.')
    parser.add_argument('--dropout', help='dropout for the model', type=float, default=0.0)
    parser.add_argument('--seed', help='random seed', type=int, default=123)
    parser.add_argument('--warmup_iters', help='Number of steps for warmup of lr', type=int, default=5000)
    parser.add_argument('--decay_iters', help='Number of steps for cosine decay of lr', type=int, default=50000)
    parser.add_argument('--weight_decay', help='weight decay for optimizer', type=float, default=0.05)
    parser.add_argument('--accumulation', help='Number of batches to accumulate before backward step.', type=int, default=1)

    ##BC: /mmfs1/data/fangxian/WassersteinVQ/VQVAE/;
    parser.add_argument('--checkpoint_dir', default="/mmfs1/data/fangxian/WassersteinVQ/VQVAE/checkpoint/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--results_dir', default="/mmfs1/data/fangxian/WassersteinVQ/VQVAE/results/", type=str, help='the directory of results.')
    parser.add_argument('--saver_dir', default="/mmfs1/data/fangxian/WassersteinVQ/VQVAE/saver/", type=str, help='the directory of saver.')
    args = parser.parse_args()

    args.workers = min(max(0, args.workers), args.batch_size)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset_name)
    args.results_dir = os.path.join(args.results_dir, args.dataset_name)
    args.saver_dir = os.path.join(args.saver_dir, args.dataset_name)

    if args.dataset_name == "FFHQ":
        args.epochs = 30
        args.eval_epochs = 5
    elif args.dataset_name == "ImageNet":
        args.epochs = 4
        args.eval_epochs = 1
        
    args.data_pre = '{}'.format(args.dataset_name)
    args.model_pre = 'model_{}_{}_{}'.format(args.codebook_size, args.codebook_dim, args.factor)
    args.loss_pre = 'loss_{}_{}_{}'.format(args.alpha, args.beta, args.gamma)
    args.training_pre = '{}_{}'.format(args.quantizer_name, args.epochs)
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