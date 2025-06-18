import gc
import os
import shutil
import sys
import time
import warnings
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn, optim
import math
import json
import random
import scipy.io as sio
from torch.nn import functional as F
from scipy.io import savemat
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from data.dataloader import build_dataloader
import torchvision.models as torchvision_models
from torchvision import models, datasets, transforms
from torch import distributed as tdist 
import itertools

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate
from data import dataloader
from model.vqvae import VQVAE
from metric.metric import PSNR, SSIM

import warnings
warnings.filterwarnings('ignore')

def eval_one_epoch(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    ssim, psnr, rec_loss_scalar, quantization_error, utilization, perplexity, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0  
    if args.quantizer_name == 'wasserstein_quantizer':
        wasserstein_loss_scalar = 0.0 
              
    histogram_all: torch.Tensor = 0.0
    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda()
        batch_size = x.size(0)
        total_num += batch_size

        with torch.no_grad():
            if args.quantizer_name == 'wasserstein_quantizer':
                x_rec, rec_loss, wasserstein_loss, quant_error, histogram = model.collect_eval_info(x)
            else:
                x_rec, rec_loss, quant_error, histogram = model.collect_eval_info(x)
    
            histogram_all += histogram
            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0

            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        ssim += batch_ssim.item()
        psnr += batch_psnr.item()
        rec_loss_scalar += rec_loss.item() * batch_size
        quantization_error += quant_error.item() * batch_size
        if args.quantizer_name == 'wasserstein_quantizer':
            wasserstein_loss_scalar += wasserstein_loss.item() * batch_size
    
    codebook_usage_counts = (histogram_all > 0).float().sum()
    utilization  = codebook_usage_counts.item() / args.codebook_size

    avg_probs = histogram_all/histogram_all.sum(0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_utilization = utilization
    eval_perplexity = perplexity.item()
    eval_rec_loss = rec_loss_scalar/total_num
    eval_quantization_error = quantization_error/total_num

    if args.quantizer_name == 'wasserstein_quantizer':
        eval_wasserstein_loss = wasserstein_loss_scalar/total_num

    model.train()
    if args.quantizer_name == 'wasserstein_quantizer':
        return Pack(psnr=eval_psnr, ssim=eval_ssim, quant_error=eval_quantization_error, utilization=eval_utilization, perplexity=eval_perplexity, rec_loss=eval_rec_loss, wasserstein_loss=eval_wasserstein_loss)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, quant_error=eval_quantization_error, utilization=eval_utilization, perplexity=eval_perplexity, rec_loss=eval_rec_loss)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main_worker(args):
    model = VQVAE(args).cuda()
    train_dataloader, val_dataloader, len_train_set, len_val_set = build_dataloader(args)

    if args.quantizer_name == 'ema_quantizer':
        model_para = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.pre_quant_proj.parameters()) + list(model.post_quant_proj.parameters())
    elif args.quantizer_name == 'vanilla_quantizer' or args.quantizer_name == 'online_quantizer':
        model_para = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.pre_quant_proj.parameters()) + list(model.post_quant_proj.parameters()) + list(model.quantizer.embedding.parameters())
    elif args.quantizer_name == 'wasserstein_quantizer':
        model_para = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.pre_quant_proj.parameters()) + list(model.post_quant_proj.parameters())
        code_para = list(model.quantizer.embedding.parameters())

    if args.quantizer_name == 'wasserstein_quantizer':
        optimizer = torch.optim.Adam([{'params': model_para}, {'params': code_para, 'lr': 0.02}], lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.Adam(model_para, lr=args.lr, betas=(0.9, 0.95))

    if args.quantizer_name == 'wasserstein_quantizer':
        results = {'total_loss':[], 'vq_loss':[], 'rec_loss': [], 'wasserstein_loss':[], 'quant_error':[], 'utilization':[], 'perplexity':[]}
        results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'rec_loss': [], 'wasserstein_loss':[], 'quant_error':[], 'utilization':[], 'perplexity':[]}
    else:
        results = {'total_loss':[], 'vq_loss':[], 'rec_loss': [], 'quant_error':[], 'utilization':[], 'perplexity':[]}
        results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'rec_loss': [], 'quant_error':[], 'utilization':[], 'perplexity':[]}

    train_loss = LossManager()
    print("Start training...")
    start_epoch = 1 

    for epoch in range(start_epoch, args.epochs+1):
        print("epoch:%d, cur_lr:%4f"%(epoch, optimizer.param_groups[0]["lr"]))

        loss_scalar, vq_loss, rec_loss, quant_error, utilization, perplexity, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        if args.quantizer_name == 'wasserstein_quantizer':
            wasserstein_loss = 0

        model.train()
        start_time = time.time()
        for step, (x, _) in enumerate(train_dataloader):
            cur_iter = len(train_dataloader) * (epoch-1) + step
            if args.dataset_name == "ImageNet":
                lr = adjust_learning_rate(optimizer, cur_iter, args)
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                x = x.cuda()
                batch_size = x.size(0)
                loss, loss_pack = model(x)
        
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                if args.quantizer_name == 'wasserstein_quantizer':
                    torch.nn.utils.clip_grad_norm_(code_para, 1.0)
                optimizer.step()
                
            train_loss.add_loss(loss_pack)
            total_num += batch_size
            loss_scalar += loss_pack.loss.item() * batch_size
            rec_loss += loss_pack.rec_loss.item() * batch_size
            vq_loss += loss_pack.vq_loss.item() * batch_size
            quant_error += loss_pack.quant_error.item() * batch_size
            perplexity += loss_pack.codebook_perplexity.item() * batch_size
            utilization += loss_pack.codebook_utilization * batch_size

            if args.quantizer_name == 'wasserstein_quantizer':
                wasserstein_loss += loss_pack.wasserstein_loss.item() * batch_size

            if (step+1) %10 ==0:
                print(train_loss.pprint(window=50, prefix='Train Epoch: [{}/{}] Iters:[{}/{}]'.format(epoch, args.epochs, step+1, len(train_dataloader))))

        train_loss.clear()
        ######################### start conducting statistical analysis per epoch on training dataset ##########
        print("######### start conducting statistical analysis per epoch on training dataset #########")
        results['total_loss'].append(loss_scalar/total_num)
        results['rec_loss'].append(rec_loss/total_num)
        results['vq_loss'].append(vq_loss/total_num)
        results['quant_error'].append(quant_error/total_num)
        results['utilization'].append(utilization/total_num)
        results['perplexity'].append(perplexity/total_num)
        if args.quantizer_name == 'wasserstein_quantizer':
            results['wasserstein_loss'].append(wasserstein_loss/total_num)

        #save statistics
        results_len = len(results['total_loss'])
        data_frame = pd.DataFrame(data=results, index=range(1, results_len + 1))
        data_frame.to_csv('{}/train_{}_statistics.csv'.format(args.results_dir, args.saver_name_pre), index_label='epoch')

        if epoch % args.eval_epochs == 0:
            with torch.no_grad():
                results_pack = eval_one_epoch(args, model, epoch, val_dataloader, len_val_set)

            results_eval['epoch'].append(epoch)
            results_eval['psnr'].append(results_pack.psnr)
            results_eval['ssim'].append(results_pack.ssim)
            results_eval['rec_loss'].append(results_pack.rec_loss)
            results_eval['quant_error'].append(results_pack.quant_error)
            results_eval['utilization'].append(results_pack.utilization)
            results_eval['perplexity'].append(results_pack.perplexity)
            if args.quantizer_name == 'wasserstein_quantizer':
                results_eval['wasserstein_loss'].append(results_pack.wasserstein_loss)
                
            results_val_len = len(results_eval['epoch'])
            data_frame = pd.DataFrame(data=results_eval, index=range(1, results_val_len+1))
            data_frame.to_csv('{}/eval_{}_rec_results.csv'.format(args.results_dir, args.saver_name_pre), index_label='index')

    print("######### saving checkpoint #########")
    model.train()
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'.pth.tar')
    save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path)
    

if __name__ == '__main__':
    args = config.parse_arg()
    dict_args = vars(args)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    for k, v in zip(dict_args.keys(), dict_args.values()):
        print("{0}: {1}".format(k, v))

    main_worker(args)  