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
from copy import deepcopy

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate
from data import dataloader
from metric.metric import PSNR, LPIPS, SSIM
import warnings
warnings.filterwarnings('ignore')

## for var quantizer
def eval_one_epoch_var(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    if args.stage == "transplant":
        ssim, psnr, lpips, rec_loss, quant_error, utilization, perplexity, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        histogram_all: torch.Tensor = 0.0
    if args.stage == "refinement":
        ssim, psnr, lpips, rec_loss, total_num = 0.0, 0.0, 0.0, 0.0, 0  

    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            if args.stage == "transplant":
                x_rec, rec_loss_eval, quant_error_eval, histogram_eval = model.module.collect_eval_info_transplant(x)
                info_pack = Pack(rec_loss=rec_loss_eval, quant_error=quant_error_eval)
                histogram_all += histogram_eval
            else:
                x_rec, rec_loss_eval = model.module.collect_eval_info_refinement(x)
                info_pack = Pack(rec_loss=rec_loss_eval)

            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0
            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        handler1 = tdist.all_reduce(batch_lpips, async_op=True)
        handler2 = tdist.all_reduce(batch_psnr, async_op=True)
        handler3 = tdist.all_reduce(batch_ssim, async_op=True)
        handler1.wait()
        handler2.wait()
        handler3.wait()

        if int(os.environ['LOCAL_RANK']) == 0:
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            rec_loss += info_pack.rec_loss.item() * batch_size
            if args.stage == "transplant":
                quant_error += info_pack.quant_error.item() * batch_size
                
    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    if args.stage == "transplant":
        eval_quant_error = quant_error/total_num
        codebook_usage_counts = (histogram_all > 0).float().sum()
        eval_utilization  = codebook_usage_counts.item() / args.codebook_size

        avg_probs = histogram_all/histogram_all.sum(0)
        eval_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

    model.train()
    if args.stage == "transplant":
        model.module.encoder.eval()
        model.module.decoder.eval()
        model.module.quant_conv.eval()
        model.module.post_quant_conv.eval()
        model.module.perceptual_loss.eval()
    else:
        model.module.encoder.eval()
        model.module.quantizer.eval()
        model.module.quant_conv.eval()

    if args.stage == "transplant":
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, quant_error=eval_quant_error, utilization=eval_utilization, perplexity=eval_perplexity)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss)