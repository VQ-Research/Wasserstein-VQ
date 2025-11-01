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
from torch import distributed as dist
import itertools
from copy import deepcopy
from torch.nn.parallel import DistributedDataParallel as DDP
import ruamel.yaml as yaml

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate, save_checkpoint
from data import dataloader
from models.pq_model import PQModel
from models.vq_model import VQModel
from models.vq_loss import VQLoss
from metric.metric import PSNR, LPIPS, SSIM
from eval_tokenizer import eval_one_epoch_vq, eval_one_epoch_pq

from timm.scheduler import create_scheduler_v2 as create_scheduler
from utils.distributed import init_distributed_mode
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.optim import param_groups_weight_decay
from eval_reconstruction import eval_reconstruction, eval_reconstruction_epoch

os.environ["TORCHDYNAMO_LOGLEVEL"] = "INFO"
os.environ["TORCHDYNAMO_VERBOSE"] = "1" 

import warnings
warnings.filterwarnings('ignore')

def main_worker(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "ema_vq" or args.VQ == "online_vq" or args.VQ == "mmd_vq":
        if args.pq == 1:
            vq_model = VQModel(args)
        else:
            vq_model = PQModel(args)
    else:
        pass 

    total_para = 0
    for p in vq_model.encoder.parameters():
        total_para += p.numel()
    for p in vq_model.decoder.parameters():
        total_para += p.numel()

    print("VQ Model Parameters:", total_para)
    vq_model = vq_model.to(device)
    vq_model = nn.SyncBatchNorm.convert_sync_batchnorm(vq_model)
    vq_loss = VQLoss(args).to(device)

    model_para = list(vq_model.decoder.parameters()) + list(vq_model.projector_out.parameters()) + list(vq_model.post_quant_conv.parameters())
    disc_para = vq_loss.discriminator.parameters()
    optimizer = torch.optim.AdamW(model_para, lr=args.lr_refinement, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    optimizer_disc = torch.optim.AdamW(disc_para, lr=args.lr_refinement, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    
    train_dataloader, val_dataloader, train_sampler, len_train_set, len_val_set = build_dataloader(args)
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=False)
    vq_model.train()
    vq_model.module.encoder.eval()
    vq_model.module.quant_conv.eval()
    vq_model.module.quantizer1.eval()
    vq_model.module.quantizer2.eval()
    vq_model.module.projector_in.eval()

    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()
    vq_loss.module.perceptual_loss.eval()

    results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'lpips':[], 'rec_loss': []}
    train_loss = LossManager()
    print("Start training...")
    start_epoch = 1 
    total_steps = len(train_dataloader) * args.refinement_epochs
    for epoch in range(start_epoch, args.refinement_epochs+1):
        train_sampler.set_epoch(epoch)
        print("epoch:%d, cur_lr:%4f"%(epoch, optimizer.param_groups[0]["lr"]))
        start_time = time.time()
        for step, (x, _) in enumerate(train_dataloader):
            cur_iter = len(train_dataloader) * (epoch-1) + step
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                x = x.to(device, non_blocking=True)
                optimizer.zero_grad()
                x_rec = vq_model.module.refinement(x)
                gen_loss, gen_loss_pack = vq_loss(x, x_rec, optimizer_idx=0, cur_epoch=epoch, last_layer=vq_model.module.decoder.last_layer)
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                optimizer.step()

                optimizer_disc.zero_grad()
                d_loss, loss_pack = vq_loss(x, x_rec, optimizer_idx=1, cur_epoch=epoch)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc_para, 1.0)
                optimizer_disc.step()
                
                torch.cuda.synchronize()
                gen_loss_pack.add(loss_pack)

            train_loss.add_loss(gen_loss_pack)
            if int(os.environ['LOCAL_RANK']) == 0 and (step+1) %10 ==0:
                print(train_loss.pprint(window=50, prefix='Train Epoch: [{}/{}] Iters:[{}/{}]'.format(epoch, args.refinement_epochs, step+1, len(train_dataloader))))

        train_loss.clear()
        if epoch % args.eval_epochs == 0 and int(os.environ['LOCAL_RANK']) == 0:
            vq_model.train()
            checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'-'+str(epoch)+'.pth.tar')
            save_checkpoint({'epoch': epoch, 'model': vq_model.module.state_dict(), 'optimizer': optimizer.state_dict(), "discriminator": vq_loss.module.discriminator.state_dict(), 'optimizer_disc': optimizer_disc.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path) 
            if args.dataset_name == "ImageNet":
                eval_reconstruction_epoch(args, vq_model, epoch)
        torch.distributed.barrier()

        if epoch % args.eval_epochs == 0:
            with torch.no_grad():
                if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "ema_vq" or args.VQ == "online_vq" or args.VQ == "mmd_vq":
                    if args.pq == 1:
                        results_pack = eval_one_epoch_vq(args, vq_model, epoch, val_dataloader, len_val_set)
                    else:
                        results_pack = eval_one_epoch_pq(args, vq_model, epoch, val_dataloader, len_val_set)
                else:
                    pass 

            if int(os.environ['LOCAL_RANK']) == 0:
                results_eval['epoch'].append(epoch)
                results_eval['psnr'].append(results_pack.psnr)
                results_eval['ssim'].append(results_pack.ssim)
                results_eval['lpips'].append(results_pack.lpips)
                results_eval['rec_loss'].append(results_pack.rec_loss)
                
                results_val_len = len(results_eval['epoch'])
                data_frame = pd.DataFrame(data=results_eval, index=range(1, results_val_len+1))
                data_frame.to_csv('{}/eval_{}_rec_results.csv'.format(args.results_dir, args.saver_name_pre), index_label='index')

    print("######### saving checkpoint #########")
    vq_model.train()
    if int(os.environ['LOCAL_RANK']) == 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'.pth.tar')
        save_checkpoint({'epoch': epoch, 'model': vq_model.module.state_dict(), 'optimizer': optimizer.state_dict(), "discriminator": vq_loss.module.discriminator.state_dict(), 'optimizer_disc': optimizer_disc.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path) 
        if args.dataset_name != "ImageNet":
            eval_reconstruction(args, vq_model)
    vq_model.eval() 
    dist.destroy_process_group()

if __name__ == '__main__':
    os.environ['NCCL_TIMEOUT_IN_MS'] = '7200000'
    args = config.parse_arg()
    dict_args = vars(args)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    if int(os.environ['LOCAL_RANK']) == 0:
        for k, v in zip(dict_args.keys(), dict_args.values()):
            print("{0}: {1}".format(k, v))
    main_worker(args)