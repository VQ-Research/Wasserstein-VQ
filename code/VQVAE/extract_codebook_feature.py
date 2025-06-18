import os
import torch
import warnings
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from scipy.io import savemat

import config
from data.dataloader import build_dataloader
from model.vqvae import VQVAE
from utils.util import Logger, LossManager, Pack

def main_worker(args):
    model = VQVAE(args)
    model = model.cuda()
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    args.batch_size = 8
    train_dataloader, val_dataloader, len_train_set, len_val_set = build_dataloader(args)

    z_cat: List[torch.Tensor] = []
    embed = model.quantizer.embedding.weight

    count = 0
    with torch.no_grad():
        for idx, (x, _) in enumerate(val_dataloader):
            x = x.cuda()
            z = model.obtain_feature(x)
            
            if count<10:
                z_cat.append(z)
            else:
                break
            count += 1
    
    z_cat = torch.cat(z_cat, 0).contiguous()
   
    saver_name = str(args.quantizer_name)+'_feature_embed.mat'
    saver_path = os.path.join("/mmfs1/data/fangxian/WassersteinVQ/VQVAE/extract_data/", saver_name)  
    savemat(saver_path, {'z_cat':z_cat.cpu().data.numpy(), 'embed':embed.cpu().data.numpy()})

if __name__ == '__main__':
    args = config.parse_arg()
    main_worker(args)  

