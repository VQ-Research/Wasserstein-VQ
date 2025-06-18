import os
import torch
import warnings
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import InterpolationMode, transforms
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from data.augmentation import random_crop_arr, center_crop_arr

import config
from data.dataloader import build_dataloader_reconstruction
from model.vqgan import VQGAN
from utils.util import Logger, LossManager, Pack
from cleanfid import fid
from pytorch_image_generation_metrics import (
    get_inception_score_from_directory,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory)

def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)

def denormalize_pm1_into_01(x):  # denormalize from [-1, 1] to [0, 1]
    return x.add(1).mul_(0.5)

def get_transform(resolution=256):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform

def load_dataset(data_path, batch_size=16):
    transform = get_transform(resolution=256)
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)
    return dataloader

def save_transformed_image_ffhq(img, img_name, save_dir):
    img = img.squeeze(0).permute(1, 2, 0).cpu()  # Remove batch dim and reorder channels
    img = denormalize_pm1_into_01(img).clamp(0, 1)  # Denormalize and clamp to [0, 1]
    img = Image.fromarray((img.numpy() * 255).astype('uint8'))

    filename = os.path.basename(img_name)
    save_path = os.path.join(save_dir,  filename)
    img.save(save_path)

def save_transformed_image_imagenet(img, img_name, save_dir):
    img = img.squeeze(0).permute(1, 2, 0).cpu() 
    img = denormalize_pm1_into_01(img).clamp(0, 1)  
    img = Image.fromarray((img.numpy() * 255).astype('uint8'))

    filename = os.path.basename(img_name)
    filename = filename.replace(".JPEG", ".png")
    save_path = os.path.join(save_dir,  filename)
    img.save(save_path)

def main_worker(args):
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group(backend='nccl')

    model = VQGAN(args)
    model = model.cuda(int(os.environ['LOCAL_RANK']))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=False, broadcast_buffers=True)
    
    if args.codebook_size == 16384:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-wasserstein_quantizer_200_60_FFHQ_256_model_16384_8_16_loss_1.0_0.1_0.5_0.2-140.pth.tar')
    elif args.codebook_size == 50000:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-wasserstein_quantizer_200_60_FFHQ_256_model_50000_8_16_loss_1.0_0.1_0.5_0.2-160.pth.tar')
    elif args.codebook_size == 100000:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-wasserstein_quantizer_200_60_FFHQ_256_model_100000_8_16_loss_1.0_0.1_0.5_0.2-140.pth.tar')

    loc = 'cuda:{}'.format(int(os.environ['LOCAL_RANK']))
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    args.batch_size = 8
    data_path = os.path.join(args.dataset_dir, 'FFHQ')
    val_dataloader = load_dataset(data_path, args.batch_size)

    base_dir = os.path.join("/mmfs1/data/fangxian/WassersteinVQ/VQGAN2/reconstruction_data/", args.dataset_name)
    base_dir = os.path.join(base_dir, "resolution_"+str(args.resolution))
    os.makedirs(base_dir, exist_ok=True)
    prefix_dir = 'rec_{}_{}'.format(args.factor, args.codebook_size)
    save_dir = os.path.join(base_dir, prefix_dir)
    os.makedirs(save_dir, exist_ok=True)

    raw_base_dir = os.path.join(os.path.join("/mmfs1/data/fangxian/WassersteinVQ/VQGAN2/reconstruction_data/", args.dataset_name), "raw")
    raw_base_dir = os.path.join(raw_base_dir, "resolution_"+str(args.resolution))
    os.makedirs(raw_base_dir, exist_ok=True)

    for idx, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)

        with torch.no_grad():
            if args.quantizer_name == 'wasserstein_quantizer':
                x_rec, _, _, _, _ = model.module.collect_eval_info(x)
            else:
                x_rec, _, _, _ = model.module.collect_eval_info(x)
        
        for i, org_img in enumerate(x):
            image_name = val_dataloader.dataset.samples[idx * val_dataloader.batch_size + i][0]
            if args.dataset_name == "FFHQ":
                save_transformed_image_ffhq(org_img, image_name, raw_base_dir)
        
        for i, rec_img in enumerate(x_rec):
            image_name = val_dataloader.dataset.samples[idx * val_dataloader.batch_size + i][0]
            if args.dataset_name == "FFHQ":
                save_transformed_image_ffhq(rec_img, image_name, save_dir)
        

    FID = fid.compute_fid(save_dir, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
    print("FID: "+str(FID))

if __name__ == '__main__':
    args = config.parse_arg()
    main_worker(args)