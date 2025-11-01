import os
import torch
import random
import numpy as np
import torch, torchvision
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import itertools
from data.augmentation import center_crop_arr
from data.lsun_church import LSUNChurchesDataset
from data.lsun_bedroom import LSUNBedroomsDataset
from metric.metric import PSNR, LPIPS, SSIM

paths = {
    "ImageNet": "imagenet",
    "FFHQ": "FFHQ",
    "CelebAHQ":"CelebAHQ",
    "Bedrooms":"LSUN-Bedrooms",
    "Churches": "LSUN-Churches",
}

def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def load_dataset(args, batch_size=16):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    data_path = os.path.join(args.dataset_dir, paths[args.dataset_name])
    
    if args.dataset_name == "ImageNet":
        val_set = ImageFolder(root=os.path.join(data_path, 'val'), transform=transform)
    elif args.dataset_name == "FFHQ":
        val_set = ImageFolder(root=data_path, transform=transform)
    elif args.dataset_name == "CelebAHQ":
        val_set = ImageFolder(root=data_path, transform=transform)
    elif args.dataset_name == "Bedrooms":
        val_set = LSUNBedroomsDataset(root=data_path, split='train', transform=transform)
    elif args.dataset_name == "Churches":
        val_set = LSUNChurchesDataset(root=data_path, split='train', transform=transform)

    len_val_set = len(val_set)
    dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6, drop_last=False)
    return dataloader, len_val_set

def eval_reconstruction(args, model):
    val_dataloader, len_val_set = load_dataset(args, batch_size=16)
    if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "mmd_vq" or args.VQ == "online_vq" or args.VQ == "ema_vq": 
        if args.pq == 1:
            reconstruction_name = '{}_{}_{}_{}'.format(args.VQ, args.stage, args.codebook_size, args.use_multiscale)
        else:
            reconstruction_name = '{}_{}_{}'.format(args.VQ, args.stage, args.pq)
    elif args.VQ == 'bsq' or args.VQ == 'fsq' or args.VQ ==  'lfq':
        reconstruction_name = '{}_{}_{}_{}'.format(args.VQ, args.stage, args.project_dim, args.L)

    reconstruction_path = os.path.join(args.reconstruction_dir, reconstruction_name)
    os.makedirs(reconstruction_path, exist_ok=True)

    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    ssim, psnr, lpips = 0.0, 0.0, 0.0
    total = 0
    for idx, (x, _) in enumerate(val_dataloader):
        x = x.cuda()
        with torch.no_grad():
            x_rec = model.module.reconstruction(x)
            samples = torch.clamp(127.5 * x_rec + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            input_samples = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i + total
                Image.fromarray(sample).save(f"{reconstruction_path}/{index:06d}.png")

            total += 16
            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0

            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    print("PSNR:"+str(eval_psnr)+"  SSIM:"+str(eval_ssim)+ "  LPIPS:"+str(eval_lpips))

    if args.dataset_name == "ImageNet":
        create_npz_from_sample_folder(reconstruction_path)

def eval_reconstruction_epoch(args, model, epoch):
    val_dataloader, len_val_set = load_dataset(args, batch_size=16)
    if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "mmd_vq" or args.VQ == "online_vq" or args.VQ == "ema_vq": 
        if args.pq == 1:
            reconstruction_name = '{}_{}_{}_{}_{}'.format(args.VQ, args.stage, args.codebook_size, args.use_multiscale, epoch)
        else:
            reconstruction_name = '{}_{}_{}_{}'.format(args.VQ, args.stage, args.pq, epoch)
    elif args.VQ == 'bsq' or args.VQ == 'fsq' or args.VQ ==  'lfq':
        reconstruction_name = '{}_{}_{}_{}_{}'.format(args.VQ, args.stage, args.project_dim, args.L, epoch)
    
    reconstruction_path = os.path.join(args.reconstruction_dir, reconstruction_name)
    os.makedirs(reconstruction_path, exist_ok=True)

    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    ssim, psnr, lpips = 0.0, 0.0, 0.0
    total = 0
    for idx, (x, _) in enumerate(val_dataloader):
        x = x.cuda()
        with torch.no_grad():
            x_rec = model.module.reconstruction(x)
            
            samples = torch.clamp(127.5 * x_rec + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            input_samples = torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            
            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i + total
                Image.fromarray(sample).save(f"{reconstruction_path}/{index:06d}.png")

            total += 16
            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0

            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    print("PSNR:"+str(eval_psnr)+"  SSIM:"+str(eval_ssim)+ "  LPIPS:"+str(eval_lpips))

    if args.dataset_name == "ImageNet":
        create_npz_from_sample_folder(reconstruction_path)