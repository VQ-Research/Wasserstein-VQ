import os
import sys  
import torch
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.augmentation import random_crop_arr, center_crop_arr

paths = {
    "ImageNet": "ImageNet",
    "FFHQ": "FFHQ",
}

def build_train_transform(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform

def build_eval_transform(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform

def build_dataloader(args):
    data_path = os.path.join(args.dataset_dir, paths[args.dataset_name])
    train_transform = build_train_transform(args)
    eval_transform = build_eval_transform(args)

    if args.dataset_name == "ImageNet":
        train_set = ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
        val_set = ImageFolder(root=os.path.join(data_path, 'val'), transform=eval_transform)
    elif args.dataset_name == "FFHQ":
        train_set = ImageFolder(root=data_path, transform=train_transform)
        val_set = ImageFolder(root=data_path, transform=eval_transform)

    print("dataset name:", args.dataset_name)
    print("len train_set:", len(train_set))
    print("len val_set:", len(val_set))

    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_dataloader = DataLoader(
        dataset=train_set, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_set, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size*2, shuffle=False, drop_last=False
    )
    print("len train_dataloader:", len(train_dataloader))
    print("len val_dataloader:", len(val_dataloader))
    return train_dataloader, val_dataloader, len(train_set), len(val_set)