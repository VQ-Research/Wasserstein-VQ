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
from torchvision.transforms import InterpolationMode, transforms
from data.augmentation import random_crop_arr, center_crop_arr
from data.lsun_church import LSUNChurchesDataset
from data.lsun_bedroom import LSUNBedroomsDataset

paths = {
    "ImageNet": "imagenet",
    "FFHQ": "FFHQ",
    "CelebAHQ":"CelebAHQ",
    "Bedrooms":"LSUN-Bedrooms",
    "Churches": "LSUN-Churches",
}

def build_train_transform(args):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.resolution)),
        transforms.RandomHorizontalFlip(),
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
    elif args.dataset_name == "CelebAHQ":
        train_set = ImageFolder(root=data_path, transform=train_transform)
        val_set = ImageFolder(root=data_path, transform=eval_transform)
    elif args.dataset_name == "Bedrooms":
        train_set = LSUNBedroomsDataset(root=data_path, split='train', transform=train_transform)
        val_set = LSUNBedroomsDataset(root=data_path, split='train', transform=eval_transform)
    elif args.dataset_name == "Churches":
        train_set = LSUNChurchesDataset(root=data_path, split='train', transform=train_transform)
        val_set = LSUNChurchesDataset(root=data_path, split='train', transform=eval_transform)

    print("dataset name:", args.dataset_name)
    print("len train_set:", len(train_set))
    print("len val_set:", len(val_set))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_dataloader = DataLoader(
        dataset=train_set, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=False,
        sampler=train_sampler, drop_last=True
    )

    eval_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    val_dataloader = DataLoader(
        dataset=val_set, num_workers=args.workers, pin_memory=True,
        batch_size=args.batch_size, shuffle=False,
        sampler=eval_sampler, drop_last=False
    )
    return train_dataloader, val_dataloader, train_sampler, len(train_set), len(val_set)