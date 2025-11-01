import os
import torch
import warnings
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
import config
from cleanfid import fid

input_dir = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/FFHQ"
mmd_transplant_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Transplant/FFHQ/mmd_vq_transplant_16384_False"
mmd_transplant_32768 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Transplant/FFHQ/mmd_vq_transplant_32768_False"
wasserstein_transplant_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Transplant/FFHQ/wasserstein_vq_transplant_16384_False"
wasserstein_transplant_32768 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Transplant/FFHQ/wasserstein_vq_transplant_32768_False"

mmd_refinement_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/FFHQ/mmd_vq_refinement_16384_False"
mmd_refinement_32768 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/FFHQ/mmd_vq_refinement_32768_False"
wasserstein_refinement_16384 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False"
wasserstein_refinement_32768 = "/projects/yuanai/projects/VQ-Transplant3/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False"


print("#################transplant-stage###########################")
print(mmd_transplant_16384)
FID = fid.compute_fid(mmd_transplant_16384, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(mmd_transplant_32768)
FID = fid.compute_fid(mmd_transplant_32768, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_transplant_16384)
FID = fid.compute_fid(wasserstein_transplant_16384, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_transplant_32768)
FID = fid.compute_fid(wasserstein_transplant_32768, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print("#################Refinement-stage###########################")
print(mmd_refinement_16384)
FID = fid.compute_fid(mmd_refinement_16384, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(mmd_refinement_32768)
FID = fid.compute_fid(mmd_refinement_32768, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_refinement_16384)
FID = fid.compute_fid(wasserstein_refinement_16384, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_refinement_32768)
FID = fid.compute_fid(wasserstein_refinement_32768, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

