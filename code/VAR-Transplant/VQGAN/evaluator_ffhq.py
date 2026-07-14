import os
import torch
import warnings
import random
import numpy as np
import PIL.Image as PImage
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image, ImageOps, ImageFilter
from cleanfid import fid

wasserstein_16384_5 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_5"
wasserstein_16384_10 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_10"
wasserstein_16384_15 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_15"
wasserstein_16384_20 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_20"
wasserstein_16384_25 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_25"
wasserstein_16384_30 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_30"
wasserstein_16384_35 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_35"
wasserstein_16384_40 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_40"
wasserstein_16384_45 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_45"
wasserstein_16384_50 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_16384_False_50_50"

wasserstein_32768_5 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_5"
wasserstein_32768_10 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_10"
wasserstein_32768_15 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_15"
wasserstein_32768_20 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_20"
wasserstein_32768_25 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_25"
wasserstein_32768_30 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_30"
wasserstein_32768_35 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_35"
wasserstein_32768_40 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_40"
wasserstein_32768_45 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_45"
wasserstein_32768_50 = "/project/6105494/sunset/VQ-Projects/WassersteinVQ/VQGAN/reconstruction/Refinement/FFHQ/wasserstein_vq_refinement_32768_False_50_50"

print("#################Refinement-stage###########################")
print(wasserstein_16384_5)
FID = fid.compute_fid(wasserstein_16384_5, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_10)
FID = fid.compute_fid(wasserstein_16384_10, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_15)
FID = fid.compute_fid(wasserstein_16384_15, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_20)
FID = fid.compute_fid(wasserstein_16384_20, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_25)
FID = fid.compute_fid(wasserstein_16384_25, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_30)
FID = fid.compute_fid(wasserstein_16384_30, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_35)
FID = fid.compute_fid(wasserstein_16384_35, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_40)
FID = fid.compute_fid(wasserstein_16384_40, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_45)
FID = fid.compute_fid(wasserstein_16384_45, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_16384_50)
FID = fid.compute_fid(wasserstein_16384_50, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_5)
FID = fid.compute_fid(wasserstein_32768_5, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_10)
FID = fid.compute_fid(wasserstein_32768_10, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_15)
FID = fid.compute_fid(wasserstein_32768_15, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_20)
FID = fid.compute_fid(wasserstein_32768_20, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_25)
FID = fid.compute_fid(wasserstein_32768_25, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_30)
FID = fid.compute_fid(wasserstein_32768_30, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_35)
FID = fid.compute_fid(wasserstein_32768_35, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_40)
FID = fid.compute_fid(wasserstein_32768_40, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_45)
FID = fid.compute_fid(wasserstein_32768_45, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))

print(wasserstein_32768_50)
FID = fid.compute_fid(wasserstein_32768_50, dataset_name="ffhq", dataset_res=256,  mode="clean", dataset_split="trainval70k")
print("FID: "+str(FID))



