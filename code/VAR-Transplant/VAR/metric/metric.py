import pyiqa
import torch
import torch.nn as nn
import piq
from pytorch_image_generation_metrics import get_inception_score_from_directory
from pytorch_image_generation_metrics import get_fid_from_directory
from pytorch_image_generation_metrics import get_inception_score_and_fid_from_directory

## PSNR and LPIPS are computed by pyiqa (https://github.com/chaofengc/IQA-PyTorch). pip install pyiqa
###### data range (0, 1) 
class PSNR():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)

###### data range (0, 1)   
class SSIM():
    def __call__(self, real, fake):
        return piq.ssim(real, fake, data_range=1., reduction='none') 

###### data range (-1, 1) 
class LPIPS():
    def __init__(self, device=None):
        self.iqa_metric = pyiqa.create_metric('lpips', device=device)
    
    def __call__(self, real, fake):
        return self.iqa_metric(real, fake)

