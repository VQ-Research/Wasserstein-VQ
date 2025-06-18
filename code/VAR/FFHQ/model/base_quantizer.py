from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as tdist 
import numpy as np


### Adopted code from the https://github.com/FoundationVision/VAR/blob/main/models/quant.py
class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

class BaseQuantizer(nn.Module):
    def __init__(self, args):
        super(BaseQuantizer, self).__init__()
        self.args = args
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
        self.embedding.weight.data.normal_(0, 0.01)
        self.embedding.weight.requires_grad = True
        #self.phi = PhiPartiallyShared(nn.ModuleList([(Phi(self.codebook_dim, 0.5)) for _ in range(4)]))

    ## continous feature (from encoder) into multi-scale image token
    ## r1, r2, r3, ..., rK
    def obtain_multiscale_image_token(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone() 

        ## output, multis_cale_image_token
        ret : List[torch.Tensor] = [] 
        levels = len(self.args.ms_token_size)
        patch_hws =  [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in self.args.ms_token_size] 
        for level, (ph, pw) in enumerate(patch_hws):
            z_downscale = F.interpolate(z_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != len(patch_hws) -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)

            ## distance [B*ph*pw, vocab_size]
            distance = torch.sum(z_downscale.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            distance.addmm_(z_downscale, self.embedding.weight.data.T, alpha=-2, beta=1)

            ## token [B*ph*pw]
            token = torch.argmin(distance, dim=1)
            token_Bhw = token.view(B, ph, pw)

            z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != len(patch_hws) -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
            z_upscale = self.phi[level/(levels-1)](z_upscale)

            z_rest.sub_(z_upscale)
            ret.append(token.reshape(B, ph*pw))
        return ret

    ## continous feature (from encoder) quantized feature (to decoder)
    ## \hat{z1}, \hat{z1}+\hat{z2},..., \hat{z1}+...+\hat{zK}
    def obtain_multiscale_quantized_feature(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        ## output, multis_cale_quantized_feature
        ret : List[torch.Tensor] = []
        levels = len(self.args.ms_token_size)
        patch_hws =  [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in self.args.ms_token_size] 
        for level, (ph, pw) in enumerate(patch_hws):
            z_downscale =  F.interpolate(z_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != len(patch_hws) -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
            ## distance [B*ph*pw, vocab_size]
            distance = torch.sum(z_downscale.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            distance.addmm_(z_downscale, self.embedding.weight.data.T, alpha=-2, beta=1)

            ## token [B*ph*pw]
            token = torch.argmin(distance, dim=1)

            token_Bhw = token.view(B, ph, pw)
            z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != len(patch_hws) -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
            z_upscale = self.phi[level/(levels-1)](z_upscale)

            z_dec.add_(z_upscale)
            z_rest.sub_(z_upscale)
            ret.append(z_dec.clone())
        return ret

    ## r1, r2, r3, ..., rK to \hat{z1}, \hat{z1}+\hat{z2},..., \hat{z1}+...+\hat{zK} (z_dec)
    def multiscale_token_to_multiscale_quantized_feature(self, multiscale_token):
        B = multiscale_token[0].shape[0]
        H = W = self.args.ms_token_size[-1] ## H = W = 16
        C = self.args.codebook_dim

        ## multis_cale_quantized_feature
        ret : List[torch.Tensor] = []
        levels = len(self.args.ms_token_size)
        z_dec = multiscale_token[0].new_zeros(B, C, H, W, dtype=torch.float32)
        for level, pn in enumerate(self.args.ms_token_size): # from small to large
            token = multiscale_token[level].view(B, pn, pn)
            z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != self.args.ms_token_size -1) else self.embedding(token).permute(0, 3, 1, 2).contiguous()
            z_upscale = self.phi[level/(levels-1)](z_upscale)

            z_dec.add_(z_upscale)
            ret.append(z_dec.clone())
        return ret

    ### For training GPT models
    def obtain_contextualized_embedding(self, multiscale_token):
        next_scales = []
        B = multiscale_token[0].shape[0]
        C = self.args.codebook_dim
        H = W = self.args.ms_token_size[-1]
        num_level = len(self.args.ms_token_size)

        token_embedding = multiscale_token[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.args.ms_token_size[0]
        for level in range(num_level-1):
            level_embedding = F.interpolate(self.embedding(multiscale_token[level]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
            level_embedding = self.phi[level/(levels-1)](level_embedding)

            token_embedding.add_(level_embedding)
            pn_next = self.args.ms_token_size[level+1]
            next_scales.append(F.interpolate(token_embedding, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) 

    ### for VAR inference (generation phase)
    def obtain_next_autoregressive_input(self, level, f_hat, predicted_token):
        H = W = self.args.ms_token_size[-1]
        pn = self.args.ms_token_size[level]
        levels = len(self.args.ms_token_size)
        if level != len(self.args.ms_token_size)-1:
            h = F.interpolate(self.embedding(predicted_token).transpose_(1, 2).view(B, C, pn, pn), size=(H, W), mode='bicubic')
            h = self.phi[level/(levels-1)](h)

            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.args.ms_token_size[level+1], self.args.ms_token_size[level+1]), mode='area')
        else:
            h = self.embedding(predicted_token).transpose_(1, 2).view(B, C, pn, pn)
            f_hat.add_(h)
            return f_hat, f_hat