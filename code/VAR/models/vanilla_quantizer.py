import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from models.base_quantizer import MultiscaleVectorQuantizer
    
##### multi-scale quantizer
class VanillaVARQuantizer(MultiscaleVectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            multi_vq_loss: torch.Tensor = 0.0
            levels = len(self.args.ms_token_size)
            ms_token_size =  self.args.ms_token_size

            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                
                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
                
                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                embed = self.embedding(token)

                token_cat.append(token)                  
                token_Bhw = token.view(B, pn, pn)

                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec = z_dec + z_upscale
                z_rest = z_rest - z_upscale

                multi_vq_loss += self.alpha * F.mse_loss(z_dec, z_no_grad) * self.args.importance[level]
                
            multi_vq_loss *= 1. / sum(self.args.importance)
            token_cat = torch.cat(token_cat, 0)
            z_dec = z_enc + (z_dec-z_enc).detach()
            
            ## Criterion Triple defined in the paper
            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
                
            codebook_usage_counts = (histogram > 0).float().sum()
            codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            
            avg_probs = histogram/histogram.sum(0)
            codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            loss = multi_vq_loss 
        return z_dec, loss, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            levels = len(self.args.ms_token_size)
            ms_token_size =  self.args.ms_token_size
            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_cat.append(token)

                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec.add_(z_upscale)
                z_rest.sub_(z_upscale)

            token_cat = torch.cat(token_cat, 0)
            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()

        return z_dec, histogram

    def collect_reconstruction(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        with torch.cuda.amp.autocast(enabled=False):
            levels = len(self.args.ms_token_size)
            ms_token_size =  self.args.ms_token_size
            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec.add_(z_upscale)
                z_rest.sub_(z_upscale)

        return z_dec
