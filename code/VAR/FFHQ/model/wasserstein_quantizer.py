import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
import os
from model.base_quantizer import BaseQuantizer

class Queue(nn.Module):
    def __init__(self, args):
        super(Queue, self).__init__()
        self.args = args
        self.queue_size = 174080
        self.register_buffer("queue", torch.randn(self.queue_size, args.codebook_dim)*0.01)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, key):
        batch_size = key.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        self.queue[ptr:ptr + batch_size, :] = key
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  

    @torch.no_grad()
    def obtain_feature_from_queue(self):
        return self.queue.detach().clone()

class Wasserstein_Quantizer(BaseQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.beta = args.beta
        self.alpha = args.alpha
        self.gamma = args.gamma ##wasserstein loss coefficient
        self.queue = Queue(args)

    def calc_wasserstein_loss(self, z=None):
        if z==None:
            z = self.queue.obtain_feature_from_queue()

        N = z.size(0)
        D = z.size(1)

        c = self.embedding.weight
        std = c.std(dim=0).max().detach()

        # Normalize z and c
        z = z / (std + 1e-8)
        c = c / (std + 1e-8)

        z_mean = z.mean(0).detach()
        z_covariance = torch.cov(z.t()) + 1e-6 * torch.eye(D, device=z.device)  ##z_covariance = torch.mm((z - torch.mean(z, dim=0, keepdim=True)).t(), z - torch.mean(z, dim=0, keepdim=True))/(N-1)
        z_covariance = z_covariance.detach()

        ### compute the mean and covariance of codebook vectors
        c_mean = c.mean(0)
        c_covariance = torch.cov(c.t()) + 1e-6 * torch.eye(D, device=z.device)  ##c_covariance = torch.mm((c - torch.mean(c, dim=0, keepdim=True)).t(), c - torch.mean(c, dim=0, keepdim=True))/(self.codebook_size-1)
        
        ### calculation of part1
        part_mean =  torch.sum(torch.multiply(z_mean - c_mean, z_mean - c_mean))

        ### 1/2 covariance
        S1, Q1 = torch.linalg.eigh(z_covariance)
        sqrt_S1 = torch.diag(torch.sqrt(F.relu(S1)+ 1e-8))
        temp = torch.mm(Q1, sqrt_S1)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = torch.mm(temp, Q1.T)
        z_sqrt_covariance = torch.nan_to_num(z_sqrt_covariance, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = z_sqrt_covariance.detach()
        
        ### 1/2 covariance
        temp = torch.mm(z_sqrt_covariance, c_covariance)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

        covariance = torch.mm(temp, z_sqrt_covariance)
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

        S2, Q2 = torch.linalg.eigh(covariance)
        sqrt_S2 = torch.sqrt(F.relu(S2)+ 1e-8)

        #############calculation of part2
        part_covariance = F.relu(torch.trace(z_covariance.detach() + c_covariance) - 2.0 * sqrt_S2.sum())
        wasserstein_loss = torch.sqrt(part_mean + part_covariance + 1e-10)
        return wasserstein_loss

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        z_rest = z_enc
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            vq_loss: torch.Tensor = 0.0
            commit_loss: torch.Tensor = 0.0
            wasserstein_loss: torch.Tensor = 0.0
            
            levels = len(self.args.ms_token_size)
            for level, pn in enumerate(self.args.ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale.detach())
                
                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
                
                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                embed = self.embedding(token)
                
                ## the multi-scale vector quantization loss
                commit_loss += (F.mse_loss(embed.detach(), z_downscale).mul_(self.beta) + F.mse_loss(embed, z_downscale.detach())) * self.args.ms_token_size[level] 

                token_cat.append(token)                  
                token_Bhw = token.view(B, pn, pn)

                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                #z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec = z_dec + z_upscale
                z_rest = z_rest - z_upscale
            
            ## residual quantization loss
            vq_loss =  F.mse_loss(z_dec.data, z_enc).mul_(self.beta) + F.mse_loss(z_dec, z_enc.data)

            commit_loss *= 1. / sum(self.args.ms_token_size)
            vq_loss = vq_loss + commit_loss 

            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)
            with torch.no_grad():
                self.queue.dequeue_and_enqueue(z_cat.detach())

            ## Criterion Triple defined in the paper
            z_dec = (z_dec - z_enc).detach().add_(z_enc)
            quant_error = (z_dec.detach()-z_enc.detach()).square().sum(1).mean()

            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
                
            codebook_usage_counts = (histogram > 0).float().sum()
            codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            
            avg_probs = histogram/histogram.sum(0)
            codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            ### compute wasserstein distance
            wasserstein_loss = self.calc_wasserstein_loss()
            loss = vq_loss + self.gamma * wasserstein_loss

        return z_dec, loss, wasserstein_loss, quant_error, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            levels = len(self.args.ms_token_size)
            for level, pn in enumerate(self.args.ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_cat.append(token)

                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                #z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec.add_(z_upscale)
                z_rest.sub_(z_upscale)

            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)
            wasserstein_loss = self.calc_wasserstein_loss(z_cat.detach())

            quant_error = (z_dec.detach()-z_enc.detach()).square().sum(1).mean()
            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
        return z_dec, wasserstein_loss, quant_error, histogram