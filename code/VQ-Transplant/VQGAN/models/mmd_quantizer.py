import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from models.base_quantizer import VectorQuantizer, ProductQuantizer

#### not the multi-scale quantizer and no residual quantization
class MMDVectorQuantizer(VectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.sqrt_d = math.sqrt(self.codebook_dim)

    def calc_gaussian_mmd_loss(self, z):
        z = z.detach()
        c = self.embedding.weight
        N = z.size(0) + c.size(0)

        dxx = (torch.sum(z.detach()**2, dim=1, keepdim=True) + torch.sum(z.detach()**2, dim=1) - 2*torch.matmul(z.detach(), z.detach().t())).div(self.sqrt_d)
        dxy = (torch.sum(z.detach()**2, dim=1, keepdim=True) + torch.sum(c**2, dim=1) - 2*torch.matmul(z.detach(), c.t())).div(self.sqrt_d)
        dyy = (torch.sum(c**2, dim=1, keepdim=True) + torch.sum(c**2, dim=1) - 2*torch.matmul(c, c.t())).div(self.sqrt_d)
        bandwidth = (dxx.sum() + 2*dxy.sum() + dyy.sum()).detach() / (N**2 -N)

        pxx = -dxx / bandwidth
        pxy = -dxy / bandwidth
        pyy = -dyy / bandwidth

        XX = torch.exp(pxx).mean() + torch.exp(pxx/2).mean()
        XY = torch.exp(pxy).mean() + torch.exp(pxy/2).mean()
        YY = torch.exp(pyy).mean() + torch.exp(pyy/2).mean()

        mmd_loss = XX.detach() - 2 * XY + YY
        return mmd_loss

    def forward(self, z_enc):
        # reshape z_enc -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  
        mmd_loss = self.calc_gaussian_mmd_loss(z_flat.detach())
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flat.detach().pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flat.detach(), self.embedding.weight.data) # 'n d -> d n'

        token = torch.argmin(d, dim=1)
        z_dec = self.embedding(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        commit_loss = self.beta * F.mse_loss(z_dec.detach(), z_enc) + self.alpha * F.mse_loss(z_dec, z_enc.detach())

        histogram = token.bincount(minlength=self.args.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        utilization = codebook_usage_counts.item() / self.args.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_dec = z_enc + (z_dec - z_enc).detach()
        loss = commit_loss + self.args.gamma * mmd_loss
        return z_dec, loss, utilization, perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_dec = self.embedding(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        histogram = token.bincount(minlength=self.args.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return z_dec, histogram

    def collect_reconstruction(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_dec = self.embedding(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        return z_dec

class MMDProductQuantizer(ProductQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.sqrt_d = math.sqrt(self.codebook_dim)
        self.total_codebook_size = args.codebook_size * args.codebook_size

    def calc_gaussian_mmd_loss(self, z, codebook):
        z = z.detach()
        c = codebook.weight
        N = z.size(0) + c.size(0)

        dxx = (torch.sum(z.detach()**2, dim=1, keepdim=True) + torch.sum(z.detach()**2, dim=1) - 2*torch.matmul(z.detach(), z.detach().t())).div(self.sqrt_d)
        dxy = (torch.sum(z.detach()**2, dim=1, keepdim=True) + torch.sum(c**2, dim=1) - 2*torch.matmul(z.detach(), c.t())).div(self.sqrt_d)
        dyy = (torch.sum(c**2, dim=1, keepdim=True) + torch.sum(c**2, dim=1) - 2*torch.matmul(c, c.t())).div(self.sqrt_d)
        bandwidth = (dxx.sum() + 2*dxy.sum() + dyy.sum()).detach() / (N**2 -N)

        pxx = -dxx / bandwidth
        pxy = -dxy / bandwidth
        pyy = -dyy / bandwidth

        XX = torch.exp(pxx).mean() + torch.exp(pxx/2).mean()
        XY = torch.exp(pxy).mean() + torch.exp(pxy/2).mean()
        YY = torch.exp(pyy).mean() + torch.exp(pyy/2).mean()

        mmd_loss = XX.detach() - 2 * XY + YY
        return mmd_loss

    def sub_quantizer(self, z_enc, codebook):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, codebook.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_dec = codebook(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        return z_dec, token

    def train_sub_quantizer(self, z_enc, codebook):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  
        mmd_loss = self.calc_gaussian_mmd_loss(z_flat.detach(), codebook)

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, codebook.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_dec = codebook(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        return z_dec, token, mmd_loss

    def forward(self, z_enc):
        z_enc_1, z_enc_2 = torch.chunk(z_enc, 2, dim=1)
        z_dec_1, token_1, mmd_loss_1 = self.train_sub_quantizer(z_enc_1, self.embedding_1)
        z_dec_2, token_2, mmd_loss_2 = self.train_sub_quantizer(z_enc_2, self.embedding_2)

        token = token_1 + token_2 * self.args.codebook_size
        z_dec = torch.cat((z_dec_1, z_dec_2), dim=1)
        commit_loss = self.beta * F.mse_loss(z_dec.detach(), z_enc) + self.alpha * F.mse_loss(z_dec, z_enc.detach())

        histogram = token.bincount(minlength=self.total_codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        utilization = codebook_usage_counts.item() / self.total_codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        z_dec = z_enc + (z_dec - z_enc).detach()
        loss = commit_loss + self.args.gamma * 0.5 * (mmd_loss_1 + mmd_loss_2)
        return z_dec, loss, utilization, perplexity

    def collect_eval_info(self, z_enc):
        z_enc_1, z_enc_2 = torch.chunk(z_enc, 2, dim=1)
        z_dec_1, token_1 = self.sub_quantizer(z_enc_1, self.embedding_1)
        z_dec_2, token_2 = self.sub_quantizer(z_enc_2, self.embedding_2)

        token = token_1 + token_2 * self.args.codebook_size
        z_dec = torch.cat((z_dec_1, z_dec_2), dim=1)

        histogram = token.bincount(minlength=self.total_codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return z_dec, histogram

    def collect_reconstruction(self, z_enc):
        z_enc_1, z_enc_2 = torch.chunk(z_enc, 2, dim=1)
        z_dec_1, _ = self.sub_quantizer(z_enc_1, self.embedding_1)
        z_dec_2, _ = self.sub_quantizer(z_enc_2, self.embedding_2)
        z_dec = torch.cat((z_dec_1, z_dec_2), dim=1)
        return z_dec