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

#### vector quantizer
class WassersteinVectorQuantizer(VectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def calc_wasserstein_loss(self, z=None):
        if z == None:
            z = self.queue.obtain_feature_from_queue()

        N = z.size(0)
        D = z.size(1)

        std = z.std(dim=0).max().detach()
        z = z / (std + 1e-8)
        z_mean = z.mean(0).detach()
        z_covariance = torch.cov(z.t()) + 1e-8 * torch.eye(D, device=z.device) 
        z_covariance = z_covariance.detach()

        ### compute the mean and covariance of codebook vectors
        c = self.embedding.weight /  (std + 1e-8)
        c_mean = c.mean(0)
        c_covariance = torch.cov(c.t()) + 1e-8 * torch.eye(D, device=z.device)
        
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
        # reshape z_enc -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        with torch.no_grad():
            self.queue.dequeue_and_enqueue(z_flat.detach())
        wasserstein_loss = self.calc_wasserstein_loss()

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flat.detach().pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flat.detach(), self.embedding.weight.data) # 'n d -> d n'
                
        token = torch.argmin(d, dim=1)
        z_dec = self.embedding(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        commit_loss = self.beta * F.mse_loss(z_dec.detach(), z_enc) +  self.alpha * F.mse_loss(z_dec, z_enc.detach())

        histogram = token.bincount(minlength=self.args.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        utilization = codebook_usage_counts.item() / self.args.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_dec = z_enc + (z_dec - z_enc).detach()
        loss = commit_loss + self.args.gamma * wasserstein_loss
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

#### product quantizer
class WassersteinProductQuantizer(ProductQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.total_codebook_size = args.codebook_size * args.codebook_size

    def calc_wasserstein_loss(self, z, codebook):
        N = z.size(0)
        D = z.size(1)

        std = z.std(dim=0).max().detach()
        z = z / (std + 1e-8)
        z_mean = z.mean(0).detach()
        z_covariance = torch.cov(z.t()) + 1e-8 * torch.eye(D, device=z.device) 
        z_covariance = z_covariance.detach()

        ### compute the mean and covariance of codebook vectors
        c = codebook.weight /  (std + 1e-8)
        c_mean = c.mean(0)
        c_covariance = torch.cov(c.t()) + 1e-8 * torch.eye(D, device=z.device)
        
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

    def train_sub_quantizer(self, z_enc, codebook, queue):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        with torch.no_grad():
            queue.dequeue_and_enqueue(z_flat.detach())
        z_queue = queue.obtain_feature_from_queue()
        wasserstein_loss = self.calc_wasserstein_loss(z_queue, codebook)

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, codebook.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_dec = codebook(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        return z_dec, token, wasserstein_loss

    def forward(self, z_enc):
        z_enc_1, z_enc_2 = torch.chunk(z_enc, 2, dim=1)
        z_dec_1, token_1, wasserstein_loss_1 = self.train_sub_quantizer(z_enc_1, self.embedding_1, self.queue_1)
        z_dec_2, token_2, wasserstein_loss_2 = self.train_sub_quantizer(z_enc_2, self.embedding_2, self.queue_2)

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
        loss = commit_loss + self.args.gamma * 0.5 * (wasserstein_loss_1 + wasserstein_loss_2)
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
